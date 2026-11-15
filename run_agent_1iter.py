#!/usr/bin/env python
"""
MolLingo — lead optimization with a single-iteration budget (1-iteration ablation)
=================================================================================

Runs stage 4 (lead optimization, §2.3.3) on a set of molecules with the full
pipeline intact — the oracle stays in the loop and still selects — but the
search budget is spent all at once instead of being spread over rounds:

    * fixed budget: 1 iteration, 30 proposed modifications
    * every proposed molecule is scored by the ADMET oracles and only
      molecules beating the parent are kept, exactly as in run_agent.py
    * there is no second round, so nothing is carried forward and the shared
      memory of prior edits never comes into play: with one iteration there is
      no "prior" for the Chemist Agent to build on

Against the paper's 15 x 2 setting this isolates breadth from depth. Both
budgets cost ~30 proposals, so the comparison asks whether two rounds of 15
that refine a winner beat one round of 30 that cannot. `run_agent.py` with
--variants 15 --iterations 2 is the reference arm.

The budget is pinned deliberately: the orchestrator's own budget reasoning
(OrchestratorAgent.plan_optimization_budget) is bypassed, because a benchmark
arm has to be the same size for every molecule. That is why this script drives
the ChemistAgent directly rather than going through the orchestrator.

Usage
-----
One molecule, the same free-text form run_agent.py takes:
    python run_agent_1iter.py "optimize dili for O=C(O)c1cc(-c2ccccc2)nc2ccccc12"
    python run_agent_1iter.py --smiles "O=C(O)c1cc(-c2ccccc2)nc2ccccc12"

A whole set of molecules, one single-round optimization each:
    python run_agent_1iter.py --input results/DILI_gpt-5.4_top3_candidates_tp67.json
    python run_agent_1iter.py            # the same 67 molecules, by default

Other inputs / budgets:
    python run_agent_1iter.py --input mols.smi --variants 30 --iterations 1
    python run_agent_1iter.py --limit 5 --jobs 1          # smoke test

One LLM call per molecule dominates the wall clock, so molecules are optimized
in parallel threads (--jobs); oracle inference is serialized, since the six
ADMET models are shared torch modules. Partial results are written to --save
after every molecule; rerun with --resume to pick up where a run stopped.

Needs the project's environment (torch + torch_geometric + rdkit) and
gpt_key.json in the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Optional

from rdkit import Chem, RDLogger

from agent import ChemistAgent
from memory import SharedMemory
from run_agent import extract_smiles

RDLogger.DisableLog("rdApp.*")

# Evaluation runs are search-free. The full pipeline (run_agent.py) keeps web
# search on -- the Literature Agent needs it -- but for a benchmark arm the
# model must reason from the structure it is given: these molecules are named
# withdrawn drugs, and letting it look up their known hepatotoxicity would
# make the measured gain partly a retrieval result.
import toolset.call_gpt as call_gpt
call_gpt.WEB_SEARCH_DEFAULT = False

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "DILI_gpt-5.4_top3_candidates_tp67.json",
)

# Candidates recorded per molecule, matching the top-3 layout of the repo's
# existing DILI result files.
TOP_K = 3

# The ADMET oracles are module-level torch models shared by every thread, so
# inference is serialized while the LLM calls stay parallel. The LLM call is
# what the wall clock is made of; GPU inference is milliseconds by comparison.
_ORACLE_LOCK = threading.Lock()


class SerializedOracleAgent(ChemistAgent):
    """ChemistAgent whose oracle calls are safe to make from several threads."""

    def admet_profiles(self, smiles_list: list) -> dict:
        with _ORACLE_LOCK:
            return super().admet_profiles(smiles_list)


# ====================================================================== #
# Input / output                                                         #
# ====================================================================== #

def load_input_smiles(path: str) -> list:
    """
    Read the molecules to optimize.

    Accepts the repo's result files (a dict keyed by the parent SMILES, or
    {"molecules": [...]} / a list of records carrying "smiles_before") and
    plain .smi/.txt files with one SMILES per line.
    """
    if path.lower().endswith((".smi", ".txt", ".csv")):
        raw = []
        with open(path) as handle:
            for line in handle:
                token = line.strip().split()[0] if line.strip() else ""
                if token and not token.startswith("#"):
                    raw.append(token)
    else:
        with open(path) as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "molecules" in data:
            data = data["molecules"]
        if isinstance(data, dict):
            raw = list(data)
        elif isinstance(data, list):
            raw = [record.get("smiles_before") for record in data
                   if isinstance(record, dict)]
        else:
            raise ValueError(f"Unsupported input structure in {path}")

    molecules, seen = [], set()
    for smiles in raw:
        if not smiles or Chem.MolFromSmiles(smiles) is None:
            print(f"[WARNING] skipping unparseable SMILES: {smiles!r}")
            continue
        if smiles in seen:
            continue
        seen.add(smiles)
        molecules.append(smiles)
    return molecules


def task_from_prompt(prompt: Optional[str]) -> Optional[str]:
    """
    The endpoint named in a free-text prompt, e.g. "optimize dili for CCO".

    Matched case-insensitively against the oracle names. No LLM round-trip:
    guessing an endpoint the user did not name would silently change what is
    being optimized.
    """
    if not prompt:
        return None
    for endpoint in ("DILI", "hERG", "AMES", "Pgp", "HIA", "BBBP"):
        if re.search(rf"\b{endpoint}\b", prompt, flags=re.IGNORECASE):
            return endpoint
    return None


def save_results(path: str, config: dict, molecules: list):
    """Write the result file (atomically, so an interrupt cannot truncate it)."""
    payload = {"config": config, "molecules": molecules}
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def load_previous(path: str) -> dict:
    """Completed records from an earlier run of --save, keyed by input SMILES."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"[WARNING] could not read {path} for --resume: {exc}")
        return {}
    molecules = data.get("molecules", data) if isinstance(data, dict) else data
    done = {}
    for record in molecules or []:
        if isinstance(record, dict) and record.get("smiles_before"):
            done[record["smiles_before"]] = record
    return done


# ====================================================================== #
# One molecule                                                           #
# ====================================================================== #

def optimize_one(smiles: str, task: str, variants: int, iterations: int,
                 bbbp_desired: Optional[bool], use_memory: bool) -> dict:
    """
    Run the single-round optimization for one molecule.

    Returns a record carrying the top-K candidates in the field layout the
    repo's DILI result files use, so an arm from this script can be compared
    against them directly.
    """
    started = time.time()
    memory = SharedMemory()
    memory.main_task = task
    memory.bbbp_desired = bbbp_desired
    memory.add_lead(smiles, hit_origin=smiles, docking_score=0.0)

    agent = SerializedOracleAgent(memory)
    agent.optimize_leads(num_leads=1, variants=variants,
                         iterations=iterations, use_memory=use_memory)

    # The lead's own profile is recorded by optimize_leads, so the "before"
    # value does not need a second oracle call.
    lead_profile = (memory.leads.get(smiles, {}) or {}).get("admet_profile") or {}
    before = lead_profile.get(task)

    ranked = sorted(
        (item for item in memory.candidates.items()
         if item[1].get("admet_score") is not None),
        key=lambda item: item[1]["admet_score"], reverse=True,
    )[:TOP_K]

    candidates = []
    for rank, (candidate, entry) in enumerate(ranked, start=1):
        after = (entry.get("admet_profile") or {}).get(task)
        edits = entry.get("edits") or []
        candidates.append({
            "smiles_before": smiles,
            "smiles_after": candidate,
            "smiles_after_valid": Chem.MolFromSmiles(candidate) is not None,
            "rank": rank,
            "modification": "; ".join(
                f"{edit['original_block']} -> {edit['substitute_block']}"
                for edit in edits) or None,
            "rationale": " | ".join(
                str(edit.get("rationale") or "") for edit in edits).strip(" |") or None,
            f"{task}_before": before,
            f"{task}_after": after,
            f"{task.lower()}_reduction": (
                before - after if before is not None and after is not None else None),
            "admet_score": entry.get("admet_score"),
            "tanimoto_similarity": entry.get("similarity_to_lead"),
            "SA_after": entry.get("sa_score"),
            "QED_after": entry.get("qed"),
        })

    return {
        "smiles_before": smiles,
        f"{task}_before": before,
        "candidates": candidates,
        "n_candidates_total": len(memory.candidates),
        "seconds": round(time.time() - started, 1),
        "trace": [
            f"{record.get('agent')}/{record.get('step')}: {record.get('content')}"
            for record in getattr(memory, "trace", []) or []
        ],
    }


# ====================================================================== #
# Reporting                                                              #
# ====================================================================== #

def print_molecule(record: dict, task: str):
    before = record.get(f"{task}_before")
    print(f"\n  {record['smiles_before']}")
    print(f"    {task} before : "
          + (f"{before:.4f}" if before is not None else "unavailable")
          + f"    candidates: {record['n_candidates_total']}"
          + f"    {record['seconds']}s")
    if not record["candidates"]:
        print("    no candidate improved on the input molecule")
        return
    for candidate in record["candidates"]:
        after = candidate.get(f"{task}_after")
        similarity = candidate.get("tanimoto_similarity")
        print(f"    #{candidate['rank']}  {task} "
              + (f"{after:.4f}" if after is not None else "n/a")
              + (f"  (-{candidate[f'{task.lower()}_reduction']:.4f})"
                 if candidate.get(f"{task.lower()}_reduction") is not None else "")
              + (f"  sim {similarity:.3f}" if similarity is not None else "")
              + (f"  QED {candidate['QED_after']:.3f}"
                 if candidate.get("QED_after") is not None else "")
              + (f"  SA {candidate['SA_after']:.2f}"
                 if candidate.get("SA_after") is not None else ""))
        print(f"        {candidate['smiles_after']}")
        if candidate.get("modification"):
            print(f"        edit: {candidate['modification']}")


def print_summary(records: list, task: str, variants: int, iterations: int):
    width = 74
    print("\n" + "=" * width)
    print(f"  MolLingo — {iterations} iteration x {variants} modifications  ({task})")
    print("=" * width)

    improved, reductions = 0, []
    for record in records:
        best = record["candidates"][0] if record["candidates"] else None
        reduction = best.get(f"{task.lower()}_reduction") if best else None
        if reduction is not None and reduction > 0:
            improved += 1
            reductions.append(reduction)

    print(f"  Molecules       : {len(records)}")
    print(f"  Improved        : {improved}"
          + (f"  ({100.0 * improved / len(records):.1f}%)" if records else ""))
    if reductions:
        print(f"  Mean {task} drop  : {mean(reductions):.4f}"
              f"   (best {max(reductions):.4f})")
    similarities = [c["tanimoto_similarity"] for r in records
                    for c in r["candidates"][:1]
                    if c.get("tanimoto_similarity") is not None]
    if similarities:
        print(f"  Mean similarity : {mean(similarities):.3f}")
    total = sum(r.get("seconds", 0) for r in records)
    print(f"  Wall clock      : {total / 60:.1f} min of per-molecule work")
    print("=" * width)


# ====================================================================== #
# CLI                                                                    #
# ====================================================================== #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MolLingo lead optimization with a single-iteration "
                    "budget (1 iteration x 30 modifications).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Free-text prompt for a single molecule, e.g. "
                             "\"optimize dili for O=C(O)c1ccccc1\". The "
                             "SMILES and the endpoint are read out of it; "
                             "--input is ignored.")
    parser.add_argument("--smiles", default=None, metavar="SMILES",
                        help="Single molecule to optimize (same as giving it "
                             "in the prompt).")
    parser.add_argument("--input", default=DEFAULT_INPUT, metavar="PATH",
                        help="Molecules to optimize: a repo result JSON keyed "
                             "by parent SMILES, a {'molecules': [...]} file, "
                             "or one SMILES per line (.smi/.txt/.csv). "
                             "Default: the 67 DILI molecules in "
                             "results/DILI_gpt-5.4_top3_candidates_tp67.json")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="Where to write the results. Default: "
                             "results/DILI_1iter_30variants.json for a batch "
                             "run, and nothing written for a single molecule.")
    parser.add_argument("--variants", type=int, default=30, metavar="N",
                        help="Modifications proposed in the single round "
                             "(default: 30).")
    parser.add_argument("--iterations", type=int, default=1, metavar="N",
                        help="Optimization iterations per molecule "
                             "(default: 1 — this is the arm's definition).")
    parser.add_argument("--task", default="DILI",
                        choices=["DILI", "hERG", "AMES", "Pgp", "HIA", "BBBP"],
                        help="Endpoint to optimize (default: DILI).")
    parser.add_argument("--bbbp", default=None,
                        choices=["high", "low"],
                        help="Assess BBB penetration separately against this "
                             "direction. Omitted, the BBBP gate is inactive, "
                             "which is how the published DILI arms were run.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only the first N input molecules (smoke test).")
    parser.add_argument("--jobs", type=int, default=4, metavar="N",
                        help="Molecules optimized in parallel (default: 4).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep molecules already present in --save and "
                             "only run the missing ones.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-molecule agent trace.")
    return parser


def main():
    args = build_parser().parse_args()

    single = args.smiles or extract_smiles(args.prompt)
    if args.prompt and not single:
        print(f"[ERROR] No valid SMILES in the prompt: {args.prompt!r}")
        print("  e.g. python run_agent_1iter.py "
              "\"optimize dili for O=C(O)c1cc(-c2ccccc2)nc2ccccc12\"")
        sys.exit(1)

    if single:
        if Chem.MolFromSmiles(single) is None:
            print(f"[ERROR] '{single}' is not a valid SMILES.")
            sys.exit(1)
        source = "prompt" if args.prompt else "--smiles"
        molecules = [single]
        named_task = task_from_prompt(args.prompt)
        if named_task and named_task != args.task:
            args.task = named_task
    else:
        source = args.input
        molecules = load_input_smiles(args.input)

    if args.limit:
        molecules = molecules[:args.limit]
    if not molecules:
        print("[ERROR] No valid molecules in the input.")
        sys.exit(1)

    if args.save is None and not single:
        args.save = os.path.join("results", "DILI_1iter_30variants.json")

    bbbp_desired = {"high": True, "low": False}.get(args.bbbp)

    done = load_previous(args.save) if (args.resume and args.save) else {}
    todo = [s for s in molecules if s not in done]

    print("[MolLingo] Lead optimization, single-iteration budget")
    print(f"[MolLingo] Input      : {source}")
    print(f"[MolLingo] Molecules  : {len(molecules)}"
          + (f"  ({len(done)} already done, {len(todo)} to run)" if done else ""))
    print(f"[MolLingo] Budget     : {args.iterations} iteration(s) x "
          f"{args.variants} modifications, pinned (the orchestrator's own "
          f"budget reasoning is bypassed)")
    print(f"[MolLingo] Objective  : {args.task}, oracle in the loop")
    print(f"[MolLingo] BBBP gate  : "
          + ("inactive" if bbbp_desired is None
             else f"active, {'higher' if bbbp_desired else 'lower'} wanted"))
    print(f"[MolLingo] Web search : disabled for evaluation "
          f"(run_agent.py keeps it enabled)")
    print(f"[MolLingo] Parallelism: {args.jobs} molecule(s) at a time\n")

    config = {
        "script": os.path.basename(__file__),
        "input": source,
        "task": args.task,
        "variants": args.variants,
        "iterations": args.iterations,
        "bbbp_desired": bbbp_desired,
        "use_memory": args.iterations > 1,
        "top_k": TOP_K,
        "web_search": call_gpt.WEB_SEARCH_DEFAULT,
    }

    records = [done[s] for s in molecules if s in done]
    lock = threading.Lock()

    def run(smiles):
        try:
            return optimize_one(smiles, args.task, args.variants,
                                args.iterations, bbbp_desired,
                                use_memory=args.iterations > 1)
        except Exception as exc:
            print(f"[ERROR] {smiles}: {type(exc).__name__}: {exc}")
            return {"smiles_before": smiles, f"{args.task}_before": None,
                    "candidates": [], "n_candidates_total": 0, "seconds": 0,
                    "error": f"{type(exc).__name__}: {exc}", "trace": []}

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run, smiles): smiles for smiles in todo}
        for done_count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with lock:
                records.append(record)
                if not args.quiet:
                    print_molecule(record, args.task)
                print(f"    [{done_count}/{len(todo)}]")
                if args.save:
                    save_results(args.save, config, records)

    print_summary(records, args.task, args.variants, args.iterations)
    if args.save:
        print(f"  Results → {args.save}")


if __name__ == "__main__":
    main()
