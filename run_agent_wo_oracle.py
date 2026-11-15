#!/usr/bin/env python
"""
MolLingo — lead optimization without oracle feedback (no-oracle ablation)
========================================================================

Runs stage 4 (lead optimization, §2.3.3) on a set of molecules with the
property oracle removed from the loop:

    * fixed budget: 3 iterations, 1 proposed modification per iteration
    * no ADMET/DILI prediction for the intermediate molecules. The Chemist
      Agent sees the BFE block decomposition of the current molecule and
      nothing else, proposes one block substitution, the molecule is
      reconstructed, and it is carried into the next iteration unconditionally
      -- improved or not, nobody checks.
    * the DILI oracle is called exactly once per molecule, at the end, on the
      original and on the final molecule, to measure what the blind chain
      achieved.

This is the ablation that isolates how much of MolLingo's gain comes from
oracle-in-the-loop selection as opposed to the LLM's chemical priors. The
full pipeline (run_agent.py) scores every enumerated variant each iteration
and only carries forward molecules that beat their parent; here nothing is
scored and nothing is rejected.

Usage
-----
One molecule, the same free-text form run_agent.py takes:
    python run_agent_wo_oracle.py "optimize dili for O=C(O)c1cc(-c2ccccc2)nc2ccccc12"
    python run_agent_wo_oracle.py --smiles "O=C(O)c1cc(-c2ccccc2)nc2ccccc12"

A whole set of molecules, one blind chain each (this is the batch form of the
single-molecule call above):
    python run_agent_wo_oracle.py --input results/DILI_gpt-5.4_top3_candidates_tp67.json
    python run_agent_wo_oracle.py            # the same 67 molecules, by default

Other inputs / budgets:
    python run_agent_wo_oracle.py --input mols.smi --iterations 3 --variants 1
    python run_agent_wo_oracle.py --limit 5 --jobs 1          # smoke test

The LLM calls dominate the runtime (iterations x molecules of them), so
molecules are optimized in parallel threads (--jobs) and the DILI oracle runs
once at the end, batched on the GPU. Partial results are written to --save
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

from agent import ChemistAgent, _canonical
from memory import SharedMemory
from toolset.ADMET import PredictDILI
from toolset.calculate_druggability import calculate_druggability
from toolset.mol_similarity import tanimoto_from_smiles
from toolset.sascorer import calculateScore
from run_agent import extract_smiles
from run_agent_1iter_smiles import RawSmilesAgent

RDLogger.DisableLog("rdApp.*")

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "DILI_gpt-5.4_top3_candidates_tp67.json",
)

# The profile line of the optimization prompt, in place of the oracle
# probabilities the full pipeline puts there.
NO_PROFILE_TEXT = (
    "not available — no property predictor is used in this run, so propose "
    "edits from medicinal-chemistry reasoning alone"
)


# ====================================================================== #
# The agent                                                              #
# ====================================================================== #

class BlindChemistAgent(ChemistAgent):
    """
    ChemistAgent that never consults the oracles while optimizing.

    Only the prompt is changed, so the block decomposition, the proposal
    parser, the attachment-point checks and the molecule reconstruction are
    the same code the full pipeline runs. Two differences, both deliberate:

      * the predicted-profile line is replaced by NO_PROFILE_TEXT instead of
        ADMET probabilities;
      * the "edits already explored" block of the prompt is rebuilt without
        outcomes. The parent writes "improved / did not improve" per edit,
        which is oracle feedback by another name; here the agent is told only
        which edits it has already applied, so it does not repeat itself.
    """

    def __init__(self, memory: Optional[SharedMemory] = None):
        super().__init__(memory)
        self.applied_edits: list = []

    def _optimization_prompt(self, smiles: str, structure: str,
                             profile_text: str, variants: int,
                             history: list) -> str:
        prompt = super()._optimization_prompt(
            smiles, structure, NO_PROFILE_TEXT, variants, history=[]
        )
        if self.applied_edits:
            lines = "\n".join(
                f"- iteration {record['iteration']}: {record['original']} "
                f"-> {record['substitute']}"
                for record in self.applied_edits
            )
            prompt += (
                "\nEdits already applied to this molecule earlier in the "
                "campaign. No property feedback is available for them, so "
                "they are listed only so you do not repeat one:\n"
                f"{lines}\n"
            )
        return prompt

    # -- raw-SMILES counterpart ----------------------------------------- #

    def optimize_blind_raw(self, smiles: str, iterations: int = 3,
                           variants: int = 1) -> dict:
        """
        The same blind chain over the raw SMILES string.

        The prompt and the proposal parser are `RawSmilesAgent.raw_prompt` /
        `propose_raw` from run_agent_1iter_smiles.py, reused verbatim so this
        arm differs from the block arm only in what the LLM is shown -- the
        same asymmetry documented there (the raw prompt states the one-edit
        restraint that MAX_BLOCKS_EDITED enforces in code) applies here too.
        No block decomposition and no reconstruction: the molecule the LLM
        writes is the molecule carried forward, after an RDKit validity check.
        """
        start = _canonical(smiles) or smiles
        current = start
        traces: list = []
        stopped = None

        for iteration in range(1, max(1, iterations) + 1):
            self.log("raw_representation",
                     f"{current} (raw SMILES, no block decomposition)")
            proposals = self.propose_raw(current, {}, variants, history=None)
            if not proposals:
                stopped = f"iteration {iteration}: no usable proposal"
                break

            chosen = proposals[0]
            if len(proposals) > 1:
                self.log("blind_pick_first",
                         f"Iteration {iteration}: {len(proposals)} molecules "
                         f"proposed, taking the first (no oracle to rank them).")
            traces.append({
                "round": iteration,
                "smiles_before": current,
                "smiles_after": chosen["smiles"],
                "original_block": "",
                "substitute_block": "",
                "rationale": chosen.get("rationale", ""),
            })
            self.log("blind_iteration_complete",
                     f"Iteration {iteration}: {current} -> {chosen['smiles']}; "
                     f"carried forward unscored.")
            current = chosen["smiles"]

        return {
            "smiles_before": start,
            "smiles_after": current,
            "rounds_completed": len(traces),
            "stopped_reason": stopped,
            "reasoning_traces": traces,
        }

    # -- one blind optimization chain ----------------------------------- #

    def optimize_blind(self, smiles: str, iterations: int = 3,
                       variants: int = 1) -> dict:
        """
        Apply `iterations` unguided block edits to `smiles`, one at a time.

        Returns {'smiles_before', 'smiles_after', 'rounds_completed',
        'reasoning_traces', 'stopped_reason'}. No property is predicted here;
        `smiles_after` is whatever the last successful edit produced (the
        input itself if no iteration completed).
        """
        start = _canonical(smiles) or smiles
        current = start
        traces: list = []
        stopped = None

        for iteration in range(1, max(1, iterations) + 1):
            decomposition = self._decompose(current)
            if decomposition is None:
                stopped = f"iteration {iteration}: BFE decomposition failed"
                break
            structure, _block_seq, blocks = decomposition

            # Empty profile: _propose_block_edits formats it into the prompt,
            # and BlindChemistAgent._optimization_prompt discards it anyway.
            proposals = self._propose_block_edits(
                current, structure, {}, variants, history=[]
            )
            if not proposals:
                stopped = f"iteration {iteration}: no usable proposal"
                break

            enumerated = self._enumerate_variants(current, blocks, proposals)
            if not enumerated:
                stopped = (f"iteration {iteration}: no molecule could be "
                           f"reconstructed")
                break

            # With variants=1 the agent proposes a single edit and this is a
            # single molecule. If a larger --variants produced several, the
            # first is taken: with no oracle there is nothing to rank them by,
            # and picking by any property would reintroduce the feedback this
            # ablation removes.
            chosen = enumerated[0]
            if len(enumerated) > 1:
                self.log("blind_pick_first",
                         f"Iteration {iteration}: {len(enumerated)} molecules "
                         f"enumerated, taking the first (no oracle to rank "
                         f"them).")

            original_block, substitute_block = chosen["edits"][0]
            rationale = self._edit_rationales.get(
                (original_block, substitute_block), ""
            )
            traces.append({
                "round": iteration,
                "smiles_before": current,
                "smiles_after": chosen["smiles"],
                "original_block": original_block,
                "substitute_block": substitute_block,
                "rationale": rationale,
            })
            self.applied_edits.append({
                "iteration": iteration,
                "original": original_block,
                "substitute": substitute_block,
            })
            self.log("blind_iteration_complete",
                     f"Iteration {iteration}: {current} -> {chosen['smiles']} "
                     f"({original_block} -> {substitute_block}); carried "
                     f"forward unscored.")
            current = chosen["smiles"]

        return {
            "smiles_before": start,
            "smiles_after": current,
            "rounds_completed": len(traces),
            "stopped_reason": stopped,
            "reasoning_traces": traces,
        }


class BlindRawSmilesAgent(BlindChemistAgent):
    """
    BlindChemistAgent shown the raw SMILES instead of its block decomposition.

    `raw_prompt` and `propose_raw` are taken from the oracle-in-the-loop raw
    arm (run_agent_1iter_smiles.RawSmilesAgent) unchanged, so the two
    representation arms ask the LLM for the same thing in the same words. Only
    the profile line differs: no predictor runs here, so NO_PROFILE_TEXT takes
    the place of the ADMET probabilities, exactly as in the block arm.
    """

    _raw_prompt_impl = RawSmilesAgent.raw_prompt
    propose_raw = RawSmilesAgent.propose_raw

    def raw_prompt(self, smiles: str, profile_text: str, variants: int,
                   history: Optional[list] = None) -> str:
        return BlindRawSmilesAgent._raw_prompt_impl(
            self, smiles, NO_PROFILE_TEXT, variants, None
        )


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

    Matched case-insensitively against the oracle names, so "DILI", "dili"
    and "hERG"/"herg" all land on the right endpoint. No LLM round-trip: the
    prompt this script takes is "optimize <endpoint> for <SMILES>", and
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
# Scoring (the single oracle call per molecule)                          #
# ====================================================================== #

def score_endpoints(records: list, batch_size: int = 32) -> None:
    """
    Fill the DILI before/after fields, in place, for records missing them.

    This is the only place the oracle is used: once per molecule, on the
    original and the final molecule of its chain. Everything is predicted in
    one batched pass so the checkpoint is loaded once.
    """
    pending = [r for r in records if r.get("dili_after") is None]
    if not pending:
        return

    unique, seen = [], set()
    for record in pending:
        for smiles in (record["smiles_before"], record["smiles_after"]):
            if smiles not in seen and Chem.MolFromSmiles(smiles) is not None:
                seen.add(smiles)
                unique.append(smiles)

    print(f"[MolLingo] Scoring DILI for {len(unique)} molecules "
          f"({len(pending)} pairs) ...")
    predictions = {}
    for start in range(0, len(unique), batch_size):
        chunk = unique[start:start + batch_size]
        for smiles, probability in zip(chunk, PredictDILI(chunk)):
            predictions[smiles] = float(probability)

    for record in pending:
        before = predictions.get(record["smiles_before"])
        after = predictions.get(record["smiles_after"])
        if before is None or after is None:
            continue
        record["dili_before"] = before
        record["dili_after"] = after
        record["dili_improvement"] = before - after
        record["dili_improvement_pct"] = 100.0 * (before - after) / before
        if record["smiles_after"] == record["smiles_before"]:
            continue
        try:
            record["tanimoto_similarity"] = float(
                tanimoto_from_smiles(record["smiles_before"],
                                     record["smiles_after"])
            )
            record["SA_after"] = float(
                calculateScore(Chem.MolFromSmiles(record["smiles_after"]))
            )
            record["QED_after"] = float(
                calculate_druggability(record["smiles_after"])
            )
        except Exception as exc:
            print(f"[WARNING] descriptors failed for "
                  f"{record['smiles_after']}: {exc}")



# ====================================================================== #
# Reporting                                                              #
# ====================================================================== #

def print_molecule(record: dict):
    """Report one blind chain: the edits, then original vs final DILI."""
    print()
    print("═" * 64)
    print("  MolLingo — optimization without oracle feedback")
    print("═" * 64)
    print(f"  Molecule   : {record['smiles_before']}")
    for trace in record["reasoning_traces"]:
        print(f"\n  [iteration {trace['round']}] "
              f"{trace['original_block']} -> {trace['substitute_block']}")
        print(f"    {trace['smiles_before']}")
        print(f"      -> {trace['smiles_after']}")
        if trace.get("rationale"):
            print(f"    Rationale : {trace['rationale']}")
    if record.get("stopped_reason"):
        print(f"\n  Stopped    : {record['stopped_reason']}")
    print(f"\n  Final      : {record['smiles_after']}  "
          f"({record['rounds_completed']} edit(s) applied, none scored)")
    if record.get("dili_after") is not None:
        print(f"  DILI       : {record['dili_before']:.4f} -> "
              f"{record['dili_after']:.4f}  "
              f"({record['dili_improvement']:+.4f}, "
              f"{record['dili_improvement_pct']:+.2f}%; lower is better)")
    if record.get("tanimoto_similarity") is not None:
        print(f"  Tanimoto   : {record['tanimoto_similarity']:.3f}   "
              f"SA {record['SA_after']:.2f}   "
              f"QED {record['QED_after']:.3f}")
    print("═" * 64)


def print_summary(records: list, iterations: int, variants: int):
    scored = [r for r in records if r.get("dili_after") is not None]
    print()
    print("═" * 64)
    print("  MolLingo — lead optimization without oracle feedback")
    print("═" * 64)
    print(f"  Budget        : {iterations} iteration(s) x {variants} "
          f"modification(s), no intermediate scoring")
    print(f"  Molecules     : {len(records)} "
          f"({len(scored)} scored, "
          f"{sum(1 for r in records if r['rounds_completed'] == iterations)} "
          f"completed all {iterations} rounds)")
    if not scored:
        print("  No scored molecules.")
        print("═" * 64)
        return

    improved = [r for r in scored if r["dili_improvement"] > 0]
    print(f"  DILI before   : {mean(r['dili_before'] for r in scored):.4f}")
    print(f"  DILI after    : {mean(r['dili_after'] for r in scored):.4f}")
    print(f"  Improvement   : {mean(r['dili_improvement'] for r in scored):+.4f}"
          f"   ({mean(r['dili_improvement_pct'] for r in scored):+.2f}%)")
    print(f"  Improved      : {len(improved)}/{len(scored)} "
          f"({100.0 * len(improved) / len(scored):.1f}%)")
    with_descriptors = [r for r in scored if r.get("tanimoto_similarity") is not None]
    if with_descriptors:
        print(f"  Tanimoto      : "
              f"{mean(r['tanimoto_similarity'] for r in with_descriptors):.4f}")
        print(f"  SA (after)    : "
              f"{mean(r['SA_after'] for r in with_descriptors):.4f}")
        print(f"  QED (after)   : "
              f"{mean(r['QED_after'] for r in with_descriptors):.4f}")
    print("═" * 64)


# ====================================================================== #
# CLI                                                                    #
# ====================================================================== #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MolLingo lead optimization with the oracle removed from "
                    "the loop (no intermediate property feedback).",
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
                             "results/DILI_no_oracle_3iter.json for a batch "
                             "run, and nothing written for a single molecule "
                             "given as a prompt or --smiles.")
    parser.add_argument("--iterations", type=int, default=3, metavar="N",
                        help="Blind optimization iterations per molecule "
                             "(default: 3).")
    parser.add_argument("--variants", type=int, default=1, metavar="N",
                        help="Modifications proposed per iteration "
                             "(default: 1).")
    parser.add_argument("--representation", default="blocks",
                        choices=["blocks", "raw_smiles"],
                        help="What the agent is shown: the BFE block "
                             "decomposition (default) or the raw SMILES "
                             "string. raw_smiles is the representation "
                             "ablation and reuses the prompt of "
                             "run_agent_1iter_smiles.py.")
    parser.add_argument("--task", default="DILI",
                        choices=["DILI", "hERG", "AMES", "Pgp", "HIA", "BBBP"],
                        help="Endpoint named in the prompt (default: DILI). "
                             "The final scoring below is DILI either way.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only the first N input molecules (smoke test).")
    parser.add_argument("--jobs", type=int, default=4, metavar="N",
                        help="Molecules optimized in parallel; each one is a "
                             "chain of LLM calls (default: 4).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep molecules already present in --save and "
                             "only run the missing ones.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-call agent trace.")
    parser.add_argument("--web-search", action="store_true",
                        help="Let the Chemist Agent use web search while "
                             "proposing block substitutions. Off by default "
                             "(toolset/call_gpt.py): these molecules are named "
                             "withdrawn drugs, so with search on part of the "
                             "measured gain can come from retrieving the "
                             "literature on the exact compound rather than "
                             "from reasoning over its blocks.")
    return parser


def main():
    args = build_parser().parse_args()

    # A prompt or --smiles means one molecule; otherwise the input file is a
    # batch of them, which is the same run repeated per molecule.
    single = args.smiles or extract_smiles(args.prompt)
    if args.prompt and not single:
        print(f"[ERROR] No valid SMILES in the prompt: {args.prompt!r}")
        print("  e.g. python run_agent_wo_oracle.py "
              "\"optimize dili for O=C(O)c1cc(-c2ccccc2)nc2ccccc12\"")
        sys.exit(1)

    if single:
        if Chem.MolFromSmiles(single) is None:
            print(f"[ERROR] '{single}' is not a valid SMILES.")
            sys.exit(1)
        source = "prompt" if args.prompt else "--smiles"
        molecules = [single]
        # An endpoint named in the prompt wins over the --task default, so
        # "optimize hERG for ..." does not silently optimize for DILI.
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

    # Saving is the point of a batch run and incidental for a single molecule,
    # whose result is printed either way.
    if args.save is None and not single:
        args.save = os.path.join("results", "DILI_no_oracle_3iter.json")

    done = load_previous(args.save) if (args.resume and args.save) else {}
    todo = [s for s in molecules if s not in done]

    print(f"[MolLingo] Lead optimization without oracle feedback")
    print(f"[MolLingo] Input      : {source}")
    print(f"[MolLingo] Molecules  : {len(molecules)}"
          + (f"  ({len(done)} already done, {len(todo)} to run)" if done else ""))
    print(f"[MolLingo] Budget     : {args.iterations} iterations x "
          f"{args.variants} modification(s), no intermediate "
          f"{args.task} prediction")
    print(f"[MolLingo] Represent. : "
          f"{'raw SMILES string' if args.representation == 'raw_smiles' else 'BFE block decomposition'}")
    print(f"[MolLingo] Oracle     : DILI, once per molecule, at the end")
    print(f"[MolLingo] Web search : "
          f"{'on' if args.web_search else 'off'} for the optimization prompts")
    print(f"[MolLingo] Parallelism: {args.jobs} molecule(s) at a time\n")

    # gpt_api defaults to no web search; the Chemist Agent's proposal call goes
    # through agent.gpt_api, so --web-search re-enables it for this run only
    # (and only for that call path) rather than changing the shared default.
    # Set the process-wide default rather than passing web_search per call:
    # every prompt in this run goes through gpt_api, and toolset/call_gpt.py
    # ships with WEB_SEARCH_DEFAULT = True for the full pipeline. A run that
    # patched only its own call path would inherit that default everywhere
    # else and report "web search off" while searching -- which is how an
    # earlier run of this script came out mislabelled.
    import toolset.call_gpt as call_gpt
    call_gpt.WEB_SEARCH_DEFAULT = bool(args.web_search)

    config = {
        "script": os.path.basename(__file__),
        "input": source,
        "task": args.task,
        "representation": args.representation,
        "web_search": bool(args.web_search),
        "iterations": args.iterations,
        "variants_per_iteration": args.variants,
        "selection": "none, every edit is kept",
        "intermediate_oracle_feedback": False,
        "memory_of_prior_edits": "edits only, no outcomes",
        "num_input_molecules": len(molecules),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    records = [done[s] for s in molecules if s in done]
    lock = threading.Lock()
    counter = {"n": 0}

    def run_one(smiles: str) -> dict:
        memory = SharedMemory()
        memory.main_task = args.task
        memory.user_prompt = (f"Optimize {smiles} for lower "
                              f"{args.task} risk (no oracle feedback)")
        raw = args.representation == "raw_smiles"
        agent = (BlindRawSmilesAgent if raw else BlindChemistAgent)(memory)
        if args.quiet:
            memory.log = lambda *a, **k: None   # type: ignore[assignment]
        optimize = agent.optimize_blind_raw if raw else agent.optimize_blind
        result = optimize(smiles, args.iterations, args.variants)
        result.update({
            "dili_before": None,
            "dili_after": None,
            "dili_improvement": None,
            "dili_improvement_pct": None,
            "tanimoto_similarity": None,
            "SA_after": None,
            "QED_after": None,
        })
        return result

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_one, s): s for s in todo}
        for future in as_completed(futures):
            smiles = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                print(f"[ERROR] {smiles}: {exc}")
                record = {
                    "smiles_before": smiles, "smiles_after": smiles,
                    "rounds_completed": 0, "stopped_reason": f"failed: {exc}",
                    "reasoning_traces": [], "dili_before": None,
                    "dili_after": None, "dili_improvement": None,
                    "dili_improvement_pct": None,
                    "tanimoto_similarity": None, "SA_after": None,
                    "QED_after": None,
                }
            with lock:
                records.append(record)
                counter["n"] += 1
                if args.save:
                    save_results(args.save, config, records)
                print(f"[MolLingo] {counter['n']}/{len(todo)} done "
                      f"({record['rounds_completed']} rounds): "
                      f"{record['smiles_before']} -> "
                      f"{record['smiles_after']}")

    print(f"\n[MolLingo] Optimization finished in "
          f"{(time.time() - started) / 60:.1f} min")

    # The one oracle pass: original vs final, for every molecule.
    score_endpoints(records)

    # Keep the output in input order, so it lines up with the input file.
    order = {smiles: i for i, smiles in enumerate(molecules)}
    records.sort(key=lambda r: order.get(r["smiles_before"], len(order)))
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.save:
        save_results(args.save, config, records)
        print(f"[MolLingo] Saved {len(records)} records to {args.save}")

    if single:
        print_molecule(records[0])
    else:
        print_summary(records, args.iterations, args.variants)


if __name__ == "__main__":
    main()
