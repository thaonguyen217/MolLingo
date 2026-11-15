#!/usr/bin/env python
"""
MolLingo — single-iteration lead optimization on raw SMILES (representation ablation)
====================================================================================

The same arm as run_agent_1iter.py — 1 iteration, 30 proposed modifications,
oracle in the loop — with the block-based representation removed. This is the
ablation that isolates MolLingo's central claim: that decomposing a molecule
into named building blocks helps an LLM reason about it better than the raw
string does.

What changes, and only this:

    * the Chemist Agent is shown the raw SMILES of the molecule. No BFE
      tokenization, no block sequence, no common chemical names, no corpus
      frequencies.
    * it proposes whole modified SMILES rather than block substitutions, so
      nothing is reconstructed from blocks -- the molecule the LLM writes is
      the molecule that gets scored, after an RDKit validity check.
    * it is instructed to change exactly one group per proposal. The block arm
      is restricted to a single block edit by MAX_BLOCKS_EDITED, enforced in
      code; a free-form rewrite has no such limit, so the restraint is stated
      in the prompt instead. Without it the arms are compared under different
      limits on how much of the molecule may change at once -- which is a
      confound, not a property of the representation. Any run before this was
      added is recorded with "single_edit_instruction" absent from its config.

Known asymmetry, deliberate and to be reported: the single-edit instruction
above has no counterpart in the block arm's prompt, because the block arm
enforces its one-edit limit in code (MAX_BLOCKS_EDITED). The two are not
exactly equivalent in strength -- one block can be a large fragment, whereas
"one functional group" is smaller -- so this arm is, if anything, held to the
tighter standard. Measured on 5 molecules it cost the raw arm about a third of
its DILI gain (0.183 -> 0.118) without raising similarity (0.737 -> 0.729).
Any comparison drawn from these two arms has to state that their prompts are
not identical.

What stays identical to run_agent_1iter.py:

    * budget: 1 iteration x 30 modifications, pinned
    * every proposal scored by the same ADMET oracles, and only molecules
      beating the parent are kept (ChemistAgent._evaluate_variants is reused
      verbatim, so selection, similarity, QED/SA and the BBBP gate behave the
      same way)
    * the same top-3 record layout, the same input set, the same resume logic

Usage
-----
One molecule, the same free-text form run_agent.py takes:
    python run_agent_1iter_smiles.py "optimize dili for O=C(O)c1cc(-c2ccccc2)nc2ccccc12"
    python run_agent_1iter_smiles.py --smiles "O=C(O)c1cc(-c2ccccc2)nc2ccccc12"

The benchmark set:
    python run_agent_1iter_smiles.py            # the 67 DILI molecules
    python run_agent_1iter_smiles.py --limit 5 --jobs 1       # smoke test

Pair it with `python run_agent_1iter.py` on the same input to get the
block-based arm; the two result files differ only in the representation the
Chemist Agent was given.

Needs the project's environment (torch + torch_geometric + rdkit) and
gpt_key.json in the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from rdkit import Chem, RDLogger

from agent import _canonical, _extract_structured, _scalar
from memory import SharedMemory
from toolset.call_gpt import gpt_api
from toolset.optimize_mol import smiles_post_processing
from run_agent import extract_smiles
# The shared benchmark scaffolding: only the representation differs, so the
# input loading, record layout, resume logic and reporting are the same.
from run_agent_1iter import (
    DEFAULT_INPUT,
    TOP_K,
    SerializedOracleAgent,
    load_input_smiles,
    load_previous,
    print_molecule,
    print_summary,
    save_results,
    task_from_prompt,
)

RDLogger.DisableLog("rdApp.*")

# Evaluation runs are search-free. The full pipeline (run_agent.py) keeps web
# search on -- the Literature Agent needs it -- but for a benchmark arm the
# model must reason from the structure it is given: these molecules are named
# withdrawn drugs, and letting it look up their known hepatotoxicity would
# make the measured gain partly a retrieval result.
import toolset.call_gpt as call_gpt
call_gpt.WEB_SEARCH_DEFAULT = False


class RawSmilesAgent(SerializedOracleAgent):
    """
    Chemist Agent that reasons over the raw SMILES string.

    Only the proposal step is replaced. Scoring, selection, the candidate
    records written to shared memory and the gates all come from
    ChemistAgent._evaluate_variants, so this differs from the block-based arm
    in what the LLM is shown and in nothing else.
    """

    def raw_prompt(self, smiles: str, profile_text: str, variants: int,
                   history: Optional[list] = None) -> str:
        """
        The block-based prompt's counterpart, with the SMILES in its place.

        Kept line-for-line equivalent to ChemistAgent._optimization_prompt
        apart from the representation and the output format, including the
        indication/target context and the record of edits already tried. Those
        two sections are empty for the default 1-iteration benchmark run, but
        omitting them would quietly deny the raw arm the memory the block arm
        gets at --iterations > 1, and the ablation would then be measuring two
        things at once.
        """
        task = getattr(self.memory, "main_task", "ADMET") or "ADMET"
        objectives = {
            "AMES": "lower Ames mutagenicity (reduced bacterial reverse mutation risk)",
            "DILI": "lower drug-induced liver injury risk",
            "hERG": "lower hERG inhibition (reduced cardiotoxicity risk)",
            "Pgp": "a lower P-glycoprotein inhibition profile",
            "HIA": "higher human intestinal absorption",
            "BBBP": ("higher blood-brain barrier permeability"
                     if getattr(self.memory, "bbbp_desired", True)
                     else "lower blood-brain barrier permeability"),
        }
        objective = objectives.get(task, f"an improved {task} profile")

        context = []
        if self.memory.disease:
            context.append(f"intended indication: {self.memory.disease}")
        if self.memory.protein_name:
            context.append(
                f"target: {self.memory.protein_name}"
                + (f" (UniProt: {self.memory.uniprot_id})"
                   if self.memory.uniprot_id else ""))
        context_text = ("This molecule is a candidate for "
                        + "; ".join(context) + ".\n") if context else ""

        memory_text = ""
        if history:
            lines = []
            for record in history[-40:]:
                outcome = ("improved" if record.get("improved")
                           else "did not improve")
                best = record.get("best_score")
                lines.append(
                    f"- iteration {record['iteration']}: "
                    f"{record['original']} -> {record['substitute']} "
                    f"({outcome}"
                    + (f", best score {best:.3f}" if best is not None else "")
                    + f"). Rationale given: {record.get('rationale', '')}")
            memory_text = (
                "\nEdits already explored in this campaign — do not propose "
                "any of them again, and move away from directions the oracles "
                "did not support:\n" + "\n".join(lines) + "\n")

        return (
            "You are an expert medicinal chemist optimizing a lead compound.\n"
            f"{context_text}"
            f"The molecule is given as a SMILES string:\n{smiles}\n\n"
            f"Predicted ADMET profile (probabilities): {profile_text}\n\n"
            f"Goal: modify the molecule for {objective}, while preserving as "
            "much of the original structure as possible (aim for Tanimoto "
            f"similarity above {self.MIN_SIMILARITY}).\n"
            f"{memory_text}"
            f"\nPropose {variants} structural modifications. For each one give:\n"
            "  1. Modified SMILES: the complete SMILES of the modified "
            "molecule, not a fragment.\n"
            "  2. Rationale: why this change should improve the stated "
            "objective.\n"
            # The block arm is held to one block per variant by
            # MAX_BLOCKS_EDITED, enforced in code. Nothing constrains a
            # free-form rewrite, so the equivalent restraint has to be stated
            # here; otherwise the two arms are compared under different limits
            # on how much of the molecule may change at once.
            "- Change exactly ONE functional group, substituent or ring "
            "position per proposal, and leave the rest of the molecule "
            "character-for-character identical to the input. Do not rewrite "
            "the scaffold, do not renumber rings, and do not restructure the "
            "SMILES string.\n"
            "- Every molecule must be a valid, synthetically accessible "
            "structure.\n"
            "- In the conclusion, return only a Python list of dictionaries:\n"
            "```python\n"
            "[\n"
            "  {\n"
            "    'Modified SMILES': 'Cc1cccc(F)c1',\n"
            "    'Rationale': 'Replacing the methyl with fluorine ...'\n"
            "  }\n"
            "]\n"
            "```\n"
        )

    def propose_raw(self, smiles: str, profile: dict, variants: int,
                    history: Optional[list] = None) -> list:
        """Whole modified molecules from the LLM, validated and de-duplicated."""
        profile_text = ", ".join(
            f"{k}: {v:.3f}" for k, v in (profile or {}).items()
        ) or "unavailable"

        try:
            response = gpt_api(
                self.raw_prompt(smiles, profile_text, variants, history))
        except Exception as exc:
            self.log("llm_call_failed", exc)
            return []

        parsed = _extract_structured(response)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            self.log("proposal_parse_failed",
                     f"Unparseable response for {smiles}: {str(response)[:200]}")
            return []

        proposals, seen = [], {_canonical(smiles)}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            candidate = _scalar(item.get("Modified SMILES")
                                or item.get("modified_smiles")
                                or item.get("SMILES"))
            if not isinstance(candidate, str):
                continue
            # The same shorthand repair the block arm applies to a substitute
            # block, so a malformed reply is not counted against either arm.
            candidate = smiles_post_processing(candidate.strip())
            canonical = _canonical(candidate)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            proposals.append({
                "smiles": canonical,
                "rationale": _scalar(item.get("Rationale")
                                     or item.get("rationale")) or "",
                # No block edit took place; the field is kept so the records
                # from the two arms have the same shape.
                "edits": [],
            })
        return proposals

    def optimize_raw(self, lead: str, variants: int, iterations: int) -> list:
        """
        The stage-4 loop over raw SMILES.

        Mirrors ChemistAgent._optimize_lead, minus the decomposition step, and
        delegates scoring and recording to the shared _evaluate_variants.
        """
        profiles = self.admet_profiles([lead])
        parent_profile = profiles.get(lead, {})
        parent_score = self.admet_score(parent_profile)
        if parent_score is None:
            self.log("optimize_lead_skipped", f"Could not profile lead {lead}.")
            return []

        lead_entry = self.memory.leads.get(lead)
        if lead_entry is not None:
            lead_entry["admet_profile"] = parent_profile
            lead_entry["admet_score"] = parent_score

        endpoint = self._main_task
        self.log("lead_baseline",
                 f"{lead}: {self.memory.main_task} score {parent_score:.3f}"
                 + (f", {endpoint} {parent_profile[endpoint]:.4f}"
                    if endpoint and endpoint in parent_profile else ""))
        self.log("raw_representation", f"{lead} (raw SMILES, no block decomposition)")

        current, current_score, current_profile = lead, parent_score, parent_profile
        produced, history = [], []

        for iteration in range(1, max(1, iterations) + 1):
            proposals = self.propose_raw(current, current_profile, variants,
                                         history)
            if not proposals:
                self.log("optimize_lead_stopped",
                         f"Iteration {iteration}: no usable proposals for {current}.")
                break
            self.log("variants_enumerated",
                     f"Iteration {iteration}: {len(proposals)} proposed "
                     f"molecule(s) parsed as valid SMILES")

            improved = self._evaluate_variants(
                proposals, lead, current, current_score, iteration)
            produced.extend(improved)

            # Keep the LLM's reasoning on the record, as the block arm does.
            for proposal in proposals:
                entry = self.memory.candidates.get(proposal["smiles"])
                if entry is not None and proposal["rationale"]:
                    entry["rationale"] = proposal["rationale"]

            # The block arm records what each edit achieved and replays it in
            # the next prompt; the whole-molecule equivalent is the rewrite
            # itself, from the current molecule to the proposed one.
            scored = {item["smiles"]: item["score"] for item in improved}
            for proposal in proposals:
                history.append({
                    "iteration": iteration,
                    "original": current,
                    "substitute": proposal["smiles"],
                    "improved": proposal["smiles"] in scored,
                    "best_score": scored.get(proposal["smiles"]),
                    "rationale": proposal["rationale"],
                })

            if not improved:
                self.log("optimize_lead_stopped",
                         f"Iteration {iteration}: no improvement on "
                         f"{current_score:.3f}.")
                break
            current, current_score = self._next_seed(improved)
            current_profile = self.memory.candidates[current]["admet_profile"]
            self.log("iteration_complete",
                     f"Iteration {iteration}: {len(improved)} improved "
                     f"candidates; carrying forward {current} "
                     f"({current_score:.3f})")

        return produced


def optimize_one(smiles: str, task: str, variants: int, iterations: int,
                 bbbp_desired: Optional[bool]) -> dict:
    """One molecule, raw-SMILES representation. Mirrors run_agent_1iter."""
    started = time.time()
    memory = SharedMemory()
    memory.main_task = task
    memory.bbbp_desired = bbbp_desired
    memory.add_lead(smiles, hit_origin=smiles, docking_score=0.0)

    agent = RawSmilesAgent(memory)
    agent.log("optimize_leads",
              f"Optimizing 1 lead for {task} over {iterations} iteration(s), "
              f"{variants} proposed molecules each (raw SMILES)")
    agent.optimize_raw(smiles, variants=variants, iterations=iterations)

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
        candidates.append({
            "smiles_before": smiles,
            "smiles_after": candidate,
            "smiles_after_valid": Chem.MolFromSmiles(candidate) is not None,
            "rank": rank,
            "modification": None,          # whole-molecule rewrite, not a block edit
            "rationale": entry.get("rationale"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MolLingo single-iteration lead optimization on raw "
                    "SMILES (block representation ablated).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Free-text prompt for a single molecule.")
    parser.add_argument("--smiles", default=None, metavar="SMILES",
                        help="Single molecule to optimize.")
    parser.add_argument("--input", default=DEFAULT_INPUT, metavar="PATH",
                        help="Molecules to optimize (same formats as "
                             "run_agent_1iter.py). Default: the 67 DILI "
                             "molecules.")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="Where to write the results. Default: "
                             "results/DILI_1iter_30variants_smiles.json for a "
                             "batch run.")
    parser.add_argument("--variants", type=int, default=30, metavar="N",
                        help="Molecules proposed in the single round "
                             "(default: 30).")
    parser.add_argument("--iterations", type=int, default=1, metavar="N",
                        help="Optimization iterations (default: 1).")
    parser.add_argument("--task", default="DILI",
                        choices=["DILI", "hERG", "AMES", "Pgp", "HIA", "BBBP"],
                        help="Endpoint to optimize (default: DILI).")
    parser.add_argument("--bbbp", default=None, choices=["high", "low"],
                        help="Assess BBB penetration separately against this "
                             "direction. Omitted, the gate is inactive.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only the first N input molecules.")
    parser.add_argument("--jobs", type=int, default=4, metavar="N",
                        help="Molecules optimized in parallel (default: 4).")
    parser.add_argument("--resume", action="store_true",
                        help="Only run molecules missing from --save.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-molecule trace.")
    return parser


def main():
    args = build_parser().parse_args()

    single = args.smiles or extract_smiles(args.prompt)
    if args.prompt and not single:
        print(f"[ERROR] No valid SMILES in the prompt: {args.prompt!r}")
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
        args.save = os.path.join("results", "DILI_1iter_30variants_smiles.json")

    bbbp_desired = {"high": True, "low": False}.get(args.bbbp)
    done = load_previous(args.save) if (args.resume and args.save) else {}
    todo = [s for s in molecules if s not in done]

    print("[MolLingo] Lead optimization on RAW SMILES (block representation ablated)")
    print(f"[MolLingo] Input      : {source}")
    print(f"[MolLingo] Molecules  : {len(molecules)}"
          + (f"  ({len(done)} already done, {len(todo)} to run)" if done else ""))
    print(f"[MolLingo] Budget     : {args.iterations} iteration(s) x "
          f"{args.variants} proposed molecules, pinned")
    print(f"[MolLingo] Objective  : {args.task}, oracle in the loop")
    print(f"[MolLingo] Represent. : raw SMILES string, no blocks or names")
    print(f"[MolLingo] Web search : disabled for evaluation "
          f"(run_agent.py keeps it enabled)")
    print(f"[MolLingo] Parallelism: {args.jobs} molecule(s) at a time\n")

    config = {
        "script": os.path.basename(__file__),
        "representation": "raw_smiles",
        "single_edit_instruction": True,
        "input": source,
        "task": args.task,
        "variants": args.variants,
        "iterations": args.iterations,
        "bbbp_desired": bbbp_desired,
        "top_k": TOP_K,
        "web_search": call_gpt.WEB_SEARCH_DEFAULT,
    }

    records = [done[s] for s in molecules if s in done]
    lock = threading.Lock()

    def run(smiles):
        try:
            return optimize_one(smiles, args.task, args.variants,
                                args.iterations, bbbp_desired)
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
