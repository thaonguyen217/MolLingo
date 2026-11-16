#!/usr/bin/env python
"""
MolLingo — Autonomous LLM Medicinal Chemist Agent
==============================================
Runs the full drug discovery pipeline from a free-text research prompt.

Usage
-----
Interactive:
    python run_agent.py

One-liner:
    python run_agent.py "Design a novel drug for Alzheimer's disease"

With overrides:
    python run_agent.py "Treat Parkinson's targeting LRRK2" \\
        --task BBBP --hits 20 --save results.json

Skip to lead optimization (known target, known hit):
    python run_agent.py "Optimize imatinib for CML" \\
        --uniprot P00519 --protein "ABL1" --task DILI --no-fragment-growing
"""

import argparse
import re
import sys
from typing import Optional

from rdkit import Chem
import textwrap
from ast import literal_eval

from toolset.call_gpt import gpt_api
from toolset.tokenizer import get_blocks
from agent import ChemistAgent
from planner import OrchestratorAgent, ReActOrchestrator
from memory import SharedMemory


# ------------------------------------------------------------------ #
# Prompt parser                                                        #
# ------------------------------------------------------------------ #

_PARSE_PROMPT = """\
You are a biomedical expert assistant. Extract structured drug discovery \
parameters from the research prompt below.

Rules:
- Return ONLY a valid Python dictionary literal — no explanation, no markdown.
- Keys must be exactly as shown in the example.
- For main_task: choose the most relevant single endpoint from \
[ADMET, DILI, hERG, AMES, Pgp, HIA, BBBP]. \
Use ADMET (overall) unless the prompt specifically mentions a single endpoint.
- For bbbp_desired: set True for CNS/brain diseases (Alzheimer, Parkinson, \
depression, epilepsy, schizophrenia, glioma, MS …), False otherwise.
- For smiles: copy the SMILES string verbatim if the prompt supplies a specific \
molecule to optimize, otherwise None. Do not invent or complete one.
- Use None for any field not mentioned or inferable.

Example output:
{{
    "disease": "Alzheimer's disease",
    "disease_subtype": None,
    "uniprot_id": None,
    "protein_name": "GSK-3β",
    "gene_mutation": None,
    "pathway": None,
    "main_task": "BBBP",
    "bbbp_desired": True,
    "smiles": None
}}

Research prompt: {prompt}
"""


def extract_smiles(prompt: str) -> Optional[str]:
    """
    The longest token in the prompt that RDKit reads as a molecule.

    The GPT parser is asked for a SMILES too, but a prompt that names a
    molecule explicitly should not depend on an LLM round-trip, and whatever
    either route returns has to parse before the pipeline treats it as a lead.
    """
    best = None
    for token in re.split(r"[\s,;]+", prompt or ""):
        token = token.strip(".?!\"\'()").strip()
        if len(token) < 5 or not re.search(r"[A-Za-z]", token):
            continue
        if Chem.MolFromSmiles(token) is None:
            continue
        if best is None or len(token) > len(best):
            best = token
    return best


def parse_prompt(prompt: str) -> dict:
    """Use GPT to extract structured pipeline parameters from a free-text prompt."""
    response = gpt_api(_PARSE_PROMPT.format(prompt=prompt))

    # Try to pull out the first dict literal from the response
    try:
        # Strip markdown fences if present
        text = response
        for fence in ["```python", "```json", "```"]:
            if fence in text:
                text = text.split(fence)[1].split("```")[0]
                break
        params = literal_eval(text.strip())
        assert isinstance(params, dict)
        return params
    except Exception:
        print("[WARNING] Could not parse GPT response as a dict. Using defaults.")
        print(f"  Raw response: {response[:300]}")
        return {}


# ------------------------------------------------------------------ #
# Result printer                                                       #
# ------------------------------------------------------------------ #

def _divider(char="─", width=60):
    print(char * width)

def print_summary(memory: SharedMemory):
    # The orchestrator records this when the prompt supplied the molecule, in
    # which case stages 2 and 3 never ran and reporting "0 hits" would read as
    # a failed search rather than a step that was not needed.
    given_molecule = any(
        record.get("step") == "entering_at_lead_optimization"
        for record in getattr(memory, "trace", []) or []
    )

    _divider("═")
    print("  MolLingo  —  Pipeline Summary")
    _divider("═")

    # Research context
    if memory.user_prompt:
        prompt_short = textwrap.shorten(memory.user_prompt, width=72, placeholder="…")
        print(f"  Prompt    : {prompt_short}")
    if memory.disease:
        print(f"  Disease   : {memory.disease}"
              + (f"  [{memory.disease_subtype}]" if memory.disease_subtype else ""))
    if memory.gene_mutation:
        print(f"  Mutation  : {memory.gene_mutation}")
    if memory.pathway:
        print(f"  Pathway   : {memory.pathway}")

    # Target
    if memory.protein_name:
        print(f"  Target    : {memory.protein_name}  (UniProt: {memory.uniprot_id})")
    elif not given_molecule:
        print("  Target    : not identified")
    if memory.target_reference:
        ref = textwrap.shorten(memory.target_reference, width=70, placeholder="…")
        print(f"  Reference : {ref}")

    # Optimization goal
    print(f"  Task      : {memory.main_task}"
          + (f"  |  BBBP desired: {memory.bbbp_desired}"
             if memory.bbbp_desired is not None else ""))
    _divider()

    # Stage 2 — Hits
    if given_molecule and not memory.hits:
        print("\n  [Stage 2] Hit Identification  —  not needed "
              "(molecule supplied in the prompt)")
    else:
        print(f"\n  [Stage 2] Hit Identification  —  {len(memory.hits)} hits")
    best_hit = memory.best_hit()
    if best_hit:
        h = memory.hits[best_hit]
        print(f"    Best hit : {best_hit}")
        print(f"    Docking  : {h['docking_score']}")
        if h['admet_profile']:
            admet_str = "  ".join(f"{k}: {v:.2f}" for k, v in h['admet_profile'].items())
            print(f"    ADMET    : {admet_str}")
    elif not given_molecule:
        print("    No hits found.")

    # Stage 3 — Leads
    if given_molecule:
        print("\n  [Stage 3] Hit-to-Lead  —  not needed "
              "(the supplied molecule is the lead)")
    else:
        print(f"\n  [Stage 3] Hit-to-Lead  —  {len(memory.leads)} leads")
    best_lead = memory.best_lead()
    if best_lead:
        ld = memory.leads[best_lead]
        print(f"    Best lead : {best_lead}")
        if ld.get("docking_score"):      # 0.0 is the seeding placeholder
            print(f"    Docking   : {ld['docking_score']}")
        # Both only exist when fragment growing actually produced the lead.
        if ld.get('fragment'):
            print(f"    Fragment  : {ld['fragment']}")
        if ld.get('rationale'):
            rationale = textwrap.shorten(ld['rationale'], width=80, placeholder="…")
            print(f"    Rationale : {rationale}")
    else:
        print("    No leads generated.")

    # Stage 4 — Candidates
    print(f"\n  [Stage 4] Lead Optimization  —  {len(memory.candidates)} candidates")
    best_cand = memory.best_candidate()
    if best_cand:
        cd = memory.candidates[best_cand]
        lead = cd.get("lead_origin")
        print(f"    Lead           : {lead}")
        print(f"    Best candidate : {best_cand}")
        print(f"    ADMET score    : {cd['admet_score']:.3f}")

        # The endpoint actually being optimized, before and after. The lead's
        # profile is recorded when optimization starts.
        endpoint = memory.main_task if memory.main_task != "ADMET" else None
        lead_profile = (memory.leads.get(lead, {}) or {}).get("admet_profile") or {}
        if endpoint and endpoint in lead_profile and endpoint in (cd['admet_profile'] or {}):
            before, after = lead_profile[endpoint], cd['admet_profile'][endpoint]
            direction = "lower is better" if endpoint != "HIA" else "higher is better"
            print(f"    {endpoint} before/after : {before:.4f} -> {after:.4f}  "
                  f"({after - before:+.4f}, {direction})")
        if cd['admet_profile']:
            admet_str = "  ".join(f"{k}: {v:.2f}" for k, v in cd['admet_profile'].items())
            print(f"    ADMET profile  : {admet_str}")

        # Binding is only meaningful against a target, so the whole block is
        # omitted when none was identified.
        lead_dock = (memory.leads.get(lead, {}) or {}).get("docking_score")
        cand_dock = cd.get("docking_score")
        if memory.uniprot_id and lead_dock:
            target = memory.protein_name or memory.uniprot_id
            print(f"    Docking vs {target} (kcal/mol, lower binds stronger):")
            print(f"      lead         : {lead_dock}")
            if cand_dock is None:
                print("      candidate    : not docked")
            else:
                delta = cand_dock - lead_dock
                verdict = ("stronger" if delta < 0 else
                           "weaker" if delta > 0 else "unchanged")
                print(f"      candidate    : {cand_dock}")
                print(f"      difference   : {delta:+.2f}  ({verdict} binding; "
                      f"candidates losing more than "
                      f"{ChemistAgent.MAX_DOCKING_LOSS} were not proposed)")

        similarity = cd.get("similarity_to_lead")
        if similarity is not None:
            print(f"    Tanimoto to lead : {similarity:.3f}  "
                  f"(difference {1.0 - similarity:.3f})")

        # BBB penetration is reported separately from the ADMET score, which
        # excludes it by design.
        lead_bbbp = lead_profile.get("BBBP")
        cand_bbbp = (cd['admet_profile'] or {}).get("BBBP")
        if memory.bbbp_desired is not None and lead_bbbp is not None and cand_bbbp is not None:
            wanted = "higher" if memory.bbbp_desired else "lower"
            print(f"    BBBP before/after : {lead_bbbp:.4f} -> {cand_bbbp:.4f}  "
                  f"({cand_bbbp - lead_bbbp:+.4f}; {wanted} is wanted for this "
                  f"indication, excluded from the ADMET score)")

        qed, sa = cd.get("qed"), cd.get("sa_score")
        if qed is not None or sa is not None:
            parts = []
            if qed is not None:
                parts.append(f"QED {qed:.3f} (drug-likeness, higher is better)")
            if sa is not None:
                parts.append(f"SA {sa:.2f} (synthetic accessibility, 1 easy - 10 hard)")
            print("    " + "   ".join(parts))
    else:
        print("    No candidate survived. The counts below show what was tried.")

    # Always report what stage 4 did, especially when it produced nothing: a
    # bare "0 candidates" cannot distinguish "the LLM proposed nothing usable"
    # from "everything improved but the gates rejected it".
    stats = getattr(memory, "optimization_stats", None)
    if stats:
        print(f"    Leads attempted   : {stats['leads_attempted']}")
        print(f"    Iterations run    : {stats['iterations_run']}")
        print(f"    Variants built    : {stats['variants_enumerated']}")
        print(f"    Variants scored   : {stats['variants_scored']}")
        print(f"    Beat the parent   : {stats['improved_before_gates']}")
        rejected = (f"{stats['rejected_implausible']} impossible, "
                    f"{stats['rejected_bbbp']} on BBBP drift, "
                    f"{stats['rejected_binding']} on binding loss")
        print(f"    Rejected          : {rejected}")
        if (stats['improved_before_gates'] and not stats['candidates_kept']):
            print("    -> every molecule that improved the objective was "
                  "rejected by a gate; loosen ChemistAgent.MAX_DOCKING_LOSS "
                  "or MAX_BBBP_DRIFT to admit them.")

    _divider("═")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MolLingo: Autonomous LLM Medicinal Chemist Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "prompt", nargs="?", default=None,
        help="Free-text research prompt (prompted interactively if omitted).",
    )

    # Manual overrides — all optional
    p.add_argument("--smiles", default=None, metavar="SMILES",
                   help="Molecule to optimize. Also picked up from the prompt, "
                        "e.g. \"optimize DILI for CC(=O)Oc1ccccc1C(=O)O\".")
    p.add_argument("--disease", default=None, help="Disease name (overrides parsed value).")
    p.add_argument("--subtype", default=None, help="Disease subtype.")
    p.add_argument("--uniprot", default=None, metavar="ID",
                   help="UniProt ID (skips target identification).")
    p.add_argument("--protein", default=None, metavar="NAME",
                   help="Protein name (required with --uniprot).")
    p.add_argument("--mutation", default=None, metavar="MUT",
                   help="Gene mutation context (e.g. 'GBA1 loss-of-function').")
    p.add_argument("--pathway", default=None, help="Pathway context.")
    p.add_argument("--task", default=None,
                   choices=["ADMET", "DILI", "hERG", "AMES", "Pgp", "HIA", "BBBP"],
                   help="ADMET optimization target (default: ADMET).")
    p.add_argument("--no-bbbp", action="store_true",
                   help="Set bbbp_desired=False (peripheral/non-CNS drug).")

    # Pipeline scale
    p.add_argument("--hits", type=int, default=30, metavar="N",
                   help="Number of ChEMBL hit candidates to retrieve (default: 30).")
    p.add_argument("--growing-hits", type=int, default=5, metavar="N",
                   help="Number of top hits to run fragment growing on (default: 5).")
    p.add_argument("--frags", type=int, default=5, metavar="N",
                   help="GPT fragment proposals per hotspot (default: 5).")
    p.add_argument("--leads", type=int, default=3, metavar="N",
                   help="Number of leads to pass to lead optimization (default: 3).")
    p.add_argument("--variants", type=int, default=15, metavar="N",
                   help="Block substitution modifications proposed per lead "
                        "optimization iteration (default: 15, the setting "
                        "reported in the paper).")
    p.add_argument("--iterations", type=int, default=2, metavar="N",
                   help="Lead optimization iterations (default: 2, the "
                        "setting reported in the paper).")
    p.add_argument("--plan-budget", action="store_true",
                   help="Let the orchestrator choose the number of "
                        "modifications and iterations from the lead's "
                        "headroom instead of using the fixed 15 x 2. This is "
                        "an extension beyond the paper, which fixes both "
                        "values for every reported experiment; it overrides "
                        "--variants and --iterations.")
    p.add_argument("--no-fragment-growing", action="store_true",
                   help="Skip fragment growing and go directly to lead optimization.")
    p.add_argument("--react", action="store_true",
                   help="Run the Orchestrator as the Appendix J ReAct loop: it "
                        "reasons about each next action and selects the agent or "
                        "tool itself, instead of following the fixed stage order.")

    # Output
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Save full memory (hits, leads, candidates, trace) to a JSON file.")

    return p


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Get prompt
    prompt = args.prompt
    if not prompt:
        print("MolLingo — Autonomous LLM Medicinal Chemist Agent")
        print("Enter your research prompt (e.g. 'Design a drug for Alzheimer's disease'):")
        prompt = input("> ").strip()
        if not prompt:
            print("No prompt provided. Exiting.")
            sys.exit(1)

    print(f"\n[MolLingo] Research prompt: \"{prompt}\"")
    print("[MolLingo] Parsing prompt ...\n")

    # Parse prompt with GPT
    params = parse_prompt(prompt)

    # Apply CLI overrides (explicit args always win over GPT-parsed values)
    if args.disease:
        params["disease"] = args.disease
    if args.subtype:
        params["disease_subtype"] = args.subtype
    if args.uniprot:
        params["uniprot_id"] = args.uniprot
    if args.protein:
        params["protein_name"] = args.protein
    if args.mutation:
        params["gene_mutation"] = args.mutation
    if args.pathway:
        params["pathway"] = args.pathway
    if args.task:
        params["main_task"] = args.task
    if args.no_bbbp:
        params["bbbp_desired"] = False

    # A molecule named in the prompt is resolved before validation, because it
    # changes what the run needs: optimizing a given molecule for a stated
    # endpoint is complete on its own, whereas discovering one from scratch
    # requires a disease to find a target for.
    molecule = args.smiles or params.get("smiles") or extract_smiles(prompt)
    if molecule and Chem.MolFromSmiles(molecule) is None:
        print(f"[WARNING] '{molecule}' is not a valid SMILES; ignoring it.\n")
        molecule = None

    if not params.get("disease") and not molecule:
        print("[ERROR] Could not identify a disease or a molecule from the prompt.")
        print("  Tip: name a disease, e.g. 'Design a drug for Parkinson\\'s disease',")
        print("       or give a molecule, e.g. 'optimize DILI for CC(=O)Oc1ccccc1C(=O)O'.")
        print("  Or use --disease <name> / --smiles <SMILES> directly.")
        sys.exit(1)

    # Echo parsed parameters
    print("[MolLingo] Extracted parameters:")
    for k, v in params.items():
        if v is not None:
            print(f"  {k:<20}: {v}")
    print()

    # Build agent and write ALL parsed entities into shared memory up front
    orchestrator_cls = ReActOrchestrator if args.react else OrchestratorAgent
    agent = orchestrator_cls(
        main_task=params.get("main_task", "ADMET"),
        bbbp_desired=params.get("bbbp_desired", True),
    )

    mem = agent.memory
    mem.user_prompt    = prompt
    mem.disease        = params.get("disease")
    mem.disease_subtype= params.get("disease_subtype")
    mem.gene_mutation  = params.get("gene_mutation")
    mem.pathway        = params.get("pathway")

    if params.get("uniprot_id") and params.get("protein_name"):
        mem.set_target(params["protein_name"], params["uniprot_id"])

    # A molecule named in the prompt enters as the lead, and the orchestrator
    # then skips hit identification and fragment growing.
    if molecule:
        mem.add_lead(molecule, hit_origin=molecule, docking_score=0.0)
        print(f"[MolLingo] Optimizing the supplied molecule: {molecule}")
        print("[MolLingo] Stages 2-3 (hit identification, fragment growing) "
              "are not needed.\n")
        # The block decomposition is what the LLM actually reasons over, so
        # show it before the run rather than leaving it implicit.
        print("[MolLingo] BFE tokenization of the input:")
        get_blocks(molecule, verbose=True)
        print()

    if args.react:
        agent.run_react(objective=prompt, save_path=args.save)
    else:
        agent.run(
            num_hits=args.hits,
            num_growing_hits=args.growing_hits,
            num_fragment_candidates=args.frags,
            num_leads=args.leads,
            # None hands the decision to the orchestrator; a number pins it.
            # The paper fixes 15 x 2 for every reported experiment, so that is
            # what a plain run uses.
            variants=None if args.plan_budget else args.variants,
            iterations=None if args.plan_budget else args.iterations,
            skip_fragment_growing=args.no_fragment_growing,
            save_path=args.save,
    )

    print()
    print_summary(agent.memory)


if __name__ == "__main__":
    main()
