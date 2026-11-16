"""
MolLingo — specialized agents
=============================

Implements the two specialized agents of the MolLingo framework. They are
coordinated by `OrchestratorAgent` (planner.py) and communicate exclusively
through `SharedMemory` (memory.py).

    LiteratureAgent
        Stage 1 — Target identification (§2.1, Appendix G)
        Retrieval-augmented target summarization over PubMed / UniProt /
        ChEMBL. Takes a disease (plus optional subtype, pathway and mutation
        context) and returns reported disease-target associations with
        supporting references. Organizes existing evidence; it does not
        establish novel target validity.

    ChemistAgent
        Stage 2 — Hit identification (§2.3.1)
            ChEMBL binder retrieval -> Butina clustering for chemical
            diversity -> ranking by binding affinity and drug-likeness,
            with ADMET profiling of the retained representatives.
        Stage 3 — Hit-to-lead (§2.3.2, Algorithm 3)
            Docks the hit, derives growth hotspots (available volume +
            neighboring residues) from the pose, and proposes fragment
            growths conditioned on that structural context.
        Stage 4 — Lead optimization (§2.3.3)
            Iteratively edits the block-based (BFE) representation to improve
            the predicted ADMET objective while monitoring Tanimoto
            similarity to the starting lead. With memory enabled, prior
            edits, their rationales and their oracle scores are included in
            subsequent prompts (Appendix H).

Every tool call and design rationale is written back to shared memory, so the
pipeline can be inspected, resumed or restarted at any stage.
"""

from __future__ import annotations

import re
import json
import math
from ast import literal_eval
from itertools import product
from typing import Any, Optional

from rdkit import Chem

from memory import SharedMemory

from toolset.ADMET import (ADMET_Profile, ADMET_based_scoring,
                          admet_score_threshold)
from toolset.Butina_clustering import Butina_clustering
from toolset.autodockvina_docking import autodockvina_docking
from toolset.biological_target_identification import biological_target_identification
from toolset.drug_like_filtering import implausible_reason
from toolset.calculate_druggability import calculate_druggability
from toolset.call_gpt import gpt_api
from toolset.fragment_growing import fragment_growing
from toolset.mol_similarity import tanimoto_from_smiles
from toolset.optimize_mol import smiles_post_processing
from toolset.protein2binders import protein2binders
from toolset.sascorer import calculateScore
from toolset.tokenizer import reconstruct_mol, smiles2NLdesc, vocab
from toolset.uniprotID2name import uniprotID2name


# ====================================================================== #
# Helpers                                                                #
# ====================================================================== #

def _canonical(smiles: Any) -> Optional[str]:
    """Return the canonical SMILES of `smiles`, or None if it is not valid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _scalar(value: Any) -> Optional[Any]:
    """Unwrap single-element containers — the LLM may return a list or a str."""
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    if isinstance(value, (list, tuple, dict)):
        return None
    return value


def _number(value: Any) -> Optional[float]:
    """Coerce to float, returning None for anything non-finite or non-numeric."""
    try:
        out = float(_scalar(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_structured(response: str):
    """
    Pull the structured block out of an LLM response.

    Follows the convention used across MolLingo's toolset: the answer ends with
    a fenced ```python or ```json block. Falls back to the outermost list or
    object literal in the text. Returns None if nothing parses.
    """
    if not isinstance(response, str):
        return None

    candidates = []
    for fence in ("```python", "```json", "```"):
        if fence in response:
            parts = response.split(fence)
            for part in parts[1:]:
                candidates.append(part.split("```")[0].strip())

    for opening, closing in (("[", "]"), ("{", "}")):
        start, end = response.find(opening), response.rfind(closing)
        if 0 <= start < end:
            candidates.append(response[start:end + 1].strip())

    for text in candidates:
        if not text:
            continue
        for parser in (literal_eval, json.loads):
            try:
                return parser(text)
            except Exception:
                continue
    return None


def _bare_block(block: str) -> str:
    """Strip attachment-point wildcards, for matching blocks by their core."""
    return (str(block).replace("[1*]", "").replace("[2*]", "")
            .replace("[*]", "").replace("*", "").replace("()", "").strip())


# Any attachment point: [1*], [2*], [*] or a bare *.
_ATTACHMENT_RE = re.compile(r"\[\d*\*\]|(?<!\[)\*")


def _attachment_points(block: str) -> tuple:
    """
    The isotope labels of a block's attachment points, sorted.

    ('1', '2') for a linker, ('1',) or ('2',) for a terminal block. Two blocks
    are interchangeable only if these match, otherwise the chain cannot be
    rebuilt with the same connectivity.
    """
    return tuple(sorted(re.findall(r"\[(\d*)\*\]", str(block))))


class _Agent:
    """Shared read/write interface onto `SharedMemory`."""

    name = "Agent"

    def __init__(self, memory: Optional[SharedMemory] = None):
        self.memory = memory or SharedMemory()

    def log(self, step: str, content: Any):
        self.memory.log(self.name, step, content)

    # -- ADMET objective ------------------------------------------------ #

    @property
    def _main_task(self) -> Optional[str]:
        """`None` means the aggregate ADMET objective (Appendix D)."""
        task = getattr(self.memory, "main_task", "ADMET") or "ADMET"
        return None if task == "ADMET" else task

    def admet_profiles(self, smiles_list: list) -> dict:
        """
        Predicted ADMET profiles, keyed by SMILES. Failures are dropped.

        The oracles return numpy float32; values are cast to built-in floats so
        that profiles written to shared memory stay JSON-serializable.
        """
        if not smiles_list:
            return {}
        try:
            raw = ADMET_Profile(list(smiles_list)) or {}
        except Exception as exc:
            self.log("admet_profile_failed", exc)
            return {}
        return {
            smiles: {endpoint: _number(value)
                     for endpoint, value in profile.items()}
            for smiles, profile in raw.items()
        }

    def admet_score(self, profile: dict) -> Optional[float]:
        """
        Scalar objective for the configured task; higher is always better.

        `ADMET_based_scoring` takes the complement of the endpoints where a
        lower probability is desirable (AMES, DILI, hERG, Pgp), so a single
        ">" comparison is valid for every task. The aggregate objective runs
        0-5 and excludes BBBP, which `bbbp_desired` handles separately; a
        single endpoint runs 0-1. `admet_score_threshold` gives the matching
        hard-filter level.
        """
        if not profile:
            return None
        try:
            score, _max_val = ADMET_based_scoring(
                profile,
                BBBP=bool(getattr(self.memory, "bbbp_desired", True)),
                main_task=self._main_task,
            )
        except Exception as exc:
            self.log("admet_scoring_failed", exc)
            return None
        return _number(score)

    def dock(self, smiles: str) -> Optional[float]:
        """Docking score in kcal/mol (lower is better), or None on failure."""
        if not self.memory.uniprot_id:
            return None
        try:
            result = autodockvina_docking(
                uniprot_id=self.memory.uniprot_id, smiles=smiles, sbdd=False
            )
        except Exception as exc:
            self.log("docking_failed", f"{smiles}: {exc}")
            return None
        if not result:
            self.log("docking_failed", f"No structure available for {smiles}")
            return None
        score = _number(result[1])
        # Belt and braces: the parse site already refuses a non-negative
        # affinity, but nothing downstream should ever treat one as a
        # measurement. A positive score means no pose could be placed.
        if score is not None and score >= 0:
            self.log("docking_invalid",
                     f"{smiles}: discarded a non-negative docking score "
                     f"({score:+.1f} kcal/mol); treating it as no pose.")
            return None
        return score


# ====================================================================== #
# Stage 1 — Literature Agent                                             #
# ====================================================================== #

class LiteratureAgent(_Agent):
    """
    Retrieves and synthesizes biological evidence for the discovery objective
    (§2.1, Appendix G).

    Queries incorporate disease subtype, pathway and known mutations, because
    the disease label alone can conflate molecular subgroups and yield targets
    that are not actionable for all patients.
    """

    name = "LiteratureAgent"

    def identify_target(
        self,
        disease: str,
        subtype: Optional[str] = None,
        pathway: Optional[str] = None,
        gene_mutation: Optional[str] = None,
    ) -> dict:
        """
        Retrieve reported protein targets for `disease` and write the primary
        one to shared memory.

        Returns
        -------
        dict
            {protein_name: {'uniprot_id': str,
                            'reference': str | None,
                            'uniprot_name': str | None}}
            Empty if no target could be retrieved.
        """
        context = ", ".join(
            f"{label}: {value}"
            for label, value in (("subtype", subtype),
                                 ("pathway", pathway),
                                 ("mutation", gene_mutation))
            if value
        )
        self.log("identify_target",
                 f"Retrieving targets for '{disease}'"
                 + (f" ({context})" if context else ""))

        try:
            retrieved = biological_target_identification(
                disease=disease,
                subtype=subtype,
                pathway=pathway,
                gene_mutation=gene_mutation,
            ) or {}
        except Exception as exc:
            self.log("identify_target_failed", exc)
            return {}

        targets = self._normalize(retrieved)
        if not targets:
            self.log("identify_target_failed",
                     f"No target with a UniProt ID returned for '{disease}'.")
            return {}

        # Annotate with the UniProt recommended name (evidence cross-check).
        for name, record in targets.items():
            try:
                record["uniprot_name"] = uniprotID2name(record["uniprot_id"])
            except Exception:
                record["uniprot_name"] = None

        primary = next(iter(targets))
        record = targets[primary]
        self.memory.set_target(primary, record["uniprot_id"], record["reference"])

        self.log("target_identified",
                 f"{primary} (UniProt: {record['uniprot_id']}"
                 + (f", {record['uniprot_name']}" if record["uniprot_name"] else "")
                 + ")")
        if record["reference"]:
            self.log("target_evidence", f"Reference: {record['reference']}")
        if len(targets) > 1:
            others = ", ".join(
                f"{n} ({r['uniprot_id']})" for n, r in list(targets.items())[1:]
            )
            self.log("additional_targets", others)

        return targets

    @staticmethod
    def _normalize(retrieved: dict) -> dict:
        """Normalize the retrieval output to {name: {uniprot_id, reference}}."""
        targets = {}
        if not isinstance(retrieved, dict):
            return targets
        for name, record in retrieved.items():
            if not isinstance(record, dict):
                continue
            uniprot_id = _scalar(
                record.get("Uniprot ID")
                or record.get("UniProt ID")
                or record.get("uniprot_id")
            )
            if not uniprot_id:
                continue
            reference = _scalar(record.get("Reference") or record.get("reference"))
            targets[str(name).strip()] = {
                "uniprot_id": str(uniprot_id).strip(),
                "reference": str(reference).strip() if reference else None,
            }
        return targets


# ====================================================================== #
# Stages 2-4 — Chemist Agent                                             #
# ====================================================================== #

class ChemistAgent(_Agent):
    """
    Performs molecular retrieval, evaluation and modification (§2.3).

    Hit identification retrieves known binders; hit-to-lead grows the hit
    inside the docked pocket; lead optimization edits the block-based
    representation under an ADMET objective with a similarity constraint.
    """

    name = "ChemistAgent"

    # Appendix K: Butina distance threshold for deduplicating ChEMBL binders.
    CLUSTER_CUTOFF = 0.7
    # Hit-to-lead prompt guideline (Appendix K): keep analogs conservative.
    MIN_SIMILARITY = 0.4
    # How many blocks one enumerated variant may change at once. §2.3.3 asks
    # for conservative, single-block edits, so each accepted proposal becomes
    # one molecule. Raise it to explore combinations of proposals, at the cost
    # of larger structural deviation from the lead.
    MAX_BLOCKS_EDITED = 1

    # Populated per iteration by _enumerate_variants; declared here so the
    # record-writing path works even if enumeration never ran.
    _edit_rationales: dict = {}

    # Most binding affinity a candidate may give up against its lead, in
    # kcal/mol. Docking scores are negative and lower is stronger, so this
    # caps how much less negative a candidate's score may be. §2.3.3 requires
    # the binding mode to be preserved through optimization; an ADMET gain
    # bought by abolishing binding is not a lead.
    MAX_DOCKING_LOSS = 1.0

    # How many of the improved candidates are docked each iteration, highest
    # ADMET score first. Docking is ~25 s per molecule, so this bounds the
    # cost while still covering whichever candidate would be reported as best.
    MAX_DOCKED_PER_ITERATION = 3

    # How far BBB penetration may drift against the indication, as a
    # probability difference from the lead. Appendix D excludes BBBP from the
    # aggregate score and assesses it separately, which is what this does.
    MAX_BBBP_DRIFT = 0.2

    # Fewest ChEMBL binders that can serve as a screening library. Below it,
    # §2.3.1's de novo route (DrugCLIP) supplies the hits instead.
    MIN_CHEMBL_BINDERS = 5

    # Fallback budget when the ChemistAgent is driven directly rather than by
    # the orchestrator: the settings reported in the paper's appendix.
    PAPER_VARIANTS = 15
    PAPER_ITERATIONS = 2

    # ------------------------------------------------------------------ #
    # Stage 2 — Hit identification (§2.3.1)                              #
    # ------------------------------------------------------------------ #

    def identify_hits(self, num_candidates: int = 30, dock: bool = False) -> list:
        """
        Retrieve candidate binders for the target in shared memory.

        ChEMBL binders are clustered with the Butina algorithm and one
        representative per cluster is kept, so the hits forwarded to
        hit-to-lead are chemically diverse rather than redundant. Survivors
        are ranked by binding affinity combined with drug-likeness (QED) and
        profiled with the ADMET oracles.

        Parameters
        ----------
        num_candidates : How many ranked hits to keep.
        dock           : Dock each retained hit. Off by default — hit-to-lead
                         docks the hits it grows anyway.

        Returns
        -------
        list of hit SMILES, best first.
        """
        uniprot_id = self.memory.uniprot_id
        if not uniprot_id:
            self.log("identify_hits_failed", "No target set in shared memory.")
            return []

        self.log("identify_hits",
                 f"Querying ChEMBL for binders of {self.memory.protein_name} "
                 f"(UniProt: {uniprot_id})")

        affinities = self._chembl_binders(uniprot_id)
        # Where each hit came from, so the saved run says which stage-1 route
        # produced it. Without this the de novo hits were all written out as
        # source "chembl", which is exactly the claim the de novo route is
        # meant to support.
        sources = {smiles: "chembl" for smiles in affinities}
        # "Experimental data unavailable" in practice means too sparse to
        # screen, not strictly zero: GPR15 returns a single ChEMBL binder,
        # which survives Butina trivially (1 -> 1) and gives the ranking
        # nothing to work with. Below this many binders the de novo route is
        # the better source of hits.
        if len(affinities) < self.MIN_CHEMBL_BINDERS:
            # §2.3.1: with no experimental binding data -- the normal case for
            # a novel or understudied target -- fall back to ligand-based
            # virtual screening over the curated library with DrugCLIP.
            self.log("chembl_empty",
                     f"Only {len(affinities)} ChEMBL binder(s) for "
                     f"{uniprot_id}, fewer than the "
                     f"{self.MIN_CHEMBL_BINDERS} needed to screen; falling "
                     f"back to DrugCLIP (de novo route).")
            screened = self._drugclip_hits(uniprot_id, num_candidates)
            # Keep whatever ChEMBL did return: a known binder is still a
            # legitimate hit, it just cannot carry the stage on its own.
            if screened:
                # A ChEMBL binder that is also retrieved by the screen stays
                # attributed to ChEMBL: it is a known binder either way.
                for smiles in screened:
                    sources.setdefault(smiles, "drugclip")
                affinities = {**affinities, **screened}
            if not affinities:
                self.log("identify_hits_failed",
                         f"Neither ChEMBL nor DrugCLIP produced hits for "
                         f"{uniprot_id}.")
                return []

        representatives = self._diverse_representatives(list(affinities))
        self.log("butina_clustering",
                 f"{len(affinities)} binders -> {len(representatives)} "
                 f"diverse representatives (cutoff {self.CLUSTER_CUTOFF})")

        # Rank by affinity and drug-likeness before ADMET profiling (§2.3.1).
        ranked = []
        for smiles in representatives:
            try:
                qed = _number(calculate_druggability(smiles))
            except Exception:
                qed = None
            if qed is None:
                continue
            ranked.append((smiles, 0.5 * qed + 0.5 * affinities[smiles], qed))
        ranked.sort(key=lambda row: row[1], reverse=True)
        ranked = ranked[:num_candidates]

        if not ranked:
            self.log("identify_hits_failed", "No candidate survived ranking.")
            return []

        profiles = self.admet_profiles([row[0] for row in ranked])

        hits = []
        for smiles, combined, qed in ranked:
            profile = profiles.get(smiles, {})
            score = self.admet_score(profile)
            self.memory.add_hit(
                smiles,
                docking_score=self.dock(smiles) if dock else None,
                admet_profile=profile,
                admet_score=score,
                source=sources.get(smiles, "chembl"),
            )
            entry = self.memory.hits[smiles]
            entry["qed"] = qed
            entry["affinity_rank_score"] = combined
            try:
                entry["sa_score"] = _number(calculateScore(Chem.MolFromSmiles(smiles)))
            except Exception:
                entry["sa_score"] = None
            hits.append(smiles)

        # Appendix D hard-filter level: 2.5 of 5 for the aggregate objective,
        # 0.5 of 1 for a single endpoint. The previous ">" against 0 could
        # never be satisfied for a toxicity endpoint, since those scores were
        # negative on the old scale.
        threshold = admet_score_threshold(self._main_task)
        passing = sum(
            1 for s in hits
            if (self.memory.hits[s]["admet_score"] or -9e9) > threshold
        )
        self.log("hits_identified",
                 f"{len(hits)} hits recorded; {passing} with a favorable "
                 f"predicted {self.memory.main_task} profile. "
                 f"Best: {hits[0]}")
        return hits

    def _drugclip_hits(self, uniprot_id: str, num_candidates: int) -> dict:
        """
        {SMILES: score} from a DrugCLIP screen of the target's pocket.

        The de novo route of §2.3.1. Scores are DrugCLIP similarities rather
        than normalized potencies, so they are not comparable to the ChEMBL
        affinities -- but they are only used to rank, and ranking is all the
        caller does with them. Returns {} on any failure, with the reason
        logged, so a target that cannot be screened on this machine degrades
        to "no hits" instead of raising.
        """
        try:
            from toolset.drugclip_screen import drugclip_screen
        except Exception as exc:
            self.log("drugclip_unavailable",
                     f"{type(exc).__name__}: {exc}")
            return {}

        try:
            hits = drugclip_screen(
                uniprot_id,
                protein_name=self.memory.protein_name,
                top_k=max(num_candidates, 1),
            )
        except Exception as exc:
            self.log("drugclip_failed", f"{type(exc).__name__}: {exc}")
            return {}

        if hits:
            best = max(hits.values())
            worst = min(hits.values())
            self.log("drugclip_screened",
                     f"{len(hits)} molecule(s) retrieved for {uniprot_id} "
                     f"(DrugCLIP score {worst:.3f}-{best:.3f}); these enter "
                     f"the pipeline as de novo hits.")
        return hits

    def _chembl_binders(self, uniprot_id: str) -> dict:
        """
        {SMILES: normalized affinity in [0, 1]} for the target's ChEMBL binders.

        Potency is normalized as 1 - value/1000 over the retrieval threshold
        (nM), matching the convention used elsewhere in the toolset.
        """
        try:
            df = protein2binders(uniprot_id=uniprot_id)
        except Exception as exc:
            self.log("chembl_retrieval_failed", exc)
            return {}
        if df is None or len(df) == 0:
            return {}

        affinities: dict = {}
        for raw_smiles, raw_value in zip(df["canonical_smiles"].values,
                                         df["standard_value"].values):
            smiles = _canonical(raw_smiles)
            # Fragment-sized records are not useful starting points here.
            if smiles is None or len(smiles) <= 20:
                continue
            value = _number(raw_value)
            if value is None or value < 0:
                continue
            normalized = max(0.0, min(1.0, 1.0 - value / 1000.0))
            # Keep the most potent measurement per compound.
            if normalized > affinities.get(smiles, -1.0):
                affinities[smiles] = normalized
        return affinities

    def _diverse_representatives(self, smiles_list: list) -> list:
        """One representative (the cluster centroid) per Butina cluster."""
        if len(smiles_list) < 2:
            return list(smiles_list)
        try:
            clusters, valid_idx = Butina_clustering(
                smiles_list, cutoff=self.CLUSTER_CUTOFF
            )
        except Exception as exc:
            self.log("butina_clustering_failed", exc)
            return list(smiles_list)

        representatives = []
        for cluster in clusters:
            if not len(cluster):
                continue
            # Butina returns the cluster centroid first.
            representatives.append(smiles_list[valid_idx[cluster[0]]])
        return representatives or list(smiles_list)

    # ------------------------------------------------------------------ #
    # Stage 3 — Hit-to-lead: context-aware fragment growing (§2.3.2)     #
    # ------------------------------------------------------------------ #

    def grow_hits(self, num_hits: int = 5, num_fragment_candidates: int = 5,
                  hits: Optional[list] = None) -> list:
        """
        Grow the top hits inside the target's binding site (Algorithm 3).

        For each hit, the molecule is docked, growth hotspots are derived from
        the pose (available volume and neighboring residues per heavy atom),
        and fragment extensions are proposed for each hotspot and re-docked.

        Parameters
        ----------
        num_hits               : How many top hits to grow.
        num_fragment_candidates: Fragment proposals per hotspot.
        hits                   : Grow exactly these hits instead of the top
                                 `num_hits`. Used by the orchestrator when it
                                 backtracks to a hit it has not tried yet.

        Returns
        -------
        list of grown lead SMILES.
        """
        uniprot_id = self.memory.uniprot_id
        if not uniprot_id:
            self.log("grow_hits_failed", "No target set in shared memory.")
            return []

        hits = list(hits) if hits else self.memory.top_hits(num_hits)
        if not hits:
            self.log("grow_hits_failed", "No hits in shared memory.")
            return []

        leads = []
        for hit in hits:
            # Baseline docking score of the hit, for relative improvement.
            baseline = _number(self.memory.hits.get(hit, {}).get("docking_score"))
            if baseline is None:
                baseline = self.dock(hit)
                if hit in self.memory.hits:
                    self.memory.hits[hit]["docking_score"] = baseline

            self.log("fragment_growing",
                     f"Growing {hit} (hit docking: {baseline}) with up to "
                     f"{num_fragment_candidates} proposals per hotspot")
            try:
                grown = fragment_growing(
                    smiles=hit,
                    uniprot_id=uniprot_id,
                    num_candidates=num_fragment_candidates,
                ) or []
            except Exception as exc:
                self.log("fragment_growing_failed", f"{hit}: {exc}")
                continue

            accepted = 0
            for record in grown:
                lead = self._record_lead(record, hit, baseline)
                if lead:
                    leads.append(lead)
                    accepted += 1
            self.log("fragment_growing_done",
                     f"{hit}: {accepted} valid leads from {len(grown)} proposals")

        if leads:
            best = self.memory.best_lead()
            self.log("hit_to_lead_complete",
                     f"{len(leads)} leads generated. Best: {best} "
                     f"(docking: {self.memory.leads[best]['docking_score']})")
        else:
            self.log("hit_to_lead_complete", "No leads generated.")
        return leads

    def _record_lead(self, record: Any, hit: str,
                     baseline: Optional[float]) -> Optional[str]:
        """
        Write one fragment-growing proposal to memory.

        `fragment_growing` yields
        [grown SMILES, attachment atom index, attached block, rationale,
         docking score].
        """
        if not isinstance(record, (list, tuple)) or len(record) < 5:
            return None

        smiles = _canonical(record[0])
        docking_score = _number(record[4])
        if smiles is None or docking_score is None:
            return None
        # A grown molecule that could not be placed is not a lead: recording
        # its positive score put failures into lead selection, where one
        # outranked a genuine -10.0 binder.
        if docking_score >= 0:
            self.log("lead_rejected",
                     f"{smiles}: grown molecule could not be docked "
                     f"({docking_score:+.1f} kcal/mol); not recorded as a lead.")
            return None

        self.memory.add_lead(
            smiles,
            hit_origin=hit,
            docking_score=docking_score,
            fragment=record[2],
            rationale=record[3],
        )
        entry = self.memory.leads[smiles]
        entry["growth_atom_index"] = record[1]
        entry["hit_docking_score"] = baseline
        # Docking scores are negative; a more negative score is an improvement.
        if baseline not in (None, 0):
            entry["docking_improvement_pct"] = \
                100.0 * (baseline - docking_score) / abs(baseline)
        else:
            entry["docking_improvement_pct"] = None
        try:
            entry["similarity_to_hit"] = _number(
                tanimoto_from_smiles(hit, smiles)
            )
        except Exception:
            entry["similarity_to_hit"] = None
        return smiles

    # ------------------------------------------------------------------ #
    # Stage 4 — Lead optimization (§2.3.3)                               #
    # ------------------------------------------------------------------ #

    def optimize_leads(
        self,
        num_leads: int = 3,
        variants: Optional[int] = None,
        iterations: Optional[int] = None,
        use_memory: bool = True,
        leads: Optional[list] = None,
    ) -> list:
        """
        Iteratively refine the top leads over the block-based representation.

        At each iteration the agent receives the current molecule and its
        predicted ADMET profile, proposes block substitutions with a rationale,
        reconstructs the resulting molecules, and keeps the ones the oracles
        score above the parent. Tanimoto similarity to the starting lead is
        monitored throughout.

        Parameters
        ----------
        num_leads  : How many leads to optimize.
        variants   : Block substitutions requested per iteration.
        iterations : Optimization iterations per lead (paper default: 2).
        use_memory : Include prior edits, rationales and oracle scores in
                     later prompts. Setting this False reproduces the
                     no-memory ablation (Appendix L.2).

        Returns
        -------
        list of candidate SMILES recorded in shared memory.
        """
        # Stage-4 statistics, so a run that ends with no candidate can still
        # say what it tried. Kept on shared memory for the summary to read.
        stats = {
            "leads_attempted": 0, "iterations_run": 0,
            "variants_enumerated": 0, "variants_scored": 0,
            "improved_before_gates": 0, "rejected_implausible": 0,
            "rejected_bbbp": 0, "rejected_binding": 0, "candidates_kept": 0,
        }
        self.memory.optimization_stats = stats

        # `leads` lets the orchestrator optimize one hit's leads at a time
        # while backtracking, instead of the global top-N across every hit it
        # has ever grown.
        # Rank whatever was handed over the same way _select_leads does.
        # Taking the list as given meant insertion order: on the GPR15 run the
        # -7.1 kcal/mol lead was optimized and the -8.6 one, reported in the
        # same breath as the best lead of the stage, was never touched.
        starts = (self._rank_leads(leads)[:num_leads] if leads
                  else self._select_leads(num_leads))
        if not starts:
            self.log("optimize_leads_failed",
                     "No leads or hits in shared memory.")
            return []

        # The search budget is the orchestrator's decision
        # (OrchestratorAgent.plan_optimization_budget). Reaching here without
        # one means the ChemistAgent was driven directly, so fall back to the
        # values reported in the paper -- and say so, rather than letting a
        # silent default stand in for a decision nobody made.
        if variants is None or iterations is None:
            variants = self.PAPER_VARIANTS if variants is None else variants
            iterations = self.PAPER_ITERATIONS if iterations is None else iterations
            self.log("budget_not_supplied",
                     f"No budget from the orchestrator; falling back to the "
                     f"paper's {variants} modifications x {iterations} "
                     f"iterations.")

        self.log("optimize_leads",
                 f"Optimizing {len(starts)} lead(s) for "
                 f"{self.memory.main_task} over {iterations} iteration(s), "
                 f"{variants} proposed modifications each "
                 f"(memory {'on' if use_memory else 'off'})")

        # Deliberately NOT setting memory.min_similarity: §2.3.3 selects the
        # final candidate as "the molecule with the highest ADMET score", and
        # treats structural deviation as something monitored across iterations
        # rather than a hard constraint on the final pick. Similarity still
        # steers which molecule is carried between iterations, and it is
        # reported alongside the winner so a large deviation is visible.
        candidates = []
        for lead in starts:
            stats["leads_attempted"] += 1
            candidates.extend(
                self._optimize_lead(lead, variants, iterations, use_memory)
            )
        stats["candidates_kept"] = len(self.memory.candidates)

        if candidates:
            best = self.memory.best_candidate()
            entry = self.memory.candidates[best]
            self.log("lead_optimization_complete",
                     f"{len(candidates)} candidates. Best: {best} "
                     f"({self.memory.main_task} score "
                     f"{entry['admet_score']:.3f}, similarity to lead "
                     f"{entry.get('similarity_to_lead')})")
        else:
            self.log("lead_optimization_complete",
                     "No candidate improved on its lead.")
        return candidates

    def _rank_leads(self, candidates) -> list:
        """
        Order leads by docking score, strongest binder first.

        Leads with no score keep their original order behind the scored ones:
        an unscored lead is not evidence of weak binding, but it cannot be
        ranked against one that was measured.
        """
        scored, unscored = [], []
        for smiles in candidates:
            entry = self.memory.leads.get(smiles) or {}
            value = _number(entry.get("docking_score"))
            (scored if value is not None else unscored).append((smiles, value))
        scored.sort(key=lambda row: row[1])
        return [s for s, _ in scored] + [s for s, _ in unscored]

    def _select_leads(self, num_leads: int) -> list:
        """Top leads by docking score, falling back to the ranked hits."""
        if self.memory.leads:
            return self._rank_leads(self.memory.leads)[:num_leads]
        return self.memory.top_hits(num_leads)

    def _optimize_lead(self, lead: str, variants: int,
                       iterations: int, use_memory: bool) -> list:
        """Run the optimization loop for a single lead."""
        profiles = self.admet_profiles([lead])
        parent_profile = profiles.get(lead, {})
        parent_score = self.admet_score(parent_profile)
        if parent_score is None:
            self.log("optimize_lead_skipped",
                     f"Could not profile lead {lead}.")
            return []

        # Keep the lead's own profile, so a before/after comparison does not
        # have to re-run the oracles or guess the starting value.
        lead_entry = self.memory.leads.get(lead)
        if lead_entry is not None:
            lead_entry["admet_profile"] = parent_profile
            lead_entry["admet_score"] = parent_score

        endpoint = self._main_task
        baseline_note = (f", {endpoint} {parent_profile[endpoint]:.4f}"
                         if endpoint and endpoint in parent_profile else "")
        self.log("lead_baseline",
                 f"{lead}: {self.memory.main_task} score {parent_score:.3f}"
                 + baseline_note)

        current, current_score, current_profile = lead, parent_score, parent_profile
        history: list = []
        produced = []

        for iteration in range(1, max(1, iterations) + 1):
            decomposition = self._decompose(current)
            if decomposition is None:
                break
            structure, block_seq, blocks = decomposition
            freqs = [vocab.get(re.sub(r"\[\d+\*\]", "[*]", b)) for b in blocks]
            self.log("block_representation",
                     f"{current} -> {len(blocks)} blocks: "
                     + " ^ ".join(blocks)
                     + "  (corpus frequencies: "
                     + ", ".join("below cutoff" if f is None else f"{f:,}"
                                 for f in freqs)
                     + f") -> {block_seq}")

            proposals = self._propose_block_edits(
                current, structure, current_profile, variants,
                history if use_memory else [],
            )
            if not proposals:
                self.log("optimize_lead_stopped",
                         f"Iteration {iteration}: no usable proposals for "
                         f"{current}.")
                break

            enumerated = self._enumerate_variants(current, blocks, proposals)
            if not enumerated:
                self.log("optimize_lead_stopped",
                         f"Iteration {iteration}: no molecule could be "
                         f"reconstructed for {current}.")
                break

            stats = getattr(self.memory, "optimization_stats", None)
            if stats is not None:
                stats["iterations_run"] += 1
                stats["variants_enumerated"] += len(enumerated)
            self.log("variants_enumerated",
                     f"Iteration {iteration}: {len(proposals)} edits -> "
                     f"{len(enumerated)} molecules")

            improved = self._evaluate_variants(
                enumerated, lead, current, current_score, iteration
            )
            produced.extend(improved)

            if use_memory:
                history.extend(
                    self._edit_outcomes(proposals, enumerated,
                                        current_score, iteration)
                )

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

    def _decompose(self, smiles: str):
        """BFE decomposition of `smiles` as (description, block sequence, blocks)."""
        try:
            structure, block_seq, blocks = smiles2NLdesc(smiles)
        except Exception as exc:
            self.log("block_representation_failed",
                     f"{smiles}: {exc}. Check the BFE vocabulary path in "
                     "toolset/tokenizer.py.")
            return None
        if not blocks:
            self.log("block_representation_failed",
                     f"{smiles}: no blocks returned.")
            return None
        return structure, block_seq, blocks

    # -- proposal ------------------------------------------------------- #

    def _propose_block_edits(self, smiles: str, structure: str,
                             profile: dict, variants: int,
                             history: list) -> list:
        """
        Ask the Chemist Agent for block substitutions (§2.3.3).

        Returns a list of {'Original block', 'Substitute block', 'Rationale'}.
        The rationales are design hypotheses; each candidate is evaluated by
        the property oracles afterwards.
        """
        profile_text = ", ".join(
            f"{k}: {v:.3f}" for k, v in (profile or {}).items()
        ) or "unavailable"

        prompt = self._optimization_prompt(
            smiles, structure, profile_text, variants, history
        )
        try:
            response = gpt_api(prompt)
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

        proposals = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            original = _scalar(item.get("Original block") or item.get("original_block"))
            substitute = _scalar(item.get("Substitute block") or item.get("substitute_block"))
            if not isinstance(original, str) or not isinstance(substitute, str):
                continue
            substitute = smiles_post_processing(substitute.strip())
            if Chem.MolFromSmiles(substitute) is None:
                continue
            proposals.append({
                "Original block": original.strip(),
                "Substitute block": substitute,
                "Rationale": _scalar(item.get("Rationale")
                                     or item.get("rationale")) or "",
            })
        return proposals

    def _optimization_prompt(self, smiles: str, structure: str,
                             profile_text: str, variants: int,
                             history: list) -> str:
        """Build the lead-optimization prompt (Appendix K, block-based form)."""
        task_descriptions = {
            "AMES": "lower Ames mutagenicity (reduced bacterial reverse mutation risk)",
            "DILI": "lower drug-induced liver injury risk",
            "hERG": "lower hERG inhibition (reduced cardiotoxicity risk)",
            "Pgp": "a lower P-glycoprotein inhibition profile",
            "HIA": "higher human intestinal absorption",
            "BBBP": ("higher blood-brain barrier permeability"
                     if getattr(self.memory, "bbbp_desired", True)
                     else "lower blood-brain barrier permeability"),
        }
        task = getattr(self.memory, "main_task", "ADMET") or "ADMET"
        if task == "ADMET":
            objective = (
                "better overall ADMET properties: lower Ames mutagenicity, "
                "lower hERG inhibition, lower drug-induced liver injury risk, "
                "a lower P-glycoprotein inhibition profile, higher human "
                "intestinal absorption, and "
                + ("higher" if getattr(self.memory, "bbbp_desired", True)
                   else "lower")
                + " blood-brain barrier permeability"
            )
        else:
            objective = task_descriptions.get(task, f"an improved {task} profile")

        context = []
        if self.memory.disease:
            context.append(f"intended indication: {self.memory.disease}")
        if self.memory.protein_name:
            context.append(
                f"target: {self.memory.protein_name}"
                + (f" (UniProt: {self.memory.uniprot_id})"
                   if self.memory.uniprot_id else "")
            )
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
                    + f"). Rationale given: {record.get('rationale', '')}"
                )
            memory_text = (
                "\nEdits already explored in this campaign — do not propose "
                "any of them again, and move away from directions the oracles "
                "did not support:\n" + "\n".join(lines) + "\n"
            )

        return (
            "You are an expert medicinal chemist optimizing a lead compound.\n"
            f"{context_text}"
            f"The molecule is given as a sequence of chemical building blocks:\n"
            f"{structure}\n\n"
            f"Predicted ADMET profile (probabilities): {profile_text}\n\n"
            f"Goal: modify the molecule for {objective}, while preserving as "
            "much of the original structure as possible (aim for Tanimoto "
            f"similarity above {self.MIN_SIMILARITY}).\n"
            f"{memory_text}"
            f"\nPropose {variants} structural modifications. For each one give:\n"
            "  1. Original block: one of the blocks listed above, copied exactly, "
            "including its [1*] / [2*] attachment points.\n"
            "  2. Substitute block: the replacement block, keeping the same "
            "attachment points so molecular connectivity is preserved.\n"
            "  3. Rationale: why this change should improve the stated "
            "objective.\n"
            "- Favor conservative, single-block edits over wholesale redesign.\n"
            "- Every substitute block must be a valid, synthetically "
            "accessible structure.\n"
            "- In the conclusion, return only a Python list of dictionaries:\n"
            "```python\n"
            "[\n"
            "  {\n"
            "    'Original block': '[1*]Cc1cccc(F)c1',\n"
            "    'Substitute block': '[1*]Cc1cccc(Cl)c1',\n"
            "    'Rationale': 'Replacing fluorine with chlorine ...'\n"
            "  }\n"
            "]\n"
            "```\n"
        )

    # -- enumeration ---------------------------------------------------- #

    def _enumerate_variants(self, smiles: str, blocks: list,
                            proposals: list) -> list:
        """
        Apply the proposed substitutions and reconstruct whole molecules.

        Returns a list of {'smiles', 'edits'}, where 'edits' are the
        (original block, substitute block) pairs actually applied — the
        attribution used to score each edit in memory.
        """
        # Map each proposal onto a block of the current molecule: exact match
        # first, then a match ignoring attachment-point wildcards.
        substitutions = {block: [] for block in blocks}
        bare = {_bare_block(block): block for block in blocks}
        # (original block, substitute block) -> the LLM's stated reason, kept
        # so it can be persisted onto each candidate in shared memory.
        self._edit_rationales = {}
        rejected = 0
        for proposal in proposals:
            original, substitute = proposal["Original block"], proposal["Substitute block"]
            block = original if original in substitutions \
                else bare.get(_bare_block(original))
            if block is None:
                continue
            # The prompt requires the substitute to keep the same attachment
            # points, since connectivity depends on them. A mismatch either
            # breaks reconstruction or, when the substitute has an extra
            # wildcard, yields a molecule that still contains a dummy atom.
            if _attachment_points(substitute) != _attachment_points(block):
                rejected += 1
                continue
            if substitute not in substitutions[block]:
                substitutions[block].append(substitute)
            self._edit_rationales.setdefault(
                (block, substitute), proposal.get("Rationale", "")
            )
        if rejected:
            self.log("proposal_rejected",
                     f"{rejected} proposal(s) dropped: substitute block did "
                     f"not preserve the attachment points of the block it "
                     f"replaces.")

        if not any(substitutions.values()):
            self.log("proposal_unmatched",
                     "No proposed edit matched a block of the current molecule.")
            return []

        # §2.3.3 describes conservative modifications that keep the molecule
        # close to the lead, with scaffold changes "permitted but discouraged",
        # and reports "15 structural modifications proposed per lead
        # optimization iteration" -- i.e. roughly one molecule per proposal.
        # Taking the full cross-product instead turned each iteration's
        # proposals into every combination of them: 19 proposals became 323
        # molecules, 61% of which replaced all three blocks at once. That is
        # wholesale redesign the LLM never proposed, and it dragged mean
        # similarity to the lead below the 0.4 the prompt asks for.
        if self.MAX_BLOCKS_EDITED <= 1:
            combos = []
            for position, block in enumerate(blocks):
                for substitute in substitutions[block]:
                    combo = list(blocks)
                    combo[position] = substitute
                    combos.append(tuple(combo))
        else:
            choices = [[block] + substitutions[block] for block in blocks]
            combos = (
                combo for combo in product(*choices)
                if 0 < sum(1 for b, c in zip(blocks, combo) if c != b)
                <= self.MAX_BLOCKS_EDITED
            )

        seen, enumerated = {_canonical(smiles)}, []
        for combo in combos:
            edits = [(block, chosen)
                     for block, chosen in zip(blocks, combo) if chosen != block]
            if not edits:
                continue
            try:
                variant = _canonical(reconstruct_mol("^".join(combo)))
            except Exception:
                continue
            if variant is None or variant in seen:
                continue
            # A leftover attachment point means the blocks did not zip into a
            # whole molecule. RDKit parses such a SMILES happily, so without
            # this guard a dummy-atom-bearing pseudo-molecule reaches the
            # ADMET oracles and is scored as if it were a real compound.
            if _ATTACHMENT_RE.search(variant):
                continue
            seen.add(variant)
            enumerated.append({"smiles": variant, "edits": edits})
        return enumerated

    # -- evaluation ----------------------------------------------------- #

    def _evaluate_variants(self, enumerated: list, lead: str, parent: str,
                           parent_score: float, iteration: int) -> list:
        """
        Score every enumerated molecule and record those that beat the parent.

        Returns a list of {'smiles', 'score', 'similarity'} for the improved
        molecules, which are also written to shared memory as candidates.

        Molecules that cannot exist are dropped before the oracles see them.
        RDKit parsing is not a plausibility test, and the oracles have no
        applicability domain: a hypervalent sulfur or a radical sanitizes
        cleanly and is then scored as though it were a drug, which is how a
        raw-SMILES run produced an apparent 0.89 DILI improvement from
        [SH](C)=O. Screening here covers every arm, since both the block-based
        and the raw-SMILES paths funnel through this method.
        """
        screened, rejected = [], []
        for item in enumerated:
            reason = implausible_reason(item["smiles"])
            if reason:
                rejected.append((item["smiles"], reason))
            else:
                screened.append(item)

        stats = getattr(self.memory, "optimization_stats", None)
        if stats is not None:
            stats["rejected_implausible"] += len(rejected)
        if rejected:
            self.log("implausible_rejected",
                     f"Iteration {iteration}: {len(rejected)} of "
                     f"{len(enumerated)} proposed molecule(s) excluded as "
                     f"chemically impossible, before scoring. "
                     + "; ".join(f"{smiles} ({reason})"
                                 for smiles, reason in rejected[:5])
                     + (" ..." if len(rejected) > 5 else ""))
        enumerated = screened
        if not enumerated:
            return []

        profiles = self.admet_profiles([item["smiles"] for item in enumerated])

        scores, improved = [], []
        for item in enumerated:
            smiles = item["smiles"]
            profile = profiles.get(smiles, {})
            score = self.admet_score(profile)
            if score is None:
                continue
            item["score"] = score
            scores.append(score)
            if score <= parent_score:
                continue

            try:
                similarity = _number(tanimoto_from_smiles(lead, smiles))
            except Exception:
                similarity = None

            self.memory.add_candidate(
                smiles,
                lead_origin=lead,
                admet_score=score,
                admet_profile=profile,
            )
            entry = self.memory.candidates[smiles]
            entry["parent"] = parent
            entry["iteration"] = iteration
            entry["similarity_to_lead"] = similarity
            entry["score_improvement"] = score - parent_score
            # §2.3.3: the reasoning trace stored in shared memory covers the
            # identified liabilities, the proposed modifications AND their
            # predicted outcomes. The rationale is the liability/hypothesis
            # part, so it is persisted with the edit rather than left in the
            # loop-local history that is discarded when the lead finishes.
            entry["edits"] = [
                {"original_block": original,
                 "substitute_block": substitute,
                 "rationale": self._edit_rationales.get((original, substitute), "")}
                for original, substitute in item["edits"]
            ]
            try:
                entry["qed"] = _number(calculate_druggability(smiles))
                entry["sa_score"] = _number(
                    calculateScore(Chem.MolFromSmiles(smiles))
                )
            except Exception:
                pass

            improved.append({"smiles": smiles, "score": score,
                             "similarity": similarity})

        improved_before_gates = len(improved)
        stats = getattr(self.memory, "optimization_stats", None)
        if stats is not None:
            stats["variants_scored"] += len(scores)
            stats["improved_before_gates"] += improved_before_gates

        # BBB penetration is deliberately absent from the aggregate score
        # (Appendix D) because its desirability depends on the indication, and
        # is instead "assessed separately" here against bbbp_desired. This
        # costs nothing -- the probabilities are already predicted -- so it
        # runs before the docking gate, which is the expensive one.
        improved = self._reject_bbbp_drift(improved, lead, iteration)

        # Binding check. An ADMET gain is worthless if the molecule no longer
        # binds the target, so a candidate that loses more than
        # MAX_DOCKING_LOSS kcal/mol against the lead is not proposed at all.
        # Only runs when a target is known; docking scores are negative and
        # lower is better, so a loss is an increase.
        improved = self._reject_weak_binders(improved, lead, iteration)

        if scores:
            gated = improved_before_gates - len(improved)
            self.log("variants_scored",
                     f"Iteration {iteration}: {len(scores)} scored "
                     f"(max {max(scores):.3f}, mean "
                     f"{sum(scores) / len(scores):.3f}, parent "
                     f"{parent_score:.3f}); "
                     f"{improved_before_gates} beat the parent "
                     f"({100.0 * improved_before_gates / len(scores):.1f}%)"
                     + (f", of which {gated} were then rejected by the "
                        f"BBBP/binding gates, leaving {len(improved)}"
                        if gated else ""))
        return improved

    def _reject_bbbp_drift(self, improved: list, lead: str,
                           iteration: int) -> list:
        """
        Assess BBB penetration separately, per the indication.

        Appendix D keeps BBBP out of the aggregate score because a higher
        probability is desirable for a CNS target and undesirable for a
        peripherally-acting drug, so no single direction can be baked into a
        shared objective. The direction comes from `bbbp_desired`, and a
        candidate drifting more than MAX_BBBP_DRIFT the wrong way against the
        lead is not proposed.

        Skipped when the indication is unstated, when BBBP is itself the
        objective (the score already handles it), or when the lead's own BBBP
        is unknown.
        """
        desired = getattr(self.memory, "bbbp_desired", None)
        if not improved or desired is None or self._main_task == "BBBP":
            return improved

        lead_profile = (self.memory.leads.get(lead, {}) or {}).get("admet_profile") or {}
        lead_bbbp = lead_profile.get("BBBP")
        if lead_bbbp is None:
            return improved

        kept, rejected = [], 0
        for item in improved:
            entry = self.memory.candidates.get(item["smiles"], {})
            bbbp = (entry.get("admet_profile") or {}).get("BBBP")
            if bbbp is None:
                kept.append(item)
                continue
            # Positive drift = more CNS penetration than the lead.
            drift = bbbp - lead_bbbp
            against = drift if not desired else -drift
            entry["bbbp"] = bbbp
            entry["bbbp_drift"] = drift
            item["bbbp_drift"] = drift
            if against > self.MAX_BBBP_DRIFT:
                self.memory.candidates.pop(item["smiles"], None)
                rejected += 1
                continue
            kept.append(item)

        if rejected:
            wanted = "higher" if desired else "lower"
            self.log("bbbp_filter",
                     f"Iteration {iteration}: {rejected} candidate(s) rejected "
                     f"for BBB penetration drifting more than "
                     f"{self.MAX_BBBP_DRIFT} away from the {wanted} "
                     f"penetration this indication wants "
                     f"(lead BBBP {lead_bbbp:.2f})")
        return kept

    def lead_docking_score(self, lead: str):
        """
        The lead's docking score against the current target, docked on demand.

        A molecule supplied in the prompt is seeded with a placeholder of 0.0,
        so it has to be docked here before it can serve as the baseline that
        candidates are compared against.
        """
        if not self.memory.uniprot_id:
            return None
        entry = self.memory.leads.get(lead, {})
        existing = entry.get("docking_score")
        # Only a negative score is a real affinity. 0.0 is the seeding
        # placeholder, and a positive value is a failed docking -- using one
        # as the baseline made every candidate look ~16 kcal/mol better and
        # silently disabled the binding filter.
        if existing is not None and existing < 0:
            return existing
        score = self.dock(lead)
        if score is not None and entry:
            entry["docking_score"] = score
            self.log("lead_docked",
                     f"{lead}: {score} kcal/mol against "
                     f"{self.memory.protein_name or self.memory.uniprot_id}")
        return score

    def _reject_weak_binders(self, improved: list, lead: str,
                             iteration: int) -> list:
        """
        Drop candidates whose binding falls off relative to the lead.

        Docking scores are in kcal/mol and more negative is stronger, so a
        candidate is rejected when its score exceeds the lead's by more than
        MAX_DOCKING_LOSS. Candidates that cannot be docked are kept, with no
        docking recorded, rather than being discarded on a tooling failure.
        """
        if not improved or not self.memory.uniprot_id:
            return improved

        baseline = self.lead_docking_score(lead)
        if baseline is None:
            self.log("docking_gate_skipped",
                     "No docking score for the lead; binding filter not applied.")
            return improved

        # Docking costs ~25 s per molecule, so it goes to the candidates most
        # likely to be proposed: those that improved on the parent -- which is
        # everything in `improved` -- ranked by ADMET score and capped at
        # MAX_DOCKED_PER_ITERATION.
        #
        # Gating on the absolute Appendix D filter instead (0.5 of 1 for a
        # single endpoint) meant nothing was ever docked on a DILI campaign:
        # a molecule starting at DILI 0.92 scores 0.076, and two iterations
        # reach ~0.45, so the threshold was unreachable and the binding rule
        # never applied to anything. Ranking guarantees the best candidate is
        # always among the docked set, so the reported winner always carries a
        # docking comparison.
        ranked = sorted(improved, key=lambda item: item["score"], reverse=True)
        shortlist = {item["smiles"] for item in ranked[:self.MAX_DOCKED_PER_ITERATION]}
        kept, rejected, docked = [], 0, 0
        for item in improved:
            smiles = item["smiles"]
            if smiles not in shortlist:
                kept.append(item)
                continue
            docked += 1
            score = self.dock(smiles)
            entry = self.memory.candidates.get(smiles, {})
            if score is None:
                entry["docking_score"] = None
                entry["docking_delta"] = None
                kept.append(item)
                continue
            delta = score - baseline          # positive = weaker binding
            entry["docking_score"] = score
            entry["docking_delta"] = delta
            item["docking_score"] = score
            item["docking_delta"] = delta
            if delta > self.MAX_DOCKING_LOSS:
                self.memory.candidates.pop(smiles, None)
                rejected += 1
                continue
            kept.append(item)

        stats = getattr(self.memory, "optimization_stats", None)
        if stats is not None:
            stats["rejected_binding"] += rejected
        if docked:
            self.log("binding_filter",
                     f"Iteration {iteration}: docked the top {docked} of "
                     f"{len(improved)} improved candidate(s) by ADMET score; "
                     f"{rejected} rejected for losing more than "
                     f"{self.MAX_DOCKING_LOSS} kcal/mol against the lead "
                     f"(baseline {baseline} kcal/mol)")
        else:
            self.log("binding_filter",
                     f"Iteration {iteration}: no improved candidate to dock.")
        return kept

    def _next_seed(self, improved: list) -> tuple:
        """
        Pick the molecule to carry into the next iteration.

        Prefers the best-scoring candidate that stays within the similarity
        constraint, since similarity alone does not guarantee that binding is
        preserved and runaway edits are not useful leads.
        """
        conservative = [
            item for item in improved
            if item["similarity"] is not None
            and item["similarity"] >= self.MIN_SIMILARITY
        ]
        pool = conservative or improved
        best = max(pool, key=lambda item: (item["score"],
                                           item["similarity"] or 0.0))
        return best["smiles"], best["score"]

    @staticmethod
    def _edit_outcomes(proposals: list, enumerated: list,
                       parent_score: float, iteration: int) -> list:
        """
        Summarize each proposed edit for the next prompt (Appendix H).

        Records the edit, its rationale, and the best oracle score reached by
        any molecule that used it, so later iterations neither repeat an exact
        edit nor pursue a direction the oracles contradicted.
        """
        best_by_edit: dict = {}
        for item in enumerated:
            score = item.get("score")
            if score is None:
                continue
            for edit in item["edits"]:
                if score > best_by_edit.get(edit, -float("inf")):
                    best_by_edit[edit] = score

        outcomes = []
        for proposal in proposals:
            original = proposal["Original block"]
            substitute = proposal["Substitute block"]
            best = next(
                (score for (orig, sub), score in best_by_edit.items()
                 if sub == substitute and (orig == original
                                           or _bare_block(orig) == _bare_block(original))),
                None,
            )
            outcomes.append({
                "iteration": iteration,
                "original": original,
                "substitute": substitute,
                "rationale": proposal.get("Rationale", ""),
                "best_score": best,
                "improved": best is not None and best > parent_score,
            })
        return outcomes
