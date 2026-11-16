import re
import sys
from typing import Optional

from memory import SharedMemory
from agent import LiteratureAgent, ChemistAgent, _extract_structured
from rdkit import Chem

from toolset.call_gpt import gpt_api, parse_json_response


class OrchestratorAgent:
    """
    Coordinates the full MolLingo drug discovery pipeline through a shared memory module:

      Stage 1 — Target identification   (LiteratureAgent)
      Stage 2 — Hit identification       (ChemistAgent)
      Stage 3 — Hit-to-lead              (ChemistAgent — context-aware fragment growing)
      Stage 4 — Lead optimization        (ChemistAgent — block-based ADMET refinement)

    Each stage writes its results to SharedMemory, which all agents read from.
    The pipeline can be started at any stage by pre-populating the memory.

    Example — full run from disease name:
        agent = OrchestratorAgent(
            disease="Alzheimer's disease",
            gene_mutation="GBA1 loss-of-function",
            bbbp_desired=True,
        )
        summary = agent.run(save_path="results.json")

    Example — skip target identification (known target):
        agent = OrchestratorAgent(
            disease="CML",
            uniprot_id="P00519", protein_name="ABL1",
            main_task="DILI",
        )
        summary = agent.run()

    Example — run stages manually for inspection:
        agent = OrchestratorAgent(disease="Parkinson's disease", gene_mutation="LRRK2")
        agent.run_target_identification()
        hits = agent.run_hit_identification(num_candidates=50)
        leads = agent.run_hit_to_lead(num_hits=5)
        candidates = agent.run_lead_optimization(num_leads=3)
        print(agent.memory.summary())
    """

    def __init__(
        self,
        disease: Optional[str] = None,
        disease_subtype: Optional[str] = None,
        gene_mutation: Optional[str] = None,
        pathway: Optional[str] = None,
        uniprot_id: Optional[str] = None,
        protein_name: Optional[str] = None,
        main_task: str = 'ADMET',
        bbbp_desired: bool = True,
        memory: Optional[SharedMemory] = None,
    ):
        self.memory = memory or SharedMemory()
        self.memory.main_task = main_task
        self.memory.bbbp_desired = bbbp_desired

        if disease:
            self.memory.disease = disease
            self.memory.disease_subtype = disease_subtype
        if gene_mutation:
            self.memory.gene_mutation = gene_mutation
        if pathway:
            self.memory.pathway = pathway
        if uniprot_id and protein_name:
            self.memory.set_target(protein_name, uniprot_id)

        self.literature_agent = LiteratureAgent(self.memory)
        self.chemist_agent = ChemistAgent(self.memory)

    # ------------------------------------------------------------------ #
    # Individual stage runners                                             #
    # ------------------------------------------------------------------ #

    def run_target_identification(self) -> dict:
        assert self.memory.disease, "Set disease before running target identification."
        return self.literature_agent.identify_target(
            disease=self.memory.disease,
            subtype=self.memory.disease_subtype,
            pathway=self.memory.pathway,
            gene_mutation=self.memory.gene_mutation,
        )

    def run_hit_identification(self, num_candidates: int = 30) -> list:
        assert self.memory.uniprot_id, "Target must be set before hit identification."
        return self.chemist_agent.identify_hits(num_candidates=num_candidates)

    def run_hit_to_lead(
        self, num_hits: int = 5, num_fragment_candidates: int = 5,
        hits: Optional[list] = None
    ) -> list:
        assert self.memory.hits, "Identify hits before running hit-to-lead."
        return self.chemist_agent.grow_hits(
            num_hits=num_hits,
            num_fragment_candidates=num_fragment_candidates,
            hits=hits,
        )

    # Anchors for the adaptive budget. PAPER_VARIANTS_PER_BLOCK * 3 blocks = 15
    # and PAPER_ITERATIONS = 2, the settings reported in the appendix, so a
    # typical three-block lead reproduces them exactly.
    VARIANTS_PER_BLOCK = 5
    VARIANTS_BOUNDS = (10, 30)
    PAPER_ITERATIONS = 2
    ITERATIONS_BOUNDS = (1, 3)

    def _heuristic_budget(self, leads: Optional[list] = None) -> dict:
        """
        Choose `variants` and `iterations` from the molecule and its profile.

        Two signals, both known before any LLM call:

        * breadth — the number of blocks the lead decomposes into. Each block
          is a substitution site, so a larger molecule needs more proposals to
          cover its sites at the same density. Five per block puts a
          three-block lead at 15, the value used in the paper.
        * headroom — how far the lead's current score sits below the maximum
          attainable for the objective (1.0 for a single endpoint, 5.0 for the
          aggregate). A molecule with little left to gain does not warrant
          extra rounds; one starting from a poor profile does.

        The loop also stops early on its own when an iteration yields no
        improvement, so `iterations` is an upper bound rather than a quota.
        """
        leads = leads or list(self.memory.leads) or list(self.memory.hits)
        lead = leads[0] if leads else None

        n_blocks, score, max_score = None, None, 1.0
        if lead:
            decomposition = self.chemist_agent._decompose(lead)
            if decomposition:
                n_blocks = len(decomposition[2])
            profile = self.chemist_agent.admet_profiles([lead]).get(lead, {})
            score = self.chemist_agent.admet_score(profile)
            max_score = 5.0 if self.chemist_agent._main_task is None else 1.0

        low, high = self.VARIANTS_BOUNDS
        variants = (min(max(self.VARIANTS_PER_BLOCK * n_blocks, low), high)
                    if n_blocks else self.VARIANTS_PER_BLOCK * 3)

        if score is None:
            iterations = self.PAPER_ITERATIONS
            deficit = None
        else:
            deficit = max(0.0, (max_score - score) / max_score)
            if deficit > 0.6:
                iterations = self.PAPER_ITERATIONS + 1
            elif deficit >= 0.3:
                iterations = self.PAPER_ITERATIONS
            else:
                iterations = self.PAPER_ITERATIONS - 1
        lo_it, hi_it = self.ITERATIONS_BOUNDS
        iterations = min(max(iterations, lo_it), hi_it)

        return {"variants": variants, "iterations": iterations,
                "n_blocks": n_blocks, "score": score, "max_score": max_score,
                "deficit": deficit,
                "rationale": "deterministic fallback: 5 proposals per block, "
                             "iterations from the headroom to the objective"}

    # -- the Orchestrator's own reasoning step --------------------------- #

    # Appendix J: "The Orchestrator is implemented as a prompted GPT-5.4 model"
    # following a ReAct-style loop, reasoning about the next action from the
    # objective, the completed subgoals and the relevant memory context. The
    # search budget is one such decision, so it is reasoned about rather than
    # computed.
    _BUDGET_PROMPT = """\
You are the Orchestrator of MolLingo, an autonomous drug discovery system.
Your role is to decompose the given objective into subgoals and coordinate the
Literature Agent, Chemist Agent, and available tools.

Current objective: {objective}
Completed subgoals: {subgoals}
Relevant context from memory:
{context}

Decision required: size the search budget for the lead optimization stage.
  - variants: how many block substitutions the Chemist Agent proposes per
    iteration. Each one is scored by the ADMET oracles, so this is the cost of
    breadth within a single round (allowed: {v_low}-{v_high}).
  - iterations: how many successive rounds of refinement to run, each starting
    from the best molecule of the previous round (allowed: {i_low}-{i_high}).

Balance exploration against exploitation:
  - A molecule with a poor starting profile and much headroom to gain, or with
    many substitutable blocks, justifies exploration: more proposals per round.
  - A molecule already close to the attainable optimum, or one where early
    rounds are likely to find the gain quickly, justifies exploitation: fewer,
    deeper rounds that refine a promising edit.
  - Every proposal costs an oracle evaluation, and every iteration costs an
    LLM call plus the risk of drifting away from the starting structure, so do
    not spend budget that the molecule does not warrant.

Return ONLY a Python dictionary literal, no markdown and no explanation:
{{"variants": <int>, "iterations": <int>, "rationale": "<one or two sentences>"}}
"""

    def plan_optimization_budget(self, leads: Optional[list] = None) -> dict:
        """
        Let the Orchestrator reason about the search budget.

        NOT a reproduction of anything in the paper. The appendix fixes 15
        modifications x 2 iterations for every reported experiment and
        describes both as user-configurable settings, and the Appendix J
        action set (literature agent, chemist agent, tool call, memory
        read/write, human feedback, terminate) contains no budget action. This
        extends the Orchestrator's remit by analogy with the workflow-level
        adaptation the paper does claim, and is reached only via
        --plan-budget.

        The decision is made by the prompted model from the objective, the
        completed subgoals and the memory context. The numbers are clamped to
        the configured bounds, and any failure (an unparseable reply, a value
        out of range, an API error) falls back to `_heuristic_budget`, so a run
        is never blocked on the LLM.
        """
        reference = self._heuristic_budget(leads)
        leads = leads or list(self.memory.leads) or list(self.memory.hits)
        lead = leads[0] if leads else None

        objective = (f"optimize {self.memory.main_task} for "
                     f"{lead or 'the current lead'}"
                     + (f", indicated for {self.memory.disease}"
                        if self.memory.disease else ""))
        subgoals = ", ".join(
            record.get("step", "") for record in (getattr(self.memory, "trace", []) or [])
            if record.get("agent") == "OrchestratorAgent") or "none yet"

        context = [f"- lead: {lead}"]
        if reference["n_blocks"]:
            context.append(f"- the lead decomposes into {reference['n_blocks']} "
                           f"substitutable building blocks")
        if reference["score"] is not None:
            context.append(
                f"- its current {self.memory.main_task} score is "
                f"{reference['score']:.3f} out of a maximum {reference['max_score']:.0f}"
                f" ({reference['deficit']:.0%} of the range still to gain)")
        if self.memory.protein_name:
            context.append(f"- target: {self.memory.protein_name}")
        context.append(f"- a comparable deterministic rule would choose "
                       f"{reference['variants']} x {reference['iterations']}")

        v_low, v_high = self.VARIANTS_BOUNDS
        i_low, i_high = self.ITERATIONS_BOUNDS
        prompt = self._BUDGET_PROMPT.format(
            objective=objective, subgoals=subgoals, context="\n".join(context),
            v_low=v_low, v_high=v_high, i_low=i_low, i_high=i_high,
        )

        decision, rationale = None, None
        try:
            parsed = _extract_structured(gpt_api(prompt))
            if isinstance(parsed, dict):
                variants = int(parsed["variants"])
                iterations = int(parsed["iterations"])
                rationale = str(parsed.get("rationale", "")).strip()
                clamped_v = min(max(variants, v_low), v_high)
                clamped_i = min(max(iterations, i_low), i_high)
                if (clamped_v, clamped_i) != (variants, iterations):
                    self.memory.log("OrchestratorAgent", "budget_clamped",
                                    f"model asked for {variants} x {iterations}; "
                                    f"clamped to {clamped_v} x {clamped_i}")
                decision = {"variants": clamped_v, "iterations": clamped_i}
        except Exception as exc:
            self.memory.log("OrchestratorAgent", "budget_reasoning_failed",
                            f"{type(exc).__name__}: {exc}; using the "
                            f"deterministic rule instead")

        if decision is None:
            decision = {"variants": reference["variants"],
                        "iterations": reference["iterations"]}
            rationale = reference["rationale"]
            source = "deterministic rule"
        else:
            source = "orchestrator (LLM)"

        self.memory.log(
            "OrchestratorAgent", "optimization_budget",
            f"{decision['variants']} modifications x {decision['iterations']} "
            f"iteration(s), decided by the {source}"
            + (f" -- {rationale}" if rationale else ""))
        return decision

    def run_lead_optimization(self, num_leads: int = 3,
                              variants: Optional[int] = None,
                              iterations: Optional[int] = None,
                              leads: Optional[list] = None) -> list:
        """
        Stage 4. Passing `variants`/`iterations` pins the configuration, which
        is what run_agent.py does by default with the paper's 15 x 2.

        Leaving either as None hands the decision to
        `plan_optimization_budget`. That is an extension beyond the paper: the
        appendix fixes both values for every reported experiment and gives the
        Orchestrator no budget-setting action, so it is reached only via
        --plan-budget.
        """
        assert self.memory.hits or self.memory.leads, \
            "Hits or leads must exist before lead optimization."
        if variants is None or iterations is None:
            planned = self.plan_optimization_budget()
            variants = planned["variants"] if variants is None else variants
            iterations = planned["iterations"] if iterations is None else iterations
        return self.chemist_agent.optimize_leads(
            num_leads=num_leads, variants=variants, iterations=iterations,
            leads=leads,
        )

    # ------------------------------------------------------------------ #
    # Full pipeline                                                        #
    # ------------------------------------------------------------------ #

    def run(
        self,
        num_hits: int = 30,
        num_growing_hits: int = 5,
        num_fragment_candidates: int = 5,
        num_leads: int = 3,
        variants: Optional[int] = None,
        iterations: Optional[int] = None,
        skip_fragment_growing: bool = False,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Run the full MolLingo pipeline end-to-end and return a result summary.
        All task context (disease, target, gene_mutation, pathway, main_task, etc.)
        must be set on self.memory before calling run().

        Parameters
        ----------
        num_hits                : How many ChEMBL candidates to retrieve.
        num_growing_hits        : How many top hits to run fragment growing on.
        num_fragment_candidates : GPT fragment proposals per hotspot per hit.
        num_leads               : How many leads to pass to lead optimization.
        variants                : Block substitution variants to explore per lead.
        skip_fragment_growing   : Go directly from hits to lead optimization.
        save_path               : If set, saves full memory to this JSON path.
        """
        self.memory.log("OrchestratorAgent", "pipeline_start",
                        f"MolLingo starting for '{self.memory.disease}'")

        # Stage 1 — Target identification
        # A molecule supplied in the prompt is already a lead, so there is
        # nothing to discover: hit identification and fragment growing are
        # about finding and elaborating a starting scaffold. The pipeline can
        # be entered at any stage by pre-populating memory, so the orchestrator
        # routes straight to optimization.
        starts_from_given_molecule = bool(self.memory.leads)

        if starts_from_given_molecule:
            self.memory.log(
                "OrchestratorAgent", "entering_at_lead_optimization",
                f"{len(self.memory.leads)} lead(s) supplied; skipping hit "
                f"identification and fragment growing.")
            # Target context is optional here -- it only enriches the prompt --
            # so a failure to identify one must not stop the run.
            if not self.memory.uniprot_id and self.memory.disease:
                try:
                    self.run_target_identification()
                except Exception as exc:
                    self.memory.log("OrchestratorAgent", "target_identification_skipped",
                                    f"{type(exc).__name__}: {exc}")
        else:
            if not self.memory.uniprot_id:
                self.run_target_identification()

            if not self.memory.uniprot_id:
                self.memory.log("OrchestratorAgent", "pipeline_abort",
                                "Could not identify a target. Stopping.")
                return self.memory.summary()

            # Stage 2 — Hit identification
            self.run_hit_identification(num_candidates=num_hits)

            if not self.memory.hits:
                self.memory.log("OrchestratorAgent", "pipeline_abort",
                                "No hits found. Stopping.")
                return self.memory.summary()

            # Stages 3 and 4, with backtracking over hits.
            #
            # A hit can dead-end: it grows into a strong binder and then every
            # ADMET improvement costs more binding affinity than
            # MAX_DOCKING_LOSS allows, so stage 4 keeps nothing. That is the
            # gate doing its job -- a molecule that stops binding is not a
            # lead -- but it used to end the run with zero candidates while
            # the remaining hits were never elaborated at all. The hits are
            # ranked, so the orchestrator now moves to the next one and tries
            # again, which is the "adapting the workflow based on intermediate
            # results" of §2.2 rather than a fixed forward pass.
            ranked_hits = self.memory.top_hits(num_hits)
            attempts = max(1, min(num_growing_hits, len(ranked_hits))) \
                if skip_fragment_growing else len(ranked_hits)
            candidates = []

            for position, hit in enumerate(ranked_hits[:attempts], start=1):
                if skip_fragment_growing:
                    leads = [hit]
                    self.memory.log(
                        "OrchestratorAgent", "skipping_fragment_growing",
                        f"Hit {position}/{attempts}: optimizing {hit} directly.")
                else:
                    leads = self.run_hit_to_lead(
                        num_fragment_candidates=num_fragment_candidates,
                        hits=[hit],
                    )
                    if not leads:
                        self.memory.log(
                            "OrchestratorAgent", "hit_abandoned",
                            f"Hit {position}/{attempts} ({hit}): fragment "
                            f"growing produced no lead; moving to the next hit.")
                        continue

                candidates = self.run_lead_optimization(
                    num_leads=num_leads, variants=variants,
                    iterations=iterations, leads=leads,
                )
                if candidates:
                    self.memory.log(
                        "OrchestratorAgent", "hit_succeeded",
                        f"Hit {position}/{attempts} ({hit}) yielded "
                        f"{len(candidates)} candidate(s).")
                    break

                stats = getattr(self.memory, "optimization_stats", {}) or {}
                self.memory.log(
                    "OrchestratorAgent", "hit_abandoned",
                    f"Hit {position}/{attempts} ({hit}): no candidate survived "
                    f"({stats.get('improved_before_gates', 0)} beat the "
                    f"objective but were rejected -- "
                    f"{stats.get('rejected_binding', 0)} on binding loss, "
                    f"{stats.get('rejected_bbbp', 0)} on BBBP drift). "
                    f"Moving to the next hit.")

            if not candidates:
                self.memory.log(
                    "OrchestratorAgent", "all_hits_exhausted",
                    f"Tried {attempts} hit(s); none produced a candidate that "
                    f"improved the objective while holding binding and BBBP.")

        # Stage 4 — Lead optimization.
        # On the discovery route this already ran inside the backtracking loop
        # above, once per hit tried; running it again here would re-optimize
        # the last hit's leads. Only the supplied-molecule route still needs
        # it, since that branch skips stages 2 and 3 entirely.
        if starts_from_given_molecule:
            self.run_lead_optimization(num_leads=num_leads, variants=variants,
                                       iterations=iterations)

        best = (self.memory.best_candidate()
                or self.memory.best_lead()
                or self.memory.best_hit())
        self.memory.log("OrchestratorAgent", "pipeline_complete",
                        f"Best molecule: {best}")

        if save_path:
            self.memory.save(save_path)
            self.memory.log("OrchestratorAgent", "saved", f"Results → {save_path}")

        return self.memory.summary()


# ====================================================================== #
# Appendix J — Orchestrator ReAct loop                                   #
# ====================================================================== #

# Verbatim from the Appendix J prompt template. Fields in braces are filled
# from the task context and shared memory at each step.
_REACT_PROMPT = """\
You are the Orchestrator of MolLingo, an autonomous drug discovery system.
Your role is to decompose the given objective into subgoals and coordinate
the Literature Agent, Chemist Agent, and available tools to complete the drug
discovery workflow.
Current objective: {objective}
Completed subgoals: {subgoals}
Relevant context from memory: {context}
Available actions:
- invoke_literature_agent(task): retrieve and synthesize biological knowledge
  for a given task or target.
- invoke_chemist_agent(task): perform molecular reasoning, hit identification,
  fragment growing, or lead optimization. The task must be one of
  "hit identification", "fragment growing", or "lead optimization".
- call_tool(tool_name, inputs): invoke a specific tool. Available tools:
  {tools}.
- request_human_feedback(question): ask the user for clarification,
  constraints, or high-level guidance.
- terminate(): end the session and return final candidates.
Based on the current state, reason step by step about what needs to be done
next, then select the most appropriate action. Respond in the following
format:
Reasoning: {{step-by-step reasoning}}
Action: {{selected action and inputs}}
"""


class ReActOrchestrator(OrchestratorAgent):
    """
    The Orchestrator as Appendix J describes it: a prompted model following a
    ReAct-style loop (Algorithm 4), rather than a hardcoded stage order.

    At each step the model receives the objective, the subgoals already
    completed, and the relevant memory context, reasons about what to do next,
    and selects one action. The result is written to shared memory and the loop
    repeats until the model terminates or MAX_STEPS is reached.

    `OrchestratorAgent.run()` is left untouched: the fixed stage order is what
    reproduces the paper's reported experiments, and this loop is the
    general-purpose coordinator.
    """

    MAX_STEPS = 10
    # How many times the same action may be selected before the loop stops.
    MAX_REPEATS = 3
    # Tools the model may call. Anything outside this mapping is refused, so a
    # hallucinated tool name cannot reach arbitrary code.
    TOOLS = ("docking", "admet", "chembl_binders", "tokenize")

    def _memory_context(self) -> str:
        lines = []
        if self.memory.disease:
            lines.append(f"disease: {self.memory.disease}")
        if self.memory.protein_name or self.memory.uniprot_id:
            lines.append(f"target: {self.memory.protein_name} "
                         f"(UniProt: {self.memory.uniprot_id})")
        lines.append(f"optimization objective: {self.memory.main_task}")
        lines.append(f"hits: {len(self.memory.hits)}, "
                     f"leads: {len(self.memory.leads)}, "
                     f"candidates: {len(self.memory.candidates)}")
        if self.memory.leads:
            lead = list(self.memory.leads)[0]
            lines.append(f"current lead: {lead}")
        best = self.memory.best_candidate()
        if best:
            entry = self.memory.candidates[best]
            lines.append(f"best candidate so far: {best} "
                         f"({self.memory.main_task} score "
                         f"{entry['admet_score']:.3f})")
        return "\n".join(f"- {line}" for line in lines)

    def _dispatch(self, action: str, argument: str):
        """Execute one selected action; returns an observation string."""
        if action == "invoke_literature_agent":
            result = self.run_target_identification()
            return (f"target: {self.memory.protein_name} "
                    f"(UniProt: {self.memory.uniprot_id})"
                    if self.memory.uniprot_id else f"no target found ({result})")

        if action == "invoke_chemist_agent":
            task = str(_first_str(_as_payload(argument), ("task",))
                       or argument or "").lower()
            if "hit" in task:
                hits = self.run_hit_identification()
                return f"{len(hits)} hits recorded"
            if "fragment" in task or "grow" in task:
                if not self.memory.hits:
                    return "refused: fragment growing needs hits first"
                leads = self.run_hit_to_lead()
                return f"{len(leads)} leads produced"
            if "optim" in task or "lead" in task:
                if not (self.memory.hits or self.memory.leads):
                    return "refused: lead optimization needs a hit or a lead first"
                before = len(self.memory.candidates)
                self.run_lead_optimization(num_leads=1)
                return (f"{len(self.memory.candidates) - before} new candidates; "
                        f"best is now {self.memory.best_candidate()}")
            return (f"refused: unknown chemist task {argument!r}; expected "
                    f"hit identification, fragment growing or lead optimization")

        if action == "call_tool":
            name, payload = _split_tool_call(argument)
            if name not in self.TOOLS:
                return f"refused: {name!r} is not an available tool"

            if name in ("docking", "admet", "tokenize"):
                molecules = _smiles_from(payload) or (
                    [list(self.memory.leads)[0]] if self.memory.leads else [])
                if not molecules:
                    return ("refused: no SMILES found in the inputs; pass "
                            "a SMILES string or {\"smiles\": \"...\"}")
                if name == "docking":
                    if not self.memory.uniprot_id:
                        return "refused: docking needs a target; identify one first"
                    scores = {sm: self.chemist_agent.dock(sm) for sm in molecules}
                    return "docking scores (kcal/mol): " + ", ".join(
                        f"{sm} = {score}" for sm, score in scores.items())
                if name == "admet":
                    profiles = self.chemist_agent.admet_profiles(molecules)
                    if not profiles:
                        return "ADMET prediction failed for all inputs"
                    return " | ".join(
                        f"{sm}: " + ", ".join(f"{k} {v:.2f}" for k, v in prof.items())
                        for sm, prof in profiles.items())
                decompositions = []
                for sm in molecules:
                    result = self.chemist_agent._decompose(sm)
                    decompositions.append(
                        f"{sm} -> {result[1]}" if result else f"{sm} -> failed")
                return " | ".join(decompositions)

            if name == "chembl_binders":
                target = (_first_str(payload, ("target", "uniprot", "uniprot_id"))
                          or self.memory.uniprot_id)
                if not target:
                    return "refused: no UniProt id given and none in memory"
                binders = self.chemist_agent._chembl_binders(target)
                return f"{len(binders)} ChEMBL binders retrieved for {target}"

        if action == "request_human_feedback":
            if not sys.stdin.isatty():
                return ("unavailable: running non-interactively, so no human "
                        "feedback can be collected; proceed on your own")
            print(f"\n[Orchestrator asks] {argument}")
            try:
                return f"human replied: {input('> ').strip()}"
            except EOFError:
                return "unavailable: no input stream"

        return f"refused: unknown action {action!r}"

    def run_react(self, objective: Optional[str] = None,
                  save_path: Optional[str] = None) -> dict:
        """Algorithm 4: reason, act, observe, write memory, repeat."""
        objective = objective or self.memory.user_prompt or (
            f"optimize {self.memory.main_task} for the molecule in memory")
        self.memory.log("OrchestratorAgent", "react_start", objective)

        completed, signatures = [], []
        for step in range(1, self.MAX_STEPS + 1):
            prompt = _REACT_PROMPT.format(
                objective=objective,
                subgoals="; ".join(completed) or "none yet",
                context=self._memory_context(),
                tools=", ".join(self.TOOLS),
            )
            try:
                reply = gpt_api(prompt)
            except Exception as exc:
                self.memory.log("OrchestratorAgent", "react_failed",
                                f"{type(exc).__name__}: {exc}")
                break

            reasoning = ""
            match = re.search(r"Reasoning:\s*(.+?)(?=\n\s*Action:|\Z)",
                              reply, re.S | re.I)
            if match:
                reasoning = " ".join(match.group(1).split())
            action, argument = _parse_action(reply)
            if action is None:
                self.memory.log("OrchestratorAgent", "react_unparsed",
                                f"Step {step}: no action found in: "
                                f"{reply.strip()[:200]}")
                break


            # The reasoning trace and the selected action are written to shared
            # memory at each step, as Appendix J requires for an audit trail.
            self.memory.log("OrchestratorAgent", f"react_step_{step}",
                            f"ACTION {action}({argument})"
                            + (f" | REASONING {reasoning}" if reasoning else ""))

            if action == "terminate":
                self.memory.log("OrchestratorAgent", "react_terminate",
                                f"Step {step}: the orchestrator ended the session")
                break

            observation = self._dispatch(action, argument)

            # The first run spent 5 of its 10 steps re-invoking lead
            # optimization on a converged molecule, each round yielding one or
            # two candidates and never terminating. Repetition is reported back
            # as part of the observation so the model can see its own loop, and
            # becomes a hard stop if it persists.
            signature = f"{action}:{argument}"
            repeats = sum(1 for seen in signatures if seen == signature)
            signatures.append(signature)
            if repeats >= 1:
                observation += (f" [note: this is repeat {repeats + 1} of the "
                                f"same action; it is not making new progress, "
                                f"so either try a different action or "
                                f"terminate()]")
            if repeats + 1 >= self.MAX_REPEATS:
                self.memory.log("OrchestratorAgent", "react_repetition_stop",
                                f"Step {step}: {signature} repeated "
                                f"{repeats + 1} times; ending the loop.")
                self.memory.log("OrchestratorAgent",
                                f"react_observation_{step}", observation)
                break

            self.memory.log("OrchestratorAgent", f"react_observation_{step}",
                            observation)
            completed.append(f"{action}({argument}) -> {observation}")
        else:
            self.memory.log("OrchestratorAgent", "react_step_limit",
                            f"Stopped after {self.MAX_STEPS} steps without "
                            f"terminating.")

        best = (self.memory.best_candidate() or self.memory.best_lead()
                or self.memory.best_hit())
        self.memory.log("OrchestratorAgent", "pipeline_complete",
                        f"Best molecule: {best}")
        if save_path:
            self.memory.save(save_path)
        return self.memory.summary()


# ---------------------------------------------------------------------- #
# Parsing the Orchestrator's replies                                      #
# ---------------------------------------------------------------------- #
#
# Appendix J asks for "Action: {selected action and inputs}" in free text, so
# the model writes whatever reads naturally. Observed replies included
#   call_tool("docking", {"target": "P47989", "ligands": [{"smiles": "..."}]})
#   invoke_chemist_agent(task="lead optimization")
# The first version split on the first comma and handed the raw JSON to dock()
# as if it were a SMILES, so every tool call silently returned nothing. These
# helpers accept a bare string, a key=value, or arbitrary nested JSON.

# Parameter names the model may use to introduce a value.
_PARAM_KEYS = "task|smiles|molecule|ligand|ligands|target|uniprot|uniprot_id|"\
              "question|tool_name|tool|inputs|input|name"

_ACTION_NAMES = ("invoke_literature_agent", "invoke_chemist_agent",
                 "call_tool", "request_human_feedback", "terminate")


def _parse_action(reply: str):
    """(action, argument string) from a reply, or (None, None)."""
    if not isinstance(reply, str):
        return None, None
    text = reply
    marker = re.search(r"Action\s*:", text, re.I)
    if marker:
        text = text[marker.end():]

    best = None
    for name in _ACTION_NAMES:
        # Tolerate the spaced spelling used in the appendix's action list.
        pattern = name.replace("_", r"[_ ]")
        match = re.search(pattern, text, re.I)
        if match and (best is None or match.start() < best[1]):
            best = (name, match.start(), match.end())
    if best is None:
        return None, None

    name, _, end = best
    rest = text[end:].lstrip()
    if not rest.startswith("("):
        return name, ""
    # Balance the parentheses, so nested JSON braces and brackets survive.
    depth, argument = 0, []
    for char in rest:
        if char == "(":
            depth += 1
            if depth == 1:
                continue
        elif char == ")":
            depth -= 1
            if depth == 0:
                break
        argument.append(char)
    return name, "".join(argument).strip()


def _as_payload(argument):
    """A dict/list/str view of an action's inputs, whatever form it arrived in."""
    if argument is None:
        return ""
    text = str(argument).strip()
    if not text:
        return ""
    parsed = parse_json_response(text)
    if isinstance(parsed, (dict, list)):
        return parsed
    # key=value, e.g. task="lead optimization". Restricted to real parameter
    # names: a SMILES like O=C(O)c1ccccc1 also matches a bare key=value
    # pattern, and stripping its "O=" silently corrupts the molecule.
    match = re.match(rf'^\s*({_PARAM_KEYS})\s*=\s*(.*)$', text, re.I | re.S)
    if match:
        text = match.group(2).strip()
    return text.strip().strip("'\"").rstrip(")").strip()


def _split_tool_call(argument):
    """(tool name, payload) from call_tool's inputs."""
    text = str(argument or "").strip()
    match = re.match(r'^\s*[\'"]?([a-z_]+)[\'"]?\s*(?:,\s*(.*))?$', text, re.I | re.S)
    if not match:
        return "", ""
    return match.group(1).strip().lower(), _as_payload(match.group(2))


def _smiles_from(payload):
    """Every value in `payload` that RDKit reads as a molecule, in order."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            for token in re.split(r"[\s,;]+", node):
                token = token.strip().strip("'\"()")
                if len(token) >= 4 and Chem.MolFromSmiles(token) is not None:
                    if token not in found:
                        found.append(token)

    walk(payload)
    return found


def _first_str(payload, keys):
    """
    The first value stored under any of `keys`, searching nested structures.

    A bare string payload is itself the value -- call_tool(chembl_binders,
    P47989) has nowhere to put a key. Nested strings, though, are only
    accepted when they sit under one of the requested keys: recursing into
    every string returned a ligand SMILES as the "target".
    """
    if isinstance(payload, str):
        return payload.strip() or None
    return _nested_str(payload, keys)


def _nested_str(payload, keys):
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _nested_str(value, keys)
            if found:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _nested_str(value, keys)
            if found:
                return found
    return None
