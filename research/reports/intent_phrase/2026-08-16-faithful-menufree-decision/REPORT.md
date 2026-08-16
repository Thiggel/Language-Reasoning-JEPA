# Faithful iGSM menu-free planning: state of evidence and decision point (2026-08-16)

_All AUC numbers are candidate-privileged oracle-labeled diagnostics
(`scripts/probe_faithful_cycle_auc.py`); all planning numbers are the frozen
band/budget protocol (ID nec 8-15 @ caps 32/40/[3,32], OOD nec 20-28 @
48/64/[3,48], budget = multiples of necessary, 200 episodes, attempted-mask
scoped to current state)._

## What we set out to test

The paper narrative needs a menu-free (no feasibility oracle) planning result
on faithful iGSM. On stylized iGSM this exists: the LDAD cycle-consistency
score separates feasible from infeasible actions at AUC ~.94 with no training,
and `ldad_cycle` planning is the menu-free result. The question of this
campaign: does that transfer to faithful iGSM, and if not, can it be trained
in?

## Evidence chain (all measured this campaign)

1. **Menu-trained checkpoints have no menu-free feasibility, in any family.**
   At 5x necessary budget on the ID band: LDAD .510, GoalHead .505, TD-JEPA
   .585 vs RANDOM .655; all with invalid-proposal rates above random. Token
   LM: above random at 1.5-3x ID budgets only (.310 vs .250 at 3x), converges
   to random at 5x (.630 vs .655), BELOW random OOD (.255 vs .480). Sentence
   LM below random everywhere. No method beats blind search out of
   distribution.
2. **The cycle score is at chance on faithful, even in-distribution** (.475
   train dist, .503 ID band; stylized control through the same code path:
   .846). Mechanism (2026-08-14 diagnosis): the LDAD decoder is perfect from
   REAL displacements (1.00 both tracks), but the faithful predictor's
   IMAGINED one-step displacement is off-manifold (cos .11 to real vs .74
   stylized), so the cycle reads noise.
3. **Fix attempt 1 — train the decode from the predictor displacement**
   (`observed_action_ldad_predictor_cycle`, gradients into the predictor;
   commit d74d9c6): loss trains cleanly (1.94 -> 0.66) but final cycle AUC
   .490 = chance. Matched training caps alone: .481 = chance.
4. **Fix attempt 2 — counterfactual cycle-score contrast** (observed action
   must outscore counterfactuals; contract-compliant ranking): with the
   default counterfactual pool at weights 1.0 and 4.0, contrast loss falls
   below neutral yet AUC stays chance (.490 / .530). Diagnosed train/eval
   mismatch: the default infeasible pool is dominated by already-RESOLVED
   variables (easy negatives, visible in the step history), while planning
   must reject unresolved-with-unmet-prerequisites (hard negatives).
5. **Fix attempt 3 — hard negatives only** (`invalid_counterfactual_
   unresolved_only`, k=8, weight 4; commit a26f28e): the ONLY variant that
   moved the needle. AUC trajectory over training .495 -> .538 -> .538 ->
   **.570 final** (every other cell: .43-.53). Real but weak — far below the
   ~.85+ that made stylized planning work. A 5x-budget planning check on this
   checkpoint is running (ldad_cycle + full_catalogue ID rows).

## Reading

The cycle-consistency feasibility signal is NOT an intrinsic property of the
LDAD recipe; on stylized it emerges for free, on faithful it does not, and
even direct self-supervised training recovers only weak separation (.57). The
pattern matches the 2026-08-11 state-readout finding: the conjunctive
parent-lookup ("are all prerequisites of X resolved?") is the hard part, and
one-step displacement decoding is too narrow a bottleneck to carry it.
Meanwhile the with-menu faithful results, the stylized menu-free mechanism
story, and the OOD budget-curve instrument are all solid.

## Options (owner decision)

(a) **Push the hard-negative recipe** (k=16, longer training, contrast weight
    schedule, possibly compositional action phrases so the decode has more to
    hang on). Evidence: the only rising AUC curve of the campaign. Cost:
    ~1.5 days per iteration, uncertain ceiling.
(b) **A feasibility mechanism with more capacity than the cycle bottleneck**,
    trained through the dynamics (e.g. history-attention head scoring
    (state, candidate) with the predictor in the loop, ranking observed vs
    hard-negative counterfactuals — same self-supervision, wider readout).
    The oracle probe ceiling is .935 from the state, so capacity is not the
    limit; the objective was. Untested on faithful.
(c) **Documented-limitation framing**: menu-free faithful iGSM is an honest
    negative for ALL methods (incl. LMs at high budget and OOD); the paper
    leads with with-menu faithful + stylized menu-free mechanism + transfer
    domains, and reports this campaign as the limitation analysis. Zero
    additional compute; weakest narrative.

No further faithful menu-free variants will be launched until the owner
picks a direction. All numbers, cells, and probes are recorded in
CAMPAIGN_LOG.md (2026-08-14 through 2026-08-16 entries).
