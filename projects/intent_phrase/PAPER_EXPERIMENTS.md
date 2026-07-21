# Geometry-first intent-phrase paper experiment contract

_Frozen 2026-07-21 before paper-scale selection runs._

## Central claim

An action-conditioned JEPA can reason by predicting the latent consequence of
an observed language action and comparing that predicted state with a goal.
Its action value is constrained to be
`Q(s,a,g) = V(F(s,a), g)`: the value loss and ranking loss operate on the same
predicted-consequence energy. There is no direct action-only policy head in
the JEPA reranker. A separately reported proposal prior maps the current
history to a distribution over an outcome-free action catalogue.

## Datasets and information boundary

The final comparison uses four domains: official/faithful iGSM,
ProofWriter, PlanBench Blocksworld, and text-only ALFWorld. Stylized iGSM is a
development and mechanism environment, not a fifth headline dataset.

- ProofWriter formal rules and PlanBench PDDL are symbolic environment
  implementations used to execute and verify actions. They are not model
  inputs beyond their natural-language renderings.
- ALFWorld's `admissible_commands` and expert plan are privileged collection
  labels. Deployment uses a separately generated grounded catalogue; the
  adapter fails if that catalogue omits the expert action.
- Candidate order is shuffled at evaluation. `data.shuffle_actions` is not
  candidate-order shuffling; it destroys intent/outcome alignment and is used
  only as a named negative control.
- Future availability, optimal continuation, terminal state, and symbolic
  distance are never model inputs. Any experiment using them as teacher
  labels is labelled privileged-teacher supervision.

## Main model rows

The learned rows are token LM, sentence LM, sentence LM plus next-sentence
latent MSE, their three weight-shared recurrent counterparts, and the final
geometry JEPA. Random is evaluated but has no optimizer or learning rate.
Sentence-plus-latent is one trained model; decoder likelihood and latent
distance are two predeclared evaluation rules, not two opportunities to tune.

Every learned row is trained at widths 128, 256, and 512. Each
model–dataset–width cell uses the exact learning-rate grid
`{5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3}` with seeds
`{0,1,2}`. The learning rate with highest mean validation success is selected;
ties choose the lower rate. Seeds `{3,4}` are then trained at that rate, so
the reported result uses all five seeds `{0,1,2,3,4}`. Test results never
participate in selection.

This is 2,268 LR-selection trainings and 168 post-selection confirmation
trainings before ablations. It is a staged campaign, not one controller round.

## Evaluation

Report task success (solved problems) at zero, one, two, and four excess
actions, area under the success-versus-budget curve, invalid-action rate, and
mean excess actions among solved episodes. Report curves by held-out reasoning
length/plan length. For recurrent LMs, evaluate loops `{1,2,4,8,16}`. For
JEPA, evaluate simulated transition depths `{1,2,4,8,16}` at matched measured
FLOPs/wall time as well as raw depth.

## Representative-domain ablations

Mechanism ablations run on iGSM, with one confirmatory endpoint on
ProofWriter. All plotted/traced paper cells use five seeds. They reuse the
selected model–dataset learning rate unless the loss scale or gradient path
changes; such changes receive a local LR cross-check.

1. Geometry value loss: ranking weight `{0,0.25,1,4}` crossed with direct
   pairwise-advantage MSE weight `{0,0.0625,0.25,1}`, excluding both zero.
2. Advantage horizon: `N={1,2,4,8,16}` crossed with continuation beam width
   `{1,4,8}`. Report teacher quality, student calibration, rollout drift, and
   closed-loop success; do not imply monotonicity.
3. Counterfactual breadth: `K={1,2,4,8}` with matched loss normalization.
4. Dense recursive dynamics: rollout depth `{0,1,2,4,8}` and discount
   `{0.5,0.7,1}` only around viable nonzero depths.
5. Proposal/planning compute: proposal top-M `{1,2,4,8,16}`, beam width
   `{1,4,8}`, and simulation depth `{1,2,4,8,16}`.
6. Causal falsifiers: shuffled action/outcome alignment, history masking,
   goal permutation, no transition loss, no proposal prior, and true-state
   versus recursively predicted-state scoring.
7. Scaling: width `{128,256,512}` in the main table and a representative
   depth curve `{2,4,8}` layers at width 256. Parameters, tokens, optimizer
   updates, FLOPs, peak memory, and wall time are reported.

## Representation analysis as a headline result

Representation analysis is a co-equal paper pillar, run on the selected
five-seed checkpoints rather than on a single favorable seed.  Features are
always frozen.  JEPA, token-LM, and sentence-LM features are sampled at the
same causal boundary immediately before the next intent; no readout receives
future text.  Probe hyperparameters use a problem-grouped inner validation
split, and final scores use held-out problem identities.

1. Linear numeric probes measure remaining optimal steps, observed progress,
   trajectory length, current feasible-action count, and domain-specific
   quantities such as arithmetic value or proof depth.
2. Linear categorical probes measure operation/rule/action type, necessity,
   feasibility, goal relevance, and domain-specific symbolic factors.  Report
   balanced accuracy and majority/label-permutation controls.
3. Frozen-feature decoders reconstruct the next intent phrase and next
   observed consequence.  Every source uses the same decoder capacity.  Each
   readout receives the paper learning-rate grid on seeds `{0,1,2}` and adds
   seeds `{3,4}` at the validation-selected rate; report token cross-entropy,
   perplexity, token accuracy, and exact match.
4. Geometry diagnostics report effective rank, covariance spectrum, linear
   centered-kernel alignment between matched models/seeds, k-nearest-neighbor
   purity, and clustering agreement with preregistered factors.  UMAP/PCA are
   illustrations only, never quantitative evidence.
5. Counterfactual pairs hold surface form or state fixed while changing goal,
   action, outcome, prerequisite satisfaction, or distractors.  Measure which
   changes move the latent, whether action displacements compose, goal
   distance calibration, path straightness, and recursive rollout drift.
6. Causal controls include random initialization, label permutation, prompt or
   history masking, shuffled action/outcome alignment, and equal-dimensional
   random projections.  Probe accuracy alone is not interpreted as causal
   use; it is related to closed-loop behavior across seeds and interventions.

The full probe battery runs on iGSM and ProofWriter.  A smaller preregistered
factor set runs on PlanBench and ALFWorld because their symbolic annotations
and observation interfaces differ.  The analysis compares matched widths and
training data; it does not compare independently selected layers or probe
capacities.

## Admission gates

No full LR sweep is admitted for a dataset until: schema validation passes;
expert actions have 100% non-oracle-catalogue recall; the executor reaches
every recorded goal; train/validation/test identities are disjoint; a tiny
model overfits a tiny set; random and oracle bounds are sensible; action
shuffle hurts; EMA targets remain in eval mode; all dropout is zero; and
closed-loop evaluation works without future action availability. ALFWorld
also requires deterministic replay/counterfactual collection checks.
