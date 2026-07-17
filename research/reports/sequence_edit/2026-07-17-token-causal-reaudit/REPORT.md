# Token-level editing is causal, but content is the weakest action component

## The one-sentence answer

All five frozen models react strongly to edit type and current-buffer position, while whole-sequence averaging undercounts the one-token content effect; a local component audit is required before selecting a recipe.

## First, the idea in everyday language

Imagine checking whether an editor follows “replace *this* word with *apples*.” If we average the consequence over a 175-word document, changing one word looks almost invisible. The old audit did exactly that after first averaging the document into one vector. We now compare predicted and target representations at every token. Shuffling an action should make error rise if the model uses that action.

## Why this question matters

Search can propose operation, pointer, and content, but planning only works if the learned transition responds to all three. Scaling an action-blind transition would waste compute. This audit decides whether training may advance to data and capacity tests.

## What we tested

We re-evaluated five completed seed-zero checkpoints: fixed and freshly corrupted mixed training, fresh mask training, fresh curriculum training, and curriculum with VICReg 0.02 plus LDAD 20 at learning rate 0.0001. Each saw 256 validation trajectories from mixed, mask, replacement, and removal corruption. We changed no weights.

## What a fair comparison means here

The current buffer, prompt, target, checkpoint, and example set remain fixed. Only the observed action tuple is deranged. Errors use valid token positions shared by prediction and EMA target. These are process-valid diagnostics, but one seed does not establish uncertainty across training runs.

## What happened

| Model / mixed evaluation | Matched token error | Shuffled token error | Ratio | Recursive token error |
|---|---:|---:|---:|---:|
| Mixed, fixed | 0.184 | 0.605 | 3.30 | 0.308 |
| Mixed, fresh | 0.185 | 0.606 | 3.28 | 0.308 |
| Curriculum, fresh | 0.186 | 0.606 | 3.26 | 0.310 |
| Mask, fresh | 0.196 | 0.611 | 3.12 | 0.375 |
| LDAD 20, lower LR | 0.184 | 0.337 | 1.83 | 0.214 |

Fresh versus repeated corruption again makes essentially no difference. LDAD improves recursive error but reduces action separation. In single-operation mask and replacement evaluation, whole-sequence token ratios are near 1.03. A small component smoke test shows why: operation ratio 3.55, pointer ratio 2.02, but content ratio 1.02 globally and 1.40 within two tokens of the edit.

## The intuitive picture

![A long row of tokens with a bright five-token neighborhood around one edit; whole-document averaging makes the local difference look small.](local_metric.svg)

The figure shows why both global and local metrics are needed: operation and length changes affect much of the sequence, while replacement content acts locally.

## The technical details

The token-aligned predictor deterministically constructs a delete/insert/replace latent scaffold at a pointer into the current sequence, adds relative pointer embeddings and action conditioning, and contextualizes it with a dropout-free bidirectional Transformer. Targets come from the EMA encoder in evaluation mode. The corrected causal statistic is mean layer-normalized L1 error after action derangement divided by matched-action error. The next audit deranges operation, pointer, or content separately and additionally scores the union of radius-two neighborhoods around original and perturbed pointers. Raw artifacts are under `runs/autonomy/sequence_edit/2026-07-17-structured-edit-token-metric-reaudit-wave4b/`.

The global statistic scores every valid token, so it remains informative for insertions and deletions that shift sequence alignment. The local statistic scores only five-token neighborhoods and is the primary content diagnostic. Matching and shuffling use the same checkpoint and batch. No terminal goal, symbolic repair label, candidate-quality label, or future action is exposed to the predictor. The clean target remains privileged supervision and is unavailable during deployment.

## What we can conclude

The structured transition is not broadly action-blind. Mixed-edit operation and pointer changes have large causal effects. Fresh exposure is not the missing ingredient.

## What we cannot conclude

We cannot yet rank recipes for content fidelity, claim planning success, or infer a scale law. The component result currently has only an eight-example smoke test, and all trained checkpoints use one seed.

## What happens next

Run the component-local audit on all five frozen checkpoints. If content-local ratios remain meaningfully above one, select the best matched-error/sensitivity tradeoff for controlled data and counterfactual ablations. If not, add an explicit local content target before scale.

## Words used in this report

- **EMA target:** A slowly updated copy of the encoder used to make stable training targets.
- **Pointer:** A location in the current token sequence, rather than a permanent numbered position.
- **Causal falsifier:** A test that changes the action while holding the observed state fixed.

## Questions for you

- Should recipe selection prioritize lowest recursive error or strongest content-local action separation when they disagree?
- After the component gate, should the first budget go to unique-data scale or counterfactual breadth?
