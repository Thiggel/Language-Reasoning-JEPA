# Intent-phrase JEPA method handbook

This directory is the living specification of the intent-phrase subproject.
Its purpose is to make every important information flow, target, loss,
privilege, and evaluation decision explicit enough that we can inspect and
change them together.

The handbook describes the implementation as of code commit `aa3a564`. It is
more authoritative about current mechanics than old run names or historical
reports. Empirical numbers can change; the documents say when a statement is a
design, an implementation fact, an experimental privilege, or an open choice.

## Reading order

1. [Task and information interfaces](01_task_and_interfaces.md)
2. [Model and latent states](02_model_and_latent_states.md)
3. [Geometric Advantage Ranking](03_gar.md)
4. [Multi-step Energy teacher](04_multistep_energy_teacher.md)
5. [Planning and beam search](05_planning.md)
6. [Training objectives and gradient routes](06_training_objectives.md)
7. [Evaluation and success metrics](07_evaluation.md)
8. [Baselines, ablations, and controls](08_baselines_and_controls.md)
9. [Representation and geometry analysis](09_representation_analysis.md)
10. [Notation and glossary](10_notation_and_glossary.md)
11. [Open design questions](11_open_design_questions.md)

## Three horizons that must not be confused

The project currently has three independent notions of depth:

- **Teacher horizon H:** how far the training-time teacher looks after a
  candidate first action before creating its target.
- **Predictor rollout depth:** how many times the learned JEPA transition model
  is applied to imagine a particular action sequence.
- **Planner beam depth D:** how many actions each test-time candidate plan
  contains before its terminal Energy is compared with other beams.

An experiment called `teacher-h8` has H=8. It can still be evaluated with
D=1, D=4, or D=16. H and D are not the same setting.

## Current short answer about “multi-step Energy”

Yes. For H=N, the teacher does the following for every candidate first action:

1. Take that action in latent imagination.
2. Search for an approximately best continuation of N−1 more actions.
3. Measure the final imagined latent state against the encoded goal.
4. Assign that endpoint distance, or its change from the current distance, as
   the target attached to the first transition.

The phrase “optimal continuation” needs a qualifier. It is not selected using
the environment's exact shortest-path value. It is the best continuation found
by a root-balanced beam according to the current EMA JEPA geometry. The
environment supplies symbolic feasible-action menus, so H>1 is
candidate-privileged. Full details are in
[04_multistep_energy_teacher.md](04_multistep_energy_teacher.md).

## How to propose a correction

When a document disagrees with the intended method, edit the design statement
first and list the unresolved implementation change in
[11_open_design_questions.md](11_open_design_questions.md). Code and new
experiments should then cite that decision. This keeps terminology from being
silently redefined by a runner or an old checkpoint.
