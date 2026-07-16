# TextJEPA research map

The stable repository-level entry points for all three subprojects are under
[`projects/`](../projects/README.md):

- [`projects/intent_phrase/`](../projects/intent_phrase/README.md): observed
  intent phrases are the action interface. Detailed records remain here under
  [`intent_phrase/`](intent_phrase/README.md).
- [`projects/token_igsm/`](../projects/token_igsm/README.md): no intent
  annotations; tokens and text spans are primitive and macro actions. Detailed
  records remain under [`hard_text/`](hard_text/README.md) for compatibility.
- [`projects/sequence_edit/`](../projects/sequence_edit/README.md): latent
  planning over mutable reasoning buffers. Its research index is
  [`sequence_edit/`](sequence_edit/README.md); historical raw logs remain in
  the local archive.

Each subproject contains:

- `waves/`: one post-hoc record per experiment wave, including runs, results,
  interpretation, and disposition;
- `logs/`: the raw console logs, physically separated by subproject;
- `BACKLOG.md`: the staged experimental plan.

The sequence-edit project currently has no active experiment cycle; this does
not make it part of the token or intent-phrase claims. Machine-generated tables
remain in `runs/` because reporting scripts regenerate them there.

The current active controller decision concerns token-level causal hierarchy.
The intent-phrase paper matrix is a separate flat-model completion track;
hierarchy is explicitly excluded from its paper-facing recipe.
