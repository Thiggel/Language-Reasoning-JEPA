---
name: literature-review
description: Perform a narrow, decision-relevant literature investigation for TextJEPA using primary sources. Use when an experiment depends on uncertain prior methods, stabilizer ranges, hierarchy definitions, evaluation validity, data construction, novelty, or a potentially wrong scientific assumption.
---

# Review literature for one decision

State the exact experimental choice the search could change. Search recent
primary papers and official code/documentation; use surveys only to discover
primary sources. Verify publication date/version, method details, tuning
protocol, datasets, scale, and evaluation conditions.

Extract only decision-relevant claims. Distinguish direct evidence from an
analogy to TextJEPA, and note incompatible assumptions such as observed actions,
continuous latents, fixed boundaries, privileged negatives, or reconstruction.
Do not copy a coefficient without reconciling normalization, representation
dimension, batch size, and loss scale.

Update `research/LITERATURE.md` with date, query, sources, applicable claim,
limitations, and resulting design change. Preserve short citations and links;
do not create an undirected paper summary.

