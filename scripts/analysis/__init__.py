"""Representation-analysis battery (ICLR paper, 2026-08).

Modules here are read-only diagnostics over frozen checkpoints:

* :mod:`common` -- distance metrics, AUC, probe trainers, encoder registry.
* ``consequence_geometry.py`` -- paraphrase collapse / negation separation.
* ``probe_battery.py`` -- linear+MLP probes with random-init controls.
* ``embedding_maps.py`` -- t-SNE / UMAP embedding maps (``.venv2`` only).
"""
