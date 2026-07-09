"""TextJEPA: joint-embedding predictive architectures for language.

Two experiment tracks share one JEPA core (encoders, EMA targets,
action-conditioned latent predictors, stabilization objectives):

- ``discourse``: autoregressive JEPA over reasoning-step chunks; actions are
  compressed codes of the next discourse move.
- ``edits``: JEPA over text-buffer states; actions are span edits.
"""

__version__ = "0.1.0"
