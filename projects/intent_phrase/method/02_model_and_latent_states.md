# Model

## Data flow

```text
prompt + observed outcomes -> chunk encoder -> discourse encoder -> z_t
intent phrase -> action encoder -> u(a)
z_t, u(a_1), ..., u(a_H) -> repeated predictor -> z_hat_H
z_t, z_hat_H, z_0, H -> endpoint head -> Energy
```

Lower Energy is better.

## Encoders

A token Transformer maps each sentence or intent phrase to a chunk vector. A
discourse encoder maps the prompt and ordered outcome chunks to a 256-wide
state. The action encoder produces a 16-wide intent embedding.

## Predictor

The validated paper recipe uses one residual state-action MLP repeatedly:

```text
z_hat_1 = F(z_t, u(a_1))
z_hat_2 = F(z_hat_1, u(a_2))
...
z_hat_H = F(z_hat_(H-1), u(a_H))
```

The causal-Transformer predictor is an architectural follow-up, not part of
the validated headline result.

## Endpoint Energy

```text
E_H = Head(root=z_t, endpoint=z_hat_H, problem=z_0, horizon=H)
```

The head is an MLP over the three latent vectors and `log(1+H)/4`. It never
receives an encoded solved state at test time. The initial state `z_0`
contains the prompt and query.

## Online and target networks

Online encoders and predictor receive gradients. Exponential-moving-average
copies produce latent targets. Target modules remain frozen and in evaluation
mode. All model dropout is zero.

## What is not in the primary method

- token generation;
- a feasibility head;
- an action prior;
- a local GAR action head;
- a Bellman value function;
- hierarchy;
- dense latent rollout supervision.
