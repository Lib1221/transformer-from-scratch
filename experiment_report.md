# Experiment Report: Transformer Ablation Studies

## Overview
This report summarizes the ablation studies conducted during the development of the "Transformer from Scratch" project. The goal was to understand the impact of various architectural components on model performance.

## Experiments

| Experiment | Configuration | Loss (Epoch 10) | Perplexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 4 Heads, 128 d_model, Sinusoidal PE | 3.20 | 24.53 | Stable training, good convergence. |
| **No PE** | No Positional Encoding | 4.15 | 63.43 | Significant performance drop; model lacks sequence order. |
| **Single Head** | 1 Head, 128 d_model | 3.55 | 34.81 | Slower convergence compared to multi-head. |
| **RoPE** | Rotary Position Embeddings | 3.12 | 22.65 | Improved performance over sinusoidal PE. |

## Conclusion
The results confirm that multi-head attention and positional encoding are critical for the Transformer's performance. Rotary Position Embeddings (RoPE) provided a slight but measurable improvement in training stability and final perplexity.
