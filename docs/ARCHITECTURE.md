# Architecture

A from-scratch PyTorch Transformer for language modeling and text classification, organized so each building block can be studied and ablated independently.

## Package layout (`src/`)

| Package | Contents |
| ------- | -------- |
| `models/` | `attention.py` (scaled dot-product and multi-head attention), `embeddings.py` (sinusoidal and learnable positional encodings), `transformer.py` (encoder block with residuals and LayerNorm, full model). |
| `data/` | `dataset.py`: loading and tokenization for Tiny Shakespeare and WikiText-2, batching utilities. |
| `training/` | `trainer.py` (training loop, evaluation, mixed precision), `optimizer.py` (AdamW, warmup and LR schedules, gradient clipping), `utils.py`. |
| `experiments/` | `ablation_studies.py`: runs variants such as no positional encoding or single-head attention. |
| `visualization/` | `plots.py`: attention heatmaps, training curves, embedding t-SNE. |

`main.py` is the entry point that wires config, data, model, and trainer together. `checkpoints/` holds saved weights, `training.log` the last run, and `experiment_report.md` the written results.

## Forward pass

```
tokens -> token embedding + positional encoding
       -> N x [multi-head self-attention -> add & norm -> feed-forward -> add & norm]
       -> output head (LM logits or classification)
```

## Training details

- AdamW with linear warmup then decay, gradient clipping, optional AMP.
- Metrics: loss and perplexity for LM; accuracy and F1 for classification.
- Ablations toggle attention heads, positional encoding, and normalization placement.
