# Transformer from Scratch in PyTorch

A comprehensive implementation of the Transformer architecture from scratch using PyTorch, designed for language modeling and text classification. This project focuses on deep understanding of attention mechanisms, architecture design, optimization techniques, and training stability.

## Project Goals

- Implement self-attention manually.
- Support multi-head attention.
- Include residual connections and layer normalization.
- Support positional encoding.
- Train end-to-end on a real-world dataset.
- Provide detailed evaluation metrics.
- Include ablation experiments.
- Document architectural and training decisions.

## Project Structure

```
./
├── README.md
├── requirements.txt
├── plan.md
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py  # Data loading and preprocessing
    ├── models/
    │   ├── __init__.py
    │   ├── attention.py  # Scaled Dot-Product Attention, Multi-Head Attention
    │   ├── embeddings.py # Positional Encoding
    │   ├── transformer.py # Transformer Encoder Block, full Transformer model
    └── training/
        ├── __init__.py
        ├── optimizer.py # AdamW, LR scheduling, gradient clipping
        ├── trainer.py   # Training loop, evaluation, mixed precision
        └── utils.py     # Utility functions for training
    ├── experiments/
    │   ├── __init__.py
    │   └── ablation_studies.py # Script for running ablation studies
    └── visualization/
        ├── __init__.py
        └── plots.py     # Attention heatmaps, training curves, t-SNE
```

## 30-Day Roadmap (Backlog)

### Week 1: Foundations & Architecture
- **Day 1-2**: Implement Scaled Dot-Product Attention and Multi-Head Attention.
- **Day 3-4**: Implement Positional Encoding (Sinusoidal and Learnable).
- **Day 5-7**: Build the Transformer Encoder Block with Residual Connections and LayerNorm.

### Week 2: Data & Training Setup
- **Day 8-10**: Set up data pipelines for Tiny Shakespeare and WikiText-2.
- **Day 11-14**: Implement the Training Loop with AdamW, LR Warmup, and Gradient Clipping.

### Week 3: Evaluation & Experiments
- **Day 15-18**: Conduct ablation studies (e.g., without positional encoding, single vs. multi-head).
- **Day 19-21**: Implement evaluation metrics (Perplexity, Accuracy, F1 Score).

### Week 4: Visualization & Advanced Features
- **Day 22-25**: Create attention heatmaps and training curve visualizations.
- **Day 26-28**: Implement advanced features (RoPE, Label Smoothing, Flash Attention).
- **Day 29-30**: Final documentation and project report.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/transformer-from-scratch.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python src/training/trainer.py
   ```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
