# Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # default run
python src/training/trainer.py # direct trainer invocation
python src/experiments/ablation_studies.py
```

Checkpoints are written to `checkpoints/`; logs to `training.log`. A GPU is recommended for WikiText-2; Tiny Shakespeare trains on CPU in reasonable time.
