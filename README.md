# HW3 Nano-GPT Optimizer Benchmark (MSI-friendly)

This repo is a **self-contained** template for:
- Training a GPT-style decoder-only Transformer ("Nano-GPT" ~124M params config included)
- Comparing **3 optimizers** (Adam baseline + 2 custom ones)
- Running **9 experiments** via Slurm job array
- Logging metrics and generating plots + final tables

## Quick start (local / MSI interactive)
```bash
python -m venv nanogpt-env
source nanogpt-env/bin/activate
pip install -r requirements.txt
```

### Data layout (expected)
Put your pre-tokenized dataset here (memmap binaries), or adjust `--data_dir`:
```
data/
  train.bin   # uint16 token ids
  val.bin
```

This matches the common nanoGPT format. If your course notebook uses a different loader,
edit `src/data.py` accordingly.

## Run a single debug job (CPU or 1 GPU)
```bash
python src/train.py --steps 200 --eval_interval 50 --out_dir runs/debug --device auto
```

## Run the full benchmark (9 experiments) on MSI (A40)
Edit `scripts/run_array_a40.sbatch` and submit:
```bash
sbatch scripts/run_array_a40.sbatch
```

It uses a job array with concurrency limit (default `%2`). Change to `%3` if you can.

## Make plots + summary table
After runs finish:
```bash
python scripts/make_plots.py --runs_dir runs --out_dir report/figures
```

Outputs:
- `report/figures/loss_curves.png`
- `report/figures/val_ppl_curves.png`
- `report/figures/final_table.csv`

## Notes
- Each run writes:
  - `config.json`
  - `metrics.csv` (step, train_loss, val_loss, val_ppl)
  - `final.json` (final val loss/ppl)
- For fairness, each run sets the same seed by default. Change with `--seed`.
