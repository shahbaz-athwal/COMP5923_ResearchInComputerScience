# Which speedrun tricks help on their own?

COMP 5923 (Research in Computer Science) project, Fall 2026. Shahbaz Singh.

Keller Jordan's modded-nanoGPT speedrun stacks lots of changes on top of Karpathy's nanoGPT.
Because they are added one after another, it's hard to tell what each one does by itself.
This project takes three of them, RoPE, RMSNorm and the Muon optimizer, and tests each one
on its own against plain nanoGPT, trained on character-level Shakespeare at matched compute
and over several seeds.

**Research question.** Among RoPE, RMSNorm and the Muon optimizer, which individually reduce
validation loss relative to vanilla nanoGPT when trained at matched compute on
Shakespeare-character data?

| Condition | What changes compared with the baseline |
|---|---|
| `baseline` | nothing: nanoGPT with learned position embeddings, LayerNorm and AdamW |
| `rope` | the learned position table is removed and queries/keys are rotated by position instead |
| `rmsnorm` | every LayerNorm becomes RMSNorm (rescale only, no mean subtraction) |
| `muon` | the weight matrices inside the transformer blocks are trained with Muon instead of AdamW |

## What's in here

```
ablation/
  model.py        the GPT model, with switches for position encoding and normalization
  muon.py         the Muon optimizer
  settings.py     default settings (nanoGPT's), the four conditions, command-line parsing
  data.py         loading the data and cutting it into batches
  train.py        trains one run and writes metrics.json
scripts/
  run_experiments.py   runs every condition and seed (and can sweep one setting)
  analyze_results.py   summary table, Welch t-tests with Holm correction, figure
configs/
  shakespeare_char.py  the main experiment
  cpu_smoke_test.py    a tiny version to check the pipeline on a laptop
data/shakespeare_char/prepare.py   downloads and tokenizes the dataset
tests/            checks that each component behaves as its paper describes
paper/            the paper draft (Typst source and PDF)
check-ins/        the check-in reports as submitted
```

## Running it

Setup and a quick check on a CPU (about 20 minutes for the smoke test):

```bash
pip install -r requirements.txt
python data/shakespeare_char/prepare.py
python -m pytest tests
python scripts/run_experiments.py --settings=configs/cpu_smoke_test.py --output=runs/cpu_smoke_test --seeds 1 2 3
python scripts/analyze_results.py runs/cpu_smoke_test
```

The smoke test only shows that everything runs from data to statistics. The model is far
too small for its numbers to say anything about the tricks.

The main experiment needs a GPU (for example Colab):

```bash
# 1. Learning-rate sweep on seed 0, the same small grid for every condition
python scripts/run_experiments.py --settings=configs/shakespeare_char.py --output=runs/lr_sweep \
    --conditions baseline rope rmsnorm --seeds 0 --sweep learning_rate=3e-4,6e-4,1e-3,2e-3
python scripts/run_experiments.py --settings=configs/shakespeare_char.py --output=runs/lr_sweep \
    --conditions muon --seeds 0 --sweep muon_learning_rate=0.005,0.01,0.02,0.04
python scripts/analyze_results.py runs/lr_sweep

# 2. Main runs on seeds 1-5, each condition at its best learning rate from step 1
python scripts/run_experiments.py --settings=configs/shakespeare_char.py --output=runs/main \
    --conditions baseline --seeds 1 2 3 4 5 --learning_rate=<best>
# ...the same for rope and rmsnorm, and for muon with --muon_learning_rate=<best>
python scripts/analyze_results.py runs/main --plot paper/figures/main.png

# 3. Muon at matched training time instead of matched steps:
#    steps = floor(5000 * baseline ms_per_step / muon ms_per_step), measured on the same GPU
python scripts/run_experiments.py --settings=configs/shakespeare_char.py --output=runs/muon_matched_time \
    --conditions muon --seeds 1 2 3 4 5 --training_steps=<steps> --muon_learning_rate=<best>
python scripts/analyze_results.py runs/muon_matched_time --baseline-from runs/main
```

Any setting in `ablation/settings.py` can be overridden on the command line with
`--name=value`.

## Building the paper

```bash
typst compile paper/paper.typ
```

## Based on

- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT): the baseline model and training loop
- Keller Jordan, [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and [Muon](https://github.com/KellerJordan/Muon)
- Su et al., *RoFormer* (RoPE), and Zhang & Sennrich, *Root Mean Square Layer Normalization*
