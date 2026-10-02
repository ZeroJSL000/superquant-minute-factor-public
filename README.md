# Minute-Level Factor Discovery: Public Reconstruction

This repository is an independently written, synthetic-data example of the research workflow I studied during a quantitative research internship. It shows how expression-based factor discovery, a factor pool, and held-out evaluation can be organized. It is not the firm's code or a reproduction of its results.

The example uses fictional OHLCV data generated at runtime. It contains no internal data connector, company API, model checkpoint, factor formula, report, chart, credential, or business dataset.

## Research pipeline

```text
fictional minute OHLCV [day, minute, stock, feature]
                 |
                 v
valid expression actions -> expression tree
                 |
                 v
causal minute operators within one trading day
                 |
                 v
last minute -> one daily stock cross-section
                 |
                 v
training-only Rank IC reward + correlation-aware factor pool
                 |
                 v
training-only weight fit -> JSON checkpoint
                 |
                 v
fresh-process evaluation on held-out days
```

The policy samples syntactically valid reverse Polish expressions. A random policy works with only NumPy. An optional masked recurrent PPO policy demonstrates the policy-search structure. Both policies feed the same expression evaluator and factor-pool logic.

Each expression is evaluated on one day's minute array at a time. Rolling operators use only current and prior minutes within that day. The final minute becomes a **daily** cross-sectional signal used to predict the next day's close-to-close return. Minute-level input here does not imply minute-by-minute trading.

Candidate rewards use only training-day Rank IC. The pool limits similarity between factor outputs and fits combination weights with ridge regression on training rows. Validation is reported separately, and the test segment is evaluated only after loading the saved checkpoint. `walk-forward` repeats a train/validation year followed by a later test year; `--mode single` keeps one factor without pool weight fitting.

## Run

Requires Python 3.10 or newer. From this directory:

```bash
python -m pip install -e .
python -m minute_factor_public train
python -m minute_factor_public evaluate --split test
python -m minute_factor_public walk-forward --trials 16
python -m minute_factor_public train --mode single
```

For recurrent PPO search, install the optional dependency and choose the policy:

```bash
python -m pip install -e ".[rl]"
python -m minute_factor_public train --search ppo --trials 16
```

Training writes `artifacts/model.json`, which `.gitignore` excludes. `evaluate` reloads that file in a separate process. If you change the synthetic data dimensions or seed while training, pass the same `--days`, `--minutes`, `--stocks`, and `--data-seed` values to `evaluate`. The default command uses 84 fictional days, 32 minutes per day, and 24 fictional stocks; these sizes are deliberately small so the example runs on a laptop.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

## Code map

| File | Responsibility |
| --- | --- |
| `data.py` | Generate synthetic OHLCV and next-day labels on CPU |
| `policy.py` | Sample valid expression actions; optional masked recurrent search |
| `expression.py` | Expression tree, same-day rolling operations, daily closing signal |
| `metrics.py` | Pearson IC, Rank IC, and uncosted top-minus-bottom return |
| `discovery.py` | Training-only search, correlation filter, factor weights, checkpoint |
| `evaluation.py` | Reloaded-model scoring on an explicit time segment |
| `splits.py` | Chronological and walk-forward time boundaries |

## Interpretation and boundaries

All displayed metrics come from generated data and have no investment meaning. The top-minus-bottom return ignores transaction costs, execution, turnover, and market constraints. This example does not connect to an external evaluation service. Metric definitions and accepted expression syntax must be aligned before scores from different systems can be compared.

The code is a compact educational reconstruction. Production data connectors, external evaluation interfaces, and GPU execution are outside its scope. No employer source files were copied into this repository. The CPU-resident, one-day-at-a-time evaluation is implemented here; this small example does not establish production-scale memory behavior.
