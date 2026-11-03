# CPU smoke test (pipeline check only)

Output of `configs/cpu_smoke_test.py`: a 0.8M-parameter model trained for 1,000 steps on a
laptop CPU, four conditions × seeds 1-3. It shows that the whole pipeline runs from data
preparation through training to the statistics.

These numbers are **not results**. The model is tiny, the runs are short and no learning
rates were tuned, so they say nothing about whether a trick helps at the real scale.

```bash
python scripts/run_experiments.py --settings=configs/cpu_smoke_test.py --output=runs/cpu_smoke_test --seeds 1 2 3
python scripts/analyze_results.py runs/cpu_smoke_test
```
