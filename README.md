# PAC Reasoning: Controlling the Performance Loss for Efficient Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2510.09133-b31b1b.svg)](https://arxiv.org/abs/2510.09133)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://openreview.net/forum?id=52BT1Jih6O)

Official code for **PAC Reasoning** (NeurIPS 2026) by Hao Zeng, Jianguo Huang, Bingyi Jing, Hongxin Wei and Bo An.

Large reasoning models are accurate but expensive. PAC reasoning answers each query with a cheap
**non-thinking** model first, and calls the **thinking** model only when the non-thinking answer
looks uncertain. The switching threshold is calibrated with an upper confidence bound on the
performance loss, so that the loss relative to always thinking stays below a user-chosen tolerance
`ε` with probability at least `1 − α`.

<p align="center">
  <em>non-thinking answer if U(x) &lt; û, otherwise thinking answer, &nbsp; û = max{u : UCB<sub>α</sub>(u) ≤ ε}</em>
</p>

## Installation

```bash
git clone https://github.com/Jianguo99/PAC-Reasoning.git && cd PAC-Reasoning
pip install -e .                    # the PAC algorithm only (numpy / scipy)
pip install -e ".[experiments]"     # + generation, grading and scoring (vLLM, datasets, sympy, ...)
```

The experiments were run with Python 3.11 and vLLM 0.10.1.

## Quick start

The algorithm only needs, for each calibration query, an uncertainty score `U` of the non-thinking
answer and the loss `l` of keeping that answer instead of the thinking answer.

```python
import numpy as np
from pac_reasoning import binary_loss, pac_calibrate

# calibration data: correctness of both models and the non-thinking model's uncertainty
loss = binary_loss(thinking_correct, nonthinking_correct)   # 1 if only the thinking model is right
calib = pac_calibrate(loss, uncertainty, epsilon=0.03, alpha=0.05, pi=0.5, rng=0)

# deployment: call the thinking model only where the non-thinking answer is uncertain
use_thinking = calib.route(new_uncertainty)                 # boolean mask, U >= threshold
```

For open-ended tasks use `semantic_loss(thinking_embeddings, nonthinking_embeddings)`
(`1 − cosine similarity`). `examples/quickstart.py` and `examples/quickstart.ipynb` run the full
calibrate-and-evaluate loop on synthetic data.

## Reproducing the paper

The pipeline has three steps. All outputs go to `results/<benchmark>/`.

| Step | Script | Output |
|---|---|---|
| 1. Answer every query with both models | `scripts/generate.py` | `thinking.jsonl`, `nonthinking.jsonl` (response, tokens, latency, token probabilities, correctness) |
| 2. Uncertainty scores and semantic loss | `scripts/verbalized_confidence.py`, `scripts/semantic_loss.py` | `nonthinking.verbalized.jsonl`, `semantic_loss.jsonl` |
| 3. Calibrate and evaluate over 100 random splits | `scripts/run_pac.py`, `scripts/plot_curves.py` | `*.json` summaries, figures |

### 0. Data access

* **ZebraLogic** answers are in the gated dataset
  [`allenai/ZebraLogicBench-private`](https://huggingface.co/datasets/allenai/ZebraLogicBench-private);
  **GPQA** is gated as well ([`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)).
  Request access and run `huggingface-cli login`.
* **Arena-Hard v2.0** questions come from the
  [arena-hard-auto](https://github.com/lmarena/arena-hard-auto) data release:
  ```bash
  git clone https://huggingface.co/datasets/lmarena-ai/arena-hard-auto arena-hard-data
  export ARENA_HARD_QUESTIONS=arena-hard-data/data/arena-hard-v2.0/question.jsonl
  ```

### 1. Generate answers

Serve both models with an OpenAI-compatible server (vLLM or SGLang; do **not** enable a reasoning
parser, the experiments use the raw output including the thinking trace), then generate:

```bash
NONTHINKING_GPU=0 THINKING_GPU=1 bash scripts/reproduce/serve.sh     # in another terminal
bash scripts/reproduce/generate.sh
```

`generate.sh` uses Qwen3-4B-Instruct-2507 / Qwen3-4B-Thinking-2507 with the sampling settings of the
paper and also produces the Chain-of-Draft and NoThinking baselines. Interrupted runs resume where they
stopped. To run a single benchmark or another model pair:

```bash
python scripts/generate.py --dataset math500 --model Qwen/Qwen3-4B-Instruct-2507 \
    --base-url http://localhost:8001/v1 --temperature 0.7 --top-p 0.8 --top-k 20 \
    --output results/math500/nonthinking.jsonl
```

Supported benchmarks: `math500`, `zebralogic`, `arena-hard`, `gpqa`, `aime`.

### 2. Uncertainty scores and semantic loss

```bash
bash scripts/reproduce/scores.sh
```

* **Logits score** (white-box): `1 −` mean token probability of the non-thinking answer, recorded during
  generation (`--score logits`; `--score logprob` uses the mean log-probability instead).
* **Verbalized score** (black-box): the non-thinking model rates its own answer ten times;
  the score is `1 −` the mean stated probability.
* **Semantic loss**: `1 − cos` between Qwen3-Embedding-4B embeddings of the two answers.

### 3. Results

```bash
bash scripts/reproduce/semantic_loss_figure.sh   # Figure 1: error / ECP / STP vs. epsilon
bash scripts/reproduce/binary_loss_tables.sh     # Table 1 and the binary-loss table in the appendix
```

`run_pac.py` prints one row per method and `ε`:

```
method        epsilon              error          ECP (%)          STP (%)           Pass@1
pac            0.0800    0.0615 ± 0.0181     22.50 ± 7.47     23.13 ± 9.90    0.840 ± 0.015
naive          0.0800    0.0807 ± 0.0168     13.24 ± 3.93     35.42 ± 5.18    0.828 ± 0.016
nonthinking    0.0800    0.1163 ± 0.0102      0.00 ± 0.00     54.32 ± 0.73    0.796 ± 0.013
```

* **error**: loss of the composite model on the test split.
* **ECP**: expert call percentage, the share of test queries sent to the thinking model.
* **STP**: saved token percentage relative to always thinking. A routed query pays for both answers.

Methods: `pac` (Algorithm 1; `--bound hoeffding` gives Algorithm 3), `naive` (largest threshold whose
empirical calibration loss is at most `ε`, no guarantee), and `nonthinking` (never call the thinking
model, used for the CoD / NoThinking baselines).

Default hyper-parameters follow the paper: `α = 0.05`, auditing probability `π = 0.5`,
`m = n / π` draws, 100 random calibration/test splits.

### Your own models and scores

Any uncertainty score can be plugged in as a JSONL file with `{"id": ..., "score": ...}`:

```bash
python scripts/run_pac.py --thinking A.jsonl --nonthinking B.jsonl --loss binary \
    --score file --score-file my_scores.jsonl --epsilon 0.02 0.03 0.04
```

Add `--score-before-generation` for scores computed from the query alone (e.g. a learned router),
so that routed queries do not pay for the non-thinking answer.

## Repository layout

```
pac_reasoning/
  calibration.py     PAC threshold (Algorithms 1 and 3) and the naive baseline
  losses.py          binary and semantic losses
  experiment.py      random splits, metrics (error, ECP, STP, Pass@1), summaries
  records.py         reading generation records and score files
  benchmarks.py      benchmark loading and prompts
  grading/           answer checking for MATH-500 / AIME, GPQA and ZebraLogic
  plotting.py        figures in the style of the paper
scripts/             command-line entry points (see above)
scripts/reproduce/   end-to-end scripts for the paper's experiments
tests/               unit tests (pytest)
```

## Acknowledgements

The answer checkers and prompts build on
[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) evaluation,
[search-and-learn](https://github.com/huggingface/search-and-learn),
[ZeroEval](https://github.com/WildEval/ZeroEval) and
[arena-hard-auto](https://github.com/lmarena/arena-hard-auto).

## Citation

```bibtex
@inproceedings{zeng2026pac,
  title     = {{PAC} Reasoning: Controlling the Performance Loss for Efficient Reasoning},
  author    = {Zeng, Hao and Huang, Jianguo and Jing, Bingyi and Wei, Hongxin and An, Bo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
