# ATC Example

## Main Results

This section summarizes the ATC finetuning sweeps around four questions:
baseline performance, dataset size, model size, and ATC/ATCOSIM transfer. WER
is a percentage, so lower is better. `mean_avg_logprob` is the mean decode
average log probability, so higher values are better. Unless otherwise noted,
runs use no timestamp tokens, `--max-new-tokens 96`, `beta1=0`, `beta2=0.9999`,
linear LR scheduling, and `weight_decay=0`. The full model-size sweeps below do
not freeze encoder layers or set gradient clipping in the sweep config.
Trainable parameter counts are for the corresponding finetuned configuration.

Before comparing WER, it is useful to know the duration mix of the train and
eval sets. The utterance length histogram is in
[fig/q00_setup_utterance_length_histograms.png](fig/q00_setup_utterance_length_histograms.png),
with source data in `egs/atc/fig/q00_setup_utterance_length_histograms.tsv`.

### 1. What is the baseline performance of all models on ATC/ATCOSIM?

Unfinetuned Whisper models were decoded with the same no-timestamp/max-96
setup used for the finetuned checkpoints:

| Model | ATC val WER | ATC test WER | ATCOSIM WER |
| --- | ---: | ---: | ---: |
| small | 83.89 | 84.56 | 69.44 |
| medium | 76.84 | 73.76 | 65.25 |
| turbo | 78.64 | 76.02 | 65.25 |
| large-v1 | 79.66 | 74.63 | 65.91 |
| large-v2 | 74.31 | 72.56 | 64.13 |
| large-v3 | 78.18 | 73.96 | 64.86 |

The best unfinetuned baseline is `large-v2` on all three evals. It is still far
behind the finetuned checkpoints, which is the main reason the rest of the
writeup focuses on finetuning behavior. The baseline table is archived as
`egs/atc/fig/q01_baseline_all_models_wer_logprob.tsv`; the comparable finetuned
WER/logprob frontier is shown in
[fig/q01_baseline_context_finetuned_avg_logprob_vs_wer_by_eval.png](fig/q01_baseline_context_finetuned_avg_logprob_vs_wer_by_eval.png).

### 2. How does the optimal learning rate change with training dataset size?

These large-v3-turbo sweeps vary the amount of ATC training data. Each cell
shows the best WER and the peak LR that achieved it:

| Training set size | ATC val WER @ LR | ATC test WER @ LR | ATCOSIM WER @ LR |
| --- | ---: | ---: | ---: |
| ATC 10% | 16.78 @ 1.2e-5 | 17.16 @ 1.2e-5 | 19.93 @ 5e-6 |
| ATC 25% | 13.80 @ 1.5e-5 | 13.02 @ 1e-5 | 18.27 @ 7e-6 |
| ATC 50% | 9.66 @ 1.5e-5 | 10.25 @ 1e-5 | 17.80 @ 5e-6 |
| ATC 75% | 8.57 @ 1e-5 | 8.32 @ 1.2e-5 | 17.00 @ 5e-6 |
| ATC 100% | 7.76 @ 1.2e-5 | 8.06 @ 1e-5 | 16.48 @ 5e-6 |

For ATC validation/test, the optimum stays in a narrow band around
`1e-5`-`1.5e-5` as dataset size increases. Cross-evaluation on ATCOSIM prefers
lower LRs, mostly `5e-6`-`7e-6`.

The matching figures show the full curves behind the table:
[fig/q02_dataset_size_lr_vs_wer_by_eval.png](fig/q02_dataset_size_lr_vs_wer_by_eval.png)
plots LR against WER, and
[fig/q02_dataset_size_avg_logprob_vs_wer_by_eval.png](fig/q02_dataset_size_avg_logprob_vs_wer_by_eval.png)
shows the same runs in WER/logprob space.

### 3. How does the optimal learning rate change with model size?

These sweeps use ATC training and compare best WER by model size:

| Model | Trainable params | Width | Enc/dec depth | ATC val WER @ LR | ATC test WER @ LR | ATCOSIM WER @ LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 242M | 768 | 12/12 | 11.23 @ 5e-5 | 10.60 @ 5e-5 | 19.18 @ 2e-5 |
| medium | 764M | 1024 | 24/24 | 7.81 @ 2e-5 | 7.92 @ 1.5e-5 | 16.98 @ 7e-6 |
| turbo | 809M | 1280 | 32/4 | 7.76 @ 1.2e-5 | 8.06 @ 1e-5 | 16.48 @ 5e-6 |
| large-v1 | 1.543B | 1280 | 32/32 | 7.30 @ 1.5e-5 | 7.51 @ 1e-5 | 16.35 @ 7e-6 |
| large-v2 | 1.543B | 1280 | 32/32 | 7.12 @ 1.2e-5 | 7.22 @ 1e-5 | 15.12 @ 1.2e-5 |
| large-v3 | 1.543B | 1280 | 32/32 | 7.17 @ 1.5e-5 | 7.02 @ 1.5e-5 | 15.27 @ 1e-5 |

The optimal LR falls as model size increases: `small` prefers `5e-5`, `medium`
prefers roughly `1.5e-5`-`2e-5`, and `turbo`/`large` models mostly prefer
`1e-5`-`1.5e-5`. The best ATC validation WER is `large-v2` at `1.2e-5`; the
best ATC test WER is `large-v3` at `1.5e-5`; the best ATCOSIM cross-eval WER is
`large-v2` at `1.2e-5`.

For mean decode log probability, the best selections are:

| Eval set | Model | Peak LR | WER | mean_avg_logprob |
| --- | --- | ---: | ---: | ---: |
| ATC validation | large-v3 | 1.2e-5 | 7.22 | -0.0349 |
| ATC test | large-v2 | 2e-5 | 7.61 | -0.0356 |
| ATCOSIM test | large-v2 | 1e-5 | 15.39 | -0.0608 |

The model-size figures are organized the same way:
[fig/q03_model_size_peak_lr_vs_wer_by_eval.png](fig/q03_model_size_peak_lr_vs_wer_by_eval.png)
shows the LR/WER curves, and
[fig/q03_model_size_avg_logprob_vs_wer_by_eval.png](fig/q03_model_size_avg_logprob_vs_wer_by_eval.png)
shows the WER/logprob tradeoff.

### 4. Does training on ATCOSIM help ATC and vice versa?

| Training source | ATC val WER @ LR | ATC test WER @ LR | ATCOSIM WER @ LR |
| --- | ---: | ---: | ---: |
| ATC train | 7.76 @ 1.2e-5 | 8.06 @ 1e-5 | 16.48 @ 5e-6 |
| ATC aug4 train | 6.25 @ 1e-5 | 6.36 @ 7e-6 | 16.26 @ 7e-6 |
| ATC+ATCOSIM train | 8.30 @ 7e-6 | 8.24 @ 1e-5 | 1.21 @ 1e-5 |
| ATCOSIM train | 34.65 @ 5e-6 | 33.60 @ 5.5e-6 | 2.23 @ 8.5e-6 |

ATC training transfers moderately to ATCOSIM (`16.48` WER, or `16.26` with
aug4), but adding ATCOSIM data is much better for ATCOSIM (`1.21` WER).
ATCOSIM training does not help ATC: ATCOSIM-only training is poor on ATC, and
ATC+ATCOSIM is slightly worse than ATC-only on ATC validation/test. The ATC aug4
sweep gives the best ATC validation/test WER among these large-v3-turbo
data/source sweeps.

The transfer comparison uses the same data/source cut as the dataset-size
question. The LR/WER curves are in
[fig/q04_transfer_lr_vs_wer_by_eval.png](fig/q04_transfer_lr_vs_wer_by_eval.png),
the WER/logprob view is in
[fig/q04_transfer_avg_logprob_vs_wer_by_eval.png](fig/q04_transfer_avg_logprob_vs_wer_by_eval.png),
and the version with duration context is
[fig/q04_transfer_lr_wer_with_length_context.png](fig/q04_transfer_lr_wer_with_length_context.png).

Earlier large-v3-turbo tuning at peak LR `1.2e-5` found clip norm `20` best in
the clip sweep (`7.87` ATC validation WER) and 8 frozen encoder layers best in
the frozen-layer sweep with clip `20` (`7.88` ATC validation WER).

Main aggregate artifacts:

- Finetuned model-size table:
  `egs/atc/fig/q03_model_size_all_runs.tsv`
- Best-WER summary:
  `egs/atc/fig/q03_model_size_best_wer_by_eval.tsv`
- Baseline summary:
  `egs/atc/fig/q01_baseline_all_models_wer_logprob.tsv`
- Standalone data/source summary:
  `egs/atc/fig/q02_dataset_size_best_wer_summary.tsv`
- Utterance length histogram data:
  `egs/atc/fig/q00_setup_utterance_length_histograms.tsv`

Reference plots:

| Plot | File |
| --- | --- |
| Utterance length histograms | [PNG](fig/q00_setup_utterance_length_histograms.png) |
| Baseline context: finetuned WER/logprob frontier | [PNG](fig/q01_baseline_context_finetuned_avg_logprob_vs_wer_by_eval.png) |
| Dataset size: LR vs WER by eval | [PNG](fig/q02_dataset_size_lr_vs_wer_by_eval.png) |
| Dataset size: avg logprob vs WER by eval | [PNG](fig/q02_dataset_size_avg_logprob_vs_wer_by_eval.png) |
| Model size: best WER by model size | [PNG](fig/q03_model_size_best_wer_by_model_size.png) |
| Model size: peak LR vs WER by eval | [PNG](fig/q03_model_size_peak_lr_vs_wer_by_eval.png) |
| Model size: avg logprob vs WER by eval | [PNG](fig/q03_model_size_avg_logprob_vs_wer_by_eval.png) |
| Model size: peak LR vs mean decode logprob | [PNG](fig/q03_model_size_peak_lr_vs_mean_logprob_logprob_gt_neg0p1.png) |
| Model size: quadratic basin approximation | [PNG](fig/q03_model_size_quadratic_logprob_basin.png) |
| ATC/ATCOSIM transfer: LR vs WER by eval | [PNG](fig/q04_transfer_lr_vs_wer_by_eval.png) |
| ATC/ATCOSIM transfer: avg logprob vs WER by eval | [PNG](fig/q04_transfer_avg_logprob_vs_wer_by_eval.png) |
| ATC/ATCOSIM transfer: LR/WER with length context | [PNG](fig/q04_transfer_lr_wer_with_length_context.png) |

## Reproducing Data and Runs

The README links point to portable result artifacts committed under
`egs/atc/fig/`. The manifests, WAV files, checkpoints, and raw sweep outputs
under `egs/atc/data/` and `egs/atc/exp/` are generated locally and are not
committed.

Prepare the default ATC ASR data from
[`jacktol/ATC-ASR-Dataset`](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset):

```bash
python3 egs/atc/local/prepare_atc.py
```

This creates `egs/atc/data/train.jsonl`, `validation.jsonl`, `test.jsonl`,
split WAV lists, references, copied audio under `egs/atc/data/wav/`, and
`egs/atc/data/manifest.json`.

The dataset-size sweeps use deterministic subsets of the ATC train manifest:

```bash
python3 egs/atc/local/subsample_train.py
```

This writes `train_subsample_10pct.jsonl`, `train_subsample_25pct.jsonl`,
`train_subsample_50pct.jsonl`, and `train_subsample_75pct.jsonl` under
`egs/atc/data/`.

The ATC aug4 condition keeps the original training utterances and adds three
offline audio augmentations per utterance:

```bash
python3 egs/atc/local/augment_train.py
```

ATCOSIM is prepared from
[`Jzuluaga/atcosim_corpus`](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus)
and can be mixed with ATC for the transfer experiments:

```bash
python3 egs/atc/local/prepare_atcosim.py
python3 egs/atc/local/mix_train_sources.py \
  --inputs egs/atc/data/train.jsonl egs/atc/data/atcosim/train.jsonl \
  --out egs/atc/data/train_mixed_atcosim.jsonl
```

The baseline matrix in question 1 decodes the unfinetuned Whisper models on
ATC validation, ATC test, and ATCOSIM test:

```bash
python3 egs/atc/local/run_baseline_model_matrix.py --test-retries 1
```

Finetuning sweeps are run with `run_lr_beta_sweep.py`; each sweep writes a
timestamped directory under `egs/atc/exp/` with `results*.jsonl` files that the
plotting scripts consume. This representative mixed-source run matches the
no-timestamp/max-96 decode setup used in the tables above:

```bash
python3 egs/atc/local/run_lr_beta_sweep.py \
  --train-jsonl egs/atc/data/train_mixed_atcosim.jsonl \
  --validation-jsonl egs/atc/data/validation.jsonl \
  --sweep-name "$(date +%Y%m%d_%H%M%S)_atc_mixed_atcosim_notimestamps_max96" \
  --learning-rates 1.2e-5 \
  --beta1-values 0 \
  --beta2-values 0.9999 \
  --clip-grad-norm-values 1 \
  --frozen-encoder-layers-values 8 \
  --lr-scheduler-type linear \
  --max-new-tokens 96 \
  --expected-runs 1 \
  --skip-baseline
```

Existing checkpoints can be cross-evaluated on another manifest without
changing their original validation decodes:

```bash
python3 egs/atc/local/run_lr_beta_sweep.py \
  --eval-only \
  --model-root egs/atc/exp/20260611_230217_atc_mixed_atcosim_lr10_notimestamps_max96 \
  --sweep-name 20260612_mixed_lr10_atcosim_test \
  --train-jsonl egs/atc/data/train_mixed_atcosim.jsonl \
  --validation-jsonl egs/atc/data/atcosim/test.jsonl \
  --eval-name atcosim_test \
  --learning-rates 1e-6 2e-6 3e-6 4e-6 5e-6 5.5e-6 6e-6 7e-6 8.5e-6 1e-5 \
  --beta1-values 0 \
  --beta2-values 0.9999 \
  --clip-grad-norm-values 1 \
  --frozen-encoder-layers-values 8 \
  --lr-scheduler-type linear \
  --max-new-tokens 96 \
  --expected-runs 10 \
  --skip-baseline
```

The committed figures were copied from local `egs/atc/exp/` outputs into
`egs/atc/fig/` with question-oriented names. The plotting scripts used for the
writeup are:

- `egs/atc/local/plot_length_histograms.py`
- `egs/atc/local/plot_lr_wer_and_lengths.py`
- `egs/atc/local/plot_avg_logprob_vs_wer.py`
- `egs/atc/local/plot_lr_avg_logprob_by_eval.py`
- `egs/atc/local/plot_model_lr_comparison.py`

After the standard sweeps and cross-evaluations have been run, the standalone
data/source comparison can be regenerated with automatic discovery:

```bash
python3 egs/atc/local/plot_lr_wer_and_lengths.py --standalone \
  --out egs/atc/exp/standalone_lr_wer_and_lengths.png \
  --eval-pane-out egs/atc/exp/standalone_lr_wer_by_eval.png \
  --summary-tsv egs/atc/exp/standalone_lr_wer_summary.tsv
```
