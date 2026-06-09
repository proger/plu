# UK1E2 100-File End-to-End Recipe

This recipe builds a deterministic 100-file subset from:

- `data/segments/wav.scp`
- `data/local/text.full`

It prepares PLU JSONL labels, runs PLU-backed `+test` before training, trains a local full checkpoint, runs PLU-backed `+test` after training, and scores WER from the `+test` JSONL outputs.

Default model settings match the accelerated large-v3-turbo path and require CUDA:

- training model: `openai/whisper-large-v3-turbo`
- baseline PLU model: `large-v3-turbo`
- subset id prefix: `N` (news)
- test language: `uk`
- test dtype: CUDA `bf16`, with MXFP8-packed linear weights
- learning rate: `1e-6`
- train steps: one pass over the selected subset (`100` rows by default)

Run:

```bash
python egs/uk1e2/run.py
```

Useful overrides:

```bash
N=100 python egs/uk1e2/run.py
python egs/uk1e2/run.py --stage 3 --stop-stage 5
INIT=openai/whisper-small BASELINE_MODEL=small N_MELS=80 python egs/uk1e2/run.py
```

Outputs are written under `egs/uk1e2/exp/news_100_large-v3-turbo_bf16_b1_ebwd24_cudagraph_mxfp8/` by default:

- `data/subset.jsonl`: train/eval JSONL with `input_ids`
- `data/wav.list`: the 100 audio files passed to `+test`
- `before.test.jsonl`: baseline `+test` segments
- `before.wer.json`: baseline WER
- `train/`: PLU training output
- `after.test.jsonl`: post-training `+test` segments
- `after.wer.json`: post-training WER
- `wer.txt`: compact before/after summary

The WER scorer normalizes text with Unicode NFKC, casefolding, and punctuation removal while preserving letters, numbers, and apostrophes.
