#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PLU_OPS_BACKEND", "triton")

import torch

import plu.test as plu_test
from plu.benchmarks.bench_end_to_end import pack_mx_model


DEFAULT_EXP = Path("egs/uk1e2/exp/news_100_large-v3-turbo_bf16_b1_ebwd24_cudagraph_mxfp8/train")
DEFAULT_AUDIO = Path("data/segments/wav/S00000-N00111923961-U0000000-0000693-0002622.wav")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PLU Completion Sampler</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f5;
      --ink: #1d2329;
      --muted: #5c6872;
      --line: #cfd6dc;
      --panel: #ffffff;
      --accent: #0b6f6a;
      --accent-strong: #084f4b;
      --warn: #875400;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 16px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      overflow-wrap: anywhere;
    }
    form {
      display: grid;
      gap: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      font-weight: 600;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea {
      min-height: 96px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 160px 120px;
      gap: 12px;
      align-items: end;
    }
    button {
      height: 38px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      background: #8ba7a4;
      cursor: wait;
    }
    #status {
      min-height: 22px;
      color: var(--muted);
    }
    #status.error { color: #9b1c1c; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }
    .metric, .result {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric {
      padding: 10px 12px;
      min-height: 64px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 18px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .results {
      display: grid;
      gap: 10px;
    }
    .result {
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      flex-wrap: wrap;
    }
    .result p {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
    }
    .empty {
      color: var(--warn);
    }
    @media (max-width: 760px) {
      main { width: min(100vw - 20px, 1120px); margin: 12px auto; }
      header, .row { display: grid; grid-template-columns: 1fr; }
      .meta { text-align: left; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>PLU Completion Sampler</h1>
      <div id="config" class="meta"></div>
    </header>
    <form id="form">
      <label>Audio path
        <input id="audio" name="audio" autocomplete="off">
      </label>
      <label>Prompt
        <textarea id="prompt" name="prompt" spellcheck="false"></textarea>
      </label>
      <div class="row">
        <label>Language
          <input id="language" name="language" autocomplete="off">
        </label>
        <label>Max tokens
          <input id="maxNewTokens" name="maxNewTokens" type="number" min="1" max="448">
        </label>
        <button id="submit" type="submit">Sample</button>
      </div>
      <div id="status"></div>
    </form>
    <section id="metrics" class="metrics"></section>
    <section id="results" class="results"></section>
  </main>
  <script>
    const form = document.querySelector("#form");
    const submit = document.querySelector("#submit");
    const statusEl = document.querySelector("#status");
    const metricsEl = document.querySelector("#metrics");
    const resultsEl = document.querySelector("#results");
    const configEl = document.querySelector("#config");
    const audioEl = document.querySelector("#audio");
    const promptEl = document.querySelector("#prompt");
    const languageEl = document.querySelector("#language");
    const maxNewTokensEl = document.querySelector("#maxNewTokens");

    function setStatus(text, error = false) {
      statusEl.textContent = text;
      statusEl.className = error ? "error" : "";
    }

    function metric(label, value) {
      const node = document.createElement("div");
      node.className = "metric";
      const name = document.createElement("span");
      name.textContent = label;
      const val = document.createElement("strong");
      val.textContent = value;
      node.append(name, val);
      return node;
    }

    function render(data) {
      metricsEl.replaceChildren(
        metric("elapsed", `${data.elapsed_seconds.toFixed(3)}s`),
        metric("audio", `${data.audio_seconds.toFixed(2)}s`),
        metric("RTF", data.rtf.toFixed(4)),
        metric("tokens/s", data.tokens_per_second.toFixed(1)),
        metric("batch", String(data.decode_batch_size)),
      );
      resultsEl.replaceChildren();
      for (const row of data.completions) {
        const item = document.createElement("article");
        item.className = "result";
        const head = document.createElement("div");
        head.className = "result-head";
        head.textContent = `alt ${row.i} | ${row.strategy} | avg ${row.avg_logprob.toFixed(3)} | tokens ${row.generated_token_count} | no_speech ${row.no_speech_prob.toFixed(3)}`;
        const text = document.createElement("p");
        text.textContent = row.text || "(empty)";
        if (!row.text) text.className = "empty";
        item.append(head, text);
        resultsEl.append(item);
      }
    }

    async function loadConfig() {
      const response = await fetch("/api/config");
      const data = await response.json();
      audioEl.value = data.default_audio;
      languageEl.value = data.language || "";
      maxNewTokensEl.value = data.max_new_tokens;
      configEl.textContent = `${data.model} | ${data.device}/${data.dtype} | B=${data.decode_batch_size}`;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      submit.disabled = true;
      setStatus("sampling");
      try {
        const response = await fetch("/api/sample", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({
            audio: audioEl.value,
            prompt: promptEl.value,
            language: languageEl.value,
            max_new_tokens: Number(maxNewTokensEl.value),
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || response.statusText);
        render(data);
        setStatus("done");
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        submit.disabled = false;
      }
    });

    loadConfig().catch((error) => setStatus(error.message, true));
  </script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a PLU completion sampling demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--model", default=None, help="HF model id, alias, or local model directory. Overrides --exp.")
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    parser.add_argument("--language", default="uk")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args(argv)


def root_relative(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class Runtime:
    args: argparse.Namespace

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.loaded = False
        self.model_path: Path | None = None
        self.model = None
        self.tokenizer = None
        self.packed = None
        self.sampler = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.language = plu_test.normalize_language(self.args.language)

    def load(self) -> None:
        if self.loaded:
            return
        with self.lock:
            if self.loaded:
                return
            self.device = plu_test.resolve_device(self.args.device)
            self.dtype = plu_test.parse_dtype(self.args.dtype)
            requested = SimpleNamespace(
                exp=root_relative(self.args.exp),
                model=self.args.model,
                download_root=root_relative(self.args.download_root),
            )
            self.model_path = plu_test.resolve_requested_model(requested)
            self.model = plu_test.WhisperForConditionalGeneration.from_pretrained(self.model_path, map_location="cpu")
            self.model.eval().to(device=self.device, dtype=self.dtype)
            self.tokenizer = plu_test.configure_tokenizer(self.model_path, self.model, self.language)
            self.packed, stats = pack_mx_model(self.model, "mxfp8")
            prompt = plu_test.prompt_ids(self.tokenizer, self.language)
            suppress_tokens = [token for token in self.tokenizer.special_tokens.values() if token != self.model.config.eos_token_id]
            self.sampler = plu_test.Mxfp8KvCacheCudaGraphSampler(
                self.model,
                self.tokenizer,
                self.packed,
                prompt,
                suppress_tokens,
                decode_batch_size=self.args.decode_batch_size,
            )
            self.packed_linear_count = int(stats["mx_packed_linear_count"])
            self.loaded = True

    def sample(self, payload: dict[str, object]) -> dict[str, object]:
        self.load()
        assert self.device is not None
        assert self.dtype is not None
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.packed is not None
        assert self.sampler is not None

        audio_value = str(payload.get("audio") or self.args.audio)
        audio_path = root_relative(Path(audio_value))
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(f"audio file does not exist: {audio_value}")

        language_value = payload.get("language") or self.language
        language = plu_test.normalize_language(str(language_value)) if language_value else None
        max_new_tokens = int(payload.get("max_new_tokens") or self.args.max_new_tokens)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        prompt_text = str(payload.get("prompt") or "")
        previous_tokens = self.tokenizer.encode(prompt_text, disallowed_special=()) if prompt_text.strip() else None
        max_previous_tokens = self.model.config.max_target_positions // 2 - 1
        prompt = plu_test.prompt_ids(
            self.tokenizer,
            language,
            previous_tokens=previous_tokens,
            max_previous_tokens=max_previous_tokens,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        features, audio_seconds = plu_test.load_input_features(audio_path, self.model.config.num_mel_bins, self.device, self.dtype)
        encoder_hidden_states = plu_test.encode_packed_mxfp8(self.model, self.packed, features)
        samples = self.sampler.sample_many(encoder_hidden_states, max_new_tokens, prompt=prompt)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_seconds = time.perf_counter() - start

        completions = []
        total_generated_tokens = 0
        prompt_len = min(len(prompt), self.model.config.max_target_positions)
        for index, (token_ids, logprobs, no_speech_prob) in enumerate(samples):
            generated = token_ids[prompt_len:]
            text = self.tokenizer.batch_decode([generated], skip_special_tokens=True)[0].strip()
            generated_count = len(logprobs)
            total_generated_tokens += generated_count
            avg_logprob = sum(logprobs) / generated_count if generated_count else 0.0
            completions.append(
                {
                    "i": index,
                    "strategy": "greedy" if index == 0 else "sample",
                    "text": text,
                    "token_ids": token_ids,
                    "generated_token_ids": generated,
                    "generated_token_count": generated_count,
                    "avg_logprob": float(avg_logprob),
                    "cumulative_logprob": float(sum(logprobs)),
                    "token_logprobs": [float(value) for value in logprobs],
                    "no_speech_prob": float(no_speech_prob),
                }
            )

        return {
            "audio": str(audio_path),
            "audio_seconds": float(audio_seconds),
            "elapsed_seconds": elapsed_seconds,
            "rtf": elapsed_seconds / audio_seconds if audio_seconds else 0.0,
            "tokens_per_second": total_generated_tokens / elapsed_seconds if elapsed_seconds else 0.0,
            "decode_batch_size": self.args.decode_batch_size,
            "max_new_tokens": max_new_tokens,
            "prompt_token_count": len(prompt),
            "previous_token_count": len(previous_tokens or []),
            "mxfp8_packed_linear_count": self.packed_linear_count,
            "completions": completions,
        }

    def config(self) -> dict[str, object]:
        model = self.args.model if self.args.model is not None else str(self.args.exp)
        return {
            "model": model,
            "device": self.args.device,
            "dtype": self.args.dtype,
            "language": self.language,
            "default_audio": str(self.args.audio),
            "decode_batch_size": self.args.decode_batch_size,
            "max_new_tokens": self.args.max_new_tokens,
            "loaded": self.loaded,
        }


class DemoServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], runtime: Runtime):
        super().__init__(server_address, DemoHandler)
        self.runtime = runtime


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/config":
            self.send_json(self.server.runtime.config())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/sample":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length > 1_000_000:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON request must be an object")
            self.send_json(self.server.runtime.sample(payload))
        except Exception as exc:
            self.send_json(
                {"error": str(exc), "traceback": traceback.format_exc()},
                status=HTTPStatus.BAD_REQUEST,
            )

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    args = parse_args()
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be at least 1")
    runtime = Runtime(args)
    server = DemoServer((args.host, args.port), runtime)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"serving {url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
