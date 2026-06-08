#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP = ROOT / "egs/uk1e2/exp/context_smc_b8_kernelctl_fixed_20260607_150053"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve aligned utterances with paginated audio review.")
    parser.add_argument("--align", type=Path, default=DEFAULT_EXP / "align.jsonl")
    parser.add_argument("--decode", type=Path, default=DEFAULT_EXP / "decode.jsonl")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--default-pad", type=float, default=0.0)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--selected-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clip-cache-dir",
        type=Path,
        default=None,
        help="Directory for generated WAV clips. Defaults to '<exp-dir>/alignment_failure_clips'.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold().replace("’", "'")
    chars = []
    for char in text:
        chars.append(char if (char.isalnum() or char == "'") else " ")
    return " ".join("".join(chars).split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            cost = previous[j - 1] if ref_word == hyp_word else previous[j - 1] + 1
            current.append(min(cost, previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1]


def domain_for(recording_id: str) -> str:
    for char in recording_id:
        if char.isalpha():
            return char.upper()
    return "?"


def load_examples(align_path: Path, decode_path: Path, root: Path, *, selected_only: bool) -> list[dict]:
    examples: list[dict] = []
    with align_path.open("r", encoding="utf-8") as align_f, decode_path.open("r", encoding="utf-8") as decode_f:
        for line_number, (align_line, decode_line) in enumerate(zip(align_f, decode_f), 1):
            if not align_line.strip() or not decode_line.strip():
                continue
            row = json.loads(align_line)
            decode = json.loads(decode_line)
            if row.get("decode_id") != decode.get("id"):
                raise ValueError(f"{align_path}:{line_number}: decode_id does not match decode.jsonl")
            sample_selected = decode.get("sample_selected", decode.get("smc_selected")) is True
            if selected_only and not sample_selected:
                continue

            source_path = str(row.get("path") or "")
            media = root / source_path
            speech_start = float(row.get("speech_start") or 0.0)
            speech_end = float(row.get("speech_end") or speech_start)
            ref_text = str(row.get("reference_counterpart") or "")
            hyp_text = str(row.get("hyp_text") or "")
            ref_words = normalize(ref_text).split()
            hyp_words = normalize(hyp_text).split()
            edits = edit_distance(ref_words, hyp_words) if ref_words or hyp_words else 0
            match_kind = str(row.get("match_kind") or "")
            alignment_failed = match_kind != "overlap"
            text_error = edits > 0
            index = len(examples)
            examples.append(
                {
                    "index": index,
                    "decode_id": row.get("decode_id"),
                    "utt_id": row.get("utt_id"),
                    "recording_id": row.get("recording_id") or "",
                    "domain": domain_for(str(row.get("recording_id") or "")),
                    "alternative": row.get("alternative"),
                    "match_kind": match_kind,
                    "counterpart_method": row.get("reference_counterpart_method"),
                    "speech_start": round(speech_start, 2),
                    "speech_end": round(speech_end, 2),
                    "duration": round(max(0.0, speech_end - speech_start), 2),
                    "hyp_text": hyp_text,
                    "reference_counterpart": ref_text,
                    "reference_ids": row.get("reference_ids") or [],
                    "path": source_path,
                    "path_exists": media.exists(),
                    "avg_logprob": row.get("avg_logprob"),
                    "no_speech_prob": row.get("no_speech_prob"),
                    "sample_selected": sample_selected,
                    "decode_strategy": decode.get("decode_strategy"),
                    "ref_words": len(ref_words),
                    "hyp_words": len(hyp_words),
                    "edit_count": edits,
                    "row_wer": round(edits / len(ref_words), 4) if ref_words else None,
                    "alignment_failed": alignment_failed,
                    "text_error": text_error,
                    "review_error": alignment_failed or text_error,
                }
            )
    return examples


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def page_html(title: str, default_pad: float, default_page_size: int) -> bytes:
    escaped_title = html.escape(title)
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #202428;
      --muted: #5f6b73;
      --line: #d8dee2;
      --accent: #2f6f73;
      --warn: #8a5a20;
      --bad: #9a3412;
      --good: #2f6f4f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(246, 247, 248, 0.97);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 14px 18px; }}
    h1 {{ margin: 0 0 10px; font-size: 20px; font-weight: 650; letter-spacing: 0; }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 150px 120px 170px 150px 110px;
      gap: 10px;
      align-items: end;
    }}
    label {{ display: grid; gap: 4px; color: var(--muted); font-size: 12px; }}
    input, select, button {{
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 9px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    button:disabled {{ cursor: default; opacity: 0.45; }}
    .stats, .pager {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      color: var(--muted);
      align-items: center;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
    }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 12px 18px 32px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 160px;
      z-index: 2;
      background: #eef2f3;
      color: #39434a;
      font-size: 12px;
      font-weight: 650;
    }}
    tr:hover td {{ background: #fafafa; }}
    audio {{ width: 240px; max-width: 100%; height: 32px; }}
    .meta {{ color: var(--muted); font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }}
    .hyp {{ font-weight: 600; line-height: 1.35; }}
    .ref {{ color: #37434a; line-height: 1.35; margin-top: 6px; }}
    .empty {{ color: var(--warn); }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 1px 7px;
      color: var(--muted);
      background: #fff;
      margin: 0 4px 4px 0;
      font-size: 12px;
    }}
    .pill.bad {{ color: var(--bad); border-color: #e4b7a4; background: #fff7f2; }}
    .pill.good {{ color: var(--good); border-color: #b7d8c4; background: #f2fbf5; }}
    .missing, .audio-error {{ color: var(--bad); min-height: 16px; }}
    @media (max-width: 980px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      th {{ top: 220px; }}
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); }}
      td {{ border-bottom: 0; }}
      audio {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{escaped_title}</h1>
      <div class="toolbar">
        <label>Search
          <input id="search" type="search" placeholder="recording, text, path">
        </label>
        <label>Review
          <select id="review">
            <option value="errors" selected>Review errors</option>
            <option value="all">All selected</option>
            <option value="clean">Clean aligned</option>
            <option value="alignment_failed">Alignment failures</option>
            <option value="text_errors">Transcript errors</option>
          </select>
        </label>
        <label>Match
          <select id="match"></select>
        </label>
        <label>Sort
          <select id="sort">
            <option value="index">File order</option>
            <option value="edits_desc">Edits desc</option>
            <option value="wer_desc">Row WER desc</option>
            <option value="duration_desc">Duration desc</option>
            <option value="logprob_asc">Avg logprob asc</option>
            <option value="recording">Recording</option>
          </select>
        </label>
        <label>Domain
          <select id="domain"></select>
        </label>
        <label>Audio pad
          <input id="pad" type="number" value="{default_pad:.1f}" min="0" max="10" step="0.5">
        </label>
      </div>
      <div class="pager">
        <button id="prev">Prev</button>
        <span id="pageStatus" class="stat"></span>
        <button id="next">Next</button>
        <label>Page size
          <select id="pageSize">
            <option value="25">25</option>
            <option value="50" selected>50</option>
            <option value="100">100</option>
            <option value="200">200</option>
          </select>
        </label>
      </div>
      <div id="stats" class="stats"></div>
    </div>
  </header>
  <main>
    <table>
      <colgroup>
        <col style="width: 280px">
        <col style="width: 250px">
        <col>
      </colgroup>
      <thead>
        <tr>
          <th>Audio</th>
          <th>Source</th>
          <th>Utterance</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script>
    const defaults = {{ pageSize: {default_page_size}, pad: {default_pad:.1f} }};
    const state = {{
      search: '',
      review: 'errors',
      match: 'all',
      domain: 'all',
      sort: 'index',
      page: 1,
      pageSize: defaults.pageSize,
      pad: defaults.pad,
    }};
    const searchEl = document.getElementById('search');
    const reviewEl = document.getElementById('review');
    const matchEl = document.getElementById('match');
    const domainEl = document.getElementById('domain');
    const sortEl = document.getElementById('sort');
    const padEl = document.getElementById('pad');
    const pageSizeEl = document.getElementById('pageSize');
    const prevEl = document.getElementById('prev');
    const nextEl = document.getElementById('next');
    const pageStatusEl = document.getElementById('pageStatus');
    const rowsEl = document.getElementById('rows');
    const statsEl = document.getElementById('stats');
    let lastResponse = null;
    let debounceTimer = null;

    pageSizeEl.value = String(defaults.pageSize);

    function esc(s) {{
      return String(s ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function audioUrl(example) {{
      const pad = Number.isFinite(Number(state.pad)) ? Number(state.pad) : 0;
      return `/audio?i=${{example.index}}&pad=${{encodeURIComponent(pad)}}`;
    }}
    function fillSelect(select, values, label) {{
      const current = select.value || 'all';
      select.innerHTML = '';
      const all = document.createElement('option');
      all.value = 'all';
      all.textContent = `All ${{label}}`;
      select.appendChild(all);
      values.forEach(value => {{
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }});
      select.value = Array.from(select.options).some(opt => opt.value === current) ? current : 'all';
    }}
    async function loadOptions() {{
      const res = await fetch('/api/options');
      const data = await res.json();
      fillSelect(matchEl, data.match_kinds, 'matches');
      fillSelect(domainEl, data.domains, 'domains');
    }}
    function queryParams() {{
      const params = new URLSearchParams();
      params.set('q', state.search);
      params.set('review', state.review);
      params.set('match', state.match);
      params.set('domain', state.domain);
      params.set('sort', state.sort);
      params.set('page', String(state.page));
      params.set('page_size', String(state.pageSize));
      return params;
    }}
    async function fetchPage() {{
      const res = await fetch(`/api/examples?${{queryParams().toString()}}`);
      if (!res.ok) throw new Error(`API ${{res.status}}`);
      lastResponse = await res.json();
      render(lastResponse);
    }}
    function renderStats(data) {{
      const parts = [
        `<span class="stat">Filtered ${{data.total}} / ${{data.overall.total}}</span>`,
        `<span class="stat">errors ${{data.counts.review_errors}}</span>`,
        `<span class="stat">clean ${{data.counts.clean}}</span>`,
        `<span class="stat">alignment failures ${{data.counts.alignment_failed}}</span>`,
        `<span class="stat">text errors ${{data.counts.text_errors}}</span>`,
      ];
      statsEl.innerHTML = parts.join('');
    }}
    function render(data) {{
      renderStats(data);
      pageStatusEl.textContent = `Page ${{data.page}} / ${{data.total_pages || 1}}`;
      prevEl.disabled = data.page <= 1;
      nextEl.disabled = data.page >= data.total_pages;
      rowsEl.innerHTML = data.rows.map(e => `
        <tr>
          <td>
            <audio controls preload="none" data-index="${{e.index}}" src="${{audioUrl(e)}}"></audio>
            <div class="meta audio-error"></div>
            <div class="meta">
              <span class="pill ${{e.alignment_failed ? 'bad' : 'good'}}">${{esc(e.match_kind)}}</span>
              <span class="pill ${{e.text_error ? 'bad' : 'good'}}">${{e.edit_count}} edits</span>
              <span class="pill">A${{String(e.alternative).padStart(2, '0')}}</span>
              <span class="pill">${{esc(e.domain)}}</span>
              <div>${{e.speech_start.toFixed(2)}}-${{e.speech_end.toFixed(2)}}s (${{e.duration.toFixed(2)}}s)</div>
              <div>ref ${{e.ref_words}} · hyp ${{e.hyp_words}} · row WER ${{e.row_wer === null ? 'n/a' : (100 * e.row_wer).toFixed(1) + '%'}}</div>
              <div>avg_logprob ${{e.avg_logprob ?? ''}}</div>
              ${{e.path_exists ? '' : '<div class="missing">media path missing</div>'}}
            </div>
          </td>
          <td>
            <div class="meta"><strong>${{esc(e.recording_id)}}</strong></div>
            <div class="meta path">${{esc(e.path)}}</div>
            <div class="meta path">${{esc(e.decode_id)}}</div>
          </td>
          <td>
            <div class="hyp">${{esc(e.hyp_text) || '<span class="empty">(empty hypothesis)</span>'}}</div>
            <div class="ref">${{esc(e.reference_counterpart) || '<span class="empty">(no aligned reference)</span>'}}</div>
            ${{e.reference_ids.length ? `<div class="meta path">${{esc(e.reference_ids.join(' '))}}</div>` : ''}}
          </td>
        </tr>
      `).join('');
      rowsEl.querySelectorAll('audio').forEach(audio => {{
        audio.addEventListener('error', () => {{
          const target = audio.nextElementSibling;
          if (target) {{
            const code = audio.error ? audio.error.code : 'unknown';
            target.textContent = `Playback failed (media error ${{code}}). Try replaying; the clip endpoint is logged server-side.`;
          }}
        }});
        audio.addEventListener('playing', () => {{
          const target = audio.nextElementSibling;
          if (target) target.textContent = '';
        }});
      }});
    }}
    function updateAudioUrls() {{
      rowsEl.querySelectorAll('audio[data-index]').forEach(audio => {{
        const next = `/audio?i=${{audio.dataset.index}}&pad=${{encodeURIComponent(state.pad)}}`;
        if (audio.getAttribute('src') === next) return;
        audio.setAttribute('src', next);
        const target = audio.nextElementSibling;
        if (target) target.textContent = '';
      }});
    }}
    function resetAndFetch() {{
      state.page = 1;
      fetchPage().catch(err => {{
        rowsEl.innerHTML = `<tr><td colspan="3" class="missing">${{esc(err.message)}}</td></tr>`;
      }});
    }}
    function debounceFetch() {{
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(resetAndFetch, 180);
    }}
    searchEl.addEventListener('input', () => {{ state.search = searchEl.value; debounceFetch(); }});
    [reviewEl, matchEl, domainEl, sortEl, pageSizeEl].forEach(el => {{
      el.addEventListener('input', () => {{
        state.review = reviewEl.value;
        state.match = matchEl.value;
        state.domain = domainEl.value;
        state.sort = sortEl.value;
        state.pageSize = Number(pageSizeEl.value);
        resetAndFetch();
      }});
    }});
    padEl.addEventListener('input', () => {{
      state.pad = padEl.value;
      updateAudioUrls();
    }});
    prevEl.addEventListener('click', () => {{ if (state.page > 1) {{ state.page -= 1; fetchPage(); }} }});
    nextEl.addEventListener('click', () => {{ if (lastResponse && state.page < lastResponse.total_pages) {{ state.page += 1; fetchPage(); }} }});
    loadOptions().then(fetchPage);
  </script>
</body>
</html>
"""
    return doc.encode("utf-8")


def filter_examples(examples: list[dict], params: dict[str, list[str]]) -> list[dict]:
    query = (params.get("q", [""])[0] or "").strip().casefold()
    review = params.get("review", ["errors"])[0]
    match = params.get("match", ["all"])[0]
    domain = params.get("domain", ["all"])[0]

    rows = []
    for example in examples:
        if match != "all" and example["match_kind"] != match:
            continue
        if domain != "all" and example["domain"] != domain:
            continue
        if review == "errors" and not example["review_error"]:
            continue
        if review == "clean" and example["review_error"]:
            continue
        if review == "alignment_failed" and not example["alignment_failed"]:
            continue
        if review == "text_errors" and (example["alignment_failed"] or not example["text_error"]):
            continue
        if query:
            haystack = " ".join(
                [
                    str(example["decode_id"]),
                    str(example["recording_id"]),
                    str(example["hyp_text"]),
                    str(example["reference_counterpart"]),
                    str(example["path"]),
                ]
            ).casefold()
            if query not in haystack:
                continue
        rows.append(example)
    return rows


def sort_examples(examples: list[dict], sort_key: str) -> None:
    if sort_key == "duration_desc":
        examples.sort(key=lambda row: (-row["duration"], row["index"]))
    elif sort_key == "logprob_asc":
        examples.sort(key=lambda row: (row["avg_logprob"] if row["avg_logprob"] is not None else 999.0, row["index"]))
    elif sort_key == "recording":
        examples.sort(key=lambda row: (row["recording_id"], row["speech_start"], row["index"]))
    elif sort_key == "edits_desc":
        examples.sort(key=lambda row: (-row["edit_count"], row["index"]))
    elif sort_key == "wer_desc":
        examples.sort(key=lambda row: (-(row["row_wer"] if row["row_wer"] is not None else 999.0), row["index"]))
    else:
        examples.sort(key=lambda row: row["index"])


def summarize(examples: list[dict]) -> dict:
    match_counts = Counter(row["match_kind"] for row in examples)
    domain_counts = Counter(row["domain"] for row in examples)
    return {
        "total": len(examples),
        "review_errors": sum(1 for row in examples if row["review_error"]),
        "clean": sum(1 for row in examples if not row["review_error"]),
        "alignment_failed": sum(1 for row in examples if row["alignment_failed"]),
        "text_errors": sum(1 for row in examples if row["text_error"] and not row["alignment_failed"]),
        "match_counts": dict(sorted(match_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


class ReviewServer(BaseHTTPRequestHandler):
    server_version = "AlignmentReviewServer/1.0"

    def do_HEAD(self) -> None:
        self.handle_request(send_body=False)

    def do_GET(self) -> None:
        self.handle_request(send_body=True)

    def handle_request(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(self.server.page, "text/html; charset=utf-8", send_body=send_body)
            return
        if parsed.path == "/api/options":
            payload = {
                "domains": sorted({row["domain"] for row in self.server.examples}),
                "match_kinds": sorted({row["match_kind"] for row in self.server.examples}),
                "overall": self.server.overall,
            }
            self.send_bytes(json_bytes(payload), "application/json; charset=utf-8", send_body=send_body)
            return
        if parsed.path == "/api/examples":
            self.serve_examples(parsed.query, send_body=send_body)
            return
        if parsed.path == "/audio":
            self.serve_audio(parsed.query, send_body=send_body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_bytes(
        self,
        payload: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        send_body: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not send_body:
            return
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return

    def serve_examples(self, query: str, *, send_body: bool) -> None:
        params = parse_qs(query)
        try:
            page = max(1, int(params.get("page", ["1"])[0]))
            page_size = max(1, min(500, int(params.get("page_size", [str(self.server.default_page_size)])[0])))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid pagination")
            return

        filtered = filter_examples(self.server.examples, params)
        sort_examples(filtered, params.get("sort", ["index"])[0])
        total = len(filtered)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, total_pages)
        start = (page - 1) * page_size
        rows = filtered[start : start + page_size]
        payload = {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "rows": rows,
            "counts": summarize(filtered),
            "overall": self.server.overall,
        }
        self.send_bytes(json_bytes(payload), "application/json; charset=utf-8", send_body=send_body)

    def serve_audio(self, query: str, *, send_body: bool) -> None:
        params = parse_qs(query)
        try:
            index = int(params.get("i", [""])[0])
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid example index")
            return
        if index < 0 or index >= len(self.server.examples):
            self.send_error(HTTPStatus.NOT_FOUND, "example index not found")
            return
        try:
            pad = float(params.get("pad", [str(self.server.default_pad)])[0])
        except ValueError:
            pad = self.server.default_pad
        pad = max(0.0, min(10.0, pad))

        example = self.server.examples[index]
        source = self.server.root / example["path"]
        if not source.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"media not found: {example['path']}")
            return

        start = max(0.0, float(example["speech_start"]) - pad)
        end = max(float(example["speech_end"]) + pad, start + 0.2)
        duration = min(120.0, end - start)
        try:
            clip_path = self.ensure_clip(index, source, start, duration, pad)
        except RuntimeError as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)[:500])
            return
        self.serve_file(clip_path, "audio/wav", send_body=send_body)

    def ensure_clip(self, index: int, source: Path, start: float, duration: float, pad: float) -> Path:
        pad_ms = round(pad * 1000)
        start_cs = round(start * 100)
        duration_cs = round(duration * 100)
        clip_path = self.server.clip_cache_dir / f"clip_{index:05d}_p{pad_ms:05d}_s{start_cs:08d}_d{duration_cs:06d}.wav"
        if clip_path.exists() and clip_path.stat().st_size > 44:
            return clip_path

        self.server.clip_cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{clip_path.name}.", suffix=".tmp", dir=self.server.clip_cache_dir)
        os.close(fd)
        tmp_path = Path(tmp_name)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            tmp_path.unlink(missing_ok=True)
            message = exc.stderr.decode("utf-8", errors="replace") or str(exc)
            raise RuntimeError(message)
        if tmp_path.stat().st_size <= 44:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg produced an empty clip for {source} at {start:.2f}s")
        tmp_path.replace(clip_path)
        return clip_path

    def serve_file(self, path: Path, content_type: str, *, send_body: bool) -> None:
        size = path.stat().st_size
        status = HTTPStatus.OK
        start = 0
        end = size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            elif last:
                suffix = int(last)
                start = max(0, size - suffix)
                end = size - 1
            if start >= size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "public, max-age=3600")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        try:
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


class Server(ThreadingHTTPServer):
    def __init__(
        self,
        addr,
        handler,
        *,
        root: Path,
        examples: list[dict],
        page: bytes,
        default_pad: float,
        default_page_size: int,
        clip_cache_dir: Path,
    ):
        super().__init__(addr, handler)
        self.root = root
        self.examples = examples
        self.page = page
        self.default_pad = default_pad
        self.default_page_size = default_page_size
        self.clip_cache_dir = clip_cache_dir
        self.overall = summarize(examples)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    align_path = args.align.resolve()
    decode_path = args.decode.resolve()
    exp_dir = align_path.parent
    examples = load_examples(align_path, decode_path, root, selected_only=args.selected_only)
    title = f"Alignment review: {exp_dir.name}"
    page = page_html(title, args.default_pad, args.page_size)
    clip_cache_dir = (args.clip_cache_dir or (exp_dir / "alignment_failure_clips")).resolve()
    server = Server(
        (args.host, args.port),
        ReviewServer,
        root=root,
        examples=examples,
        page=page,
        default_pad=args.default_pad,
        default_page_size=args.page_size,
        clip_cache_dir=clip_cache_dir,
    )
    print(f"Serving {len(examples)} examples from {align_path}", flush=True)
    print(f"Decode source {decode_path}", flush=True)
    print(f"Caching clips under {clip_cache_dir}", flush=True)
    print(f"http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
