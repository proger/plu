from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


class WhisperTokenizer:
    eot = 50257
    sot = 50258
    no_speech = 50363
    no_timestamps = 50364

    def __init__(self, id_to_token: dict[int, str] | None = None, source_dir: str | os.PathLike[str] | None = None, pad_token_id: int = 50257):
        self.id_to_token = id_to_token or {}
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.pad_token_id = pad_token_id
        self.byte_decoder = {value: key for key, value in bytes_to_unicode().items()}
        self.special_token_ids = {
            self.pad_token_id,
            self.eot,
            self.sot,
            self.no_speech,
            self.no_timestamps,
        }
        for token_id, token in self.id_to_token.items():
            if token.startswith("<|") and token.endswith("|>"):
                self.special_token_ids.add(token_id)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | os.PathLike[str],
        *,
        pad_token_id: int = 50257,
    ) -> "WhisperTokenizer":
        model_path = Path(model_dir)
        if model_path.is_file():
            model_path = model_path.parent

        vocab: dict[str, int] = {}
        tokenizer_json = model_path / "tokenizer.json"
        vocab_json = model_path / "vocab.json"

        if tokenizer_json.exists():
            with tokenizer_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            vocab.update(data.get("model", {}).get("vocab", {}))
            for token in data.get("added_tokens", []):
                content = token.get("content")
                token_id = token.get("id")
                if content is not None and token_id is not None:
                    vocab[content] = int(token_id)
        elif vocab_json.exists():
            with vocab_json.open("r", encoding="utf-8") as f:
                vocab.update(json.load(f))

        id_to_token = {int(token_id): token for token, token_id in vocab.items()}
        return cls(id_to_token=id_to_token, source_dir=model_path, pad_token_id=pad_token_id)

    def _decode_pieces(self, pieces: Iterable[str]) -> str:
        byte_values: list[int] = []
        for piece in pieces:
            for char in piece:
                byte = self.byte_decoder.get(char)
                if byte is None:
                    byte_values.extend(char.encode("utf-8", errors="replace"))
                else:
                    byte_values.append(byte)
        return bytes(byte_values).decode("utf-8", errors="replace")

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.special_token_ids:
                continue
            token = self.id_to_token.get(token_id)
            if token is None:
                continue
            if skip_special_tokens and token.startswith("<|") and token.endswith("|>"):
                continue
            pieces.append(token)
        return self._decode_pieces(pieces)

    def batch_decode(self, sequences: Iterable[Iterable[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens) for sequence in sequences]

    def save_pretrained(self, output_dir: str | os.PathLike[str]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self.source_dir is None:
            return
        for filename in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
            src = self.source_dir / filename
            if src.exists():
                shutil.copy2(src, output_path / filename)
