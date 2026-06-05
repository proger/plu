from __future__ import annotations

import base64
import json
import os
import shutil
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Iterable, Optional


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

EOT = 50257
SOT = 50258
NO_SPEECH = 50363
NO_TIMESTAMPS = 50364
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


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


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_number(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _is_punctuation_piece(char: str) -> bool:
    return not char.isspace() and not _is_letter(char) and not _is_number(char)


def _tokenize_piece_pattern(text: str) -> list[str]:
    pieces: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        contraction = next((c for c in _CONTRACTIONS if text.startswith(c, i)), None)
        if contraction is not None:
            pieces.append(contraction)
            i += len(contraction)
            continue

        if text[i] == " " and i + 1 < n and _is_letter(text[i + 1]):
            j = i + 2
            while j < n and _is_letter(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if _is_letter(text[i]):
            j = i + 1
            while j < n and _is_letter(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if text[i] == " " and i + 1 < n and _is_number(text[i + 1]):
            j = i + 2
            while j < n and _is_number(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if _is_number(text[i]):
            j = i + 1
            while j < n and _is_number(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if text[i] == " " and i + 1 < n and _is_punctuation_piece(text[i + 1]):
            j = i + 2
            while j < n and _is_punctuation_piece(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        if _is_punctuation_piece(text[i]):
            j = i + 1
            while j < n and _is_punctuation_piece(text[j]):
                j += 1
            pieces.append(text[i:j])
            i = j
            continue

        j = i + 1
        while j < n and text[j].isspace():
            j += 1
        if j < n and text[j - 1] == " ":
            pieces.extend(text[k : k + 1] for k in range(i, j - 1))
            i = j - 1
        else:
            pieces.extend(text[k : k + 1] for k in range(i, j))
            i = j

    return pieces


def _asset_candidates(name: str) -> list[Path]:
    filename = f"{name}.tiktoken"
    candidates = []
    if "PLU_TOKENIZER_ASSETS" in os.environ:
        candidates.append(Path(os.environ["PLU_TOKENIZER_ASSETS"]).expanduser() / filename)
    candidates.append(Path(__file__).resolve().parent / "assets" / filename)

    for site_path in os.sys.path:
        if not site_path:
            continue
        candidates.append(Path(site_path) / "whisper" / "assets" / filename)

    candidates.append(Path(os.environ.get("PLU_TOKENIZER_CACHE", "~/.cache/plu/tokenizer")).expanduser() / filename)
    return candidates


def _download_asset(name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/{name}.tiktoken"
    with urllib.request.urlopen(url) as response, destination.open("wb") as f:
        shutil.copyfileobj(response, f)
    return destination


def _asset_path(name: str) -> Path:
    candidates = _asset_candidates(name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return _download_asset(name, candidates[-1])


def _load_tiktoken_bpe(path: Path) -> dict[bytes, int]:
    ranks = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


class Encoding:
    def __init__(
        self,
        name: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int] | None = None,
        explicit_n_vocab: int | None = None,
        id_to_token_bytes: dict[int, bytes] | None = None,
    ):
        self.name = name
        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens or {}
        self.special_tokens_set = set(self.special_tokens)
        self.explicit_n_vocab = explicit_n_vocab or (max([*mergeable_ranks.values(), *self.special_tokens.values(), -1]) + 1)
        self.eot_token = self.special_tokens.get("<|endoftext|>", EOT)

        self._token_to_bytes = {token_id: token for token, token_id in mergeable_ranks.items()}
        if id_to_token_bytes:
            self._token_to_bytes.update(id_to_token_bytes)
        for token, token_id in self.special_tokens.items():
            self._token_to_bytes[token_id] = token.encode("utf-8")

    @lru_cache(maxsize=200_000)
    def _bpe(self, token: bytes) -> tuple[int, ...]:
        rank = self.mergeable_ranks.get(token)
        if rank is not None:
            return (rank,)

        parts = [bytes([b]) for b in token]
        while len(parts) > 1:
            best_rank = None
            best_index = None
            for i in range(len(parts) - 1):
                candidate_rank = self.mergeable_ranks.get(parts[i] + parts[i + 1])
                if candidate_rank is not None and (best_rank is None or candidate_rank < best_rank):
                    best_rank = candidate_rank
                    best_index = i
            if best_index is None:
                break
            parts[best_index : best_index + 2] = [parts[best_index] + parts[best_index + 1]]

        return tuple(self.mergeable_ranks[part] for part in parts)

    def _allowed_special_set(self, allowed_special) -> set[str]:
        if allowed_special == "all":
            return set(self.special_tokens)
        if allowed_special is None:
            return set()
        return set(allowed_special)

    def _find_special(self, text: str, start: int, allowed_special: set[str]) -> str | None:
        if text[start : start + 2] != "<|":
            return None
        matches = [token for token in allowed_special if text.startswith(token, start)]
        if not matches:
            return None
        return max(matches, key=len)

    def encode(self, text: str, *, allowed_special=None, disallowed_special="all") -> list[int]:
        allowed = self._allowed_special_set(allowed_special)
        if disallowed_special == "all" and "<|" in text:
            disallowed = [token for token in self.special_tokens if token not in allowed and token in text]
            if disallowed:
                raise ValueError(f"Encountered text corresponding to disallowed special token {disallowed[0]!r}.")

        tokens: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            special = self._find_special(text, i, allowed)
            if special is not None:
                tokens.append(self.special_tokens[special])
                i += len(special)
                continue

            next_special = n
            if allowed and "<|" in text[i:]:
                positions = [text.find(token, i) for token in allowed if token in text[i:]]
                positions = [position for position in positions if position >= 0]
                if positions:
                    next_special = min(positions)

            chunk = text[i:next_special]
            for piece in _tokenize_piece_pattern(chunk):
                tokens.extend(self._bpe(piece.encode("utf-8")))
            i = next_special
        return tokens

    def decode_tokens_bytes(self, token_ids: Iterable[int]) -> list[bytes]:
        return [self._token_to_bytes.get(int(token_id), b"") for token_id in token_ids]

    def decode(self, token_ids: Iterable[int], errors: str = "replace", **_: object) -> str:
        return b"".join(self.decode_tokens_bytes(token_ids)).decode("utf-8", errors=errors)

    def encode_single_token(self, token: str | bytes) -> int:
        if isinstance(token, str):
            if token in self.special_tokens:
                return self.special_tokens[token]
            token = token.encode("utf-8")
        return self.mergeable_ranks[token]


@dataclass
class Tokenizer:
    encoding: Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    source_dir: Path | None = None
    sot_sequence: tuple[int, ...] = ()
    special_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            self.special_tokens[special] = self.encoding.encode_single_token(special)

        sot = self.special_tokens.get("<|startoftranscript|>", SOT)
        translate = self.special_tokens.get("<|translate|>")
        transcribe = self.special_tokens.get("<|transcribe|>")

        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sequence = [sot]
        if self.language is not None:
            sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            if translate is None or transcribe is None:
                raise ValueError("Tokenizer is missing Whisper task tokens")
            sequence.append(transcribe if self.task == "transcribe" else translate)
        self.sot_sequence = tuple(sequence)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | os.PathLike[str],
        *,
        pad_token_id: int = EOT,
    ) -> "Tokenizer":
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

        byte_decoder = {value: key for key, value in bytes_to_unicode().items()}
        id_to_token_bytes = {}
        special_tokens = {}
        for token, token_id in vocab.items():
            token_id = int(token_id)
            if token.startswith("<|") and token.endswith("|>"):
                special_tokens[token] = token_id
                id_to_token_bytes[token_id] = token.encode("utf-8")
            else:
                id_to_token_bytes[token_id] = bytes(byte_decoder.get(char, ord("?")) for char in token)

        special_tokens.setdefault("<|endoftext|>", pad_token_id)
        encoding = Encoding(
            name=str(model_path),
            mergeable_ranks={},
            special_tokens=special_tokens,
            explicit_n_vocab=max([*id_to_token_bytes, *special_tokens.values(), pad_token_id, -1]) + 1,
            id_to_token_bytes=id_to_token_bytes,
        )
        return cls(encoding=encoding, num_languages=100, source_dir=model_path)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        token_ids = [int(t) for t in token_ids if int(t) < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: list[int], **kwargs) -> str:
        return self.encoding.decode(token_ids, **kwargs)

    def batch_decode(self, sequences: Iterable[Iterable[int]], skip_special_tokens: bool = True) -> list[str]:
        decoded = []
        special_ids = set(self.special_tokens.values())
        for sequence in sequences:
            token_ids = [int(token_id) for token_id in sequence]
            if skip_special_tokens:
                token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
            decoded.append(self.encoding.decode(token_ids))
        return decoded

    def save_pretrained(self, output_dir: str | os.PathLike[str]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self.source_dir is None:
            return
        for filename in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
            src = self.source_dir / filename
            if src.exists():
                shutil.copy2(src, output_path / filename)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        language = language.lower()
        if language not in LANGUAGES and language in TO_LANGUAGE_CODE:
            language = TO_LANGUAGE_CODE[language]
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token
        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> tuple[int, ...]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self) -> tuple[str, ...]:
        return tuple(self.decode([token]).strip("<|>") for token in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> tuple[int, ...]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])


WhisperTokenizer = Tokenizer


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99) -> Encoding:
    ranks = _load_tiktoken_bpe(_asset_path(name))
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return Encoding(
        name=f"{name}.tiktoken",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
        explicit_n_vocab=n_vocab,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    return Tokenizer(
        encoding=get_encoding(name=encoding_name, num_languages=num_languages),
        num_languages=num_languages,
        language=language,
        task=task,
    )
