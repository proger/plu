# based on https://github.com/simonw/ttok/tree/main/ttok

import argparse
import re
import sys
from pathlib import Path

from plu.tokenizer import get_tokenizer


DESCRIPTION = """Convert text into whisper tokens

Examples:
  cat sentences.txt | +tok
  cat sentences.txt | +tok -t 100
  cat sentences.txt | +tok -t 100 -m gpt2
  cat sentences.txt | +tok --encode
  echo 9906 1917 | +tok --decode
  echo hello world | +tok --tokens
"""


def version() -> str:
    return Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=argparse.FileType("r"), default=sys.stdin, help="Input file")
    parser.add_argument("-t", "--truncate", type=int, help="Truncate to this many tokens")
    parser.add_argument("-m", "--model", default="multilingual", help="Which model to use")
    parser.add_argument("-l", "--language", default="en", help="Prepend multilingual prompt for given language to each string")
    parser.add_argument("--encode", dest="encode_tokens", action="store_true", help="Output token integers")
    parser.add_argument("--decode", dest="decode_tokens", action="store_true", help="Convert token integers to text")
    parser.add_argument("-k", "--tokens", dest="as_tokens", action="store_true", help="Output full tokens")
    parser.add_argument("-s", "--allow-special", action="store_true", help="Do not error on special tokens")
    parser.add_argument("--num-languages", type=int, default=100, help="Number of languages in the model (large-v3 has 100. Must match your model)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version()}")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def main():
    args = parse_args()
    input_file = args.input
    truncate = args.truncate
    model = args.model
    language = args.language
    encode_tokens = args.encode_tokens
    decode_tokens = args.decode_tokens
    as_tokens = args.as_tokens
    allow_special = args.allow_special

    if decode_tokens and encode_tokens:
        fail("Cannot use --decode with --encode")
    if allow_special and not (encode_tokens or as_tokens):
        fail("Cannot use --allow-special without --encode or --tokens")
    if as_tokens and not decode_tokens and not encode_tokens:
        encode_tokens = True
    try:
        tokenizer = get_tokenizer(multilingual=model == "multilingual", language=language, num_languages=args.num_languages)
        encoding = tokenizer.encoding
    except KeyError as e:
        raise SystemExit(f"Error: Invalid model: {model}") from e
    for text in input_file:
        text = text.strip()

        if decode_tokens:
            tokens = [int(token) for token in re.findall(r"\d+", text)]
            if as_tokens:
                print(encoding.decode_tokens_bytes(tokens))
            else:
                print(encoding.decode(tokens))
            return

        # Tokenize it
        kwargs = {}
        if allow_special:
            kwargs["allowed_special"] = "all"
        try:
            tokens = encoding.encode(text, **kwargs)
        except ValueError as ex:
            ex_str = str(ex)
            if "disallowed special token" in ex_str and not allow_special:
                # Just the first line, then add a hint
                ex_str = (
                    ex_str.split("\n")[0]
                    + "\n\nUse --allow-special to allow special tokens"
                )
            fail(ex_str)

        # Prepend the prompt
        if tokens:
            tokens = list(tokenizer.sot_sequence_including_notimestamps) + tokens
        else:
            tokens = [tokenizer.sot, tokenizer.no_speech]

        if truncate:
            tokens = tokens[:truncate]

        # Append the epilogue
        tokens = tokens + [tokenizer.eot]

        def wrap(x):
            return x.replace(' ', '▁')

        if encode_tokens:
            if as_tokens:
                print(" ".join(wrap(t.decode('utf-8')) for t in encoding.decode_tokens_bytes(tokens)))
            else:
                print(" ".join(str(t) for t in tokens))
        elif truncate:
            print(encoding.decode(tokens), end="")
        else:
            print(len(tokens))


if __name__ == "__main__":
    main()
