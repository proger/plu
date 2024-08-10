# based on https://github.com/simonw/ttok/tree/main/ttok

import click
import re
import sys
from whisper.tokenizer import get_tokenizer

@click.command()
@click.version_option()
@click.option("-i", "--input", "input", type=click.File("r"))
@click.option(
    "-t", "--truncate", "truncate", type=int, help="Truncate to this many tokens"
)
@click.option("-m", "--model", default="multilingual", help="Which model to use")
@click.option("-l", "--language", default="en", help="Prepend multilingual prompt for given language to each string")
@click.option(
    "encode_tokens", "--encode", "--tokens", is_flag=True, help="Output token integers"
)
@click.option(
    "decode_tokens", "--decode", is_flag=True, help="Convert token integers to text"
)
@click.option("as_tokens", "-k", "--tokens", is_flag=True, help="Output full tokens")
@click.option("-s", "--allow-special", is_flag=True, help="Do not error on special tokens")
@click.option("--num-languages", default=100, help="Number of languages in the model (large-v3 has 100. Must match your model)")
def main(
    input,
    truncate,
    model,
    language,
    encode_tokens,
    decode_tokens,
    as_tokens,
    allow_special,
    num_languages
):
    """
    Convert text into whisper tokens

    To count tokens from stdin:

        cat sentences.txt | wtok

    To truncate to 100 tokens:

        cat sentences.txt | wtok -t 100

    To truncate to 100 tokens using the gpt2 model:

        cat sentences.txt | wtok -t 100 -m gpt2

    To view token integers:

        cat sentences.txt | wtok --encode

    To convert tokens back to text:

        echo 9906 1917 | wtok --decode

    To see the details of the tokens:

        echo hello world | wtok --tokens
    """
    if decode_tokens and encode_tokens:
        raise click.ClickException("Cannot use --decode with --encode")
    if allow_special and not (encode_tokens or as_tokens):
        raise click.ClickException(
            "Cannot use --allow-special without --encode or --tokens"
        )
    if as_tokens and not decode_tokens and not encode_tokens:
        encode_tokens = True
    try:
        tokenizer = get_tokenizer(multilingual=model == "multilingual", language=language, num_languages=num_languages)
        encoding = tokenizer.encoding
    except KeyError as e:
        raise click.ClickException(f"Invalid model: {model}") from e
    for text in sys.stdin:
        text = text.strip()

        if decode_tokens:
            tokens = [int(token) for token in re.findall(r"\d+", text)]
            if as_tokens:
                click.echo(encoding.decode_tokens_bytes(tokens))
            else:
                click.echo(encoding.decode(tokens))
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
            raise click.ClickException(ex_str)

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
                click.echo(" ".join(wrap(t.decode('utf-8')) for t in encoding.decode_tokens_bytes(tokens)))
            else:
                click.echo(" ".join(str(t) for t in tokens))
        elif truncate:
            click.echo(encoding.decode(tokens), nl=False)
        else:
            click.echo(len(tokens))
