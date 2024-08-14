import argparse
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
import json
import random
import sys
from whisper.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Convert sentences into whisper tokens one line at a time", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", type=argparse.FileType("r"), default=sys.stdin, nargs="?", help="Input file")
parser.add_argument("-t", "--truncate", type=int, default=100, help="Truncate to this many tokens. Whisper maximum is 448")
parser.add_argument("-m", "--model", default="multilingual", help="Which model to use")
parser.add_argument("-l", "--language", default="en", help="Assume each string is in this language for prompt generation")
parser.add_argument("-i", "--output-ids", action="store_true", help="Output tokens as vocabulary ids")
parser.add_argument("-k", "--output-tokens", action="store_true", help="Output raw tokens as strings")
parser.add_argument("-q", "--quiet", action="store_true", help="Do not print any tokens, useful if you just want to count them")
parser.add_argument("-c", "--counts", type=int, default=0, help="Output token counts as a histogram dividing length by the given number (use 10)")
parser.add_argument("-p", "--prev", type=float, default=0, help="Context inclusion probability, uses the previous input as context")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("-r", "--resample", type=float, default=0, help="Resample the input data this many times")
parser.add_argument("--num-languages", type=int, default=100, help="Number of languages in the model (large-v3 has 100. Must match your model)")

def cdiv(x: int, y: int):
    "Ceiling division"
    return (x + y - 1) // y

def render(t, s):
    try:
        return s.decode('utf-8').replace(' ', '▁')
    except UnicodeDecodeError:
        return str(t)

@dataclass(frozen=True)
class Example:
    source: str | None # Source of the text, different sources are not contextualized
    text: str | None # Text to be tokenized, empty for no speech
    language: str | None # Language code, optional
    path: str | None # Path to the audio file
    timestamp: str # Timestamp of the audio file
    duration: float | None = None # Duration of the audio file
    input_ids: list[int] | None = None # Tokenized input (input_ids is huggingface jargon)

    @classmethod
    def from_json(cls, obj):
        return cls(obj.get("receiver_location") or obj.get("source"),
                   obj.get("text"),
                   obj.get("language"),
                   obj.get("path"),
                   obj.get("timestamp"),
                   obj.get("duration"),
                   obj.get("input_ids"))

    @classmethod
    def from_text(cls, text, language):
        if obj := json.loads(text):
            return cls.from_json(obj)
        else:
            return cls(source=None, text=text, language=language, path=None, timestamp=None, duration=None, input_ids=None)

    def to_json(self, input_ids=None):
        return json.dumps(dict(
            source=self.source,
            text=self.text,
            language=self.language,
            path=self.path,
            timestamp=self.timestamp,
            duration=self.duration,
            input_ids=input_ids or self.input_ids # override
        ), ensure_ascii=False)


def print_histogram(counter):
    density = 0
    cdf = {}
    normalizer = sum(counter[bucket] for bucket in sorted(counter)) if counter else 1
    print('bucket', 'count', 'cdf', 'histogram', sep='\t', file=sys.stderr)
    for bucket in sorted(counter):
        count = counter[bucket]
        prob = count / normalizer
        density += prob
        cdf[bucket] = density
        print(bucket, count, f'{density:.02f}', '+'*int(prob*100), sep='\t', file=sys.stderr)
    return cdf


def main():
    args = parser.parse_args()
    input = args.input
    resample = args.resample
    language = args.language
    count_buckets = args.counts
    model = args.model

    if resample:
        # buffer the input and sort according to timestamp

        samples = []
        for input_line in input:
            example = Example.from_text(input_line, language)
            samples.append(example)

        samples = sorted(samples, key=lambda x: x.timestamp)
        input, buckets = samples, defaultdict(list)
    else:
        input = (Example.from_text(input_line, language) for input_line in input)

    try:
        tokenizer = get_tokenizer(multilingual=model == "multilingual", language=language, num_languages=args.num_languages)
    except KeyError as e:
        raise Exception(f"Invalid model: {model}, available: multilingual and gpt2 (see openai-whisper)") from e
    encoding = tokenizer.encoding

    def render_ids_as_string(input_ids):
        return " ".join(render(t, s) for t, s in zip(input_ids, encoding.decode_tokens_bytes(input_ids)))

    counter = Counter()
    truncated = 0

    random.seed(args.seed)
    for example, length, input_ids in tokenize(input, tokenizer, contextualize_probability=args.prev):
        if count_buckets:
            bucket = cdiv(length, count_buckets) * count_buckets
        else:
            bucket = length

        if bucket <= args.truncate:
            counter.update([bucket])
        else:
            truncated += 1

        if resample:
            if bucket <= args.truncate:
                buckets[bucket].append((example, input_ids))
        else:
            if not args.quiet:
                if args.output_ids:
                    print(" ".join(str(i) for i in input_ids))
                elif args.output_tokens:
                    print(render_ids_as_string(input_ids))
                else:
                    print(example.to_json(input_ids))

    if count_buckets:
        print_histogram(counter)
        print(f'removed utterances that are too long (-t {args.truncate}):', truncated, file=sys.stderr)

    if resample:
        # resample from buckets uniformly
        bucket_keys = list(buckets.keys())
        for _ in range(int(len(input)*resample)):
            key = random.choice(bucket_keys)
            if not len(buckets[key]):
                print(sys.argv, 'empty bucket', key, file=sys.stderr)
            example, input_ids = random.choice(buckets[key])
            if len(input_ids) > args.truncate:
                print('too long', len(input_ids), 'in bucket', key, file=sys.stderr)
                print(render_ids_as_string(input_ids), file=sys.stderr)
            print(example.to_json(input_ids))


def tokenize(
    input,
    tokenizer,
    contextualize_probability=0,
):
    encoding = tokenizer.encoding
    context = deque(maxlen=10)

    for example in input:
        input_ids = []

        if example.input_ids:
            # user has provided the tokens already, just use them
            input_ids = example.input_ids

        if not input_ids:
            # Tokenize the contexts
            if context:
                context_copy = list(context)
                while context_copy and (prev_example := context_copy.pop()):
                    if prev_example.source != example.source:
                        # Do not include the context if the source is different
                        break
                    else:
                        if random.random() < contextualize_probability:
                            input_ids.append(tokenizer.sot_prev)
                            input_ids.extend(encoding.encode(prev_example.text, allowed_special="all"))
                        else:
                            break

            # Tokenize the input
            if example.text:
                input_ids.extend([tokenizer.sot, tokenizer.to_language_token(example.language), tokenizer.transcribe, tokenizer.no_timestamps])
                input_ids.extend(encoding.encode(example.text, allowed_special="all"))
            elif not example.text or not example.language:
                input_ids.extend([tokenizer.sot, tokenizer.no_speech])

            # Append the epilogue
            input_ids = input_ids + [tokenizer.eot]

        yield example, len(input_ids), input_ids
        context.append(example)
