import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset
import time

VOWEL_MARKS = {
    "\u0302": "circumflex",
    "\u0306": "breve",
    "\u031b": "horn",
    "\u0301": "s",
    "\u0300": "f",
    "\u0309": "r",
    "\u0303": "x",
    "\u0323": "j",
}
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?;\n]+")
MAX_WORD_VOCABULARY = 8000
MIN_BIGRAM_COUNT = 2
MIN_TRIGRAM_COUNT = 2


def vietnamese_syllable_to_telex(syllable):
    normalized = unicodedata.normalize("NFD", syllable.lower())
    output = []
    tone = ""
    index = 0

    while index < len(normalized):
        char = normalized[index]
        index += 1
        if char == "đ":
            output.append("dd")
            continue
        if not char.isalpha():
            continue

        marks = []
        while index < len(normalized) and unicodedata.combining(normalized[index]):
            marks.append(normalized[index])
            index += 1

        output.append(char)
        if char in "aeiouy":
            mark_names = [VOWEL_MARKS.get(mark) for mark in marks]
            if "circumflex" in mark_names:
                output.append(char)
            elif "breve" in mark_names or "horn" in mark_names:
                output.append("w")
            for mark in marks:
                mark_name = VOWEL_MARKS.get(mark)
                if mark_name in {"s", "f", "r", "x", "j"}:
                    tone = mark_name

    if tone:
        output.append(tone)
    return "".join(output)


def iter_telex_sentences(dataset):
    for article in dataset:
        for sentence in SENTENCE_RE.split(article["text"].lower()):
            syllables = [
                vietnamese_syllable_to_telex(token)
                for token in TOKEN_RE.findall(sentence)
            ]
            syllables = [syllable for syllable in syllables if syllable]
            if syllables:
                yield syllables


def add_nested_count(table, keys):
    current = table
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    final = keys[-1]
    current[final] = current.get(final, 0) + 1


print("Loading Vietnamese Wikipedia data...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train[:10000]")
total_articles = len(dataset)
start_time = time.time()

char_counts = defaultdict(lambda: defaultdict(Counter))
char_context_totals = Counter()
syllable_counts = Counter()
print("Pass 1/2: counting Telex character n-grams and syllables...")
for idx, syllables in enumerate(iter_telex_sentences(dataset), start=1):
    syllable_counts.update(syllables)
    text = "  " + " ".join(syllables) + " "
    for position in range(len(text) - 2):
        context = (text[position], text[position + 1])
        char_counts[context[0]][context[1]][text[position + 2]] += 1
        char_context_totals[context] += 1
    if idx % 100000 == 0:
        print(f"  Processed {idx} sentences...")

vocabulary = {
    syllable
    for syllable, _ in syllable_counts.most_common(MAX_WORD_VOCABULARY)
}
word_unigrams = {syllable: syllable_counts[syllable] for syllable in vocabulary}
word_bigrams = {}
word_trigrams = {}
bigram_context_totals = Counter()
trigram_context_totals = Counter()

print("Pass 2/2: counting Telex context n-grams...")
for idx, syllables in enumerate(iter_telex_sentences(dataset), start=1):
    segment = []

    def count_segment(items):
        if not items:
            return
        sequence = ["<BOS>", "<BOS>"] + items
        for position in range(2, len(sequence)):
            previous_two = (sequence[position - 2], sequence[position - 1])
            current = sequence[position]
            add_nested_count(word_bigrams, (sequence[position - 1], current))
            add_nested_count(word_trigrams, (*previous_two, current))
            bigram_context_totals[sequence[position - 1]] += 1
            trigram_context_totals[previous_two] += 1

    for syllable in syllables:
        if syllable in vocabulary:
            segment.append(syllable)
        else:
            count_segment(segment)
            segment = []
    count_segment(segment)
    if idx % 100000 == 0:
        print(f"  Processed {idx} sentences...")

for previous, following in list(word_bigrams.items()):
    kept = {token: count for token, count in following.items()
            if count >= MIN_BIGRAM_COUNT}
    if kept:
        word_bigrams[previous] = kept
    else:
        del word_bigrams[previous]
bigram_context_totals = Counter({
    previous: sum(following.values())
    for previous, following in word_bigrams.items()
})

for previous, second_words in list(word_trigrams.items()):
    for second, following in list(second_words.items()):
        kept = {token: count for token, count in following.items()
                if count >= MIN_TRIGRAM_COUNT}
        if kept:
            second_words[second] = kept
        else:
            del second_words[second]
    if not second_words:
        del word_trigrams[previous]
trigram_context_totals = Counter({
    (previous, second): sum(following.values())
    for previous, second_words in word_trigrams.items()
    for second, following in second_words.items()
})

plain_char_counts = {
    char1: {char2: dict(next_chars) for char2, next_chars in contexts.items()}
    for char1, contexts in char_counts.items()
}
model = {
    "format_version": 2,
    "char_counts": plain_char_counts,
    "char_context_totals": dict(char_context_totals),
    "char_vocabulary": sorted(
        {char for contexts in plain_char_counts.values()
         for next_chars in contexts.values() for char in next_chars}
        | set(plain_char_counts)
        | {char2 for contexts in plain_char_counts.values() for char2 in contexts}
        | {" "}
    ),
    "word_model": {
        "unigrams": word_unigrams,
        "bigrams": word_bigrams,
        "trigrams": word_trigrams,
        "bigram_context_totals": dict(bigram_context_totals),
        "trigram_context_totals": dict(trigram_context_totals),
        "vocabulary_size": len(vocabulary),
    },
}

model_path = Path(__file__).resolve().with_name("model_wikipedia.pkl")
print("Saving character and context models...")
temporary_path = model_path.with_suffix(".pkl.tmp")
with temporary_path.open("wb") as model_file:
    pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
temporary_path.replace(model_path)

print(f"Done in {time.time() - start_time:.2f} seconds")
print(f"Model saved to: {model_path}")