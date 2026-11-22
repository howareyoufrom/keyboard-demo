import math
import pickle
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_PATH = Path(__file__).resolve().with_name("model_wikipedia.pkl")
MODEL_REPO_ID = "tritrung28/keyboard-model"
MODEL_FILENAME = "model_wikipedia.pkl"


def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH

    print(f"Downloading model from Hugging Face: {MODEL_REPO_ID}")
    return hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )


MODEL_TO_LOAD = ensure_model()
with Path(MODEL_TO_LOAD).open("rb") as model_file:
    _loaded_model = pickle.load(model_file)

if isinstance(_loaded_model, dict) and _loaded_model.get("format_version") == 2:
    CHAR_COUNTS = _loaded_model["char_counts"]
    CHAR_CONTEXT_TOTALS = _loaded_model["char_context_totals"]
    CHAR_VOCABULARY_SIZE = len(_loaded_model["char_vocabulary"])
    WORD_MODEL = _loaded_model["word_model"]
else:
    CHAR_COUNTS = _loaded_model
    CHAR_CONTEXT_TOTALS = {}
    CHAR_VOCABULARY_SIZE = 30
    WORD_MODEL = None

CANVAS_WIDTH = 860.0
CANVAS_HEIGHT = 340.0
KEY_WIDTH = 74.0
KEY_HEIGHT = 66.0
KEY_GAP = 8.0
KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
total_height = 4 * KEY_HEIGHT + 3 * KEY_GAP
start_y = (CANVAS_HEIGHT - total_height) / 2.0 + KEY_HEIGHT / 2.0
KEY_POSITIONS = {}

for row_index, row_keys in enumerate(KEYBOARD_ROWS):
    extra_width = 1.5 * KEY_WIDTH if row_index == 2 else 0.0
    item_count = len(row_keys) + (1 if extra_width else 0)
    row_width = (
        len(row_keys) * KEY_WIDTH
        + extra_width
        + (item_count - 1) * KEY_GAP
    )
    cursor_x = (CANVAS_WIDTH - row_width) / 2.0
    center_y = start_y + row_index * (KEY_HEIGHT + KEY_GAP)
    for key in row_keys:
        KEY_POSITIONS[key] = (cursor_x + KEY_WIDTH / 2.0, center_y)
        cursor_x += KEY_WIDTH + KEY_GAP

KEY_POSITIONS[" "] = (CANVAS_WIDTH / 2.0, start_y + 3 * (KEY_HEIGHT + KEY_GAP))
CHAR_KEYS = tuple("abcdefghijklmnopqrstuvwxyz")

SPATIAL_SIGMA = 15.0
TREMOR_TIME_MS = 200.0
TREMOR_DISTANCE_MM = 3.0
CANVAS_DPI = 96.0
PX_PER_MM = CANVAS_DPI / 25.4
CHAR_LM_WEIGHT = 0.1
WORD_LM_WEIGHT = 0.45
KNOWN_WORD_BONUS = 0.5
BEAM_WIDTH = 256
SMOOTHING_ALPHA = 0.1
WORD_UNIGRAM_TOTAL = (
    sum(WORD_MODEL["unigrams"].values()) if WORD_MODEL is not None else 0
)


def _is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _spatial_candidates(touch, sigma=SPATIAL_SIGMA):
    key = str(touch.get("key", "")).lower()
    x = touch.get("x")
    y = touch.get("y")
    if key not in CHAR_KEYS or not _is_number(x) or not _is_number(y):
        return [(key, 1.0)]
    if not _is_number(sigma) or sigma <= 0:
        raise ValueError("spatial_sigma must be greater than zero")

    log_likelihoods = []
    for candidate in CHAR_KEYS:
        key_x, key_y = KEY_POSITIONS[candidate]
        distance_squared = (key_x - x) ** 2 + (key_y - y) ** 2
        log_likelihoods.append(
            (
                candidate,
                -math.log(2 * math.pi * sigma * sigma)
                - distance_squared / (2 * sigma * sigma),
            )
        )

    max_log_likelihood = max(log_likelihood for _, log_likelihood in log_likelihoods)
    weights = [
        (key, math.exp(log_likelihood - max_log_likelihood))
        for key, log_likelihood in log_likelihoods
    ]
    total_weight = sum(weight for _, weight in weights)
    return [(key, weight / total_weight) for key, weight in weights]


def _char_probability(c1, c2, c3):
    if not CHAR_CONTEXT_TOTALS:
        try:
            return CHAR_COUNTS[c1][c2].get(c3, 1e-5)
        except KeyError:
            return 1e-5

    context_total = CHAR_CONTEXT_TOTALS.get((c1, c2))
    if context_total is None:
        try:
            next_chars = CHAR_COUNTS[c1][c2]
        except KeyError:
            return 1e-5
        context_total = sum(next_chars.values())

    try:
        count = CHAR_COUNTS[c1][c2][c3]
    except KeyError:
        count = 0
    denominator = context_total + SMOOTHING_ALPHA * CHAR_VOCABULARY_SIZE
    return (count + SMOOTHING_ALPHA) / denominator


def _nested_count(table, *keys):
    current = table
    for key in keys:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
        if current is None:
            return 0
    return current if isinstance(current, (int, float)) else 0


def _word_probability(history, word):
    if WORD_MODEL is None:
        return 1e-5

    unigrams = WORD_MODEL["unigrams"]
    vocabulary_size = WORD_MODEL["vocabulary_size"] + 1
    alpha = SMOOTHING_ALPHA
    unigram_probability = (
        unigrams.get(word, 0) + alpha
    ) / (WORD_UNIGRAM_TOTAL + alpha * vocabulary_size)

    if not history:
        return unigram_probability

    previous = history[-1]
    bigram_total = WORD_MODEL["bigram_context_totals"].get(previous, 0)
    bigram_count = _nested_count(WORD_MODEL["bigrams"], previous, word)
    bigram_probability = (bigram_count + alpha) / (
        bigram_total + alpha * vocabulary_size
    )
    if len(history) < 2:
        return 0.25 * unigram_probability + 0.75 * bigram_probability

    previous_two = (history[-2], previous)
    trigram_total = WORD_MODEL["trigram_context_totals"].get(previous_two, 0)
    trigram_count = _nested_count(
        WORD_MODEL["trigrams"], previous_two[0], previous_two[1], word
    )
    trigram_probability = (trigram_count + alpha) / (
        trigram_total + alpha * vocabulary_size
    )
    return (
        0.15 * unigram_probability
        + 0.25 * bigram_probability
        + 0.60 * trigram_probability
    )


def _word_lm_score(history, word):
    probability = max(_word_probability(history, word), 1e-12)
    score = math.log(probability)
    if WORD_MODEL is not None and word in WORD_MODEL["unigrams"]:
        score += KNOWN_WORD_BONUS
    return score


def tremor_probability(
    prev_touch,
    curr_touch,
    tremor_time_ms=TREMOR_TIME_MS,
    tremor_distance_mm=TREMOR_DISTANCE_MM,
):
    if prev_touch is None:
        return {"EMIT": 0.99, "SKIP": 0.01}
    if tremor_time_ms <= 0 or tremor_distance_mm <= 0:
        raise ValueError("Tremor thresholds must be greater than zero")

    prev_t = prev_touch.get("t", prev_touch.get("timestamp_ms"))
    curr_t = curr_touch.get("t", curr_touch.get("timestamp_ms"))
    coordinates = (
        prev_touch.get("x"),
        prev_touch.get("y"),
        curr_touch.get("x"),
        curr_touch.get("y"),
    )
    if not _is_number(prev_t) or not _is_number(curr_t):
        return {"EMIT": 0.995, "SKIP": 0.005}
    if not all(_is_number(value) for value in coordinates):
        return {"EMIT": 0.995, "SKIP": 0.005}

    dt = curr_t - prev_t
    distance_px = math.hypot(
        curr_touch["x"] - prev_touch["x"],
        curr_touch["y"] - prev_touch["y"],
    )
    tremor_distance_px = tremor_distance_mm * PX_PER_MM
    if not (0.0 <= dt < tremor_time_ms and distance_px < tremor_distance_px):
        return {"EMIT": 0.995, "SKIP": 0.005}

    time_closeness = 1.0 - dt / tremor_time_ms
    distance_closeness = 1.0 - distance_px / tremor_distance_px
    p_skip = 0.88 + 0.09 * (
        0.5 * time_closeness + 0.5 * distance_closeness
    )
    return {"EMIT": 1.0 - p_skip, "SKIP": p_skip}


def viterbi_decode(
    touches,
    spatial_sigma=SPATIAL_SIGMA,
    tremor_time_ms=TREMOR_TIME_MS,
    tremor_distance_mm=TREMOR_DISTANCE_MM,
    prev_words=None,
):
    decoded, _ = _viterbi_decode(
        touches,
        spatial_sigma=spatial_sigma,
        tremor_time_ms=tremor_time_ms,
        tremor_distance_mm=tremor_distance_mm,
        prev_words=prev_words,
    )
    return decoded


def viterbi_decode_with_candidates(
    touches,
    spatial_sigma=SPATIAL_SIGMA,
    tremor_time_ms=TREMOR_TIME_MS,
    tremor_distance_mm=TREMOR_DISTANCE_MM,
    prev_words=None,
):
    return _viterbi_decode(
        touches,
        spatial_sigma=spatial_sigma,
        tremor_time_ms=tremor_time_ms,
        tremor_distance_mm=tremor_distance_mm,
        prev_words=prev_words,
    )


def _viterbi_decode(
    touches,
    spatial_sigma=SPATIAL_SIGMA,
    tremor_time_ms=TREMOR_TIME_MS,
    tremor_distance_mm=TREMOR_DISTANCE_MM,
    prev_words=None,
):
    if not touches:
        return "", []
    if spatial_sigma <= 0:
        raise ValueError("spatial_sigma must be greater than zero")

    log_skip_spatial_background = math.log(1.0 / len(CHAR_KEYS))
    context_words = [
        str(word).lower()
        for word in (prev_words or [])
        if isinstance(word, str) and word
    ][-2:]
    beam = [("", 0.0)]

    for index, touch in enumerate(touches):
        candidates = _spatial_candidates(touch, sigma=spatial_sigma)
        temporal = tremor_probability(
            touches[index - 1] if index > 0 else None,
            touch,
            tremor_time_ms=tremor_time_ms,
            tremor_distance_mm=tremor_distance_mm,
        )
        next_beam = []

        for decoded, score in beam:
            next_beam.append((
                decoded,
                score
                + math.log(temporal["SKIP"])
                + log_skip_spatial_background,
            ))

            for candidate, spatial_probability in candidates:
                previous_two_chars = decoded[-2:] if len(decoded) >= 2 else (
                    " " * (2 - len(decoded)) + decoded
                )
                language_probability = _char_probability(
                    previous_two_chars[0],
                    previous_two_chars[1],
                    candidate,
                )
                emission_score = math.log(max(temporal["EMIT"], 1e-12))
                emission_score += math.log(max(spatial_probability, 1e-12))
                emission_score += CHAR_LM_WEIGHT * math.log(
                    max(language_probability, 1e-12)
                )

                if candidate == " " and decoded and not decoded.endswith(" "):
                    completed_words = decoded.split()
                    if completed_words:
                        history = context_words + completed_words[:-1]
                        emission_score += WORD_LM_WEIGHT * _word_lm_score(
                            history, completed_words[-1]
                        )
                next_beam.append(
                    (decoded + candidate, score + emission_score)
                )

        next_beam.sort(key=lambda state: state[1], reverse=True)
        beam = next_beam[:BEAM_WIDTH]

    def final_score(state):
        decoded, score = state
        words = decoded.split()
        if not words or decoded.endswith(" "):
            return score
        return score + WORD_LM_WEIGHT * _word_lm_score(
            context_words + words[:-1], words[-1]
        )

    best_scores = {}
    for state in beam:
        decoded, _ = state
        score = final_score(state)
        if decoded not in best_scores or score > best_scores[decoded]:
            best_scores[decoded] = score
    candidates = sorted(
        (
            {"decoded_word": decoded, "log_score": score}
            for decoded, score in best_scores.items()
        ),
        key=lambda candidate: candidate["log_score"],
        reverse=True,
    )[:3]
    return candidates[0]["decoded_word"], candidates
