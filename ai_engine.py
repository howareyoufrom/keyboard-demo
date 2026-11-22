import math
import pickle
from pathlib import Path

print("Loading N-gram model...")
with (Path(__file__).resolve().parent / "model_wikipedia.pkl").open("rb") as model_file:
    NGRAM_MODEL = pickle.load(model_file)

KEY_POSITIONS = {}
for row_index, row_keys in enumerate(("qwertyuiop", "asdfghjkl", "zxcvbnm")):
    row_offset = (0, 0.5, 1.5)[row_index]
    for key_index, key in enumerate(row_keys):
        KEY_POSITIONS[key] = (row_offset + key_index, row_index)


def get_language_prob(c1, c2, c3):
    """Return P(c3 | c1, c2), using a small fallback for unseen trigrams."""
    try:
        return NGRAM_MODEL[c1][c2][c3]
    except KeyError:
        return 0.00001


def _spatial_candidates(touch):
    key = touch["key"]
    if key not in KEY_POSITIONS or not isinstance(touch.get("x"), (int, float)) or not isinstance(touch.get("y"), (int, float)):
        return [(key, 1.0)]

    x, y = touch["x"], touch["y"]
    candidates = []
    for candidate, (key_x, key_y) in KEY_POSITIONS.items():
        distance = math.hypot(key_x - x, key_y - y)
        if distance <= 1.15:
            likelihood = math.exp(-(distance ** 2) / (2 * 0.55 ** 2))
            candidates.append((candidate, likelihood))

    total_likelihood = sum(likelihood for _, likelihood in candidates)
    if not total_likelihood:
        return [(key, 1.0)]
    return [(candidate, likelihood / total_likelihood) for candidate, likelihood in candidates]


def viterbi_decode(touches):
    if not touches:
        return ""

    beam = [("", 0.0)]
    for index, touch in enumerate(touches):
        candidates = _spatial_candidates(touch)
        next_beam = []
        is_repeat = (
            index > 0
            and touch.get("group") is not None
            and touch.get("group") == touches[index - 1].get("group")
        )
        skip_probability = 0.95 if is_repeat else 0.000001

        for decoded, score in beam:
            next_beam.append((decoded, score + math.log(skip_probability)))

            for candidate, spatial_probability in candidates:
                c1 = decoded[-2] if len(decoded) >= 2 else " "
                c2 = decoded[-1] if decoded else " "
                language_probability = (
                    0.5 if candidate == " "
                    else get_language_prob(c1, c2, candidate)
                )
                emission_score = math.log(max(spatial_probability, 1e-12))
                emission_score += 0.1 * math.log(max(language_probability, 1e-12))
                next_beam.append((decoded + candidate, score + emission_score))

        next_beam.sort(key=lambda state: state[1], reverse=True)
        beam = next_beam[:256]

    return max(beam, key=lambda state: state[1])[0]
