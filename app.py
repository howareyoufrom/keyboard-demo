from time import perf_counter

from flask import Flask, render_template, request, jsonify
from ai_engine import viterbi_decode_with_candidates

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/decode', methods=['POST'])
def decode():
    started_at = perf_counter()
    data = request.json
    touches = data.get('touches', [])
    if not touches:
        return jsonify({
            "decoded_word": "",
            "candidates": [],
            "elapsed_ms": (perf_counter() - started_at) * 1000,
        })

    spatial_sigma = max(5.0, min(25.0, float(data.get("spatial_sigma", 15.0))))
    tremor_time_ms = max(20.0, min(1000.0, float(data.get("tremor_time_ms", 200.0))))
    tremor_distance_mm = max(
        0.5,
        min(10.0, float(data.get("tremor_distance_mm", 3.0))),
    )
    prev_words = data.get("prev_words", [])
    if not isinstance(prev_words, list):
        prev_words = []

    result, candidates = viterbi_decode_with_candidates(
        touches,
        spatial_sigma=spatial_sigma,
        tremor_time_ms=tremor_time_ms,
        tremor_distance_mm=tremor_distance_mm,
        prev_words=prev_words,
    )
    return jsonify({
        "decoded_word": result,
        "candidates": candidates,
        "elapsed_ms": (perf_counter() - started_at) * 1000,
    })

if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)