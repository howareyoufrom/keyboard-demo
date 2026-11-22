from flask import Flask, render_template, request, jsonify
from ai_engine import viterbi_decode

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/decode', methods=['POST'])
def decode():
    data = request.json
    touches = data.get('touches', [])
    if not touches:
        return jsonify({"decoded_word": ""})
    
    # Gọi thuật toán Viterbi
    result = viterbi_decode(touches)
    return jsonify({"decoded_word": result})

if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)