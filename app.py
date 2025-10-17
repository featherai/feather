from flask import Flask, request, jsonify
from risks import analyze_asset
from formatter import format_output

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symbol = data.get('symbol')
    period = data.get('period', '5d')
    holder_concentration = float(data.get('holder_concentration', 0.1) or 0.1)

    try:
        assessment = analyze_asset(symbol, period=period, holder_concentration=holder_concentration)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = format_output({}, assessment, sources=[])

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)