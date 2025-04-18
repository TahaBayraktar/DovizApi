from flask import Flask, jsonify
from model import get_sarima_forecast

app = Flask(__name__)

@app.route("/forecast", methods=["GET"])
def forecast():
    result = get_sarima_forecast()
    return jsonify(result)