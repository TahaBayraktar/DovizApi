from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

@app.route("/forecast", methods=["GET"])
def forecast():
    if not os.path.exists("tahmin.json"):
        return jsonify({"error": "Tahmin verisi mevcut değil. Lütfen model çalıştırılsın."}), 500
    with open("tahmin.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)