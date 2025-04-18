import requests
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import StringIO
from datetime import datetime, timedelta
import os
import urllib3

# Uyarı bastır (TCMB için SSL doğrulama devre dışı)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_sarima_forecast():
    series = "TP.DK.USD.S.YTL"
    startDate = "01-01-2021"
    endDate = datetime.now().strftime("%d-%m-%Y")

    url = (
        f"https://evds2.tcmb.gov.tr/service/evds/series={series}"
        f"&startDate={startDate}&endDate={endDate}&type=csv"
        f"&aggregationTypes=avg&formulas=0&frequency=1"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "key": os.getenv("EVDS_API_KEY")
    }

    print("🌐 Veri çekiliyor...")
    response = requests.get(url, headers=headers, verify=False)

    if response.status_code != 200:
        print(f"❌ Veri alınamadı. HTTP {response.status_code}")
        return

    print("✅ Veri çekildi.")

    df = pd.read_csv(StringIO(response.text)).dropna()
    df.set_index("Tarih", inplace=True)
    df.rename(columns={series.replace(".", "_"): "Dolar_Kuru"}, inplace=True)
    df.index = pd.to_datetime(df.index, format="%d-%m-%Y")

    # Son 180 günle sınırla
    df = df[df.index >= (datetime.now() - timedelta(days=180))]
    print(f"📊 Toplam veri sayısı: {len(df)}")

    try:
        model = SARIMAX(df["Dolar_Kuru"], order=(1,1,1), seasonal_order=(1,1,1,30))
        result = model.fit(disp=False)
        print("✅ Model başarıyla eğitildi.")
    except Exception as e:
        print(f"❌ Model eğitimi başarısız: {e}")
        return

    steps = 21
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=steps, freq="D")
    forecast = result.get_forecast(steps=steps)
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()

    json_output = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "prediction": float(pred[i]),
            "conf_low": float(conf_int.iloc[i, 0]),
            "conf_high": float(conf_int.iloc[i, 1])
        }
        for i, date in enumerate(future_dates)
    ]

    print(f"📈 Tahmin üretildi: {len(json_output)} gün")

    try:
        file_path = os.path.join(os.getcwd(), "tahmin.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        print(f"✅ tahmin.json başarıyla oluşturuldu: {file_path}")
    except Exception as e:
        print(f"❌ JSON yazım hatası: {e}")

if __name__ == "__main__":
    get_sarima_forecast()
