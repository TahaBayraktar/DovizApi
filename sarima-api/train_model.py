import requests
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import os
import urllib3
from pytz import timezone

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_forecast_for_currency(series_code, label, steps=90):  # ⬅️ 90 günlük tahmin
    startDate = "01-01-2021"
    
    # 🇹🇷 Türkiye saatiyle bugünün tarihi
    turkey_time = datetime.now(timezone("Europe/Istanbul"))
    endDate = turkey_time.strftime("%d-%m-%Y")
    
    url = (
        f"https://evds3.tcmb.gov.tr/igmevdsms-dis/series={series_code}"
        f"&startDate={startDate}&endDate={endDate}&type=json"
        f"&aggregationTypes=avg&formulas=0&frequency=1"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "key": os.getenv("EVDS_API_KEY")
    }

    print(f"🌐 {label} verisi çekiliyor...")
    response = requests.get(url, headers=headers, verify=False)
    if response.status_code != 200:
        print(f"❌ {label} verisi alınamadı. HTTP {response.status_code}")
        return []

    df = pd.DataFrame(response.json()["items"]).dropna()
    df.set_index("Tarih", inplace=True)
    df.rename(columns={series_code.replace(".", "_"): "Kur"}, inplace=True)
    df["Kur"] = df["Kur"].astype(float)
    df.index = pd.to_datetime(df.index, format="%d-%m-%Y")

    print(f"⚙️ {label} modeli eğitiliyor...")
    model = SARIMAX(df["Kur"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
    result = model.fit(disp=False)
    print(f"✅ {label} modeli eğitildi.")

    forecast = result.get_forecast(steps=steps)
    pred = forecast.predicted_mean.reset_index(drop=True)
    conf = forecast.conf_int().reset_index(drop=True)
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=steps, freq="D")

    forecast_data = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "prediction": round(float(pred[i]), 4),
            "conf_low": round(float(conf.iloc[i, 0]), 4),
            "conf_high": round(float(conf.iloc[i, 1]), 4)
        }
        for i, date in enumerate(future_dates)
    ]
    
    return forecast_data

# 🔮 Tahminleri al (artık 90 gün)
usd_forecast = get_forecast_for_currency("TP.DK.USD.S.YTL", "USD", steps=90)
eur_forecast = get_forecast_for_currency("TP.DK.EUR.S.YTL", "EUR", steps=90)

# ⏰ Türkiye saatiyle tahmin zamanı
turkey_now = datetime.now(timezone("Europe/Istanbul")).strftime("%Y-%m-%d %H:%M:%S")

output = {
    "generated_at": turkey_now,
    "forecast_days": 90,  # ⬅️ Güncellendi
    "forecasts": {
        "USD": usd_forecast,
        "EUR": eur_forecast
    }
}

# 💾 JSON dosyasına kaydet
with open("tahmin.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("📁 tahmin.json başarıyla oluşturuldu.")
