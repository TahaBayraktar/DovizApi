import requests
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import StringIO
from datetime import datetime, timedelta
import os
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    print(f"❌ Veri çekilemedi! HTTP {response.status_code}")
    exit()

print("✅ Veri çekildi.")

df = pd.read_csv(StringIO(response.text)).dropna()
df.set_index("Tarih", inplace=True)
df.rename(columns={series.replace(".", "_"): "Dolar_Kuru"}, inplace=True)
df.index = pd.to_datetime(df.index, format="%d-%m-%Y")

print("⚙️ Model eğitiliyor...")
model = SARIMAX(df["Dolar_Kuru"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
result = model.fit(disp=False)
print("✅ Model eğitildi.")

# ⏳ Bugünden itibaren 30 gün tahmin
forecast_steps = 30
forecast = result.get_forecast(steps=forecast_steps)
pred = forecast.predicted_mean.reset_index(drop=True)
conf = forecast.conf_int().reset_index(drop=True)
future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_steps, freq="D")

json_output = [
    {
        "date": date.strftime("%Y-%m-%d"),
        "prediction": round(float(pred[i]), 4),
        "conf_low": round(float(conf.iloc[i, 0]), 4),
        "conf_high": round(float(conf.iloc[i, 1]), 4)
    }
    for i, date in enumerate(future_dates)
]

# ✅ Ekstra bilgi: tahmin zamanı
output_data = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "forecast_days": forecast_steps,
    "forecasts": json_output
}

with open("tahmin.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("📁 tahmin.json başarıyla oluşturuldu.")
