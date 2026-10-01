"""Time-series risk forecasting baseline.
CSV: timestamp, field_id, temperature, humidity, rainfall, soil_moisture, risk_score
"""
import argparse, json
from pathlib import Path
import pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); a=ap.parse_args(); df=pd.read_csv(a.csv).sort_values('timestamp')
features=['temperature','humidity','rainfall','soil_moisture'];
for lag in [1,2,3]: df[f'risk_lag_{lag}']=df['risk_score'].shift(lag)
df=df.dropna(); X=df[features+[f'risk_lag_{i}' for i in [1,2,3]]]; y=df.risk_score; cut=int(len(df)*.8); Xtr,Xte,ytr,yte=X.iloc[:cut],X.iloc[cut:],y.iloc[:cut],y.iloc[cut:]
m=HistGradientBoostingRegressor(random_state=42); m.fit(Xtr,ytr); p=m.predict(Xte); out={'MAE':mean_absolute_error(yte,p),'RMSE':mean_squared_error(yte,p)**.5,'n_test':len(yte)}
Path('models').mkdir(exist_ok=True); joblib.dump(m,'models/forecast_model.joblib'); Path('experiments/results').mkdir(parents=True,exist_ok=True); Path('experiments/results/forecast.json').write_text(json.dumps(out,indent=2)); print(out)
