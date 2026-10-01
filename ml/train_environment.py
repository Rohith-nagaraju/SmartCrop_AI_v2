"""Train learned environmental risk models from a CSV.
CSV columns: temperature,humidity,rainfall,soil_moisture,risk_score
"""
import argparse, json
from pathlib import Path
import pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); a=ap.parse_args()
df=pd.read_csv(a.csv); X=df[['temperature','humidity','rainfall','soil_moisture']]; y=df['risk_score']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={'random_forest':RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),'gradient_boosting':GradientBoostingRegressor(random_state=42)}
res=[]; Path('models').mkdir(exist_ok=True); Path('experiments/results').mkdir(parents=True,exist_ok=True)
for name,m in models.items():
 m.fit(Xtr,ytr); p=m.predict(Xte); res.append({'model':name,'MAE':mean_absolute_error(yte,p),'RMSE':mean_squared_error(yte,p)**.5,'R2':r2_score(yte,p)}); joblib.dump(m,Path('models')/('environment_model.joblib' if name=='random_forest' else f'{name}_environment.joblib'))
Path('experiments/results/environment_models.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
