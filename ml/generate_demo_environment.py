import numpy as np, pandas as pd
from pathlib import Path
rng=np.random.default_rng(42); n=1000
t=rng.normal(25,5,n); h=np.clip(rng.normal(68,15,n),0,100); r=np.clip(rng.gamma(2,4,n),0,60); s=np.clip(rng.normal(55,18,n),0,100)
risk=np.clip(0.25*(1-np.abs(25-t)/20).clip(0,1)*100+0.35*np.clip((h-55)/45,0,1)*100+0.2*np.minimum(r/30,1)*100+0.2*np.clip((s-45)/55,0,1)*100+rng.normal(0,5,n),0,100)
out=pd.DataFrame({'temperature':t,'humidity':h,'rainfall':r,'soil_moisture':s,'risk_score':risk}); Path('data').mkdir(exist_ok=True); out.to_csv('data/demo_environment.csv',index=False); print('Generated demo data ONLY for pipeline testing.')
