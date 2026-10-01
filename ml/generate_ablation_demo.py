import numpy as np,pandas as pd
from pathlib import Path
rng=np.random.default_rng(42); n=1000
img=rng.random(n); env=rng.random(n); sev=rng.random(n); temp=rng.random(n); score=.35*img+.25*env+.2*sev+.2*temp; y=(score>.55).astype(int)
pd.DataFrame({'image_score':img,'environment_score':env,'severity_score':sev,'temporal_score':temp,'target':y}).to_csv('data/demo_ablation.csv',index=False); print('Generated demo ablation data ONLY for pipeline testing.')
