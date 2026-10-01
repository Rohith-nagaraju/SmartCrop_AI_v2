"""Ablation runner for a prepared CSV of component scores.
Columns should include: image_score, environment_score, severity_score, temporal_score, target.
The script trains the same classifier for each feature subset and records macro-F1.
"""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); a=ap.parse_args()
df=pd.read_csv(a.csv); target='target'; configs={'A_image_only':['image_score'],'B_environment_only':['environment_score'],'C_image_environment':['image_score','environment_score'],'D_plus_severity':['image_score','environment_score','severity_score'],'E_full_multimodal':['image_score','environment_score','severity_score','temporal_score']}
res=[]
for name,cols in configs.items():
 X=df[cols]; y=df[target]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); m=RandomForestClassifier(n_estimators=300,random_state=42); m.fit(Xtr,ytr); p=m.predict(Xte); res.append({'experiment':name,'features':cols,'accuracy':accuracy_score(yte,p),'precision_macro':precision_score(yte,p,average='macro',zero_division=0),'recall_macro':recall_score(yte,p,average='macro',zero_division=0),'f1_macro':f1_score(yte,p,average='macro',zero_division=0)})
Path('experiments/results').mkdir(parents=True,exist_ok=True); Path('experiments/results/ablation.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
