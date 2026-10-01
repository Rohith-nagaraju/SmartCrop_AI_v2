import json, argparse
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
ap=argparse.ArgumentParser(); ap.add_argument('--split',default='test'); a=ap.parse_args()
classes=json.loads(Path('models/class_names.json').read_text())
ds=tf.keras.utils.image_dataset_from_directory(Path('data/disease_dataset')/a.split,image_size=(224,224),batch_size=32,shuffle=False)
model=tf.keras.models.load_model('models/disease_model.keras'); y=np.concatenate([z.numpy() for _,z in ds]); p=model.predict(ds,verbose=0); yp=p.argmax(1)
print(classification_report(y,yp,target_names=classes,digits=4)); print('accuracy',accuracy_score(y,yp)); print('macro_f1',f1_score(y,yp,average='macro'))
Path('experiments/results').mkdir(parents=True,exist_ok=True); pd.DataFrame(confusion_matrix(y,yp),index=classes,columns=classes).to_csv('experiments/results/confusion_matrix.csv')
