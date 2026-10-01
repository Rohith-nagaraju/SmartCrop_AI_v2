"""Train EfficientNet-B0 on folder-organized images.
Expected: data/disease_dataset/{train,val,test}/{class_name}/*.jpg
"""
import json, argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=12); a=ap.parse_args()
root=Path('data/disease_dataset'); img=(224,224); bs=32
train=tf.keras.utils.image_dataset_from_directory(root/'train',image_size=img,batch_size=bs,label_mode='int',shuffle=True)
val=tf.keras.utils.image_dataset_from_directory(root/'val',image_size=img,batch_size=bs,label_mode='int',shuffle=False)
classes=train.class_names
aug=tf.keras.Sequential([layers.RandomFlip('horizontal'),layers.RandomRotation(.08),layers.RandomZoom(.12),layers.RandomContrast(.12)])
base=EfficientNetB0(include_top=False,weights='imagenet',input_shape=(*img,3)); base.trainable=False
inputs=layers.Input((*img,3)); x=aug(inputs); x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.35)(x); outputs=layers.Dense(len(classes),activation='softmax')(x)
model=tf.keras.Model(inputs,outputs); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
cb=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=2)]
model.fit(train,validation_data=val,epochs=a.epochs,callbacks=cb)
Path('models').mkdir(exist_ok=True); model.save('models/disease_model.keras'); Path('models/class_names.json').write_text(json.dumps(classes,indent=2))
# fine tune last 20 layers
base.trainable=True
for l in base.layers[:-20]: l.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train,validation_data=val,epochs=5,callbacks=cb)
model.save('models/disease_model.keras')
print('Saved model and class names:',classes)
