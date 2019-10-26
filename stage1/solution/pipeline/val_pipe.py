'''

Copyright (C) 2019 fs0c131y

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from pipeline.ckpt import callbacks

nclasses=3
size=(224,224,3)
width=size[0]
height=size[1]
depth=size[2]
batch_size=16
seed=1
initial_epoch=0
train_count=543
val_count=142

def val_pipe(val_dir,model_dir):
    print('Searching for model.........')
    model=load_model(model_dir)
    print ('Model loaded')
    print ('building pipeline for data..........')

    val_data_generator = ImageDataGenerator(horizontal_flip=False,vertical_flip=False).flow_from_directory(
    val_dir,
    target_size = (width,height),
    color_mode = 'rgb',
    classes=['random','old','new'],
    class_mode='categorical',
    batch_size = batch_size)

    print ('val Indices:',val_data_generator.class_indices)
    print('lr:',K.eval(model.optimizer.lr))
    print ('batch_size:',batch_size)
    
    history = model.evaluate_generator(
      val_data_generator,
      verbose=1)

    print ('validation_loss:{} validation_auc:{}'.format(history[0],history[-1]))
    
