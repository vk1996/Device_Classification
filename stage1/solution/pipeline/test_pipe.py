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

from tensorflow.keras.models import load_model,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy


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
classes=['no_device','old_device','new_device']

def test_pipe(test_path,model_dir,cmap=False):
    
    print('Searching for model.........')
    
    model=load_model(model_dir)
    print ('Model loaded')
    print ('building pipeline for data..........')
    
    x=cv2.cvtColor(cv2.resize(cv2.imread(test_path),(width,height)),cv2.COLOR_BGR2RGB)
    blob=np.expand_dims(x,axis=0)
    pred=model.predict(blob)
    print ('')
    print ('[RESULT]:predicted_class:{} confidence:{}'.format(classes[np.argmax(pred)],pred[:,np.argmax(pred)]))
    
    if cmap:
        plt.title('Predicted:{} Conf:{}'.format(classes[np.argmax(pred)],pred[:,np.argmax(pred)]))
        print ('')
        print ('generating visualization. Keep your eyes closed.........')
        inputs=model.input
        outputs=(model.layers[-4].output,model.layers[-1].output)
        new_model=Model(inputs=inputs,outputs=outputs)
        conv_out,pred_vec=new_model.predict(blob)

        pred=np.argmax(pred_vec)
        conv_out=np.squeeze(conv_out)
        mat = scipy.ndimage.zoom(conv_out, (32, 32, 1), order=1)
        #print (mat.shape)
        weights=new_model.layers[-1].get_weights()[0]
        weights = weights[:, np.argmax(pred_vec)]
        final_op=np.dot(mat.reshape(224*224,1664),weights).reshape(224,224)

        #print (pred_vec)
        
        plt.imshow(x)
        plt.imshow(final_op,cmap='jet',alpha=0.5)
        plt.show()
    
