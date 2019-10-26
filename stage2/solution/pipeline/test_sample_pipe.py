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
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras import backend as K
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy


nclasses=5
size=(384,384,3)
width=size[0]
height=size[1]
depth=size[2]
batch_size=16
seed=1
initial_epoch=0
train_count=1820
val_count=191
classes=['no_device','device_A','device_B','device_C','device_D']

def test_pipe(test_path,model_dir,cmap=False):
    
    print('Searching for model.........')
    
    model=load_model(model_dir)
    print ('Model loaded')
    print ('building pipeline for data of shape {}*{}..........'.format(width,height))
    
    x=load_img(test_path,target_size=(width,height))
    x=img_to_array(x)
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
        final_op=np.dot(mat.reshape(width*height,1664),weights).reshape(width,height)
        del model
        del new_model
        #print (pred_vec)
        
        plt.imshow(np.array(x,dtype=np.int32))
        plt.imshow(final_op,cmap='jet',alpha=0.5)
        plt.show()
    
