'''
To submit predictions as csv.

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras import backend as K
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

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


def submission_pipe(sub_path,model_dir):
    print('Searching for model.........')
    model=load_model(model_dir)
    print ('Model loaded')
    print ('building pipeline for data of shape {}*{}..........'.format(width,height))
    file_=[]
    no=[]
    A=[]
    B=[] 
    C=[]
    D=[]
    print('using tf.keras')
    print('target path :{}'.format(sub_path+'/*'))
    for i in tqdm(glob(sub_path+'/*')):
        #x=cv2.cvtColor(cv2.resize(cv2.imread(i),(width,height)),cv2.COLOR_BGR2RGB)
        #blob=np.expand_dims(x,axis=0)
        x=load_img(i,target_size=(width,height))
        x=img_to_array(x)
        blob=np.expand_dims(x,axis=0)
        pred=model.predict(blob)
        file_.append(i)
        no.append(np.ravel(pred[:,0])[0])
        A.append(np.ravel(pred[:,1])[0])
        B.append(np.ravel(pred[:,2])[0])
        C.append(np.ravel(pred[:,3])[0])
        D.append(np.ravel(pred[:,4])[0])
     

    df=pd.DataFrame({'filename':file_,'prob_device_A':A,'prob_device_B':B,'prob_device_C':C,'prob_device_D':D,'prob_no_device': no})
    df.to_csv('output.csv',index=False) 





