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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from glob import glob
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
nclasses=3
size=(224,224,3)
width=size[0]
height=size[1]
depth=size[2]
batch_size=16
seed=1
initial_epoch=0
train_count=806
val_count=142


def submission_pipe(sub_path,model_dir):
    print('Searching for model.........')
    model=load_model(model_dir)
    print ('Model loaded')
    print ('building pipeline for data..........')
    file_=[]
    no=[]
    old=[]
    new=[]
    print('target path :{}'.format(sub_path+'/*'))
    for i in tqdm(glob(sub_path+'/*')):
        x=cv2.cvtColor(cv2.resize(cv2.imread(i),(width,height)),cv2.COLOR_BGR2RGB)
        blob=np.expand_dims(x,axis=0)
        pred=model.predict(blob)
        file_.append(i)
        no.append(pred[:,0])
        old.append(pred[:,1])
        new.append(pred[:,2])

    df=pd.DataFrame({'filename':file_,'prob_no_device': no,'prob_old_device':old,'prob_new_device':new})
    df.to_csv('submission.csv',index=False)
