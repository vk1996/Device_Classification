'''
A Computer Vision powered command line software 
to classify device.

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
from pipeline.train_pipe import *
from pipeline.val_pipe import *
from pipeline.test_pipe import *
from pipeline.test_sample_pipe import *
import argparse



def train():
    train_path=input("Enter train directory:")
    val_path=input("Enter validation directory:")
    modelpath=input("Enter model path to be trained:")
    savepath=input("Enter path to save model:")
    train_pipe(train_path,val_path,modelpath,savepath)

def val():
    val_path=input("Enter the validation directory:")
    modelpath=input("Enter model path to be evaluated:")
    return val_pipe(val_path,modelpath)

def test():
    sub_path=input("Enter the predictions directory:")
    modelpath=input("Enter model path to be evaluated:")
    return submission_pipe(sub_path,modelpath)

def sample():
    test_path=input("Enter the filename:")
    modelpath=input("Enter model path to be tested:")
    cmap=input("Visualize Network on Test Images[Y/N] ?")
    if cmap=='Y':
        test_pipe(test_path,modelpath,True)
    else:
        test_pipe(test_path,modelpath)
            
def main(input_):
    while True:
        text = input("Press [Enter] to start  [h] for help [q] to exit !!")
        if text == "":
            print ("You just entered Device Classification Software [{}] mode".format(input_))
            if input_=='train':
                train()

            if input_=='validate':
                val()

            if input_=='predict':
                test()

            if input_=='test':
                sample()

            if input_ not in mode :
                print ('Incorrect command!!! Retry!!!')
                exit()
                
            
        elif text=='q':
            print ("Exiting!!")
            exit()
            break
       

        elif text=='h':
            print ("We are here to help you!! This software is an Computer Vision-powered tool to classify device. Press [Enter] to launch the program and [q] to quit ")

            
        else:
            print ("!!OOps!!invalid input!!!Press [h] for help")
    
mode=['train','validate','predict','sample']
parser = argparse.ArgumentParser()
parser.add_argument("input")
args = parser.parse_args()
input_=args.input
main(input_)
