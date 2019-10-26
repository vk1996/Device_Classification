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

from tensorflow.keras.callbacks import *


def callbacks(savepath):
    model_checkpoint_acc = ModelCheckpoint(filepath=savepath,
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
    model_checkpoint_loss = ModelCheckpoint(filepath=savepath,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

    model_checkpoint_auc = ModelCheckpoint(filepath=savepath,
                                   monitor='val_auc',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='max',
                                   period=1)

    csv_logger = CSVLogger(filename='devicecf.csv',
                       separator=',',
                       append=True)

    callbacks_ = [model_checkpoint_loss,model_checkpoint_acc,csv_logger,model_checkpoint_auc]
    return callbacks_
