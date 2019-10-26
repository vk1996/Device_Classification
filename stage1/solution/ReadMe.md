You just stumbled upon fs0c131y's Device Classification Software. We promise a train_AUC of 1.000 and val_AUC of 0.9999

We have trained our Densely Connected Model with classes as 0 for [No device], 1 for [Old device], 2 for [New device]

File Structure:
```
Solution

   -Device_Detection.ipynb

   -model 
       - Download 150MB model file from https://drive.google.com/file/d/1zL3_7lijDvdNnYKpS-FBxeTRAOvwzKKb/view?usp=sharing and place it here
   -onnx
       - The onnx.ipyb has instructions to convert model/model.h5 to model.onnx
```    

The model and preprocessing explanation is given separately as Markup in Device_Classification.ipynb file in submission folder.So don't expect any much docstrings in .py files 

Execution:
```
$pip3 install -r requirements.txt
$sudo apt-get install python3-opencv
$python3 fsociety.py test

```
mode belongs to any one of the [train,validate,test,submit]. From here you will be guided by the cli software.
Appart from training,validating directory, test sample and submitting predictions. You can visualize what the model sees in the input.
To visualize, open software in test mode and press [Y] for visualization question.


For all validation,submission and train mode,

```

validation_directory-       data/val
submission_directory-      data/val/*
train_directory-           data/train (if downloaded train.zip from drive link and saved in submission/data folder)
model_directory -          model/model.h5
test_sample -              path to any jpg,png,jpeg etc

```



The current model is around (150MB .h5 and 48MB .onnx) which can be optimised to 1/10th with small trade-off in accuracy.



