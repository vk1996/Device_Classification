You just stumbled upon fs0c131y's Device Classification Software. We promise a train_AUC of 1.000 and val_AUC of 0.9999

We have trained our Densely Connected Model with classes as 0 for [No device], 1 for [Old device], 2 for [New device]

File Structure:
```
Solution

   -Device_Detection.ipynb

   -model 
       - Download 150MB model file from https://drive.google.com/file/d/1zL3_7lijDvdNnYKpS-FBxeTRAOvwzKKb/view?usp=sharing and          place it here
   -onnx
       - The onnx.ipyb has instructions to convert model/model.h5 to model.onnx
```    

The model and preprocessing explanation is given separately as Markup in Device_Classification.ipynb file in submission folder.So don't expect any much docstrings in .py files 

Execution:
```
$pip3 install -r requirements.txt
$sudo apt-get install python3-opencv
$python3 fsociety.py test
-->img_path:path to any jpg,png,jpeg
-->model_path: model/model.h5
```
You can visualize what the model sees in the input.
To visualize, open software in test mode and press [Y] for visualization question.



