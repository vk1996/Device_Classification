## MULTICLASS DEVICE CLASSIFICATION STAGE2 ####

### OVERVIEW ####
You just stumbled upon fs0c131y's Device Classification Software. We promise a train_AUC of 1.000 and val_AUC of 0.99997 
We have trained our Densely Connected Model with prediction classes ordered as  1 for [device_A], 2 for [device_B], 3 for [device_C] ,4 for [device_D], 0 for [No device] .

#### The accpeted input image formats are common formats like jpg,jpeg,png.The image size should be greater than 400*400(refer preprocessing_output_details.pdf) . ###
  
#### To install required packages #
```bash
$pip3 install -r requirements.txt
$sudo apt-get install python3-opencv
```
#### Directory Structure ###
 
```
submission/  
    model/
        Download model from <https://drive.google.com/file/d/1qu1D20wVP1hlA2eKTEf6IUPeTY7riNJA/view?usp=sharing> and place here
    
    fsociety.py  
    output.csv (after prediction, the csv will be here.) 
    README.md  
    requirements.txt
    report.pdf
    preprocessing_output_details.pdf
```

The cli gets the input for filepath,modelpath from the user. The cli will guide once you get started.

#### Execution ####

``` bash
$ python3 fsociety.py test
enter img path			      : 'path to image jpg/png/jpeg'
enter model path                      : model/model.h5
Visualize Network on Test Images[Y/N] : Y

```

The report pdf consists of ipynb explaining the training process. The model conversion of onnx and coreML is explained in the report along with installation of conversion related libraries. The Preprocessing&Output_layer.pdf has details involving our simple preprocessing pipeline and what the output layer pumps out .











