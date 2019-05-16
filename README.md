# deep_learning_project

## **Perceptron Part**

_arraynumpy.py_ is a python file that represent a perceptron that learns linear classification with 2 gaussians vectors.

## **Recurrent Neural Network**

_LSTM.py_ is the recurrent neural network

### **In order to make the LSTM.py_ file works:**
- To just launch the neural network without the DATA ( all the informations if in the _f.pickle_ file),
  
  you have to launch `./download.sh` in the *same* repository as _LSTM.py_

- To have access to the *unquoted part* you will need the data,

  you can access it with `./dowload_data.sh`

- **All the Data is stored on my samba server, that is where the scripts are taking the Data**

- If you have some issues with `CuDNNLSTM` layer just replace it by `LSTM` 

## **Deep Neural Network**

_urbanNN.py_ is a python file that load wav files and feed a Neural Network in order to predict them.

The wav files are from [analyticsvidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/)

_result.csv_ is the result of the prediction of the NN
### **In order to make the _urbanNN.py_ file works:**

- download the wav files from : [analytics google drive](https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU)
- the folder _Train_, _Test_ and the excel files _train.csv_ and _test.csv_ must be in the **_same_** directory that the _urbanNN.py_


### *Sources*
- https://github.com/ashwinsamuel/Urban-Sound-Classification
- [analytics vidhya](https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/)

