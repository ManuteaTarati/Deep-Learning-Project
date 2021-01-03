# Deep-Learning-Project


## Introduction
This is the Github Repository of our Deep Learning Project, where we worked on Time Series Benchmarking. We benchmarked a few models: NBeats (a pure deep learning model
developed by Element AI), Prophets (a statistical model developed by Facebook), and ES-RNN (a hybrid model developed by Smyl, winner of the M4 competition). 
INSERT ORGANISATION OF FOLDERS
This README file only contains a summary of our findings of the various models. For a summary on the more theoretical aspects of the models, refer to the links below.

## Important links 

* Theoretical Summary: https://drive.google.com/uc?export=download&id=1cCb9SroZqGw1wGVDrGHjs3em8eX7_orD  
* Original Papers: 
  * NBeats: https://drive.google.com/file/d/1x_G4pWQpHMl-etyAt9qOxV7f1gMnfODp/view?usp=sharing
  * Prophet: https://drive.google.com/file/d/1Q2f4CQeeNt_YXlqo_E2_CJArUqDIcTq7/view?usp=sharing
  * ES-RNN: https://eng.uber.com/m4-forecasting-competition/
  * M4-Competition: https://drive.google.com/file/d/1Q2f4CQeeNt_YXlqo_E2_CJArUqDIcTq7/view?usp=sharing
  
## Analysis
Due to the size of the M4 Dataset (100k different time series), we were unable to train our models on the full dataset due to time and computational constraints. We thus took a subset of the dataset instead (100 time series from each frequency, resulting in a total of 600 time series). Because we are working with a drastically smaller subset compared to the original one, it is obvious that our models cannot achieve the same level of accuracy. We expect the accuracy of the different models that we've benchmarked to improve substantially when the full dataset is used. In addition, for statistical models such as Prophet which requires some domain knowledge of tuning, accuracy can probably be improved with careful tuning of the parameters, which we were not able to perform due to time constraints.

