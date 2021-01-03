# Deep Learning Project


## Introduction
This is the Github Repository of our Deep Learning Project, where we worked on Time Series Benchmarking. We benchmarked a few models: NBeats (a pure deep learning model), Prophets (a statistical model), and ES-RNN (a hybrid model). \
INSERT ORGANISATION OF FOLDERS \
In order to keep this `README` file short and concise, we only provide an analysis on the results. For the code, refer to the relevant `jupyter` notebook. For a theoretical summary on the models, refer to the link below.


## Important links 

* Theoretical Summary: https://drive.google.com/uc?export=download&id=1JH89berkZJsZr3Scqmxenq3A60DYOHzR
* Original Papers: 
  * NBeats: https://drive.google.com/file/d/1x_G4pWQpHMl-etyAt9qOxV7f1gMnfODp/view?usp=sharing
  * Prophet: https://drive.google.com/file/d/1Q2f4CQeeNt_YXlqo_E2_CJArUqDIcTq7/view?usp=sharing
  * ES-RNN: https://eng.uber.com/m4-forecasting-competition/
  * M4-Competition: https://drive.google.com/file/d/1Q2f4CQeeNt_YXlqo_E2_CJArUqDIcTq7/view?usp=sharing
  
## Analysis
Due to the size of the M4 Dataset (100k different time series), we were unable to train our models on the full dataset due to time and computational constraints. We thus took a subset of the dataset instead (100 time series from each frequency, resulting in a total of 600 time series). Because we are working with a drastically smaller subset compared to the original one, it is obvious that our models cannot achieve the same level of accuracy. We expect the accuracy of the different models that we've benchmarked to improve substantially when the full dataset is used. In addition, for statistical models such as Prophet which requires some domain knowledge of tuning, accuracy can probably be improved with careful tuning of the parameters, which we were not able to perform due to time constraints. 

To measure the performance of the various models, we used the same benchmark described in the M4-Competition, using an overall weighted average (OWA) based on the sMAPE and MASE. Please refer to the theoretical summary or the original M4 paper for more details.

 ### Prophet
 We start our exposition with the Prophet model. Since we are only training on a small subset of the data, we recalculate the sMAPE and MASE of the Naive 2 benchmark on our subset for each frequency. They are given by the following table:
 
|      | sMAPE     | MASE     |
|------|-----------|----------|
| Mean | 12.061665 | 2.945491 |
 
 As expected, we can see that the sMAPE and MASE for the baseline Naive 2 model is bigger than if trained on the entire 100,000 time series. The values seen here are the ones we use as the baseline reference, and thus the OWA score is 1. After running our prophet model, we get the following scores:
 
 |      | sMAPE     | MASE     |
 |------|-----------|----------|
 | Mean | 20.087469 | 5.632893 |
 
 and by OWA score of around 1.82. That is, the Prophet model performs _considerably_ worse than the Naive 2 benchmark. Although this might seem surprising, we have a few conjectures for why this is the case. Firstly, Prophet is at its heart a non-parametric statistical model. There are several parameters that can be tuned, such as the trend and seasonality parameters. Since Prophet was originally built for business forecasting purposes, it works best when the analyst using Prophet possess substantial knowledge about the time series they're working with in order to make judgements about how to tune the parameters. Unfortunately, for the case of the M4 dataset, no information was released about the nature of the time series, and using domain knowledge to tune the parameters wasn't possible for us. However, we conjecture that for an analyst with expert domain knowledge, Prophet will probably perform way better than baseline models such as the Naive 2 or ARIMA models. 
 
 In addition to solely looking at the scores, we can also take a look at the plots to see how the forecasts perform in general.
 
 ### NBeats
 
 NBeats is a _pure_ deep learning model developed by Element AI for time series forecasting. This makes it substantially different from Prophet, which is a statistical model. 
 
 
 
 ### ES-RNN
