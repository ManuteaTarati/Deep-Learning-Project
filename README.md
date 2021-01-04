# Deep Learning Project


## Introduction
This is the Github Repository of our Deep Learning Project, where we worked on Time Series Benchmarking. We benchmarked a few models: NBeats (a pure deep learning model), Prophets (a statistical model), and ES-RNN (a hybrid model). \
INSERT ORGANISATION OF FOLDERS \
In order to keep this `README` file short and concise, we only provide an analysis on the results. For the code, refer to the relevant folders. For a technical summary on the models, refer to the link below.


## Important links 

* Technical Summary: https://drive.google.com/uc?export=download&id=1YHZbR8TkujcWc_wkKIIcwTK_NbNSm71p
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
 
|   | sMAPE     | MASE     | OWA |
|---|-----------|----------|-----|
|   | 12.061665 | 2.4 | 1   |
 
 As expected, we can see that the sMAPE and MASE for the baseline Naive 2 model is bigger than if trained on the entire 100,000 time series. The values seen here are the ones we use as the baseline reference, and thus the OWA score is 1. After running our prophet model, we get the following scores:
 
|   | sMAPE     | MASE     | OWA  |
|---|-----------|----------|------|
|   | 13.38 | 4.7 | 1.86 |
 
 and by OWA score of around 1.86. That is, the Prophet model performs _considerably_ worse than the Naive 2 benchmark. Although this might seem surprising, we have a few conjectures for why this is the case. Firstly, Prophet is at its heart a non-parametric statistical model. There are several parameters that can be tuned, such as the trend and seasonality parameters. Since Prophet was originally built for business forecasting purposes, it works best when the analyst using Prophet possess substantial knowledge about the time series they're working with in order to make judgements about how to tune the parameters. Unfortunately, for the case of the M4 dataset, no information was released about the nature of the time series, and using domain knowledge to tune the parameters wasn't possible for us. However, we conjecture that for an analyst with expert domain knowledge, Prophet will probably perform way better than baseline models such as the Naive 2 or ARIMA models. 
 
 In addition to solely looking at the scores, we can also take a look at the plots to see how the forecasts perform in general. Let us take a look at the plot for the yearly series:
 
 ![Prophet Yearly Predictions](https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/Yearly_Prophet.png)
 
 The black points represents our observations in the training set, while the red dots represents observations in the test set. The blue line up to the last black dot represents our in-sample fit, and out-of-sample forecasts in the region of the red dots. The blue area around the line is the confidence interval of our fit. We can see that the predictions do not perform well - only 4 out of the 6 forecasts are within the confidence interval and only one test sample lies on the forecast line. This is in line with our performance measures, since the Prophet model performs significantly worse than even the Naive 2 benchmark. Let us look at the other plots for the different frequencies:
 
 <p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/Quarterly_Prophet.png" width="500" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/Monthly_Prophet.png" width="500" /> 
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/Daily_Prophet.png" width="500" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/Hourly_Prophet.png" width="500" />
</p>
 
 ### NBeats
 
 NBeats is a _pure_ deep learning model developed by Element AI for time series forecasting. This makes it substantially different from Prophet, which is a statistical model. The original developers have demonstrated that NBeats has outperformed even the best submissions in the M4 competition. However, since we are unable to work on the full dataset, we wanted to see how it stacks up against the other methods when there is much less data to train on. This is a test on just how flexible it is - while deep learning models like these can be extremely powerful, they require large amounts of data to train on. In addition, they suffer in interpretability and transparency relative to statistical methods due to their "black box" nature. While the authors have proposed specific configurations to make it more interpretable, we are still of the opinion that it is still not as interpretable as pure statistical methods. This is not necessarily a bad thing, since the benefits of interpretability highly depends on the application domain. Similar to before, we first look at the scores when trained on our data subset:
 
 
 
 ### ES-RNN
 
 The Exponential Smoothing-Recurrent Neural Network is a hybrid model developed by Slawek Smyl, the winner of the M4 competition. It is a hybrid model in the sense that it combines both statistical modelling and machine learning components in one model. The key idea is to use exponential smoothing (the statistical part) to model the seasonality of the data for on-the-fly pre-processing, and then pass the processed data into a LSTM type neural network for prediction. This combines the best of both the statistical and deep learning worlds, since the statistical model captures the main components of each individual series such as seasonality and level, while the LSTM network enables non-linear trends and cross-learning. Again, we look at the scores when trained on our data subset: 
 
 
 
