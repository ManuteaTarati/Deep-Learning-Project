# Deep Learning Project


## Introduction
This is the Github Repository containing the Deep Learning Project by Manutea Tarati and Sunny Wang, where we worked on Time Series Benchmarking. We benchmarked a few models: NBeats (a pure deep learning model), Prophets (a statistical model), and ES-RNN (a hybrid model) on the M4 competition dataset. \
INSERT ORGANISATION OF FOLDERS \
In order to keep this `README` file short and concise, we only provide an analysis of the results. For the code, refer to the relevant folders. Pleasea read the technical summary in the link below for a description of the models.


## Important links 

* __Technical Summary__: https://drive.google.com/uc?export=download&id=1YHZbR8TkujcWc_wkKIIcwTK_NbNSm71p
* Original Papers: 
  * NBeats: https://drive.google.com/file/d/1x_G4pWQpHMl-etyAt9qOxV7f1gMnfODp/view?usp=sharing
  * Prophet: https://drive.google.com/file/d/1Q2f4CQeeNt_YXlqo_E2_CJArUqDIcTq7/view?usp=sharing
  * ES-RNN: https://eng.uber.com/m4-forecasting-competition/
  * M4-Competition: https://drive.google.com/file/d/1Q2f4CQeeNt_YXlqo_E2_CJArUqDIcTq7/view?usp=sharing
  
## Analysis
Due to the size of the M4 Dataset (100k different time series), we were unable to train our models on the full dataset due to time and computational constraints. We thus took a subset of the dataset instead (100 time series from each frequency, resulting in a total of 600 time series). Because we are working with a drastically smaller subset compared to the original one, it is obvious that our models cannot achieve the same level of accuracy. We expect the accuracy of the different models that we've benchmarked to improve substantially when the full dataset is used. In addition, for statistical models such as Prophet which requires some domain knowledge of tuning, accuracy can probably be improved with careful tuning of the parameters, which we were not able to perform due to time constraints. However, not using the full dataset allows us to provide a fresh perspective on how these models perform in the presence of small data instead of big data. If we have used the complete dataset, no new insight would've been gleaned from this project other than learning how to implement them.

To measure the performance of the various models, we used the same benchmark described in the M4-Competition, using an overall weighted average (OWA) based on the sMAPE and MASE. Please refer to the technical summary or the original M4 paper for more details. Since we are only training on a small subset of the data, we recalculate the sMAPE and MASE of the Naive 2 benchmark on our subset for each frequency. They are given by the following table:
 
| sMAPE     | MASE     | OWA |
|-----------|----------|-----|
| 7.79 | 2.34 | 1   |
 
 As expected, we can see that the sMAPE and MASE for the baseline Naive 2 model is bigger than if trained on the entire 100,000 time series. The values seen here are the ones we use as the baseline reference, and thus the OWA score is 1.

 ### Prophet
 We start our exposition with the Prophet model. After running our prophet model, we get the following scores:
 
| sMAPE     | MASE     | OWA  |
|-----------|----------|------|
| 13.38 | 4.7 | 1.86 |
 
 and an OWA score of around 1.86. That is, the Prophet model performs _considerably_ worse than the Naive 2 benchmark. Although this might seem surprising, we have a few conjectures for why this is the case. Firstly, Prophet is at its heart a non-parametric statistical model. There are several parameters that can be tuned, such as the trend and seasonality parameters. Since Prophet was originally built for business forecasting purposes, it works best when the analyst using Prophet possess substantial knowledge about the time series they're working with in order to make judgements about how to tune the parameters. Unfortunately, for the case of the M4 dataset, no information was released about the nature of the time series, and using domain knowledge to tune the parameters wasn't possible for us. However, we conjecture that for an analyst with expert domain knowledge, Prophet will probably perform way better than baseline models such as the Naive 2 or ARIMA models. Secondly, we only used a small subset of full M4 dataset, and there might be some sparsity issues (in the sense that the sample size is not large enough) which affect the performance of this non-parametric model. Non-parametric models typically require a lot more data than fully parametric ones, since there must be sufficient data to "speak" about both the model structure _and_ its parameters.
 
 In addition to solely looking at the scores, we can also take a look at the plots to see how the forecasts perform in general. Let us take a look at the plot for the yearly series:
 
 <img src = "https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/prophet/Yearly_Prophet.png" width = "450" />
 
 The black points represents our observations in the training set, while the red dots represents observations in the test set. The blue line up to the last black dot represents our in-sample fit, and out-of-sample forecasts in the region of the red dots. The blue area around the line is the confidence interval of our fit. We can see that the predictions do not perform well - only 4 out of the 6 forecasts are within the confidence interval and only one test sample lies on the forecast line. This is in line with our performance measures, since the Prophet model performs significantly worse than even the Naive 2 benchmark. Let us look at the other plots for the different frequencies:
 
 <p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/prophet/Quarterly_Prophet.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/prophet/Monthly_Prophet.png" width="400" /> 
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/prophet/Daily_Prophet.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/prophet/Hourly_Prophet.png" width="400" />
</p>

We can see from the plots that regardless of the frequency, Prophet's predictions perform really badly and its forecast almost never aligns with the testing sample. Perhaps only the predictions from the Hourly frequency performs slightly better, since most of the test set lies in the confidence interval. Nevertheless, the point forecasts are still bad, but it looks for this particular frequency, the model is even struggling to do a good in-sample fit because of the structure of the data.
 
 ### NBeats
 
 NBeats is a _pure_ deep learning model developed by Element AI for time series forecasting. This makes it substantially different from Prophet, which is a statistical model. The original developers have demonstrated that NBeats has outperformed even the best submissions in the M4 competition. However, since we are unable to work on the full dataset, we wanted to see how it stacks up against the other methods when there is much less data to train on. This is a test on just how flexible it is - while deep learning models like these can be extremely powerful, they require large amounts of data to train on. In addition, they suffer in interpretability and transparency relative to statistical methods due to their "black box" nature. While the authors have proposed specific configurations to make it more interpretable, we are still of the opinion that it is still not as interpretable as pure statistical methods. This is not necessarily a bad thing, since the benefits of interpretability highly depends on the application domain. Similar to before, we first look at the scores when trained on our data subset:
 
 | sMAPE | MASE  | OWA  |
|-------|-------|------|
| 17.95 | 11.66 | 3.64 |

Interestingly, we can see that NBeats performs _much worse overall_ when there is very little data to train on (100 time series instead of 100k). It performs even worse than Prophet, a statistical method. This is because while NBeats did well in some frequencies (monthly and hourly), it did extremely poorly on others (10 times worse), and the overall score is an average across all frequencies. Let us now take a look at the hourly and monthly plots, which were the ones which performed well:

 <p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/NBeats/Monthly_NBeats.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/NBeats/Hourly_NBeats.png" width="400" /> 
</p>
 
 We can see that for these two frequencies, NBeats actually did a good job predicting the test data. Even though it failed to capture a lot of the fluctuations, it still predicted it in the same direction. This is already much better than Prophet's predictions, which did not even go in the same direction (i.e increasing or decreasing next). Let us now take a look at the frequencies  which were responsible for NBeat's horrible overall score:
 
  <p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/NBeats/Daily_NBeats.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/NBeats/Weekly_NBeats.png" width="400" /> 
</p>
 
We can see that for both these frequencies, it generates huge amounts of un-necessary fluctuations. For example, in the Daily plot, all it needed to do was to predict an increasing trend, but instead it generated huge amounts of noise. The converse is true for the Weekly plot - instead of simply predicting a decreasing trend, it made a huge jump downwards and starting fluctuation a lot. This shows that NBeats requires a lot more data to train on before being able to handle all frequencies well. We already know from the creator's previous benchmarks that NBeats can outperform even the best models in the M4 competition when trained on the entire dataset, but our analysis here gives a fresh perspective on how a pure machine learning model like this stacks up when there is small instead of big data, illustrating just how data hungry these pure deep learning models are.
 
 ### ES-RNN
 
 The Exponential Smoothing-Recurrent Neural Network is a hybrid model developed by Slawek Smyl, the winner of the M4 competition. It is a hybrid model in the sense that it combines both statistical modelling and machine learning components in one model. The key idea is to use exponential smoothing (the statistical part) to model the seasonality of the data for on-the-fly pre-processing, and then pass the processed data into a LSTM type neural network for prediction. This combines the best of both the statistical and deep learning worlds, since the statistical model captures the main components of each individual series such as seasonality and level, while the LSTM network enables non-linear trends and cross-learning. Again, we look at the scores when trained on our data subset: 
 
 | sMAPE | MASE  | OWA  |
|-------|-------|------|
| 9.945 | 3.459 | 1.38 |

Here, we can see that ES-RNN outperforms NBeats and Prophet on our data subset. This is an interesting result - while NBeats can do better when trained on the entire dataset, the ES-RNN model fares better when it comes to small data. Let us also take a look at the plots of our forecasts, starting off with the frequencies that performed well:

  <p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/esrnn/Daily_ESRNN.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/esrnn/Weekly_ESRNN.png" width="400" /> 
</p>
 
We can see that the forecasts are very close to the "ground truth". We take a look at the frequencies which performed moderately well:

<p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/esrnn/Yearly_ESRNN.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/esrnn/Quarterly_ESRNN.png" width="400" /> 
</p>

We can see that although the forecasts are still quite off, at least they trend in the same direction. Finally, let us look at the worst performing frequencies:

<p float="left">
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/esrnn/Monthly_ESRNN.png" width="400" />
  <img src="https://github.com/sunnywang93/Deep-Learning-Project/blob/main/images/esrnn/Hourly_ESRNN.png" width="400" /> 
</p>
 
The main problem with these forecasts is that they do not capture nearly enough fluctuations as the actual test set, which is surprising to us since the fluctuations were really pronounced even in the training sample. Our hypothesis is that the de-seasonalisation in the statistical component is primarily driving this phenomenon, and the neural network doesn't have enough training samples to learn from and correct this behaviour.

## Conclusion

In this benchmarking test, we have provided a fresh perspective on how these previously well-known models stacks up against each other when there is small instead of large amounts of data. We have found that when the dataset is sufficiently small, comprising only 100 time series of 100,000, the hybrid model ES-RNN performs the best, followed by Prophet, a statistical model, with NBeats, a pure deep learning model ranking last. It is interesting to see the reversal of results between ES-RNN and NBeats when working with small instead of large amounts of time series. 

What is even more interesting is that _all_ these models performed worst than the Naive 2 benchmark in the presence of small data! While we should be prudent in jumping to conclusions based on this small project, it nudges us to follow the motto "with small data, the simplest models work best". This project has helped us to learn a great about state-of-the-art time series methods (both the theory and practical implementation), and we have no doubt that it'll be useful for us in our careers as future data scientists.
 
 
 
