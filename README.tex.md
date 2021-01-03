
## Introduction

This is the GitHub repository of time series benchmarking on the M4 competition dataset. We test the accuracy of a few models: `NBeats` and `Prophet`. 

## Problem Statement
We now describe the objective function of our models. 

We have the _univariate_ point forecasting problem in discrete time. Given a series of observations $[y_1,...y_T] \in \mathbb{R}^T$, our goal is to predict of a vector of future values of horizon $h$, $\mathbf{y} \in \mathbb{R}^H = [y_{T+1},y_{T+2},...,y_{T+H}] \in \mathbb{R}^T$. We consider a lookback window of length $t \leq T$ ending with the last observed value $y_T$ as our model input, denoted $\mathbf{x} \in \mathbb{R}^T = [y_{T-t+1},...,y_T]$. Our forecasts of $\mathbf{y}$ are denoted $\hat{\mathbf{y}}$.

In the M4 competition datasets, we have many time series of different frequencies available. The forecasting horizon for the different frequencies are briefly discussed in the next section. The goal of this project is to benchmark a few different methods on the M4 competition dataset, which comprises of 100,000 time series, where we compare our models to seasonally-adjusted naive forecasts.


## Metrics

We use the sMAPE and MASE as building blocks for the measures of model performance. They are defined by:

$$\text{sMAPE} = \frac{2}{h} \sum_{n+1}^{n+h} \frac{|Y_t - \hat{Y_t}|}{|Y_t| + \hat{Y_t}} \times 100% $$

and

$$ \text{MASE} = \frac{1}{h} \frac{\sum_{n+1}^{n+h} |Y_t - \hat{Y_t}|}{ \frac{1}{n+m}\sum_{t=m+1}^n |Y_t - Y_{t-m}|} $$

where $\hat{Y_t}$ denotes the estimated forecast, $h$ the forecasting horizon, $n$ the number of data points available in-sample, and $m$ the time interval between successive observations considered by the organisers for each data frequency (12 for monthly, 4 for quarterly, 24 for hourly, one for weekly, yearly, daily data).

They are then combined to form the overall weighted average (OWA):

$$\text{OWA} = (\frac{\text{sMAPE}}{\text{Naive 2}} + \frac{\text{MASE}}{\text{Naive 2}}) / 2  $$

To be precise, sMAPE and MASE are calculated for each individual time series (based on the frequencies), and then averaged across all time series. The OWA is only calculated once at the end.


## NBeats

Nbeats is a pure deep learning model developed by Element AI with no time-series specific components. It is designed based on two principles:

1) The base architecture should be generic and simple, but deep.

2) It should not rely on time series-specific feature engineering, methods or scaling. This is what makes `NBeats` a 'pure' deep learning model compared to other statistical (e.g ARIMA) or hybrid (e.g ES-RNN) models.

An overview of the architecture is shown in the image below.

![NBeats Architecture](https://raw.githubusercontent.com/ElementAI/N-BEATS/master/nbeats.png)

In the following description, we focus on the $l$-th block. The $l$-th block takes an input $\mathbf{x}_l$ and outputs two vectors $\hat{\mathbf{x}}_l$ and $\hat{\mathbf{y}}_l$. In the first block, the input corresponds to the overall model input, which is the history look backwindow of pre-defined length ending with the last observation.

The length of the input window is set to be a multiple of the forecast horizon $H$, ranging from $2H$ to $7H$. The residual output is then passed as input to the rest of the blocks. Each block has two outputs: 

(i) the block's forward forecast of length $H$, $\hat{\mathbf{y}}_l$ \\
(ii) the block's best estimate of $\hat{\mathbf{x}}_l$, also known as the backcast.

Within the basic building block, there are two parts. The first part is a fully connected network that produces the forward forecast $\theta_l^f$ and backcast $\theta_l^b$. The second part consists of the backward and forward basis layers $g_l^b$ and $g_l^f$ that accepts the expansion coefficients $\theta_l^f$ and $\theta_l^b$ produced by the first part. The coefficients are then projected onto the set of basis functions, and produces the backcast $\hat{\mathbf{x}}_l$ and forecast $\hat{\mathbf{y}}_l$. In summary, the operations of the first part of the $l$-th block is characterised by the following equations

\begin{equation}
\begin{aligned}
\mathbf{h}_{l,1} = \text{FC}_{l,1}(\mathbf{x}_l), \quad \mathbf{h}_{l,2} = \text{FC}_{l,2}(\mathbf{h}_{l,1}), \mathbf{h}_{l,3}(\mathbf{h}_{l,2}), \quad 
\mathbf{h}_{l,4} = \text{FC}_{l,4}(\mathbf{h}_{l,3}) \\
\theta_l^b = \text{LINEAR}_l^b (\mathbf{h}_{l,4}), \quad \theta_l^f = \text{LINEAR}_l^f(\mathbf{h}_{l,4})
\end{aligned}
\end{equation}


where the linear layer here is a projection layer $\theta_l^f = \mathbf{W}_l^f \mathbf{h}_{l,4}$, and the FC layer is a standard fully connected RELU layer. For example, we have $\mathbf{h}_{l,1} = \text{RELU}(\mathbf{W}_{l,1}\mathbf{x}_l + \mathbf{b}_{l,1})$. The goal of this part of the architecture is to predict the forward expansion coefficients $\theta_l^f$, properly mix the basis vectors supplied by $g_l^f$, and ultimately optimise the accuracy of the forecast $\hat{\mathbf{y}}_l$.

The second part of the network maps the expansion coefficients $\theta_l^f$ and $\theta_l^b$ to outputs via the basis layers $\hat{\mathbf{y}}_l = g_l^f(\theta_l^f)$ and $\hat{\mathbf{x}}_l = g_l^b(\theta_l^b)$. This part is characterised by the following equations

\begin{equation}
\hat{\mathbf{y}}_l = \sum_{i = 1}^{\text{dim}(\theta_l^f)} \theta_{l,i}^f \mathbf{v}_i^f \\
\hat{\mathbf{x}}_l = \sum_{i=1}^{\text{dim}(\theta_l^b)} \theta_{l,i}^b \mathbf{v}_i^b
\end{equation}


where $\mathbf{v}_i^f$ and $\mathbf{v}_i^b$ are forecast and backcast basis vectors, and $\theta_{l,i}$^f is the $i$-th element of $\theta_l^f$.

Instead of a classical residual network architecture where the input is added to the output of a stack of layers, a novel double residual stacking is done in NBeats. This is seen in the middle and right portion of the Figure above. There are two branches, one running over the backcast prediction of each layer, and the other running over the forecast branch of eahc layer. It is described by the equations

\begin{equation}
\begin{aligned}
\mathbf{x}_l = \mathbf{x}_{l-1} - \hat{\mathbf{x}}_{l-1} \\
\hat{\mathbf{y}} = \sum_{l} \hat{\mathbf{y}}_l
\end{aligned}
\end{equation}

Each blocks outputs a forecast $\hat{\mathbf{y}}_l$ that is first aggregated at the stack level and then at the overall network level, providing a hierarchical decomposition, and these block forecasts are then summed to produce the final forecast $\hat{\mathbf{y}}_l$.

Finally, ensembling can be done to maximise the performance of the model.  


## Prophet

Prophet is an open-source forecasting model developed by Facebook for business forecasting applications. A decomposable time series model with three components is used: trend, seasonality and holidays, represented by the equation

\begin{equation}
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\end{equation}

where $g(t)$ is the trend function modelling non-periodic changes, $s(t)$ modelling periodic changes (weekly and yearly seasonality) , $h(t)$ is the effect of holidayas which occur on potentially irregular schedule over one or more days, and $\epsilon_t$ is an idosyncratic and Gaussian error term.


The problem is basically a non-parametric curve fitting exercise, which is more statistical in nature, compared to the 'pure' deep learning model that we've seen earlier. However, it is still quite different from parametric approaches such as ARIMA models which are often used as a baseline. This gives Prophet an advantage in interpretability relative to 'black box' deep learning models such as NBeats.

### Trend Model

There are two trend models implemented - a saturating rating growth model and a piecewise linear model.

#### Non-linear saturating growth
We first discuss the saturating rating growth model. In its basic form, growth model is a logistic one, taking the form

$$g(t) = \frac{C(t)}{1 + \text{exp}(-k + \mathbf{a}(t)^T \mathbf{\delta})(t - (m+\mathbf{a}(t)^T \mathbf{\gamma}))} $$

where $C$ is the carrying capacity, $k$ is the growth rate, and $m$ is an offset parameter. We need to be more precise on some of the variables. Our variable growth rate incorporates trend by explicitly defining changepoints where the growth rate is allowed to change. Suppose there are $S$ changepoints at times $s_j$, $j = 1,...,S$. We define a vector of rate adjustments $\mathbf{\delta} \in \mathbb{R}^s$, where $\delta_j$ is the change in rate that occurs at time $s_j$. The rate at any time t is given by a base rate plus all the adjustments up to that time, and given by a vector $\mathbf{a}(t) \in \{0,1\}^S$ such that 

\begin{equation}
\begin{cases}
a_j(t) = 1 \quad \text{if} t \geq s_j, \\
0 \quad \text{otherwise} 
\end{cases}
\end{equation}

and so the rate at time $t$ is $k + \mathbf{a}(t)^T \mathbf{\delta}$. When $k$ is adjusted, the offset parameter $m$ must also be adjusted to connect the endpoints of the segments. This is computed as

$$\delta_j = (s_j - m - \sum_{l < j}\gamma_l)(1 - \frac{k + \sum_{l < j} \delta_l}{k + \sum_{l \leq j }\delta_l}) $$

#### Linear Trend with Changepoints

Here, a piecewise constant rate of growth is used:

\begin{equation}
g(t) = (k + \mathbf{a}(t)^T \mathbf{\delta})t + (m + \mathbf{a}(t)^T \mathbf{\gamma})
\end{equation}

where $k$ is the growth rate, $\mathbf{\delta}$ has the rate adjustments, $m$ is the offset parameter, and $\delta_j$ is set to $-s_j \delta_j$ to make the function continuous.

#### Automatic Changepoint Selection 

Instead of manually selecting the changepoints, automatic selection can also be done by putting a sparse prior on $\mathbf{\delta}$. In practice, a large number of changepoints are specified, and the prior $\delta_j \sim$ Lapalce(0, $\tau$) is used. $\tau$ controls the flexibility of the model in altering its rate, and a sparse prior on the adjustments $\mathbf{\delta}$ has no impact on the primary growth rate $k$, so as $\tau \rightarrow 0$ the fit reduces to a standard logistic or linear growth model.


#### Trend Forecast Uncertainty

The uncertainty in the forecast trend can be estimated by extending the generative model forward. There are $S$ changepoints over a history of $T$, each of which has a rate change $\delta_j \sim$ Laplace(0, \tau). Future rate changes that emulate past data can be done by replacing $\tau$ with the empirical variance. For bayesians this is done with a hierarchical prior on $\tau$ to obtain the posterior, and for frequentists this is done with a maximum likelihood estimate of the rate scale parameter $\lambda = \frac{1}{S}\sum_{j=1}^s |\delta_j|$. Future changepoints are then randomly sampled in such a way that the average frequency matches that in the history.

### Seasonality

To model periodicity, Fourier series are used. We let $P$ to be the regular period we expect the time series to have (e.g $P = 7$ for weekly data). Arbitrary smooth seasonal effects can be approximated with 

$$s(t) = \sum_{n=1}^N (a_n \cos(\frac{2\pi nt}{P}) + b_n \sin (\frac{2\pi nt}{P})) $$
which is a standard Fourier series. Estimation requires $2N$ parameters $\mathbf{\beta} = [a_1,b_1,...,a_N,b_N]^T$, which is done by constructing a matrix of seasonality vectors for each value of $t$ in historical and future data, and then seasonal component is $s(t) = X(t) \mathbf{\beta}$, where $\mathbf{\beta} \sim \text{Normal}(0,\sigma^2)$. Parameter choice can be automated using model selection techniques like the AIC.

### Holidays and Events

Holidays generate predictable shocks that do not follow a periodic pattern. The data scientist can provide a custom list of past and future events (country-specific), which is incorporated into the model by assuming the effects of holidays are independent. For each holiday $i$, let $D_i$ be the set of past and future dates for that holiday, and we can add an indicator function representing whether $t$ is during holiday $i$, and assign each holiday a parameter $\kappa$_i, which is the change in the foreacst. A matrix of regressors $Z(t) = [\mathbf{1}(t \in D_1),..., \mathbf{1}(t \in D_L)]$ is generated, taking $h(t) = Z(t) \kappa$, where the prior $\mathbf{\kappa} \sim$ Normal $(0, v^2)$.

### Tuning Parameters

While there are default settings for the Prophet model, there are several places where analysts with domain knowledge can alter the model, such as capacities, changepoints, holidays & seasonality, and smoothing parameters. 


