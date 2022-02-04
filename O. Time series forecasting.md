# Time Series forecasting

### Import statsmodels requests
```
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
```

Use `statsmodels` library to be able to decompose time series

--> Break a time series down into **systematic** and **unsystematic** components.

**Systematic**: Components of the time series that have consistency or recurrence and can be described and modeled.

**Non-Systematic**: Components of the time series that cannot be directly modeled.
A given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.

These components are defined as follows:

Level: The average value in the series.
<br>Trend: The increasing or decreasing value in the series.
<br>Seasonality: The repeating short-term cycle in the series.
<br>Noise: The random variation in the series.

**A series is thought to be an aggregate or combination of these four components.** That's why there are 2 models: additive and multiplicative

The snippet below shows how to decompose a series into trend, seasonal, and residual components assuming an additive model and plot the result.
```
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot

series = df.target.values
res = seasonal_decompose(series, model='additive')

trend = res.trend
seasonal = res.seasonal
residual = res.resid
observed = res.observed

res.plot()
pyplot.show()
```
