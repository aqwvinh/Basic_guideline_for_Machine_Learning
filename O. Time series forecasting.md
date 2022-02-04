# Time Series forecasting

Import statsmodels requests
```
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
```

Use `statsmodels` library to be able to decompose time series
### Theory

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

### Code

The function below shows how to decompose a series into trend, seasonal, and residual components assuming an additive model and plot the result.
<br>```seasonal_decompose``` returns an **object** with trend, seasonal, residuals and observed attributes (each one is an array).
```
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot

# Function to decompose a target from df giving the possibility to select samples size and period
def decompose_serie(df, target, model, samples, period):
    if samples == 'all':
        #decomposing all time series timestamps
        res = seasonal_decompose(df[target].values, model=model, period=period)
    else:
        #decomposing a sample of the time series (take the n-samples last ones)
        res = seasonal_decompose(df[target].values[-samples:], model=model, period=period)
    
    observed = res.observed
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    #plot the complete time series
    fig, axs = plt.subplots(4, figsize=(16,8))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(observed)
    axs[0].grid() # Add a grid on the subplot
    
    #plot the trend of the time series
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(trend)
    axs[1].grid()
    
    #plot the seasonality of the time series. Period=24 daily seasonality | Period=24*7 weekly seasonality.
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(seasonal)
    axs[2].grid()
    
    #plot the noise of the time series
    axs[3].set_title('NOISE', fontsize=16)
    axs[3].plot(residual)
    axs[3].scatter(y=residual, x=range(len(residual)), alpha=0.5)
    axs[3].grid()
    
    plt.show()
```

Apply function
```
decompose_serie(df, "count", "additive", samples=1000, period=24)
```

## Time series forecasting

1. Convert timestamp to datetime and set it as index
2. Drop datetime column and some feature engineering (Basic: create month,day, hour columns. Advanced: create lags/shifts to use previous data --> ```df['count_prev_week_same_hour'] = df['count'].shift(24*7))```
4. Forecast using LBGM for example and show features importance

```
# Horizon is the time window we want to forecast: here predictions for the next week
def train_time_series(df, target, horizon=24*7): 
    X = df.drop(target, axis=1)
    y = df[target]
    
    #take last week of the dataset for validation
    X_train, X_valid = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
    y_train, y_valid = y.iloc[:-horizon], y.iloc[-horizon:]
    
    #create, train and do inference of the model
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    
    #calculate MAE
    mae = np.round(mean_absolute_error(y_valid, predictions), 3)    
    
    #plot reality vs prediction for the last week of the dataset
    fig = plt.figure(figsize=(16,8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_valid, color='red')
    plt.plot(pd.Series(predictions, index=y_valid.index), color='green')
    plt.xlabel('Hour', fontsize=16)
    plt.ylabel('Number of Shared Bikes', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()
    
    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()
```
