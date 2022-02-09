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

Level/Observed: The average value in the series.
<br>Trend: The increasing or decreasing value in the series. Trend is the moving average in the period.
<br>Seasonality: The repeating short-term cycle in the series. Seasonality+Noise = observed/trend for multiplicative model or = observed-trend for additive and then average to retrieve the pure seasonality
<br>Noise: The random variation in the series.

**A series is thought to be an aggregate or combination of these four components.** That's why there are 2 models: additive and multiplicative.
<br>For multiplicative model, *Time series value = trend component * seasonal component * noise component*


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
3. Make sure the time series is **stationary** (same mean, variance and covariance through times) cuz it would ensure that we'll have the same behaviour in the future
4. Split dataset: train before validation before test cuz it ensures that test data are more realistic as they are collected **after training the model**
5. Forecast using LBGM for example and show features importance

```
# Horizon is the time window we want to forecast: here predictions for the next week
def train_time_series(df, target, horizon=24*7): 
    X = df.drop(target, axis=1)
    y = df[target]
    
    #take last week of the dataset for validation (train set before)
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

### Bonus

Split manually data (70/20/10) cuz we shouldn't split data randomly --> keep training data = oldest

```
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
```


Normalize manually. use train mean and std. Make sure there are only numeric variables for this method
```
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
```

Manually retrieve trend, saisonality and noise (to understand what is behing statsmodel.seasonal_decompose)
```
df = pd.read_csv('retail_sales_used_car_dealers_us_1992_2020.csv')
fig = plt.figure(figsize=(15,5))
# OBSERVED
fig.suptitle("Observed")
df['Retail_Sales'].plot()
plt.show()

# TREND.Compute moving average. We need to shift because the period is 12 months so we compute the average between the average with 6 months before the specific month and the average with 5 months before.
df['Retail_Sales_shift'] =  df['Retail_Sales'].shift(-1)
df['left_ma'] = df['Retail_Sales'].rolling(window=12, center=True).mean()
df['right_ma'] = df['Retail_Sales_shift'].rolling(window=12, center=True).mean()
df['trend'] = (df['left_ma'] + df['right_ma']) / 2
fig = plt.figure(figsize=(15,5))
fig.suptitle("Trend")
df['trend'].plot()
plt.show()

# Saisonality. Compute the average of (saisonality and noise) components per period and divide by the average of all the average (to have a flat season at 1)
df['seasonNnoise'] = df['Retail_Sales']/df['trend']
tmp_mean = df.groupby('month')['seasonNnoise'].mean().mean()
df['season'] = df.groupby('month')['seasonNnoise'].transform("mean")/tmp_mean #Use transform to fill all the row per group with the result

```
