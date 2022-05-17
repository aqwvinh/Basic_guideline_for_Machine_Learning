# Data visualization 
```
# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import gmaps (conda install -c conda-forge gmaps)
```

# 1. Lineplot
```
# Set the width and height of the figure
plt.figure(figsize=(14,6))
#Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)
```

# 2. Bar chart
Fast plot using value_counts()
```
# Visualize the wine quality distribution
ax = wine.quality.value_counts().sort_index().plot(kind='bar')

# C'est important de toujours mettre des titres et des noms d'axes
ax.set_xlabel("Wine Quality", fontsize=16)
ax.set_ylabel("Number", fontsize=16)
ax.set_title("Nnumber of wine per quality", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)  # Pour augmenter la taille de police des axes

```
```
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
# Choose the frequency of xticks
plt.xticks(np.arange(0, len(x)+1, 3))
# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['target'])
# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
```
Tips : Plot value_counts() by sorting by index and important to add titles !
```
ax = df['beds'].value_counts().sort_index().plot.bar()
ax.set_xlabel("Number of beds", fontsize=16)
ax.set_ylabel("Number of occurences", fontsize=16)
ax.set_title("Number of listings per number of beds", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)  # To augment the font size
```

# 3. Heat map
```
sns.heatmap(data=flight_data, annot=True)
```

# 4. Scatter plots
```
sns.scatterplot(x=df['col_x'], y=df['col_y'])

# Add a linear regression line that best fits the data
sns.regplot(x=df['col_x'], y=df['col_y'])

# Relation with a third binary variable (color the points)
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
```
Alternative for scatter points (nuage de points)
```
sns.jointplot(x='beds', y='bedrooms', data=listings)
```

# 5. Histogram 
```
sns.distplot(a=df['col'], kde=False)

# Histograms for each species --> good to plot histogram for different label values but same feature
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)
# Add title
plt.title("Histogram of Petal Lengths, by Species")
# Force legend to appear
plt.legend()
```

# 6. Bonus from FUll Guideline

## 6.A Gmaps
```
gmaps.configure(api_key='YourAPI')
fig = gmaps.figure()
heatmap_layer_inside = gmaps.heatmap_layer(
  df[['lat','lng']],
  weights=df[criterion]
)
fig.add_layer(heatmap_layer_inside)
fig
```

## 6.B DOUBLE LINE GRAPH
```
ax = df.plot(x = 'month', y= ['Y1', 'Y2'], figsize = (25,10), title = 'yourTitle')
_ = plt.xticks(fontsize = 16, rotation=45) # tip for changing the ticks

# Bar plot horizontal
ax = df.sort_values('revenue', ascending=True).plot.barh(x='quartier', y='revenue', figsize = (15,15), title = 'yourTitle')
# show text on top of bars (vertical) or after bars (horizontal)
bars = ax.patches
# Get labels to show
labels = round(df.revenue,0).astype(int)
for rect, label in zip(bars, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
            ha='center', va='bottom')
  
# Bar plot from df with values as text on the graph  
ax = sns.barplot(x='category', y="price", data=df_plot)
for _, row in to_draw.iterrows(): #Add text on graph
    ax.text(row.name,row.price, round(row.price,2), color='black', ha="center")
ax.set_title('My_Title') # Add title

# COMBINED BAR AND LINE CHART (SEABORN)
fig, ax1 = plt.subplots(figsize=(20,9))
#bar plot creation
color1 = 'lightgray'
ax1.set_title('yourTitle', fontsize=24)
ax1.set_xlabel('X', fontsize=14)
ax1 = sns.barplot(x='X', y='Y', data = df_plot, color = color1)
ax1.set_ylabel('Y', fontsize=14, color = "black")
plt.xticks(rotation=45) # for better reading
#specify we want to share the same x-axis
ax2 = ax1.twinx()
#line plot creation
color2 = "green"
ax2.set_ylabel('Y2', fontsize=14, color = color2 )
ax2 = sns.lineplot(x=range(0,len(df_plot.Y2)), y='Y2', data = df_plot, sort=False, color=color2)
# Show yearly occ_rate
ax3 = ax2.twiny()
# line plot creation
color3 = 'navy'
ax3 = sns.lineplot(x=range(0,len(df_plot.Y3)), y='Y3', data=df_plot, sort=False, color = color3)
# Round values for axis
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#show plot
plt.show()
```

## 6.C PIE CHART
```
labels = df.country_name
sizes = df.ratio
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
```

# 7. Boxplot
Useful to show the distribution and detect outliers using IQR
```
sns.boxplot(x='room_type', y='accommodates', data=df)
# Detect outliers using IQR
Q1 = df['target'].quantile(0.25)
Q3 = df['target'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df['target'] < (Q1 - 1.5 * IQR)  #lower bound
df['target'] > (Q3 + 1.5 * IQR) #upper bound
```

# 8. Line and scatter plot with condition
Useful to show different colors depending on a condition
```
def plot_res(id, date_min, date_max):
    
    df['date'] = pd.to_datetime(df.date)
    df_plot = df[(df.listing_id == id) & (df.date >= date_min) & (df.date < date_max)][['date', 'lead_time', 'label', 'price']]
    df_plot = df_plot.sort_values(['date', 'lead_time', 'price'])
    
    dates = list(df_plot.date.values)
    prices = list(df_plot.price.values)
    
    
    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(dates, prices, lw=0.2, alpha=0.5)
    ax1.scatter(df_plot[df_plot.label == 0].date, df_plot[df_plot.label == 0].price, c='lime', alpha=0.8, s=5)
    ax1.scatter(df_plot[df_plot.label == 1].date, df_plot[df_plot.label == 1].price, c='red', alpha=0.8, s=5)
    
    ax2.plot(dates, df_plot.lead_time, c='black', lw=1, alpha=0.2)
    
    ax1.set_ylabel('price', color='black')
    ax2.set_ylabel('lead time', color='black')
    
    plt.title(f'Price vs Leadtime for listing {id}')
    
    plt.show()
```


## 9. Bonus

Create plot function
```
def plot_prediction(df, id):
    
    df_plot_pred = df[df.id == id].sort_values('month')
    df_plot_pred.month = df_plot_pred.month.apply(lambda x: x[0:7])
    
    dates = list(df_plot_pred.month.values)
    volexe = list(df_plot_pred['target'].values)
    max_vol = df_plot_pred.target.max()
    preds = list(df_plot_pred['prediction'].values)
    dummy = list(df_plot_pred['dummy_val'].values)
    max_preds = df_plot_pred.prediction.max()
    
    df_plot_score = df_plot_pred[(df_plot_pred.month >= "2021-03-01") & (df_plot_pred.month <= "2022-02-01")]
    volexe_score = list(df_plot_score['target'].values)
    preds_score = list(df_plot_score['prediction'].values)
    dummy_score = list(df_plot_score['dummy_val'].values)
    
    def nrmse_mean(ytrue, ypred):
        return np.sqrt(mean_squared_error(ytrue, ypred))/(np.mean(ytrue))
    
    def r2_function(ytrue, ypred):
        return r2_score(ytrue, ypred)
    
    rmse_score = nrmse_mean(volexe_score, preds_score)
    rmse_dummy = nrmse_mean(volexe_score, dummy_score)
    r2_metric = r2_function(volexe_score, preds_score)
    r2_dummy = r2_function(volexe_score, dummy_score)

    fig, ax = plt.subplots(figsize=(15, 2.8))
    line1, = ax.plot(dates, volexe, color = "black", lw=8, alpha=0.6)
    line2, = ax.plot(dates, preds, color = "red", lw=8, alpha=0.6)
    line3, = ax.plot(dates, dummy, color = "lightblue", lw=8, alpha=0.6)
    try:
        plt.ylim([-10, max(max_vol, max_preds)*1.1])
    except:
        plt.ylim([-10, 1000])
    #ax.set_ylabel('Target')
    #ax.set_xlabel('Date')
    plt.yticks(fontsize= 15)
    plt.xticks(fontsize= 15)
    plt.xticks(rotation = 10)
    loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    plt.title(f"Predictions versus target: {fdm}", fontsize=15)
    ax.legend([line1, line2, line3], ['Observed Target', 'Prediction with advanced model', 'Prediction with dummy'], fontsize=15, loc='best')# handleheight=3, handlelength=6, markerscale=0.8)
    print("RMSE and R2 for advanced and dummy models:", rmse_score, r2_metric, rmse_dummy, r2_dummy)
    plt.show()
```


Plot features importance
```
def plot_features_importance(cols_train, model):
    # settings
    plt.style.use('seaborn')
    plt.rcParams["figure.figsize"] = (4, 4)

    # Show feature importance
    l_train_features = cols_train
    feat_names = []
    feat_importance = []
    for i in model.feature_importances_.argsort()[-8:]:
        feat_names.append(l_train_features[i])
        feat_importance.append(model.feature_importances_[i])

    fig, ax = plt.subplots(figsize=(10, 6))

    plt.barh(feat_names, feat_importance)
    ax.set_xlabel('Coeff Importance', fontsize=20)
    plt.yticks(fontsize= 20)
    plt.xticks(fontsize= 14)
    plt.show()
```
