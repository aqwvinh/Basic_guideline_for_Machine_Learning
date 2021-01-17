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
```
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['target'])
# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
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
