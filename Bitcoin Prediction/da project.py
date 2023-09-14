#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[ ]:


#data information
df = pd.read_csv('bitcoin.csv')
print(df.head())
print(df.shape) 
print(df.info())
print(df.describe())


# In[ ]:


#data cleaning
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)


# In[ ]:


#Visualizing Bitcoin closing prices over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'])
plt.title('Bitcoin Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Building a model to predict Bitcoin prices

#data modeling
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Month', 'Year']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# In[ ]:


# Function to predict Bitcoin prices
def predict_price(open_price, high_price, low_price, adj_close, volume, month, year):
    price_data = np.array([[open_price, high_price, low_price, adj_close, volume, month, year]])
    price_predicted = model.predict(price_data)
    return price_predicted[0]

open_price = float(input("Enter the open price: "))
high_price = float(input("Enter the high price: "))
low_price = float(input("Enter the low price: "))
adj_close = float(input("Enter the adjusted close price: "))
volume = float(input("Enter the volume: "))
month = int(input("Enter the month: "))
year = int(input("Enter the year: "))

price_predicted = predict_price(open_price, high_price, low_price, adj_close, volume, month, year)
print("Predicted Bitcoin price:", price_predicted)


# In[ ]:





# In[ ]:




