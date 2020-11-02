import pandas as pd 
from sklearn.model_selection import train_test_split

nyc = pd.read_csv('ave_hi_nyc_jan_1895-2018.csv')
print(nyc.head(3))

## gives all dates as one dimension
print(nyc.Date.values)

## reshapes 1 dimensional array into 2D array, -1 means number of rows as there are elements
print(nyc.Date.values.reshape(-1,1)) 

## random state so we all get the same series of numbers while in class | y = targets, x = data
## x_train = training raw data          x_test= used for predicting
## y_train =                            y_train = target, for checking if correct
x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X=x_train, y=y_train)

'''
print(lr.coef_)
print(lr.intercept_)
'''
predicted = lr.predict(x_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]): ## :: checks every 5th element in array
    print(f"predicted: {p:.2f}, expected: {e:.2f}")


# lambda implements y = mx + b
predict = (lambda x: lr.coef_ * x + lr.intercept_)
print(predict(2020))
print(predict(1890))
print(predict(2021))


## creating the scatterplot 
import seaborn as sns
axes = sns.scatterplot(
    data = nyc,
    x = "Date",
    y = "Temperature",
    hue = "Temperature",
    palette = "winter",
    legend = False
)
axes.set_ylim(10,70) # scale y_axis


## creating the regression line 
import numpy as np
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt 
line = plt.plot(x,y)
plt.show()
