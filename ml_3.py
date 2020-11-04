from sklearn.datasets import fetch_california_housing
cali = fetch_california_housing()

'''
print(cali.DESCR)
print(cali.data.shape)
print(cali.target.shape)
print(cali.feature_names)
'''

import pandas as pd
pd.set_option("precision", 4) ## 4 digit percision for floats
pd.set_option("max_columns", 9) ## diplay up to 9 col in DF outputs
pd.set_option("display.width", None) ## auto detect display width for wrapping

cali_df = pd.DataFrame(cali.data, columns = cali.feature_names)
cali_df["MedHouseValue"] = pd.Series(cali.target)
print(cali_df.head())

sample_df = cali_df.sample(frac=0.1, random_state=17) ## frac means using 10% of our random data


import matplotlib.pyplot as plt 
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid")

'''
for ft in cali.feature_names:
    plt.figure(figsize=(8,4.5))
    sns.scatterplot(
        data = sample_df,
        x = ft,
        y = "MedHouseValue",
        hue = "MedHouseValue",
        palette = "cool",
        legend = False,

    )
plt.show()
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(cali.data,cali.target,random_state=11)
## train_data, test_data, train_target, test_target

'''
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
'''

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X=x_train, y=y_train)

'''
for i, name in enumerate(cali.feature_names):
    print(f"{name:>10}: {lr.coef_[i]}")
'''

predicted = lr.predict(x_test)
#print(predicted[:5])
expected = y_test
#print(expected[:5])
print(f"predicted: {predicted[::5]} expected: {expected[::5]}")



df = pd.DataFrame()
df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

import matplotlib.pyplot as plt2
figure = plt2.figure(figsize=(9,9))
axes = sns.scatterplot(
    data = df,
    x="Expected",
    y="Predicted",
    hue="Predicted",
    palette="cool",
    legend=False
    )

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
print(start, end)
axes.set_xlim(start, end) 
axes.set_ylim(start, end)

line2 = plt2.plot([start,end], [start,end], "k--") ## k-- = [ k = black, -- = dotted line ]
plt2.show()
