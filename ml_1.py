from sklearn.datasets import load_digits
digits = load_digits()

## visualizing the data
import matplotlib.pyplot as plt 
figure, axes = plt.subplots(nrows=4,ncols=6,figsize=(6,4))
for item in zip(axes.ravel(),digits.images,digits.target):
    axes, image, target =item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()

plt.show()


## training the machine
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X=x_train, y=y_train)

predicted = knn.predict(X=x_test)
expected = y_test

print(predicted[:20])
print(expected[:20])

wrong = [(p,e) for (p,e) in zip(predicted,expected) if p != e]
print(wrong)

print(format(knn.score(x_test,y_test), ".2%"))


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_true=expected, y_pred=predicted)
print(cf)

## visualizing the results
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt2

cf_df = pd.DataFrame(cf,index=range(10), columns=range(10))

fig = plt2.figure(figsize=(7,6))
axes = sns.heatmap(cf_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.show()