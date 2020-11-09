from sklearn.datasets import fetch_california_housing
cali = fetch_california_housing()

'''
print(cali.DESCR)
print(cali.data.shape)
print(cali.target.shape)
print(cali.feature_names)
'''

import pandas as pd
pd.set_option("precision", 4) ## 4 digit precision for floats
pd.set_option("max_columns", 9) ## diplay up to 9 col in DF outputs
pd.set_option("display.width", None) ## auto detect display width for wrapping

cali_df = pd.DataFrame(cali.data, columns = cali.feature_names)

## adds column to df for median house value
cali_df["MedHouseValue"] = pd.Series(cali.target)

sample_df = cali_df.sample(frac=0.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid")

for ft in cali.feature_names:
    plt.figrue()
