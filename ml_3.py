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

cali_df = pd.DataFrame(cali.data, columns = cali.feautre_names)