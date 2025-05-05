import pandas as pd
print(pd.__version__)
import tensorflow as tf
print(tf.__version__)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os 

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer Prognostic")

df = pd.read_csv("prognostic.csv", index_col=0,na_values="?")
dum_df = pd.get_dummies(df, drop_first=True)

imp_mean = IterativeImputer(random_state=2022, 
                            initial_strategy='median')

imputed = imp_mean.fit_transform(dum_df)

pd_imputed = pd.DataFrame(imputed,
                          index=dum_df.index,
                          columns=dum_df.columns)

X = pd_imputed.drop(['Time','Outcome_R'], axis=1)
y = pd_imputed[['Time','Outcome_R']]

