import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os 

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer Prognostic")

df = pd.read_csv("prognostic.csv", index_col=0,na_values="?")
dum_df = pd.get_dummies(df, drop_first=True)

imp_mean = IterativeImputer(random_state=2022)

imputed = imp_mean.fit_transform(dum_df)

pd_imputed = pd.DataFrame(imputed,
                          index=dum_df.index,
                          columns=dum_df.columns)