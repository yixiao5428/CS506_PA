import pandas as pd
df = pd.read_csv("/Users/tony/project_506/temperature_production.csv")
import numpy as np
df = df.replace(-9999,np.nan)
df = df.dropna()
df['lastyear']=np.nan
for i in range(df.shape[0]):
    if i==0:
        continue
    if df.iloc[i-1,0]==df.iloc[i,0] and df.iloc[i-1,1] == df.iloc[i,1] and df.iloc[i-1,3]==df.iloc[i,3]-1:
        df.iloc[i,-1]= df.iloc[i-1,18]
df = df.dropna()
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

df1 = df.iloc[:7000,1:13].join(df.iloc[:7000,-1]).values
# print(df1)
df2 = df.iloc[:7000,15].values
# print(df2)
df3 = df.iloc[7000:,1:13].join(df.iloc[7000:,-1]).values
df4 = df.iloc[7000:,15].values
regressor = MLPRegressor()
# print(df1.shape)
regressor.fit(df1,df2)
prediction = regressor.predict(df3)
print(regressor.score(df3,df4))

plt.plot(range(len(df4)), prediction, 'ro')
plt.plot(range(len(df4)), df4, 'bo')
plt.show()
