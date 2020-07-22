
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Area.csv')

plt.scatter(df.Area,df.Price,color='red',marker='+')
plt.xlabel('Price')
plt.ylabel('Area')
plt.title('House ploter Graph')

X = np.array(df.Area)
X = X.reshape([-1,1])
y = np.array(df.Price)

reg = LinearRegression()
reg.fit(X,y)
reg.coef_
reg.intercept_

reg.predict([[1800]])
reg.score(X,y)

df1 = pd.read_csv('Area1.csv')
Pred = reg.predict(df1[['Area']])
df1['Price']=Pred
df1.to_csv('Area11.csv',index = False)

plt.scatter(df.Area,df.Price,color='red',marker='+')
plt.xlabel('Price')
plt.ylabel('Area')
plt.title('House ploter Graph')
plt.plot(df.Area,reg.predict(X))



