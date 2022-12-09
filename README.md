# Final-stat-code
import pandas as pd
import numpy as np
df=pd.read_csv('/content/Stock market 1234.csv')
print(df.isnull())
df=df.fillna(0)
df.head()
df.info()
df.describe()
df.columns
y=df['No of trades ']
y
X=df[['PREV. CLOSE ','series','vwap ',]]
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
y.shape
y
X.shape
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred.shape
y_pred
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Actual quality:-")
plt.ylabel("Predicted quality:-")
plt.title("Actual quality: vs Predicted quality:")
plt.show
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model = LinearRegression().fit(X, y)
r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")
import pandas as pd
import numpy as np
df=pd.read_csv('/content/Stock market 1234.csv')
print(df.isnull())
df=df.fillna(0)
df.head()
df.info()
df.describe()
df.columns
y=df['No of trades ']
y
X=df[['PREV. CLOSE ','series','vwap ',]]
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
y.shape
y
X.shape
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred.shape
y_pred
from sklearn.linear_model import Ridge 
rr= Ridge(alpha=0.01)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))


import pandas as pd
import numpy as np
df=pd.read_csv('/content/Stock market 1234.csv')
print(df.isnull())
df=df.fillna(0)
df.head()
df.info()
df.describe()
df.columns
y=df['No of trades ']
y
X=df[['PREV. CLOSE ','series','vwap ',]]
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
y.shape
y
X.shape
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred.shape
y_pred
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score 
from sklearn.linear_model import Lasso
model_lasso= Lasso(alpha=0.01)
model_lasso.fit(X_train,y_train)
pred_train_lasso= model_lasso.predict(X_test)
print(mean_squared_error(y_test,pred_train_lasso))
