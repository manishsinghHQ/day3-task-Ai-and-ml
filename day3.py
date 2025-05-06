import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
dataset=pd.read_csv("/content/housing.csv")
is_null=dataset.isnull().sum()
print(is_null)
dup=dataset.duplicated().sum()
print(dup)
for i in dataset.select_dtypes(include='object').columns:
     print(dataset[i].value_counts())
     print("***"*10)

x=dataset[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y=dataset['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=100)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
accuracy=r2_score(y_true=y_test,y_pred=prediction)
accuracy1=mean_absolute_error(y_true=y_test,y_pred=prediction)
accuracy2=mean_squared_error(y_true=y_test,y_pred=prediction)
print(f"r2_scor{accuracy*100}")
print(f"mean_absolute_error{accuracy1}")
print(f"mean_squared_error{accuracy2}")

for i in['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']:
     sns.histplot(data=dataset,x=i,y='Price')
     plt.show()

sns.histplot(x=y_test,y=prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual price vs predicted price')





