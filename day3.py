import seaborn as sns
import matplotlib.pyplot
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
dataset=pd.read_csv("housing.csv")
x=dataset['Avg. Area Income  ','Avg. Area House Age  ','Avg. Area Number of Rooms  ','Avg. Area Number of Bedrooms ','Area Population ']
y=dataset[' Price ']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=100)
model=LinearRegression(random_state=100)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
accuracy=accuracy_score(y_true=y_test,y_pred=prediction)
print(accuracy)