import pandas as pd
import numpy as np
import numpy.linalg as npl
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


print("==================================================")
print("Blockchain for the Management of Internet of Things Devices   ")
print(" Medical Industry")
print("==================================================")


##1.data slection---------------------------------------------------
#def main():
df=pd.read_csv("data.csv")

print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(df.head(10))
print("----------------------------------------------")
print()


 #2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(df.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=df.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
 

#label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
number = LabelEncoder()


df['Time Stamp'] = number.fit_transform(df['Time Stamp'].astype(str))
df['Status'] = number.fit_transform(df['Status'].astype(str))
df['Prog Time'] = number.fit_transform(df['Prog Time'].astype(str))
df['Step Time'] = number.fit_transform(df['Step Time'].astype(str))
df['Procedure'] = number.fit_transform(df['Procedure'].astype(str))

print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print()


X=df
y=df['Status']
# x1=df['Time Stamp']
# y1=df['Voltage']

# f = plt.figure(figsize=(14,6))
# plt.plot(df.groupby(['Status'])['Time Stamp'].mean(), linewidth=2)
# plt.title()
# plt.plot(df.groupby(['Voltage'])['Time Stamp'].mean(), linewidth=1)
# plt.plot(df.groupby(['Current'])['Time Stamp'].mean(), linewidth=2)
# plt.plot(df.groupby(['Temperature'])['Time Stamp'].mean(), linewidth=2)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)
#----------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

model = RandomForestRegressor()
model.fit(x_train,y_train)
predicted = model.predict(x_test)
MAE = mean_absolute_error(y_test , predicted)
print('Random forest validation MAE = ', MAE)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test , predicted))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test , predicted))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test , predicted)))
from scipy.stats import norm, kurtosis
print('kurtosis:',kurtosis(predicted))
import numpy
x = numpy.std(predicted)
print('STD DEVIATION:',x)

import matplotlib.pyplot as plt
"plotting"
true_valueXGB =  y_test
predicted = predicted
plt.figure(figsize=(10,10))
plt.scatter(true_valueXGB, predicted, c='crimson')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('Randomforest  ')
plt.axis('equal')
plt.show()
#---------------------------------------------------------------------------------------------


from xgboost import XGBRegressor,XGBModel
XGBModel1 = XGBRegressor()
XGBModel1.fit(x_train,y_train)
XGBpredictions = XGBModel1.predict(x_test)
RF = mean_absolute_error(y_test , XGBpredictions)
print('XGBoost validation RF = ',RF)
#print((XGBpredictions))
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test , XGBpredictions))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test , XGBpredictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test , XGBpredictions)))
print('kurtosis:',kurtosis(XGBpredictions))
x = numpy.std(XGBpredictions)
print('STD DEVIATION:',x)
import matplotlib.pyplot as plt
"plotting"
true_valueXGB =  y_test
XGBpredictions = XGBpredictions
plt.figure(figsize=(10,10))
plt.scatter(true_valueXGB, XGBpredictions, c='crimson')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('XGBpredictions ')
plt.axis('equal')
plt.show()
#--------------------------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test , y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test , y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test , y_pred)))

print('kurtosis:',kurtosis(y_pred))
x = numpy.std(y_pred)
print('STD DEVIATION:',x)

true_valueXGB =  y_test
y_pred = y_pred
plt.figure(figsize=(10,10))
plt.scatter(true_valueXGB, y_pred, c='crimson')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('Decision Tree  ')
plt.axis('equal')
plt.show()
#----------------------------------------------------------------------------------------------
"CNN Algorithm "
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

model6 = Sequential()
model6.add(Dense(32, activation = 'relu', input_dim = 14))
model6.add(Dense(units = 32, activation = 'relu'))
model6.add(Dense(units = 32, activation = 'relu'))
model6.add(Dense(units = 1))
model6.compile(optimizer = "adam",loss = 'mean_squared_error')
history=model6.fit(x_train, y_train, batch_size = 5, epochs = 100) 
y_pred = model6.predict(x_train)

"Plotting "
true_value =  y_train.values
predicted_value =y_pred
plt.figure(figsize=(10,10))
plt.scatter(true_value, predicted_value, c='crimson')
p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('CNN Regression',fontsize=20)
plt.axis('equal')
plt.show()
#-----------------------------------------------------------------------
true_value =  y_train.values
predicted_value =y_pred
plt.figure(figsize=(10,10))
plt.scatter(true_value, predicted_value, c='crimson')
p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('SOC ',fontsize=20)
plt.axis('equal')
plt.show()

true_value =  y_train.values
predicted_value =y_pred
plt.figure(figsize=(10,10))
plt.scatter(true_value, predicted_value, c='crimson')
p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.title('SOC',fontsize=20)
plt.axis('equal')
plt.show()