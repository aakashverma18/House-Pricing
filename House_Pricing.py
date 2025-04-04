#Supervised Learning


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Load the dataset of california housing from sklearn
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
#print(housing.DESCR)
print(housing.feature_names)
# print(housing.target)
# print(housing.data)
#Preparing the data part
dataset = pd.DataFrame(housing.data, columns = housing.feature_names)

#dataset.tail()
dataset['Price'] = housing.target
dataset.head()
dataset.info()
dataset.describe() #to describe the dataset
dataset.isnull().sum()


#EDA
# print(dataset.corr())
#Average Bedrooms and Average Rooms have high +ve correlation (0.847621) and latittude and longitude have high -ve correlation (-0.924664 ).
# sns.pairplot(dataset)
#To detect the outliers, we use Boxplot
# fig, ax = plt.subplots(figsize=(15,15))
# sns.boxplot(data = dataset, ax=ax)
# plt.savefig('boxplot.jpg')

#To manage the outliers, we standardize/normalize., we convert into standard normal mean=0, std=1.
##split the data into independent and dependent features.
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
## Split the data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state =42)

## Normalization of given data points
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_norm = scaler.fit_transform(x_train)

x_test_norm = scaler.transform(x_test)
##Train set -> fit_transform(X_train)
## Test set -> transform(X_test)
##Why usually this happens?
## during training we fit by computing mean and std-dev, then transform using z-score whilst during test_set we already have the value of
## mu and sigma so we just transform in the test set.

#Model Training
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train_norm, y_train)

print(regression.coef_) #m=slope values
#Indpendent features are eight so y = m1x1 + m2x2 + m3x3 + m4x4 + m5x5 + m6x6 + m7x7 + m8x8 + c.
print(regression.intercept_)

#Model Prediction
reg_pred = regression.predict(x_test_norm)
reg_pred

##Calculate the residuals
residuals = y_test - reg_pred
residuals
## distribution plots of the residuals
sns.displot(residuals, kind='kde')

#Model Evaluation
##Model Performance using MSE and MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MSE:', mean_squared_error(y_test, reg_pred))
print('MAE:', mean_absolute_error(y_test, reg_pred))
r2 = r2_score(y_test, reg_pred)
print('R2:', r2)
adjusted_r2 = 1 - (1-r2) * ( len(y_test) -1) / (len(y_test)- x_test_norm.shape[1] -1)
print('adjusted_r2:', adjusted_r2)
##we are looking for lower error value - MSE and MAE and higher R2 score and adjusted R2 score.
x_test_norm.shape[1] #no of independent features
#RMSE
print('RMSE:', np.sqrt(mean_squared_error(y_test, reg_pred)))

#Save the model -> Pickle file.
import pickle
pickle.dump(regression, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb')) #opening a pickle file : it contains the coefficients/ intercepts i.e. m1, m2, m3... and c.
model.predict(x_test_norm)