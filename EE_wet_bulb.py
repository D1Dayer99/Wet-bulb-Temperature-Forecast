#importing the libraries
import pandas as pd #loads csv file and creates a pandas data frame for data processing
import matplotlib.pyplot as plt #visualizing data
import numpy as np #linear algebra
import time
import datetime
from statsmodels.tsa.stattools import adfuller #Augmented Dickey-Fuller Test
from sklearn.ensemble import RandomForestRegressor #Random Forest Regression
from sklearn.model_selection import RandomizedSearchCV #Cross Validation for Random Forest
from sklearn.preprocessing import StandardScaler #Scaling Data
from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.linear_model import LassoCV #Lasso Cross Validation
from sklearn.linear_model import Lasso #Lasso Regression
from sklearn.linear_model import RidgeCV #Ridge Cross Validation
from sklearn.linear_model import Ridge #Ridge Regression
import math
from sklearn import metrics #Evaluation Methods

data = pd.read_csv("dubai_climate_data_1h.csv") #creates a pandas DataFrame from the CSV file
print(data.info()) #outputs information about the DataFrame
print(data.head(20))

def dataset_filter(data):
    #Changing the data type of date field to datatime and setting it as the index
    data['date'] = pd.to_datetime(data['date']) 
    data.index=data['date']
    del data['date']

    data['time'] = data['time'].div(100) #Dividing each row in time by 100 to correspond to the hour of the day
    #dropping unnecessary columns
    data = data.drop(['loc_id','isdaytime','tempF','windspeedMiles','winddirdegree','winddir16point','weatherCode','weatherIconUrl','weatherDesc','precipInches','visibilityKm','visibilityMiles','pressureInches','HeatIndexC','HeatIndexF','DewPointC','DewPointF','WindChillC','WindChillF','WindGustMiles','WindGustKmph','FeelsLikeC','FeelsLikeF','uvIndex','windspeedKmph','precipMM','pressureMB','cloudcover'],axis=1)
    return data

data = dataset_filter(data)
print(data.head(48)) #Outputting the new dataset (date,time,temperature, humidity)

# #Checking for anomaly 
# def anomaly_check(data):
#     print(data.isnull().sum()) 
#     print(data.describe()) #outputs the mean, different percentiles and the maximum value of each field
#     for column in data:
#         plt.figure()
#         data.boxplot([column])
#         plt.show()

# anomaly_check(data)

#Splitting the multivariate data set into two data sets: Temp and Humidity
def data_split(data):
    temp_data = data.filter(['tempC','time'],axis=1)
    humid_data = data.filter(['humidity','time'],axis=1)
    return temp_data,humid_data
        
temp_data, humid_data = data_split(data)

#Augmented Dickey-Fuller Test
# def ADF_test(data_column):
#     result_temp = adfuller(data_column)
#     print(f'ADF Statistic : {result_temp[0]}')
#     print(f'p-value: {result_temp[1]}')
#     print('Critical Values: ')
#     for key, value in result_temp[4].items():
#         print(f'\t{key}: {value}')

# ADF_test(temp_data.tempC)
# ADF_test(humid_data.humidity)

def create_features(data,column):
    #Time Lag - Serial Dependence
    data['Lag_1'] = data[column].shift(1)
    data['Lag_2'] = data[column].shift(2)
    data['Lag_3'] = data[column].shift(3)
    data = data.reindex(columns=[column,'time','Lag_1','Lag_2','Lag_3'])
    data = data.iloc[3:,:]

    #Adding a new column which corresponds to seasons, hour of the day and day of year
    spring = [3,4,5]
    summer = [6,7,8]
    autumn = [9,10,11]
    winter = [12,1,2]
    data['Spring'] = data.index.get_level_values(0).month.isin(spring).astype(int)
    data['Summer'] = data.index.get_level_values(0).month.isin(summer).astype(int)
    data['Autumn'] = data.index.get_level_values(0).month.isin(autumn).astype(int)
    data['Winter'] = data.index.get_level_values(0).month.isin(winter).astype(int)
    data['dayofyear'] = pd.DatetimeIndex(data.index).dayofyear
    return data

temp_data = create_features(temp_data, "tempC")
humid_data = create_features(humid_data,"humidity")
print(temp_data.head(10))

# temp_data.plot(y='tempC', kind='line',figsize=(15,10))
# plt.ylabel("Temperature")
# plt.show()

# humid_data.plot(y='humidity', kind='line',figsize=(15,10))
# plt.ylabel("Humidity")
# plt.show()

print(temp_data.shape)

#Random Forest
def RF(X_train,X_test, y_train,n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth):
    rf_regress = RandomForestRegressor(n_estimators = n_estimators, 
                                        min_samples_split = min_samples_split, 
                                        min_samples_leaf = min_samples_leaf, 
                                        max_features = max_features, 
                                        max_depth = max_depth)

    start = time.time() #Time Complexity
    rf_regress.fit(X_train, y_train)
    y_pred_rf = rf_regress.predict(X_test)

    end = time.time()
    rf_time = end-start
    return y_pred_rf, rf_time

#Random Forest CrossValidation
def find_opt(X_train,y_train):
    n_estimators = [100,300,500]
    max_features = ['auto','sqrt','log2']
    max_depth = [2,4,6]
    min_samples_split = [2,4,6]
    min_samples_leaf = [2,4,6]

    hyperparameters = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                        }
    rfr = RandomForestRegressor(random_state = 42)
    #RandomizedSearch CV is used instead of Grid Search due to technological limitations
    rrfr = RandomizedSearchCV(estimator=rfr, param_distributions=hyperparameters, n_iter=200,random_state = 42, n_jobs = -1,verbose =10)

    rrfr.fit(X_train,y_train)
    print(rrfr.best_params_)

#Standard Scaling for Linear Regression and variants
def standard_scale(X_train,X_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train,X_test

def LR(X_train,X_test, y_train): #Linear Regression

    lr = LinearRegression()
    X_train,X_test = standard_scale(X_train,X_test)
    
    start = time.time() 
    lr.fit(X_train,y_train)

    y_pred_lr = lr.predict(X_test)
    end = time.time()

    lr_time = end-start
    return y_pred_lr, lr_time

#Lasso Cross Validation
def lasso_find_opt(X_train,y_train):
    model = LassoCV(cv=5, random_state=42,max_iter=5000)
    model.fit(X_train, y_train)

    return model.alpha_

def L1reg(X_train,X_test,y_train,alpha): #Lasso Regression

    lasso = Lasso(alpha=alpha)
    X_train,X_test = standard_scale(X_train,X_test)

    start = time.time()
    lasso.fit(X_train,y_train)

    y_pred_lasso = lasso.predict(X_test)
    end=time.time()

    lasso_time=end-start
    return y_pred_lasso,lasso_time

#Ridge Cross Validation
def ridge_find_opt(X_train,y_train,alphas):
    model = RidgeCV(cv=5,alphas=alphas)
    model.fit(X_train, y_train)

    return model.alpha_

#Ridge Regression
def L2reg(X_train,X_test,y_train,alpha):

    ridge = Ridge(alpha=alpha)
    X_train,X_test = standard_scale(X_train,X_test)

    start = time.time()
    ridge.fit(X_train,y_train)

    y_pred_ridge = ridge.predict(X_test)
    end=time.time()

    ridge_time = end-start

    return y_pred_ridge, ridge_time

def evaluate_ml(y_test,y_pred_rf,y_pred_lr, y_pred_lasso,y_pred_ridge,rf_time,lr_time,lasso_time,ridge_time):
    #MAE
    print("\nMAE:\n")
    print("RF Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred_rf), 4))
    print("LR Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred_lr), 4))
    print("Lasso Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred_lasso), 4))
    print("Ridge Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred_ridge), 4))

    #RMSE
    print("\nRMSE:\n")
    print("RF RMSE:", round(math.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)), 4))
    print("LR RMSE:", round(math.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)), 4))
    print("Lasso RMSE:", round(math.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)), 4))
    print("Ridge RMSE:", round(math.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge)), 4))

    #Time
    print("\nTime Complexity:\n")
    print(f'RF Time: {round(rf_time,3)} seconds')
    print(f'LR Time: {round(lr_time,3)} seconds')
    print(f'Lasso Time: {round(lasso_time,3)} seconds')
    print(f'Ridge Time: {round(ridge_time,3)} seconds')

def train_test_split(data, column):
    data0 = data.reset_index(drop=True)

    temp_train, temp_test= np.split(data0, [int(.60 *len(data0))]) #splitting the dataframe into 60% for training and 40% for testing, which corresponds to 6 years and 4 years
    X_train = temp_train.loc[:, temp_train.columns != column]#all columns except temperature/humidity
    X_test = temp_test.loc[:,temp_test.columns != column]
    y_train = temp_train[column]#temperature/humidity
    y_test = temp_test[column]

    return X_train, X_test, y_train, y_test

X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(temp_data, 'tempC')
X_train_humid, X_test_humid, y_train_humid, y_test_humid= train_test_split(humid_data, 'humidity')

#Finding the optimal parameters for lasso and ridge regression
def find_opt_alpha(X_train_temp, y_train_temp, X_train_humid, y_train_humid):
    alphas = np.logspace(3,-4,20)
    lasso_temp_alpha = lasso_find_opt(X_train_temp,y_train_temp)
    lasso_humid_alpha = lasso_find_opt(X_train_humid,y_train_humid)
    ridge_temp_alpha = ridge_find_opt(X_train_temp,y_train_temp,alphas)
    ridge_humid_alpha = ridge_find_opt(X_train_humid,y_train_humid,alphas)
    return lasso_temp_alpha, lasso_humid_alpha, ridge_temp_alpha, ridge_humid_alpha

lasso_temp_alpha, lasso_humid_alpha, ridge_temp_alpha, ridge_humid_alpha = find_opt_alpha(X_train_temp, y_train_temp, X_train_humid, y_train_humid)

# find_opt(X_train_temp,y_train_temp)
# {'n_estimators': 500, 'min_samples_split': 4, 'min_samples_leaf': 6, 'max_features': 'auto', 'max_depth': 6}

# find_opt(X_train_humid,y_train_humid)
# {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 6, 'max_features': 'auto', 'max_depth': 6}

#Fitting the algorithms on both data sets
y_pred_rf_humid, rf_time_humid = RF(X_train_humid, X_test_humid, y_train_humid, 300, 2, 6, 'auto', 6)
y_pred_lr_humid, lr_time_humid = LR(X_train_humid, X_test_humid, y_train_humid)
y_pred_lasso_humid, lasso_time_humid = L1reg(X_train_humid, X_test_humid, y_train_humid, lasso_humid_alpha)
y_pred_ridge_humid, ridge_time_humid = L2reg(X_train_humid, X_test_humid, y_train_humid, ridge_humid_alpha)

evaluate_ml(y_test_humid,y_pred_rf_humid,y_pred_lr_humid, y_pred_lasso_humid,y_pred_ridge_humid,rf_time_humid,lr_time_humid,lasso_time_humid,ridge_time_humid)

y_pred_rf_temp, rf_time_temp = RF(X_train_temp, X_test_temp, y_train_temp, 500, 4, 6, 'auto', 6)
y_pred_lr_temp, lr_time_temp = LR(X_train_temp, X_test_temp, y_train_temp)
y_pred_lasso_temp, lasso_time_temp = L1reg(X_train_temp, X_test_temp, y_train_temp, lasso_temp_alpha)
y_pred_ridge_temp, ridge_time_temp = L2reg(X_train_temp, X_test_temp, y_train_temp, ridge_temp_alpha)

evaluate_ml(y_test_temp,y_pred_rf_temp,y_pred_lr_temp, y_pred_lasso_temp,y_pred_ridge_temp,rf_time_temp,lr_time_temp,lasso_time_temp,ridge_time_temp)

#turning these data to numpy array in order to create a pd Dataframe
y_test_temp = y_test_temp.to_numpy()
y_test_humid = y_test_humid.to_numpy()

#Creating a pd Dataframe out of the predictions
def create_dataframes(y_pred_rf_temp,y_pred_rf_humid,y_pred_lr_temp,y_pred_lr_humid,y_pred_lasso_temp,y_pred_lasso_humid, y_pred_ridge_temp, y_pred_ridge_humid,y_test_temp,y_test_humid):
    rf_data = pd.DataFrame({'RF_tempC':y_pred_rf_temp,'RF_humidity':y_pred_rf_humid},columns = ['RF_tempC','RF_humidity'])
    lr_data = pd.DataFrame({'LR_tempC':y_pred_lr_temp,'LR_humidity':y_pred_lr_humid},columns = ['LR_tempC','LR_humidity'])
    lasso_data = pd.DataFrame({'Lasso_tempC':y_pred_lasso_temp,'Lasso_humidity':y_pred_lasso_humid},columns = ['Lasso_tempC','Lasso_humidity'])
    ridge_data = pd.DataFrame({'Ridge_tempC':y_pred_ridge_temp,'Ridge_humidity':y_pred_ridge_humid},columns = ['Ridge_tempC','Ridge_humidity'])
    actual_data = pd.DataFrame({'Real_tempC':y_test_temp,'Real_humidity':y_test_humid},columns = ['Real_tempC','Real_humidity'])
    return rf_data, lr_data, lasso_data, ridge_data, actual_data

rf_data, lr_data, lasso_data, ridge_data, actual_data = create_dataframes(y_pred_rf_temp,y_pred_rf_humid,y_pred_lr_temp,y_pred_lr_humid,y_pred_lasso_temp,y_pred_lasso_humid, y_pred_ridge_temp, y_pred_ridge_humid,y_test_temp,y_test_humid)

print(lasso_data.head(10))

#For every dataset, find the wet bulb temperature of each row using the formula and appending to the corresponding array
def find_wet_bulb(rf_data,lr_data, lasso_data, ridge_data, actual_data):
    rf_wet_bulb = []
    lr_wet_bulb = []
    lasso_wet_bulb = []
    ridge_wet_bulb = []
    real_wet_bulb = []
    rf_normal= []
    lr_normal = []
    lasso_normal = []
    ridge_normal = []
    real_normal = []
    for index,row in rf_data.iterrows():
        rf_wb_temp = row["RF_tempC"]*math.atan(0.151977*((row["RF_humidity"]+8.313659)**0.5))+math.atan(row["RF_tempC"]+row["RF_humidity"])-math.atan(row["RF_humidity"]-1.676331)+0.00391838*(row["RF_humidity"]**1.5)*math.atan(0.023101*row["RF_humidity"])-4.686035
        if rf_wb_temp >=28:
            rf_wet_bulb.append(index)
        else:
            rf_normal.append(index)
    for index,row in lr_data.iterrows():
        lr_wb_temp = row["LR_tempC"]*math.atan(0.151977*((row["LR_humidity"]+8.313659)**0.5))+math.atan(row["LR_tempC"]+row["LR_humidity"])-math.atan(row["LR_humidity"]-1.676331)+0.00391838*(row["LR_humidity"]**1.5)*math.atan(0.023101*row["LR_humidity"])-4.686035
        if lr_wb_temp >=28:
            lr_wet_bulb.append(index)
        else:
            lr_normal.append(index)
    for index,row in lasso_data.iterrows():
        lasso_wb_temp = row["Lasso_tempC"]*math.atan(0.151977*((row["Lasso_humidity"]+8.313659)**0.5))+math.atan(row["Lasso_tempC"]+row["Lasso_humidity"])-math.atan(row["Lasso_humidity"]-1.676331)+0.00391838*(row["Lasso_humidity"]**1.5)*math.atan(0.023101*row["Lasso_humidity"])-4.686035
        if lasso_wb_temp >=28:
            lasso_wet_bulb.append(index)
        else:
            lasso_normal.append(index)    
    for index,row in ridge_data.iterrows():
        ridge_wb_temp = row["Ridge_tempC"]*math.atan(0.151977*((row["Ridge_humidity"]+8.313659)**0.5))+math.atan(row["Ridge_tempC"]+row["Ridge_humidity"])-math.atan(row["Ridge_humidity"]-1.676331)+0.00391838*(row["Ridge_humidity"]**1.5)*math.atan(0.023101*row["Ridge_humidity"])-4.686035
        if ridge_wb_temp>=28:
            ridge_wet_bulb.append(index)
        else:
            ridge_normal.append(index)
    for index,row in actual_data.iterrows():
        real_wb_temp = row["Real_tempC"]*math.atan(0.151977*((row["Real_humidity"]+8.313659)**0.5))+math.atan(row["Real_tempC"]+row["Real_humidity"])-math.atan(row["Real_humidity"]-1.676331)+0.00391838*(row["Real_humidity"]**1.5)*math.atan(0.023101*row["Real_humidity"])-4.686035
        if real_wb_temp>=28:
            real_wet_bulb.append(index)
        else:
            real_normal.append(index)

    return rf_wet_bulb, lr_wet_bulb, lasso_wet_bulb,ridge_wet_bulb, real_wet_bulb,rf_normal,lr_normal,lasso_normal,ridge_normal,real_normal

rf_wet_bulb, lr_wet_bulb, lasso_wet_bulb,ridge_wet_bulb, real_wet_bulb,rf_normal,lr_normal,lasso_normal,ridge_normal,real_normal = find_wet_bulb(rf_data, lr_data, lasso_data, ridge_data, actual_data)

#Finding true/false positives/negatives for evaluation
def t_or_f_positive(data1, data2, data3, data4, real):
    rf_t = 0
    lr_t = 0
    lasso_t = 0
    ridge_t = 0
    rf_f = 0
    lr_f = 0
    lasso_f = 0
    ridge_f = 0

    for ele in data1:
        if ele in real:
            rf_t+=1
        else:
            rf_f+=1
    for ele in data2:
        if ele in real:
            lr_t+=1
        else:
            lr_f+=1
    for ele in data3:
        if ele in real:
            lasso_t+=1
        else:
            lasso_f+=1
    for ele in data4:
        if ele in real:
            ridge_t+=1
        else:
            ridge_f+=1
    
    return rf_t,lr_t,lasso_t,ridge_t,rf_f,lr_f,lasso_f,ridge_f

rf_tcorrect,lr_tcorrect,lasso_tcorrect,ridge_tcorrect,rf_fcorrect,lr_fcorrect,lasso_fcorrect,ridge_fcorrect = t_or_f_positive(rf_wet_bulb, lr_wet_bulb, lasso_wet_bulb,ridge_wet_bulb,real_wet_bulb)
rf_tneg,lr_tneg,lasso_tneg,ridge_tneg,rf_fneg,lr_fneg,lasso_fneg,ridge_fneg = t_or_f_positive(rf_normal, lr_normal, lasso_normal,ridge_normal,real_normal)

def wet_bulb_evaluate(rf_tcorrect,lr_tcorrect,lasso_tcorrect,ridge_tcorrect,rf_fcorrect,lr_fcorrect,lasso_fcorrect,ridge_fcorrect,rf_fneg,lr_fneg,lasso_fneg,ridge_fneg):
    #Precision
    print('\n')
    print(f"RF precision: {(rf_tcorrect/(rf_tcorrect+rf_fcorrect))*100}")
    print(f"LR precision: {(lr_tcorrect/(lr_tcorrect+lr_fcorrect))*100}")
    print(f"Lasso precision: {(lasso_tcorrect/(lasso_tcorrect+lasso_fcorrect))*100}")
    print(f"Ridge precision: {(ridge_tcorrect/(ridge_tcorrect+ridge_fcorrect))*100}")

    #Recall
    print("\n")
    print(f"RF recall: {(rf_tcorrect/(rf_tcorrect+rf_fneg))*100}")
    print(f"LR recall: {(lr_tcorrect/(lr_tcorrect+lr_fneg))*100}")
    print(f"Lasso recall: {(lasso_tcorrect/(lasso_tcorrect+lasso_fneg))*100}")
    print(f"Ridge recall: {(ridge_tcorrect/(ridge_tcorrect+ridge_fneg))*100}")

    #F-measure
    print("\n")
    print(f"RF F-score: {((2*rf_tcorrect)/(2*rf_tcorrect+rf_fcorrect+rf_fneg))*100}")
    print(f"LR F-score: {((2*lr_tcorrect)/(2*lr_tcorrect+lr_fcorrect+lr_fneg))*100}")
    print(f"Lasso F-score: {((2*lasso_tcorrect)/(2*lasso_tcorrect+lasso_fcorrect+lasso_fneg))*100}")
    print(f"Ridge F-score: {((2*ridge_tcorrect)/(2*ridge_tcorrect+ridge_fcorrect+ridge_fneg))*100}")

wet_bulb_evaluate(rf_tcorrect,lr_tcorrect,lasso_tcorrect,ridge_tcorrect,rf_fcorrect,lr_fcorrect,lasso_fcorrect,ridge_fcorrect,rf_fneg,lr_fneg,lasso_fneg,ridge_fneg)
