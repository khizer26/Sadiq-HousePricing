# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm
from sklearn.ensemble import RandomForestRegressor # Random Forest Regression

from sklearn.metrics import mean_absolute_percentage_error # mean absolutue percentage error
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric

sb.set_style('whitegrid') # plot style
plt.rcParams['figure.figsize'] = (20, 10) # plot size

# IMPORTING DATA

data = pd.read_csv('HousePricePrediction.csv')
data.set_index('Id', inplace = True)
print(data.head(5))

# EDA

data_new= data.dropna()   #drops null values
data_new.isnull().sum()

print(data_new.describe())       #statistical values of mean, median, standard deviation etc.
print(data_new.dtypes)

#converting float values to integer values

data_new['BsmtFinSF2'] = pd.to_numeric(data_new['BsmtFinSF2'], errors = 'coerce')
data_new['BsmtFinSF2'] = data_new['BsmtFinSF2'].astype('int64')

data_new['TotalBsmtSF'] = pd.to_numeric(data_new['TotalBsmtSF'], errors = 'coerce')
data_new['TotalBsmtSF'] = data_new['TotalBsmtSF'].astype('int64')

data_new['SalePrice'] = pd.to_numeric(data_new['SalePrice'], errors = 'coerce')
data_new['SalePrice'] = data_new['SalePrice'].astype('int64')

print(data_new.dtypes)

# DATA VISUALIZATION

# 1. Heatmap
def heat_map(data_new):
    sb.heatmap(data_new.corr(), annot = True, cmap = 'magma')

    plt.savefig('heatmap.png')
    plt.show()


# Distribution plot
def dist_plot(data_new):
    sb.distplot(data_new['SalePrice'], color = 'r')
    plt.title('Sale Price Distribution', fontsize = 16)
    plt.xlabel('Sale Price', fontsize = 14)
    plt.ylabel('Frequency', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    plt.savefig('distplot.png')
    plt.show()

# Display Plots
# heat_map(data_new)
# dist_plot(data_new)

from sklearn.preprocessing import OneHotEncoder

s = (data_new.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
#print('No. of. categorical features: ', len(object_cols))

OH_encoder = OneHotEncoder(sparse_output =False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(data_new[object_cols]))
OH_cols.index = data_new.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = data_new.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

df_final['MSZoning_C (all)'] = pd.to_numeric(df_final['MSZoning_C (all)'], errors = 'coerce')
df_final['MSZoning_C (all)'] = df_final['MSZoning_C (all)'].astype('int64')

df_final['MSZoning_FV'] = pd.to_numeric(df_final['MSZoning_FV'], errors = 'coerce')
df_final['MSZoning_FV'] = df_final['MSZoning_FV'].astype('int64')

df_final['MSZoning_RH'] = pd.to_numeric(df_final['MSZoning_RH'], errors = 'coerce')
df_final['MSZoning_RH'] = df_final['MSZoning_RH'].astype('int64')

df_final['MSZoning_RL'] = pd.to_numeric(df_final['MSZoning_RL'], errors = 'coerce')
df_final['MSZoning_RL'] = df_final['MSZoning_RL'].astype('int64')

df_final['MSZoning_RM'] = pd.to_numeric(df_final['MSZoning_RM'], errors = 'coerce')
df_final['MSZoning_RM'] = df_final['MSZoning_RM'].astype('int64')

df_final['LotConfig_Corner'] = pd.to_numeric(df_final['LotConfig_Corner'], errors = 'coerce')
df_final['LotConfig_Corner'] = df_final['LotConfig_Corner'].astype('int64')

df_final['LotConfig_CulDSac'] = pd.to_numeric(df_final['LotConfig_CulDSac'], errors = 'coerce')
df_final['LotConfig_CulDSac'] = df_final['LotConfig_CulDSac'].astype('int64')

df_final['LotConfig_FR2'] = pd.to_numeric(df_final['LotConfig_FR2'], errors = 'coerce')
df_final['LotConfig_FR2'] = df_final['LotConfig_FR2'].astype('int64')

df_final['LotConfig_FR2'] = pd.to_numeric(df_final['LotConfig_FR2'], errors = 'coerce')
df_final['LotConfig_FR2'] = df_final['LotConfig_FR2'].astype('int64')

df_final['LotConfig_FR3'] = pd.to_numeric(df_final['LotConfig_FR3'], errors = 'coerce')
df_final['LotConfig_FR3'] = df_final['LotConfig_FR3'].astype('int64')

df_final['LotConfig_Inside'] = pd.to_numeric(df_final['LotConfig_Inside'], errors = 'coerce')
df_final['LotConfig_Inside'] = df_final['LotConfig_Inside'].astype('int64')

df_final['BldgType_1Fam'] = pd.to_numeric(df_final['BldgType_1Fam'], errors = 'coerce')
df_final['BldgType_1Fam'] = df_final['BldgType_1Fam'].astype('int64')

df_final['BldgType_2fmCon'] = pd.to_numeric(df_final['BldgType_2fmCon'], errors = 'coerce')
df_final['BldgType_2fmCon'] = df_final['BldgType_2fmCon'].astype('int64')

df_final['BldgType_Duplex'] = pd.to_numeric(df_final['BldgType_Duplex'], errors = 'coerce')
df_final['BldgType_Duplex'] = df_final['BldgType_Duplex'].astype('int64')

df_final['BldgType_Twnhs'] = pd.to_numeric(df_final['BldgType_Twnhs'], errors = 'coerce')
df_final['BldgType_Twnhs'] = df_final['BldgType_Twnhs'].astype('int64')

df_final['Exterior1st_AsbShng'] = pd.to_numeric(df_final['Exterior1st_AsbShng'], errors = 'coerce')
df_final['Exterior1st_AsbShng'] = df_final['Exterior1st_AsbShng'].astype('int64')

df_final['Exterior1st_AsphShn'] = pd.to_numeric(df_final['Exterior1st_AsphShn'], errors = 'coerce')
df_final['Exterior1st_AsphShn'] = df_final['Exterior1st_AsphShn'].astype('int64')

df_final['Exterior1st_BrkComm'] = pd.to_numeric(df_final['Exterior1st_BrkComm'], errors = 'coerce')
df_final['Exterior1st_BrkComm'] = df_final['Exterior1st_BrkComm'].astype('int64')

#print(df_final.dtypes)
#print(df_final.head(5))

# FEATURE SELECTION & DATA SPLIT

X_var = df_final.drop(['SalePrice'], axis=1).values
y_var = df_final['SalePrice'].values

X_train,X_test, Y_train,Y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0) # train:test = 80:20

# print('X_train samples : ', X_train[0:5])
# print('X_test samples : ', X_test[0:5])
# print('Y_train samples : ', Y_train[0:5])
# print('Y_test samples : ',  Y_test[0:5])

# MODELING

# 1. OLS (Ordinary Least Squares)
def Ols(X_train, Y_train,X_test):
    ols = LinearRegression()
    ols.fit(X_train, Y_train)
    ols_result = ols.predict(X_test)
    return round(mean_absolute_percentage_error(Y_test, ols_result), 4)

# 2. Ridge
def Ridgee(X_train, Y_train,X_test):
    ridge = Ridge(alpha = 0.5)
    ridge.fit(X_train, Y_train)
    ridge_result = ridge.predict(X_test)
    return round(mean_absolute_percentage_error(Y_test, ridge_result), 4)

# 3. Lasso (least absolute shrinkage and selection operator)
def Lassoo(X_train, Y_train,X_test):
    lasso = Lasso(alpha = 0.01)
    lasso.fit(X_train, Y_train)
    lasso_result = lasso.predict(X_test)
    return round(mean_absolute_percentage_error(Y_test, lasso_result), 4)

# 4. Bayesian
def Bay(X_train, Y_train,X_test):
    bayesian = BayesianRidge()
    bayesian.fit(X_train, Y_train)
    bayesian_result = bayesian.predict(X_test)
    return round(mean_absolute_percentage_error(Y_test, bayesian_result), 4)

# 5. ElasticNet
def EN(X_train, Y_train,X_test):
    en = ElasticNet(alpha = 0.01)
    en.fit(X_train, Y_train)
    en_result = en.predict(X_test)
    return round(mean_absolute_percentage_error(Y_test, en_result), 4)

# 6. Random Forest Regression

def RFR(X_train, Y_train,X_test):
    rfr = RandomForestRegressor(n_estimators=10)
    rfr.fit(X_train, Y_train)
    rfr_result = rfr.predict(X_test)    
    return round(mean_absolute_percentage_error(Y_test, rfr_result), 4)
       
# EVALUATION
print('Mean Absolute Error:')

print('OLS:', Ols(X_train, Y_train,X_test))
print('Ridge:', Ridgee(X_train, Y_train, X_test))
print('Lasso:', Lassoo(X_train, Y_train,X_test))
print('Bayesian:', Bay(X_train, Y_train,X_test))
print('Elastic Net:', EN(X_train, Y_train,X_test))
print('Random Forest:', RFR(X_train, Y_train,X_test))
