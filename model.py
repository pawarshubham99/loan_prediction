# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

import pandas as pd
data = pd.read_csv('D:\\Desktop\\bank_final.csv')

col_names=[
        'DisbursementGross', 'BalanceGross',
        'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']

for i in col_names:
  data[i] = data[i].str.replace(',', '').str.replace('$', '').astype(float)

data.head()

data = data.drop(['Name','Zip','ApprovalFY','ChgOffDate','DisbursementDate','ApprovalDate','ChgOffPrinGr'],1)

data.keys()

data.dropna(axis= 0, how ='any',inplace= True)
print(data.describe(include='all'))

lis =[1,0]
data.drop(data[[x not in lis for x in data['FranchiseCode']]].index, axis=0, inplace= True)

import seaborn as sns
sns.boxplot(x=data.FranchiseCode)

lis = [0,1]
data.drop(data[[x not in lis for x in data['UrbanRural']]].index, axis=0, inplace= True)

sns.boxplot(data.UrbanRural)

lis = ['Y','N']
data.drop(data[[x not in lis for x in data['RevLineCr']]].index, axis=0, inplace= True)

data.RevLineCr.describe()

lis = [1,2]
data.drop(data[[x not in lis for x in data['NewExist']]].index, axis=0, inplace= True)

lis = ['Y','N']
data.drop(data[[x not in lis for x in data['LowDoc']]].index, axis=0, inplace= True)

data.LowDoc.describe()

data.columns

data.MIS_Status.hist()
print(data.MIS_Status.str.contains('P I F').sum())
print(data.MIS_Status.str.contains('CHGOFF').sum())

y = pd.DataFrame() 
y = data[['MIS_Status']]

print(y.describe(include='all'))

# Categorical boolean mask
categorical_feature_mask = y.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = y.columns[categorical_feature_mask].tolist()
# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()
# apply le on categorical feature columns
y[categorical_cols] = y[categorical_cols].apply(lambda col: le.fit_transform(col))
y.head()

X = data.drop(['MIS_Status'],1)

X= X.drop(['CCSC','DisbursementGross','BalanceGross'],1)
X.keys()

import numpy as np

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)

y.dtypes

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


numeric_features = ['Term', 'NoEmp','CreateJob', 'RetainedJob','GrAppv','SBA_Appv']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['City', 'State', 'Bank', 'BankState', 'NewExist','FranchiseCode', 'UrbanRural', 'RevLineCr','LowDoc']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators= 20))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train.values.ravel())
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

import joblib
joblib.dump(clf,  'model.joblib',compress = 1)
   
   