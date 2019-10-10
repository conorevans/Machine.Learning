import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# load data frame for data from which we will build our model
model_frame = pd.read_csv('withlabels.csv')

# fill NaN values in categorical fields with not_given
for column in model_frame.select_dtypes(include=['object']):
  model_frame[column] = model_frame[column].fillna('not_given')

for column in model_frame.select_dtypes(exclude=['object']).drop('Instance', axis=1):
  # cut bottom and top 10 percent of data to have a more balanced mean
  lower = np.percentile(model_frame[column].dropna(),10)
  upper = np.percentile(model_frame[column].dropna(),90)
  mean = model_frame[model_frame[column].between(lower,upper)][column].values.mean()
  model_frame[column] = model_frame[column].fillna(mean)

  # remove outliers
  #if(column == 'Income in EUR'):
   # data_frame = data_frame[data_frame[column] > lower]
    #data_frame = data_frame[data_frame[column] < upper]

# import scipy.stats as stats
# this code was used to find correlation between columns and Income, to refine which columns used
"""
correlation_columns = []
for column in data_frame.columns:
  if(column != 'Instance' and column != 'Income in EUR'):
    correlation_columns.append(column)

correlations = []
for column in correlation_columns:
  correlations.append([column,stats.pearsonr(data_frame[column], data_frame['Income in EUR'])])

for tuple in correlations:
  print(tuple)
  print("\n")
"""

# these were the resulting low correlation columns
low_correlation = ['Hair Color', 'Wears Glasses', 'Size of City']
model_frame = model_frame.drop(columns=low_correlation)


income = model_frame[['Income in EUR']]
columns = []
for column in model_frame.columns:
  if(column != 'Instance' and column != 'Income in EUR'):
    columns.append(column)
parameters = model_frame[columns]

# build transformers 

# numeric transformation is done at the start so imputing would seem to be redundant
# but it performs better this way. perhaps it overwrites my changes and simply deals with 
# n/a values better.
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

# again, imputing seems redundant, but improves performance
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='not_present')),
                                                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# get features we will transform
numeric_features = ['Year of Record','Age','Body Height [cm]']
categorical_features = ['Country','Gender','Profession']

# looking into whether encoding some typical high paying job keywords might help
"""
professions = ['senior','lead','consultant','manager','analyst','coordinator']
for profession in professions:
  data_frame[profession] = np.where(data_frame['Profession'].str.contains(profession),1,0)
data_frame = data_frame.drop(columns='Profession')
"""

# transform columns and build regression model
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])
regr = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])

# get training and test values
X_train, X_test, Y_train, Y_test = train_test_split(parameters, income, train_size = 0.8, test_size = 0.2)

regr.fit(X_train, Y_train)

# build data frame for our target dataset
target_frame = pd.read_csv('nolabels.csv').drop(columns=low_correlation)

y_predict = regr.predict(target_frame[columns])
print(np.sqrt(metrics.mean_squared_error(Y_test, regr.predict(X_test))))

# Write to File

# Instances saved to separate file for ease of access
instances = pd.read_csv('instances.csv')['Instance'].values
f = open("predictions.csv", "w")
f.write("Instance,Income\n")

for i in range(len(y_predict)):
  f.write(str(instances[i]) + "," + str(y_predict[i][0]) + "\n")
