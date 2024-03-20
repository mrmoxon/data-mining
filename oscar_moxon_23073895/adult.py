
# Part 1: Decision Trees with Categorical Attributes

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	data = pd.read_csv(data_file)
	data = data.drop(['fnlwgt'], axis=1)
	return data

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return df.columns.tolist()

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	list_nan_names = df.isnull().any()
	return list_nan_names[list_nan_names == True].index.tolist()

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	return ((df['education'].isin(['Bachelors', 'Masters']).sum()/df.shape[0])*100).round(1).astype(str) + '%'

df = read_csv_1('./adult.csv')
print(df)
num = num_rows(df)
print(num)
columns = column_names(df)
print(columns)
missing = missing_values(df)
print("missing:", missing)
columns_missing = columns_with_missing_values(df)
print("columns_missing:", columns_missing)
percentage = bachelors_masters_percentage(df)
print("percentage:", percentage)

# Return a p
# andas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	return df.dropna()

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	df_train = df.drop('class', axis=1)
	print(df_train)
	numeric_one_hot = pd.get_dummies(df_train, columns=['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'native-country'], dtype=int)
	return numeric_one_hot

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	y_train = df['class']
	# Convert to numeric
	numeric_y_train = y_train.astype('category').cat.codes
	return numeric_y_train

df = data_frame_without_missing_values(df)
print(df)
X_train = one_hot_encoding(df)
print(X_train)
y_train = label_encoding(df)
print(y_train)

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	# Convert to arrays
	X = X_train.values
	y = y_train.values
	# Initialize the DecisionTreeClassifier
	clf = DecisionTreeClassifier()
	# Fit the model
	clf.fit(X, y)
	predictions = clf.predict(X)
	predictions_series = pd.Series(predictions, name='Predicted Class')
	return predictions_series

y_pred = dt_predict(X_train, y_train)
print("Predictions:", y_pred)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	# Ensure y_pred and y_true are of the same length
    if len(y_pred) != len(y_true):
        raise ValueError("Length of predicted and true labels must be the same.")

	# Calculate the number of incorrect predictions
    incorrect_predictions = sum(y_pred_i != y_true_i for y_pred_i, y_true_i in zip(y_pred, y_true))
    
	# Calculate the error rate
    error_rate = (incorrect_predictions / len(y_true))

    return error_rate

error_rate = dt_error_rate(y_pred, y_train)
print("Error rate:", error_rate)