import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Drop rows with missing target, separate target from predictors
#Age, Cabin, Embarked
train_data.dropna(axis=0, subset=['Survived'], inplace=True)
y = train_data.Survived

# Drop target column
train_data.drop(['Survived'], axis=1, inplace=True)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in train_data.columns if train_data[cname].nunique() < 10 and train_data[cname].dtype == 'object']

# Select numerical columns
numerical_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols

train_data = train_data[my_cols].copy()
test_data = test_data[my_cols].copy()

# Preprocessing for numerical DataFrame
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical DataFrame
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical DataFrame
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model_selection
X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Use the model to make predictions
predictions = rf_model.predict(X_valid)

submission = pd.DataFrame({
    'PassengerId': test_data.PassengerId,
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)

