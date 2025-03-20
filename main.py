import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier  # Changed from Regressor to Classifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report  # Changed metrics

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Handle missing values (as you've already done)
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Cabin'] = train_data['Cabin'].fillna('X')
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data['Cabin'] = test_data['Cabin'].fillna('X')

# Feature engineering (extract cabin type)
train_data['CabinType'] = train_data['Cabin'].str[0]
test_data['CabinType'] = test_data['Cabin'].str[0]

# Select features for model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'CabinType']

# Create X (features) and y (target)
X_train = train_data[features]
y_train = train_data['Survived']

X_test = test_data[features]

# Check feature data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

# Set up preprocessing for categorical features
categorical_features = ['Sex', 'Embarked', 'CabinType']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and train model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# For validation (if splitting train data)
# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# model.fit(X_train_split, y_train_split)
# val_predictions = model.predict(X_val)
# print("Validation accuracy:", accuracy_score(y_val, val_predictions))
# print(classification_report(y_val, val_predictions))

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions.astype(int)
})

submission.to_csv('submission.csv', index=False)
print("Submission file created!")