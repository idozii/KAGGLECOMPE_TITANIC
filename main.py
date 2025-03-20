import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report  

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print(train_data.info())
print(test_data.info())

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Cabin'] = train_data['Cabin'].fillna('X')
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data['Cabin'] = test_data['Cabin'].fillna('X')

features = train_data.columns.drop(['Survived'])

X_train = train_data[features]
y_train = train_data['Survived']

X_test = test_data[features]

# Check feature data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

categorical_features = ['Sex', 'Embarked', 'Cabin', 'Name', 'Ticket']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'PassengerId']

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

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_train_split, y_train_split)
val_predictions = model.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_predictions))
print(classification_report(y_val, val_predictions))

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions.astype(int)
})

submission.to_csv('submission.csv', index=False)
