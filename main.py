import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import plotly.express as px
from sklearn.decomposition import PCA

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
test_data_copied = test_data.copy()
print(train_data.head())
print(train_data.info())
print(train_data.isnull().sum())

columns_to_drop = ["PassengerId"]
train_data = train_data.drop(columns=columns_to_drop)
test_data_copied = test_data_copied.drop(columns=columns_to_drop)

X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_features = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", KNNImputer(n_neighbors=3)),
        ("scaler", StandardScaler()),
    ]
)
cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

transformers = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", cat_pipeline, cat_features),
    ]
)

transformed_X_train = transformers.fit_transform(X_train)
transformed_X_test = transformers.transform(X_test)
transformed_test_data = transformers.transform(test_data_copied)

rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42)
nn_model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(transformed_X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="linear"),
    ]
)
nn_model.compile(Adam(learning_rate=0.001), loss=BinaryCrossentropy(from_logits=True))

rf_grid_search = GridSearchCV(rf_model, param_grid={"n_estimators": [100, 200]}, cv=3)
xgb_grid_search = GridSearchCV(xgb_model, param_grid={"n_estimators": [100, 200]}, cv=3)

nn_model.fit(transformed_X_train, y_train, epochs=100, batch_size=32, verbose=1)
rf_grid_search.fit(transformed_X_train, y_train)
xgb_grid_search.fit(transformed_X_train, y_train)

nn_predictions = nn_model.predict(transformed_X_test)
nn_predictions = tf.sigmoid(nn_predictions).numpy()
nn_predictions_final = (nn_predictions > 0.5).astype(int)
rf_best_model = rf_grid_search.best_estimator_
rf_predictions = rf_best_model.predict(transformed_X_test)
rf_predictions = (rf_predictions > 0.5).astype(int)
xgb_best_model = xgb_grid_search.best_estimator_
xgb_predictions = xgb_best_model.predict(transformed_X_test)
xgb_predictions = (xgb_predictions > 0.5).astype(int)

print(f"NN Accuracy: {classification_report(y_test, nn_predictions_final)}")
print(f"RF Accuracy: {classification_report(y_test, rf_predictions)}")
print(f"XGB Accuracy: {classification_report(y_test, xgb_predictions)}")

predictions = nn_model.predict(transformed_test_data)
predictions = tf.sigmoid(predictions).numpy()
predictions = (predictions > 0.5).astype(int)
submission = pd.DataFrame(
    {
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions.flatten(),
    }
)
submission.to_csv("data/submission.csv", index=False)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(transformed_X_train.toarray())
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color=y_train.astype(str))
fig.show()
