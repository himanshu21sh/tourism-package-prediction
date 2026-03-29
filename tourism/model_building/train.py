import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from random import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score

import joblib

import os

from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/himanshu21sh/tourism-package-prediction/processed_data/Xtrain.csv"
Xtest_path = "hf://datasets/himanshu21sh/tourism-package-prediction/processed_data/Xtest.csv"
ytrain_path = "hf://datasets/himanshu21sh/tourism-package-prediction/processed_data/ytrain.csv"
ytest_path = "hf://datasets/himanshu21sh/tourism-package-prediction/processed_data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'MaritalStatus', 'ProductPitched', 'Designation'
]

numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups',
    'DurationOfPitch', 'NumberOfChildrenVisiting'
]

# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# --- Random Forest ---
rf_params = {
    'regressor__n_estimators': randint(100, 300),
    'regressor__max_depth': randint(3, 10),
    'regressor__min_samples_split': randint(2, 10),
    'regressor__min_samples_leaf': randint(1, 20)
}

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Grid search with cross-validation
rf_grid = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=30, cv=3,
    scoring='recall',
    n_jobs=-1, random_state=42, verbose=1
)

rf_grid.fit(Xtrain, ytrain)

# Best model
best_model = rf_grid.best_estimator_
print("Best Params:\n", rf_grid.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_model_v1.joblib")

# Upload to Hugging Face
repo_id = "himanshu21sh/tourism-package-prediction"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_model_v1.joblib",
    path_in_repo="best_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
