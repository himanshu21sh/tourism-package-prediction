import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/himanshu21sh/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

df['CityTier'] = LabelEncoder().fit_transform(df['CityTier'].astype(str))

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save data locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload data to Hugging Face
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed_data/{file_path}",
        repo_id="himanshu21sh/tourism-package-prediction",
        repo_type="dataset",
    )
print("(Xtrain, Xtest, ytrain, ytest) uploaded to Hugging Face dataset.")
