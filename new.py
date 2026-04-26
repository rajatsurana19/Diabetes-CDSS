import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Dataset downloaded at:", path)

# List files inside folder
files = os.listdir(path)
print("Files in dataset:", files)