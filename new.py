import kagglehub
import os

path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Dataset downloaded at:", path)

files = os.listdir(path)
print("Files in dataset:", files)