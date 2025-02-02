import kagglehub
import shutil
import os

# Define the dataset and destination path
dataset = "promptcloud/walmart-product-reviews-dataset"
my_path = "/Users/ivanxie/Desktop/GithubProjects/pinecone_demo/input_data"

# Download latest version
path = kagglehub.dataset_download(dataset)

# Ensure destination directory exists
os.makedirs(my_path, exist_ok=True)

# Move downloaded files to the desired path
for file in os.listdir(path):
    shutil.move(os.path.join(path, file), os.path.join(my_path, file))

print("Dataset downloaded to:", my_path)