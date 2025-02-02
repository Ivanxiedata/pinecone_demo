from utils.pinecone_demo import PineconeDemo
import os
from dotenv import load_dotenv

load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
model_name = 'multilingual-e5-large'
index_name = 'pinecone-demo'
csv_path = '/Users/ivanxie/Desktop/GithubProjects/pinecone_demo/input_data/product_review_data.csv'
# Define your query
query = "This TV is absolutely fantastic."
detail = False # display full detail or just the results
data_size = 300  #control the input data size
embedding_batch = 20  #control the embedding batch size
upsert_batch_size = 10   #control the upsert batch size



PCD = PineconeDemo(
    pinecone_key = pinecone_api_key,
    model_name = model_name,
    csv_path =csv_path,
    index_name =index_name,
    query = query,
    detail = detail,
    data_size = data_size,
    embedding_batch = embedding_batch)


# result = PCD.run_program()
result = PCD.run_query()
print(result)




