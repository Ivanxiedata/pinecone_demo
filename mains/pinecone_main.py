from utils.pinecone_demo import PineconeDemo
import os
from dotenv import load_dotenv



load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
model_name = 'multilingual-e5-large'
index_name = 'pinecone-demo'
# Define your query
query = "Tell me about the tech company known as Apple."
detail = False
PCD = PineconeDemo(pinecone_api_key, model_name, index_name, query, detail)
result = PCD.run_program()
print(result)


