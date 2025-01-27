from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import os
from dotenv import load_dotenv
from loguru import logger



class PineconeDemo:
    load_dotenv()

    # 1. Get a model
    def __init__(self, pinecone_key, model_name = 'distilbert-base-nli-stsb-mean-tokens', index_name="pinecone_demo", query = "Tell me about the tech company known as Apple.", detail = True, namespace ='example-namespace'):
        self.model = model_name
        self.pinecone_key = pinecone_key
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.index_name = index_name
        self.query = query
        self.detail = detail
        self.namespace = namespace

    def print_pinecone(self):
        print(self.pinecone_key)

    def generate_vector(self):
        # 3. Generate vectors
        # Define a sample dataset where each item has a unique ID and piece of text
        data = [
            {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
            {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
            {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
            {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
            {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
            {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
        ]

        # Convert the text into numerical vectors that Pinecone can index
        embeddings = self.pc.inference.embed(
            model=self.model,
            inputs=[d['text'] for d in data],
            parameters={"input_type": "passage", "truncate": "END"}
        )

        return data, embeddings

    def index_listing(self):
        return self.pc.list_indexes()


    # 4. Create an index
    # Create a serverless index
    def create_serverless_index(self):
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logger.info(f"Index {self.index_name} creation initiated.")
        else:
            logger.info(f"Index {self.index_name} already exists.")

        # Wait for the index to be ready
        while True:
            status = self.pc.describe_index(self.index_name).status
            if status['ready']:
                logger.info(f"Index {self.index_name} is ready.")
                break
            else:
                logger.info(f"Waiting for index {self.index_name} to be ready...")

        time.sleep(5)


    #5. Upsert vectors
    # Target the index where you'll store the vector embeddings
    def upsert_vectors(self):

        # Target the index where you'll store the vector embeddings
        index = self.pc.Index(self.index_name)

        records = []
        data, embeddings = self.generate_vector()
        for d, e in zip(data, embeddings):
            records.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text']}
            })

        # Upsert the records into the index
        index.upsert(
            vectors=records,
            namespace=self.namespace

        )

        print(index.describe_index_stats(namespace=self.namespace))

        time.sleep(10)  # Wait for the upserted vectors to be indexed


    # 6. Query the index
    def run_query(self):
        index = self.pc.Index(self.index_name)
        # Convert the query into a numerical vector that Pinecone can search with
        query_embedding = self.pc.inference.embed(
            model=self.model,
            inputs=[self.query],
            parameters={
                "input_type": "query"
            }
        )

        # Search the index for the three most similar vectors
        results = index.query(
            namespace=self.namespace,
            vector=query_embedding[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )


        if results:
            logger.info('Results found:')
            logger.info('clean up index name to save space')

        else:
            logger.info('No results found')

        if self.detail:
            return results

        else:
            return [match['metadata']['text'] for match in results['matches']]


    def clean_up_index_name(self):
        # 7. Clean up
        self.pc.delete_index(self.index_name)
        logger.info(f"Deleted index: {self.index_name}")


    def run_program(self):
        self.create_serverless_index()
        self.upsert_vectors()
        result = self.run_query()
        # self.clean_up_index_name()
        # logger.info('Done')

        return result





