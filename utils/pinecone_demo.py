from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
from loguru import logger
import re
import pandas as pd

class PineconeDemo:

    def __init__(self, pinecone_key, model_name='multilingual-e5-large', index_name="pinecone_demo", namespace='example-namespace', csv_path= '', query = "Tell me about custoemr's feedback on this product 36907838", detail =False, data_size = 300, embedding_batch = 20, upsert_batch_size = 10):
        self.model = model_name  # Load SentenceTransformer model
        self.pinecone_key = pinecone_key
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.index_name = re.sub(r'[^a-z0-9-]', '-', index_name.lower().strip())
        self.namespace = namespace
        self.csv_path = csv_path  # Path to CSV dataset
        self.query =  query
        self.detail = detail
        self.data_size = data_size # Limit to first 300 rows
        self.embedding_batch = embedding_batch
        self.upsert_batch_size = upsert_batch_size


        # Initialize Spark Session
        self.spark = SparkSession.builder \
            .appName("PineconeSparkIntegration") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .getOrCreate()


    def print_pinecone(self):
        print(self.pinecone_key)

    def index_listing(self):
        return self.pc.list_indexes()

    def load_walmart_data(self):
        """Load Walmart dataset and convert reviews to embeddings in batches."""

        logger.info(f"Loading CSV file from path: {self.csv_path}")

        df = self.spark.read.csv(self.csv_path, header=True, inferSchema=True)

        df = df.select(
            col("Pageurl").cast("string").alias("ProductId"),
            col("Rating").cast("float").alias("Rating"),
            col("Review").cast("string").alias("Review")
        )

        df_pandas = df.toPandas().reset_index(drop=True)

        # Remove empty reviews
        df_pandas = df_pandas.dropna(subset=['Review'])
        df_pandas = df_pandas[df_pandas['Review'].str.strip() != '']
        df_pandas = df_pandas[df_pandas['Review'].str.len() > 5]

        df_pandas = df_pandas.iloc[:self.data_size].reset_index(drop=True)  # Limit to first 300 rows

        logger.info(f"📊 Total reviews after cleaning: {len(df_pandas)}")

        all_embeddings = []

        for i in range(0, len(df_pandas), self.embedding_batch):
            batch = df_pandas.iloc[i: i + self.embedding_batch]


            # Get embeddings from Pinecone API
            embedding_response = self.pc.inference.embed(
                model=self.model,
                inputs=batch['Review'].tolist(),
                parameters={"input_type": "passage", "truncate": "END"}
            )


            # Extract embeddings properly
            embeddings = [embedding['values'] for embedding in embedding_response.data if 'values' in embedding]

            all_embeddings.extend(embeddings)



        # Ensure DataFrame matches embedding count
        df_pandas = df_pandas.iloc[:len(all_embeddings)]
        df_pandas['embedding'] = pd.Series(all_embeddings).reset_index(drop=True)

        records = [
            {"id": str(product_id), "values": embedding, "metadata": {"text": review}}
            for product_id, embedding, review in
            zip(df_pandas['ProductId'], df_pandas['embedding'], df_pandas['Review'])
        ]


        if not records:
            raise ValueError("🚨 No valid records to upsert!")

        return records

    # 4. Create an index
    # Create a serverless index
    def create_serverless_index(self):

        if self.index_name not in self.pc.list_indexes().names():
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
        records = self.load_walmart_data()

        if not records:
            raise ValueError("🚨 No records to upsert!")

        # Debug: Print first 5 records before upserting



        #upsert data in bathces

        for i in range(0, len(records), self.upsert_batch_size):
            batch = records[i: i+ self.upsert_batch_size]
            index.upsert(vectors=batch, namespace=self.namespace)
            logger.info(f"Upserted batch {i+1}")

        logger.info('Finished upserting vectors')

        time.sleep(10)

        stats = index.describe_index_stats()
        print("📊 Pinecone Index Stats After Upsert:", stats)


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
            logger.info('Results found...')
            logger.info('clean up index name to save space')

        else:
            logger.info('No results found')

        if self.detail:
            return results

        else:
            return [match['metadata']['text'] for match in results['matches']]


    def clean_up_index_name(self):
        # 7. Clean up
        existing_index_name = [index.name for index in self.pc.list_indexes()]

        if self.index_name in existing_index_name:
            self.pc.delete_index(self.index_name)
            logger.info(f'Deleted index: {self.index_name}')

        else:
            logger.warning(f'Index {self.index_name} does not exist.')



    def run_program(self):

        index = self.pc.Index(self.index_name)
        stats = index.describe_index_stats()
        print("📊 Pinecone Index Stats:", stats)

        self.create_serverless_index()
        self.upsert_vectors()
        result = self.run_query()


        self.clean_up_index_name()
        logger.info('Done')

        return result







