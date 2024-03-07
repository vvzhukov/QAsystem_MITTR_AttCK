from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

import deeplake
import vertexai
import os
import pandas as pd
import numpy as np

from engine.txt_embed import encode_text_to_embedding_batched
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from sklearn.metrics.pairwise import cosine_similarity

# Path to your service account key file
key_path = "\\keys\\project-03042024-60650bc6b71b.json"
# Path to the json key associated with your service account from google cloud

# Create credentials object

credentials = Credentials.from_service_account_file(
    os.getcwd() + key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

if credentials.expired:
    credentials.refresh(Request())

PROJECT_ID = 'project-03042024'
REGION = 'us-central1'

# init VertexAI
vertexai.init(project=PROJECT_ID, location=REGION, credentials = credentials)


# Load the data
so_database = pd.read_csv('so_database_app.csv')

print("Shape: " + str(so_database.shape))
print(so_database)

# Load the question embeddings

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@003")
generation_model = TextGenerationModel.from_pretrained(
    "text-bison@001")

# Encode the stack overflow data

so_questions = so_database.input_text.tolist()
question_embeddings = encode_text_to_embedding_batched(
            sentences = so_questions,
            api_calls_per_second = 20/60,
            batch_size = 5)

so_database['embeddings'] = question_embeddings.tolist()

# review data with embeddings
so_database


# Semantic search
query = ['How to concat dataframes pandas']
query_embedding = embedding_model.get_embeddings(query)[0].values

# Cosine similarity might be replaced with approximate nearest neighbor search
#   scann package
cos_sim_array = cosine_similarity([query_embedding],
                                  list(so_database.embeddings.values))

index_doc = np.argmax(cos_sim_array)

context = so_database.input_text[index_doc] + \
"\n Answer: " + so_database.output_text[index_doc]

prompt = f"""Here is the context: {context}
             Using the relevant information from the context,
             provide an answer to the query: {query}."
             If the context doesn't provide \
             any relevant information, answer with 
             [I couldn't find a good match in the \
             document database for your query]
             """

t_value = 0.2
response = generation_model.predict(prompt = prompt,
                                    temperature = t_value,
                                    max_output_tokens = 1024)
print(response)

