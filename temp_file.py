import os
from pprint import pprint

from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
import json
import os
import uuid
from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv("OPENAI_API_BASE")
API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
API_VERSION = os.getenv("OPENAI_API_VERSION")
API_TYPE = os.getenv("OPENAI_API_TYPE")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")

# KEYS OF EMBEDDING
EMBEDDING_BASE_URL = os.getenv("EMBEDDINGS_OPENAI_API_BASE")
EMBEDDING_API_KEY = os.getenv("EMBEDDINGS_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDINGS_AZURE_DEPLOYMENT_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDINGS_OPENAI_MODEL")
EMBEDDING_API_TYPE = os.getenv("EMBEDDINGS_OPENAI_API_TYPE")


def load_one_vector_db_by_name(vectordb_name):
    openai_embeddings = OpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT_NAME,
        model=EMBEDDING_MODEL,
        openai_api_base=EMBEDDING_BASE_URL,
        openai_api_type=EMBEDDING_API_TYPE,
        openai_api_key=EMBEDDING_API_KEY,
        chunk_size=16)

    print("vector db loaded")
    vectordb_path = f"vector_dbs/{vectordb_name}"
    # vector_db = FAISS.load_local("Genpact_KB_faiss_index_9thOct2023", openai_embeddings)
    # vector_db = FAISS.load_local("candidate_bot_corpus_V7_03May2024", openai_embeddings)
    vector_db = FAISS.load_local(vectordb_path, openai_embeddings)
    return vector_db

# loading all dbs at once
def load_all_country_vectordbs():
    vector_dbs = {}
    vectordb_names = os.listdir("vector_dbs")
    #vectodb_names_paths = [x[0] for x in os.walk("vector_dbs")]
    for name in vectordb_names:
        vector_dbs[name] = load_one_vector_db_by_name(name)
    return vector_dbs



if __name__ == '__main__':
    vector_dbs = load_all_country_vectordbs()
    pprint(vector_dbs)