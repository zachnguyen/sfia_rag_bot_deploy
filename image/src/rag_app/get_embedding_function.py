from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_aws import BedrockEmbeddings

def get_embedding_function():
    #embeddings = BedrockEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    #embeddings = OpenAIEmbeddings()
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
