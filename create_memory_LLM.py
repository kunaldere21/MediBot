

from langchain_community.document_loaders import  DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS

# load row pdf
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls= PDFPlumberLoader)
    
    documents = loader.load()
    return documents

DATA_PATH = 'data/'
documents = load_pdf_file(data= DATA_PATH)
print('len of docs', len(documents))


# create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print('len of chunks', len(text_chunks))


# create vector embedding
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model
embedding_model = get_embedding_model()

# store embedding
DB_FAISS_PATH = 'vectorstore/db_faiss'
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)