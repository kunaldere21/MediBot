# setup LLM (Mistral with Huggingface)

import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_LLM(huggingface_repoid):
    llm = HuggingFaceEndpoint(
        repo_id=HF_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN,  # Fixed Typo
                      "max_length": 512}
    )
    return llm

# Connect with vector store and create chain
DB_FAISS_PATH = 'vectorstore/db_faiss'

CUSTOME_PROMPT_TEMPLATE = """
Use the piece of information provided in the context to answer user's question.
If you do not know the answer, just say you do not have answer, dont try make the answer.
Dont provided anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def get_custom_prompt(template):
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )
    return prompt

# Load database
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QnA chain
qa_chain = RetrievalQA.from_chain_type( 
    llm=load_LLM(HF_REPO_ID),
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 3}),  # Fixed Argument Name
    return_source_documents=True,
    chain_type_kwargs={"prompt": get_custom_prompt(CUSTOME_PROMPT_TEMPLATE)}
)

# Invoke with single query
USER_QUERY = input("Write Query Here: ")
response = qa_chain.invoke({"query": USER_QUERY})

print("******** Result **********")
print(response['result'])
print("******** source_documents **********")
print(response['source_documents'])

print("******** response **********")
print(response)
