# setup LLM (Mistral with Huggingface)

import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


# load llm
def load_LLM(repoid, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=repoid,
        temperature=0.5,
        model_kwargs={"token": hf_token,  # Fixed Typo
                      "max_length": 512}
    )
    return llm

# Connect with vector store and create chain
def get_custom_prompt(template):
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )
    return prompt

# Load database
def get_vectorstore(db_path):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Create QnA chain
def create_chain(llm, db, prompt):
    qa_chain = RetrievalQA.from_chain_type( 
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


if __name__ == "__main__":  # Fixed Typo Here
    print("Check pipeline..........")

    HF_TOKEN = os.environ.get('HF_TOKEN')
    HF_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    CUSTOME_PROMPT_TEMPLATE = """
    Use the piece of information provided in the context to answer user's question.
    If you do not know the answer, just say you do not have answer, dont try make the answer.
    Dont provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    # Create template 
    prompt = get_custom_prompt(CUSTOME_PROMPT_TEMPLATE)

    # Load db store 
    db = get_vectorstore(DB_FAISS_PATH)

    # Load LLM
    llm = load_LLM(HF_REPO_ID, HF_TOKEN) 

    # Connect to chain  
    qa_chain = create_chain(llm, db, prompt)
    
    # Invoke with single query
    USER_QUERY = input("Write Query Here: ")
    response = qa_chain.invoke({"query": USER_QUERY})


    # Print the result and source documents
    print("Answer:", response['result'])
    # print("Source Documents:", response['source_documents'])
