import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS



DB_FAISS_PATH = 'vectorstore/db_faiss'

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(template):
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )
    return prompt



def load_LLM(repoid, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=repoid,
        temperature=0.5,
        model_kwargs={"token": hf_token,  # Fixed Typo
                      "max_length": 512}
    )
    return llm


def main():
    st.title('Ask MediBot')
    
    # Initialize messages list if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Capture user input
    prompt = st.chat_input("Pass your prompt here")


    if prompt:
        st.chat_message('user').markdown(prompt)

        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # custom prompt
        CUSTOME_PROMPT_TEMPLATE = """
            Use the piece of information provided in the context to answer user's question.
            If you do not know the answer, just say you do not have answer, dont try make the answer.
            Dont provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """

        HF_TOKEN = os.environ.get('HF_TOKEN')
        HF_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

        try: 
            vectorstore  = get_vectorstore()
            if vectorstore==None:
                st.error("Fail to laod vector store ")

            qa_chain = RetrievalQA.from_chain_type( 
                llm=load_LLM(HF_REPO_ID, HF_TOKEN),
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),  # Fixed Argument Name
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOME_PROMPT_TEMPLATE)}
            )
            response = qa_chain.invoke({'query': prompt})

            result = response['result']
            source_documents = response['source_documents']



            Dynamic_response = result
            st.chat_message('assistant').markdown(Dynamic_response)

            st.session_state.messages.append({'role': 'assistant', 'content': Dynamic_response})
        except Exception as e:
            st.error("ERROR: " , e)


if __name__ == "__main__":
    main()
