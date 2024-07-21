import boto3
import streamlit as st

# Import Titan Embeddings Model to generate embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Function for data ingestion from PDF files
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Using Character splitter for better results with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create and save the vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Function to create the Claude LLM
def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={"max_tokens_to_sample": 512})
    return llm

# Function to create the LLaMA3 LLM
def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Prompt template for the LLM
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. Please summarize with at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to get response from the LLM
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="DocQuery AI", layout="wide")

    st.title("📄 DocQuery AI: AI-powered PDF Document Assistant 💼")

    st.markdown("### Ask questions directly from your PDF documents using advanced AI models!")

    user_question = st.text_input("Enter your question related to the PDF files", "")

    st.sidebar.title("Manage Vector Store")
    if st.sidebar.button("Update Vectors"):
        with st.spinner("Updating vectors..."):
            docs = data_ingestion()
            get_vector_store(docs)
            st.sidebar.success("Vectors updated successfully!")

    st.sidebar.title("Select AI Model")
    model_choice = st.sidebar.radio("Choose a model to generate responses:", ("Claude", "LLaMA3"))

    if st.button("Get Response"):
        with st.spinner(f"Generating response with {model_choice}..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            if model_choice == "Claude":
                llm = get_claude_llm()
            else:
                llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.markdown(f"### {model_choice}'s Response")
            st.write(response)
            st.success("Response generated successfully!")

if __name__ == "__main__":
    main()
