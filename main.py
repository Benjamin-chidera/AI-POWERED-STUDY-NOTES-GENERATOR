from dotenv import load_dotenv
load_dotenv()
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# pdf
from langchain_community.document_loaders import PyPDFLoader

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vectore store setup
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore

# set up streamlit
st.set_page_config(page_title="LangChain", page_icon="🤖", layout="centered", initial_sidebar_state="auto")

st.title("AI-POWERED STUDY NOTES GENERATOR")
st.subheader("Generate study notes using AI")

# pinecone setup
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

# pinecone vector store
index_name = "study-notes-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

llm = ChatOllama(model="llama3.1")

def store_pdf():
    pdf_file = st.file_uploader("Upload your notes", type=["pdf"])
    vector_btn = st.button("Store PDF")
    temp = "temp.pdf"

    if pdf_file is not None:
        data = pdf_file.getvalue()
        pdf_viewer(input=data, height=500, width=800)
        
        if vector_btn:
            try:
                with open(temp, "wb") as f:
                    f.write(data)
                
                # read pdf file
                get_pdf = PyPDFLoader(temp)
                docs = get_pdf.load()
                
                # split doc
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                doc = text_splitter.split_documents(docs)
                
                
                # embedding setup
                embedding = OllamaEmbeddings(model="llama3.1")
                
                vectore_store = PineconeVectorStore(index=index, embedding=embedding)
                
                vectore_store.add_documents(doc)
                
                st.success("PDF stored successfully")
                
            except Exception as e:
                st.error(f"error, {e}")
                
        
def get_pdf():
    
    # embedding setup
    embedding = OllamaEmbeddings(model="llama3.1")   
    vectore_store = PineconeVectorStore(index=index, embedding=embedding)
    
    query = st.text_input("Enter your query")
    
    search_btn = st.button("Search")
    
    if search_btn:
        try:
            if query:
                result = vectore_store.similarity_search(query, k=4)
                
                # for res in result:
                
                #     st.write(res.page_content)   
                
                message = [
                    (
                        "system", f"""You are an ai exam student who answers questions i ask related to the documents provided. If the answer is not in the document, respond with i don't know this one.
                        
                        {[res.page_content for res in result]}
                        """
                    ),
                    
                    (
                        "human", "{query}"
                    )
                ]
                
                chat = ChatPromptTemplate.from_messages(message)
                
                chain = chat | llm | StrOutputParser()
                
                res = chain.invoke({"query": query})
                
                st.write(res)
                    
        except Exception as e:
            st.error(f"error, {e}")


if __name__ == "__main__":
    # store_pdf()

    get_pdf()