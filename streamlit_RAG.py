import os, tempfile
from pathlib import Path

# from langchain_community.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from src.generator.llm import rag_module
# from llama_document_parser import llama_document_parser
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="Course Generation Application")
st.title("Course Generation")

if not os.path.exists('data/tmp'):
    os.makedirs('data/tmp')
def load_documents():
    # Call llamaparse for parsing
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    # file = open('./txt/handbook/handbook.txt', "r")
    # documents = file.read()
    return documents

def split_documents(documents):
    # Use markdownsplitter
    splitter = MarkdownTextSplitter(chunk_size = 2000, chunk_overlap=50)
    md_splitter = splitter.create_documents([documents[0].page_content])
    md_convert_splitter = [md.page_content for md in md_splitter]
    return md_convert_splitter

def embeddings_on_local_vectordb(texts):

    # initialize the bm25 retriever and faiss retriever
    bm25_retriever = BM25Retriever.from_texts(
        texts, metadatas=[{"source": 1}] * len(texts)
    )
    bm25_retriever.k = 12
    embedding = OpenAIEmbeddings()
    faiss_vectorstore = FAISS.from_texts(
        texts, embedding, metadatas=[{"source": 2}] * len(texts)
    )
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 12})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    print('Embedding retriever',ensemble_retriever)
    return ensemble_retriever



def query_llm(retriever, query):
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model="gpt-4o-mini",temperature=0),
    #     retriever=retriever,
    #     return_source_documents=True,
    # )
    # result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    # result = result['answer']
    content_retrieved = retriever.invoke(query)
    content_concat = "\n\n".join(doc.page_content for doc in content_retrieved)
    result = rag_module(query,content_concat)
    st.session_state.messages.append((query, result))
    return result

def input_fields():
    #
    with st.sidebar:
        #
        # if "OPENAI_API_KEY" in st.secrets:
        #     st.session_state.openai_api_key = st.secrets.OPENAI_API_KEY
        # else:
        st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:#or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    print('within temp dirctory')
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                print('after temp directory')

                texts = split_documents(documents)
                print('split text process...')
                #
                # if not st.session_state.pinecone_db:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
                # else:
                #     st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():

    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
    