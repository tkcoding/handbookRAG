import os, tempfile
from pathlib import Path

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# from src.generator.llm import rag_module
import openai
from openai import OpenAI
import ell

# from llama_document_parser import llama_document_parser
import streamlit as st

# https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="Course Generation Application")
st.title("Course Generation")
os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY
# # Initialize ell logging
# ell.init(store='./logdir', autocommit=True, verbose=False)

# # Existing ell-based LLM functions
# @ell.simple(model="gpt-4o-mini",client=OpenAI(api_key=openai.api_key))
# def rag_module(query: str, context: str) -> str:
#     """
#     You are a course content planner to help with creating course outline and course content according to handbook information.
#     Provided with handbook curriculum information and intended learning outcome with other details , user can ask information related to the course handbook.
#     If user asking to list handbook module , provide available module in the handbook.
#     If user asking to provide inetnded learning otucome, provide only the intended learning outcome without altering any words.
#     Do not hallucinate and add additional information outside of this document.
#     """
#     return f""" 
#     Given the following query and relevant context, please provide a comprehensive and accurate response:

#     Query: {query}

#     Relevant context:
#     {context}

#     Response:
#     """

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



    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        openai_api_key=st.session_state.openai_api_key
    )
    prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a course content planner to help with creating course outline and course content according to handbook information.
    Provided with handbook curriculum information and intended learning outcome with other details , user can ask information related to the course handbook.
    If user asking to list handbook module , provide available module in the handbook.
    If user asking to provide inetnded learning otucome, provide only the intended learning outcome without altering any words.
    Do not hallucinate and add additional information outside of this document.
    """),
    ("human",
     """ 
    Given the following query and relevant context, please provide a comprehensive and accurate response:

    Query: {query}

    Relevant context:
    {context}

    Response:
    """)])
    content_retrieved = retriever.invoke(query)
    content_concat = "\n\n".join(doc.page_content for doc in content_retrieved)
    chain = prompt | llm
    response =chain.invoke({
        "query":query,
        "context":content_concat
    })
    # result = rag_module(query,content_concat)
    st.session_state.messages.append((query, response.content))
    return response.content

def input_fields():
    #
    with st.sidebar:
        #
        if "OPENAI_API_KEY" in st.secrets:
            st.session_state.openai_api_key = st.secrets.OPENAI_API_KEY
        else:
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
    openai.api_key = st.session_state.openai_api_key
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
    