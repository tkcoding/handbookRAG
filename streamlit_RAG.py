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
import streamlit_authenticator as stauth
import streamlit as st
from langchain.chains import ConversationalRetrievalChain

# from llama_document_parser import llama_document_parser
import streamlit as st

import json

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pydantic.v1 import BaseModel, Field
from typing import Any, List, Tuple
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from operator import itemgetter
# https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit

chat_model = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0
)
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="Course Generation Application")
st.title("Course Generation")
os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY


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
    llm_chat = ChatOpenAI(
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

    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm_chat,
    #     retriever=retriever,
    #     combine_docs_chain_kwargs={"prompt": prompt}
    # )
    # print('xxxxx',qa_chain({
    #     "question":query,
    #     "query":query,
    #     "context":content_concat,
    #     "chat_history":st.session_state.messages
    #     }))


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
    # tab1, tab2 = st.tabs(["RAG Chat", "Course Generation"])
# with tab1:
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


 # Notice that you can forward text_input parameters naturally
def course_input():
    with st.form(key="Form1"):
        params = {}
        params.setdefault('label_visibility', 'collapsed')
        c1, c2 =  st.columns([4,9])
        c3, c4 = st.columns([4,9])
        c5, c6 = st.columns([4,9])
        c7, c8 = st.columns([4,9])
        c9, c10 = st.columns([4,9])
        c11, c12 = st.columns([4,9])
        # c7, c8, c9, c10, c11, c12 = st.columns([6,6,6,6,6,6]) 
        # with c1:
            # course_theme = st.text_input("Course theme",value="Algebra")
        with c1:
            c1.markdown('Topic: :red[*]')
            course_topic = c2.text_input('topic',value="Algebra", **params)
        # c3.markdown(")
        with c3:
            c3.markdown('Description :red[*]')
            course_description = c4.text_input('description',value="Matrix operations",**params)
        with c5:
            c5.markdown('Target Audience :red[*]')
            target_audience = c6.text_input('target_audience',value="Electrical engineering student",**params)
        with c7:
            c7.markdown('Pre-requisites :red[*]')
            pre_requisites = c8.text_input('',value="Basic algebra",**params)     
        with c9:
            c9.markdown('Allocated time :red[*]')
            allocated_time = c10.text_input('',value="2 weeks",**params)     
  
        input_fields()
        submitButton = st.form_submit_button(label = 'Generate learning outcomes')

        if submitButton:
            settings = {}
            settings['course_topic'] = course_topic
            settings['course_description'] = course_description
            settings['target_audience'] = target_audience
            settings['pre_requisites'] = pre_requisites
            settings['allocated_time'] = allocated_time
            # settings['other_features'] = "No other feature"
            GENERATE_TOC_PROMPT = """
            Here is example of a table of contents in JSON format for some course about python data science for absolute beginner for 10 weeks allocated time.
            ###
            {{'Learning Outcomes': {{
            'I. Data Foundations': 
            {{
            'Duration':2 weeks,
            'A. Define the workflow, tools and approaches data scientists use to analyse data': {{}},
            'B. Apply the Data Science Workflow to solve a task': {{}},
            'C. Navigate through directories using the command line': {{}},
            'D. Conduct arithmetic and string operations in Python': {{}}}},
            'II. Working with Data': 
            {{
            'Duration': 2 weeks,
            'A. Use DataFrames and Series to read data: {{}},
            'B. Define key principles of data visualization': {{}},
            'C. Create line plots, bar plots , histograms and box plots using Seaborn and Matplotlib': {{}},
            'D. Determine causality and sampling bias': {{}}}},
            'III. Data Science Modeling': 
            'Duration': 3 weeks
            {{'A. Define data modeling and linear regression': {{}},
            'B. Describe errors of bias and variance': {{}},
            'C. Build a k-nearest neighbors model using the scikit-learn library': {{}},
            'D. Evaluate a model using metrics such as classification accuracy/error, confusion matrix, ROC/AOC curves and loss functions': {{}}}},
            'IV. Data Science Applications':
            'Duration' : 3 weeks
            {{'A. Demonstrate how to tokenize natural language text': {{}},
            'B. Perform text classification model using scikit-learn, CountVectorizer, TfidfVectorizer, and TextBlog': {{}},
            'C. Create rolling means and plot time series data': {{}},
            'D. Explore an additional data science topic based on class interest. Options include: clustering, decision trees, robust regression and deploying model with Flask': {{}}}},
            'E. Final Project: Complete a capstone project on data science real world application': {{}}}}}}
            ###

            Your task it to estimate each section with total duration needed , the amount of all section should add up to allocated time according to the template provided above.
            Scale the duration needed for each section with respect to the complexity and expertise level.
            Generate learning objectives in the JSON format for a course about '{course_topic}' for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}

                
            """

            generate_toc_prompt = ChatPromptTemplate.from_template(GENERATE_TOC_PROMPT)
            toc_chain = (generate_toc_prompt | chat_model | StrOutputParser())
            st.info(toc_chain.invoke(settings), icon="ℹ️")

if __name__ == '__main__':

    print(st.secrets)
    _secrets_to_config = {}
    _secrets_to_config['usernames'] = {st.secrets['login']['user']:{"email":st.secrets['login']['email'],\
                                                                    "name":st.secrets['login']['name'],\
                                                                    "password":st.secrets['login']['password'],\
                                                                    }}
    authenticator = stauth.Authenticate(
        _secrets_to_config,
        st.secrets['login']['name'],
        st.secrets['cookie']['key'],
        st.secrets['cookie']['expiry_days'],
    )

    try:
        authenticator.login()
    except Exception as e:
        st.error(e)
    if st.session_state['authentication_status']:
        authenticator.logout()
        st.write(f'Welcome *{st.session_state["name"]}*')
        my_button = st.sidebar.radio(label="Choose the application ", options=('Doc Chat','Course Generation')) 
        st.markdown(
            """<style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-family: 'Roboto', sans-serif; 
            font-size: 18px;
            font-weight: 500;
            color: #091747;
            }
            </style>
            """, unsafe_allow_html=True)
        if my_button == 'Doc Chat':
            boot()
        elif my_button == 'Course Generation': 
            course_input()
        else:
            pass 

        # boot()
    elif st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your username and password')
    