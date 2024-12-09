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
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pydantic.v1 import BaseModel, Field
from typing import Any, List, Tuple
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from operator import itemgetter
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from common.utils import streamlit_utility
from common.prompt import promptTemplate

# Example of multiple credentials
# credentials:
#   usernames:
#     jsmith:
#       email: jsmith@gmail.com
#       name: John Smith
#       password: abc # To be replaced with hashed password
#     rbriggs:
#       email: rbriggs@gmail.com
#       name: Rebecca Briggs
#       password: def # To be replaced with hashed password


# Get keys for your project from the project settings page
# https://cloud.langfuse.com
# os.environ["LANGFUSE_PUBLIC_KEY"] = st.secrets.LANGFUSE_PUBLIC_KEY
# os.environ["LANGFUSE_SECRET_KEY"] = st.secrets.LANGFUSE_SECRET_KEY
# os.environ["LANGFUSE_HOST"] = st.secrets.LANGFUSE_HOST
# os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY

chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_store")
)

st.set_page_config(page_title="Course Generation Application")

if not os.path.exists("data/tmp"):
    os.makedirs("data/tmp")


def load_documents():
    # Call llamaparse for parsing
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf")
    documents = loader.load()
    return documents


def split_documents(documents):
    # Use markdownsplitter
    splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=50)
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
    return ensemble_retriever


def extract_LO(retriever, query):
    content_retrieved = retriever.invoke(query)
    content_concat = "\n\n".join(doc.page_content for doc in content_retrieved)
    return content_concat


def query_llm(retriever, query):
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model="gpt-4o-mini",temperature=0),
    #     retriever=retriever,
    #     return_source_documents=True,
    # )
    # result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    # result = result['answer']

    langfuse_handler = CallbackHandler()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        openai_api_key=st.session_state.openai_api_key,
    )
    # llm_chat = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     openai_api_key=st.session_state.openai_api_key,
    # )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a course content planner to help with creating course outline and course content according to handbook information.
    Provided with handbook curriculum information and intended learning outcome with other details , user can ask information related to the course handbook.
    If user asking to list handbook module , provide available module in the handbook.
    If user asking to provide intended learning otucome, provide only the intended learning outcome without altering any words.
    Do not hallucinate and add additional information outside of this document.
    """,
            ),
            (
                "human",
                """
    Given the following query and relevant context, please provide a comprehensive and accurate response:

    Query: {query}

    Relevant context:
    {context}

    Response:
    """,
            ),
        ]
    )
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
    response = chain.invoke(
        {"query": query, "context": content_concat},
        config={"callbacks": [langfuse_handler]},
    )
    # result = rag_module(query,content_concat)
    st.session_state.messages.append((query, response.content))
    return response.content


def input_fields():
    # Common field used by multiple tab
    with st.sidebar:
        #
        if "OPENAI_API_KEY" in st.secrets:
            st.session_state.openai_api_key = st.secrets.OPENAI_API_KEY
        else:
            st.session_state.openai_api_key = st.text_input(
                "OpenAI API key", type="password"
            )

    # st.session_state.source_docs = st.file_uploader(
    #     label="Uploads document", type="pdf", accept_multiple_files=True
    # )


def process_documents():
    # Common field used by multiple tab
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            all_doc_text = []
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(
                    delete=False, dir=TMP_DIR.as_posix(), suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(source_doc.read())
                documents = load_documents()
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                texts = split_documents(documents)
                all_doc_text += texts
            st.session_state.retriever = embeddings_on_local_vectordb(all_doc_text)
        except Exception as e:
            st.error(f"An error occurred: {e}")


def chat_widget():

    input_fields()
    openai.api_key = st.session_state.openai_api_key
    # st.button("Submit Documents", on_click=process_documents)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])
    if query := st.chat_input():
        st.chat_message("human").write(query)
        if "retriever" not in st.session_state:
            response = "No document provided"
        else:
            response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)


# Notice that you can forward text_input parameters naturally
def course_input():
    langfuse_handler = CallbackHandler()
    promptTemplateHandler = promptTemplate()
    prompt_template_with_context = promptTemplateHandler.LOPromptWithContext()
    prompt_template_wo_context = promptTemplateHandler.LOPromptWithoutContext()

    # st.subheader("Optional: Upload document related to learning outcomes", divider="gray")

    # st.session_state.source_docs = st.file_uploader(
    #         label="", type="pdf", accept_multiple_files=True)
    # # might need to replace this part.
    # st.button("Submit document",on_click=process_documents,key="LO_upload")
    st.subheader("Required: Fill in course details", divider="gray")

    with st.form(key="Form1"):
        settings = {}
        settings["context"] = "No document context"
        prompt_text_input = prompt_template_wo_context

        params = {}
        params.setdefault("label_visibility", "collapsed")
        c1, c2 = st.columns([4, 9])
        c3, c4 = st.columns([4, 9])
        c5, c6 = st.columns([4, 9])
        c7, c8 = st.columns([4, 9])
        c9, c10 = st.columns([4, 9])
        c11, c12 = st.columns([4, 9])

        with c1:
            c1.markdown("Topic: :red[*]")
            c1.markdown(
                "<span style='background-color:rgba(173, 216, 230);font-size: 12px;'>e.g. Algebra,Data Processing</span>",
                unsafe_allow_html=True,
            )
            course_topic = c2.text_input("topic", value="Algebra", **params)
        # c3.markdown(")
        with c3:
            c3.markdown("Description :red[*]")
            c3.markdown(
                "<span style='background-color:rgba(173, 216, 230);font-size: 12px;'>e.g. Introduction to matrix</span>",
                unsafe_allow_html=True,
            )
            course_description = c4.text_input(
                "description", value="Matrix operations", **params
            )
        with c5:
            c5.markdown("Target Audience :red[*]")
            c5.markdown(
                "<span style='background-color:rgba(173, 216, 230);font-size: 12px;'>e.g. Engineering student</span>",
                unsafe_allow_html=True,
            )
            target_audience = c6.text_input(
                "target_audience", value="Electrical engineering student", **params
            )
        with c7:
            c7.markdown("Pre-requisites :red[*]")
            c7.markdown(
                "<span style='background-color:rgba(173, 216, 230);font-size: 12px;'>e.g. Basic math operation</span>",
                unsafe_allow_html=True,
            )
            pre_requisites = c8.text_input("", value="Basic algebra", **params)
        with c9:
            c9.markdown("Allocated time :red[*]")
            c9.markdown(
                "<span style='background-color:rgba(173, 216, 230);font-size: 12px;'>e.g. 2 weeks, 50 hours, 2 months</span>",
                unsafe_allow_html=True,
            )
            allocated_time = c10.text_input("", value="2 weeks", **params)

        with c11:
            c11.markdown("Langugage selected :red[*]")
            language_selected = c12.selectbox(
                "", ("English", "Chinese", "Russian"), **params
            )

        # Section where it should add in what's the file input.

        # prompt_text_input = st.text_area("Prompt input", value=prompt_template_with_context)
        submitButton = st.form_submit_button(label="Generate learning outcomes")
        if submitButton:
            if len(st.session_state.source_docs) >= 1:
                extraction_query = f"""
                    Extract learning outcomes that is related to {course_description} on the topic of {course_topic}
                """
                langfuse_handler = CallbackHandler()
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    openai_api_key=st.session_state.openai_api_key,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
                            You are a course content planner to help with creating course outline and course content according to handbook information.
                            Provided with handbook curriculum information and intended learning outcome with other details , user can ask information related to the course handbook.
                            If user asking to list handbook module , provide available module in the handbook.
                            If user asking to provide intended learning otucome, provide only the intended learning outcome without altering any words.
                            Do not hallucinate and add additional information outside of this document.
                            """,
                        ),
                        (
                            "human",
                            """
                Given the following query and relevant context, please provide a comprehensive and accurate response:

                Query: {query}

                Relevant context:
                {context}

                Response:
                """,
                        ),
                    ]
                )
                chain = prompt | llm
                RAG_final_response = chain.invoke(
                    {
                        "query": extraction_query,
                        "context": extract_LO(
                            st.session_state.retriever, extraction_query
                        ),
                    },
                    config={"callbacks": [langfuse_handler]},
                )

                # Another layer of response.
                question_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
                    You are an agent to verify the context whether is relevant to the topic provided.
                    If it's relevant reply "Yes" if it s not relevant reply "No".""",
                        ),
                        (
                            "human",
                            """
                        Provided context :
                        {context}

                        topic:
                        {course_topic}

                        topic_description:
                        {course_description}
                        """,
                        ),
                    ]
                )
                chain = question_prompt | llm

                response_confirmation = chain.invoke(
                    {
                        "context": RAG_final_response.content,
                        "course_topic": course_topic,
                        "course_description": course_description,
                    },
                    config={"callbacks": [langfuse_handler]},
                )

                # Logic here might change
                if "Yes" in response_confirmation.content:
                    prompt_text_input = prompt_template_with_context
                    settings["context"] = RAG_final_response.content
            st.subheader("Retrieve context", divider="gray")
            st.info(
                settings["context"],
                icon="ℹ️",
            )
            settings["course_topic"] = course_topic
            settings["course_description"] = course_description
            settings["target_audience"] = target_audience
            settings["pre_requisites"] = pre_requisites
            settings["allocated_time"] = allocated_time
            settings["language"] = language_selected
            GENERATE_TOC_PROMPT = f"""
            {prompt_text_input}

            """
            generate_toc_prompt = ChatPromptTemplate.from_template(GENERATE_TOC_PROMPT)
            toc_chain = generate_toc_prompt | chat_model | StrOutputParser()
            st.subheader("Generated Learning Outcomes", divider="gray")

            st.info(
                toc_chain.invoke(settings, config={"callbacks": [langfuse_handler]}),
                icon="ℹ️",
            )


if __name__ == "__main__":
    _streamlit_utils = streamlit_utility()
    _streamlit_utils.environment_settings()
    authenticator = _streamlit_utils.initiate_authentication()

    try:
        authenticator.login()
    except Exception as e:
        st.error(e)
    if st.session_state["authentication_status"]:
        authenticator.logout()
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.sidebar.markdown("### Choose the application")
        my_button = st.sidebar.radio(
            label="", options=("Doc Chat", "Course Generation")
        )

        st.markdown(
            _streamlit_utils.markdown_style(),
            unsafe_allow_html=True,
        )
        if my_button == "Doc Chat":
            st.title("Document Chat")
            st.markdown("# Please ask any question regarding to the document!")
            chat_widget()
        elif my_button == "Course Generation":
            st.title("Learning outcome generation")
            course_input()
        else:
            pass
        # Declare variable.
        # Access the uploaded ref via a key.
        st.sidebar.markdown("### Upload relevant material for course")
        st.session_state.source_docs = st.sidebar.file_uploader(
            label="", type="pdf", accept_multiple_files=True
        )
        # might need to replace this part.
        st.sidebar.button(
            "Submit document", on_click=process_documents, key="LO_upload"
        )
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")
