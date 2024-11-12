from openai import OpenAI
import openai
import ell
from config.config import Config
# from ragas.metrics import faithfulness, answer_relevancy
# from datasets import Dataset

# Initialize OpenAI API key
openai.api_key = Config.OPENAI_API_KEY

# Initialize ell logging
ell.init(store='./logdir', autocommit=True, verbose=False)

# Existing ell-based LLM functions
@ell.simple(model="gpt-4o-mini",client=OpenAI(api_key=openai.api_key))
def rag_module(query: str, context: str) -> str:
    """
    You are a course content planner to help with creating course outline and course content according to handbook information.
    Provided with handbook curriculum information and intended learning outcome with other details , user can ask information related to the course handbook.
    If user asking to list handbook module , provide available module in the handbook.
    If user asking to provide inetnded learning otucome, provide only the intended learning outcome without altering any words.
    Do not hallucinate and add additional information outside of this document.
    """
    return f""" 
    Given the following query and relevant context, please provide a comprehensive and accurate response:

    Query: {query}

    Relevant context:
    {context}

    Response:
    """

@ell.simple(model="gpt-4o-mini",client=OpenAI(api_key=openai.api_key))
def rag_intendedLO(query: str, context: str) -> str:
    """
    You are a course content planner to help with creating course outline and course content according to handbook information.
    Provide the intended learning outcome for module/subject in point form.
    Do not hallucinate and add additional information outside of this document.
    """
    return f""" 
    Given the following query and relevant context, please provide a comprehensive and accurate response:

    Query: {query}

    Relevant context:
    {context}

    Response:
    """

@ell.simple(model="gpt-4o-mini")
def cot(query: str, context: str) -> str:
    """
    Chain of Thought (CoT) prompt for detailed reasoning.
    """
    return f"""
    <thinking>
    Let's break down the problem step by step to understand it better.

    Query: {query}

    Relevant context:
    {context}

    Step-by-step reasoning:
    </thinking>
    <output>
    """

@ell.simple(model="gpt-4o-mini")
def basic(query: str) -> str:
    """
    Basic prompt for straightforward queries.
    """
    return f"""
    Query: {query}
    Response:
    """

@ell.simple(model="gpt-4o-mini")
def course(query: str, context: str) -> str:
    """
    Prompt for generating course-related content.
    """
    return f"""
    Course Query: {query}
    Course Context: {context}
    Course Response:
    """

@ell.simple(model="gpt-4o-mini")
def quiz(query: str, context: str) -> str:
    """
    Prompt for generating quiz questions and answers.
    """
    return f"""
    Quiz Query: {query}
    Quiz Context: {context}
    Quiz Response:
    """


# @ell.simple(model="gpt-4o-mini")
# def generate_question(context: str):
#     """
#     Generates a question based on the provided context.
#     """
#     return [
#         {
#             "role": "user",
#             "content": f"Generate a question based on the following context: {context}"
#         }
#     ]

# @ell.simple(model="gpt-4o-mini")
# def generate_answer(question: str):
#     """
#     Generates an answer based on the provided question.
#     """
#     return [
#         {
#             "role": "user",
#             "content": f"Generate a short answer to the following question: {question}"
#         }
#     ]

# def create_dataset(question: str, context: str, answer: str):
#     """
#     Helper function to create a RAGAS-compatible dataset for evaluation.
#     """
#     data = {
#         "question": [question],
#         "answer": [answer],
#         "contexts": [[context]]
#     }
#     return Dataset.from_dict(data)

# @ell.simple(model="gpt-4o-mini")
# def evaluate_faithfulness_ragas(dataset):
#     """
#     Evaluates the faithfulness of an answer using RAGAS.
#     """
#     result = evaluate(
#         metrics=[faithfulness],
#         dataset=dataset
#     )
    
#     # Handle the case where result is a scalar or a dictionary
#     if isinstance(result, dict):
#         # Extract the faithfulness score if it's in the expected structure
#         faithfulness_score = result.get("faithfulness", {}).get("score", "N/A")
#     else:
#         # Handle case where result is a scalar (e.g., numpy.float64)
#         faithfulness_score = result
    
#     return [
#         {
#             "role": "system",
#             "content": f"Faithfulness evaluation score: {faithfulness_score}"
#         }
#     ]

# @ell.simple(model="gpt-4o-mini")
# def evaluate_answer_relevancy_ragas(dataset):
#     """
#     Evaluates the relevancy of an answer to the given question using RAGAS.
#     """
#     result = evaluate(
#         metrics=[answer_relevancy],
#         dataset=dataset
#     )
    
#     # Check if result is a scalar or a dictionary
#     if isinstance(result, dict):
#         relevancy_score = result.get("answer_relevancy", {}).get("score", "N/A")
#     else:
#         relevancy_score = result

#     return [
#         {
#             "role": "system",
#             "content": f"Answer relevancy evaluation score: {relevancy_score}"
#         }
#     ]