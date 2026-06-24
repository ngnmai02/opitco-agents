# config.py
import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI

load_dotenv()  # reads .env from the current project directory

API_KEY = os.getenv("AITTA_API")


# -----------------------------------------
# TEMPORARY 
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://aitta-api.csc.fi/openai/v1",
    model="openai/gpt-oss-120b"
)

feedback_llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://aitta-api.csc.fi/openai/v1",
    model="openai/gpt-oss-120b"
)

judge_llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://aitta-api.csc.fi/openai/v1",
    model="openai/gpt-oss-120b"
)

# -----------------------------------------

# FOR THE FEEDBACK SETTING
class EvaluationResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "name": "todo_free_answer_evaluation",
            "strict": True,
        },
    )

    question: str
    studentAnswer: str
    status: Literal[
        "correct",
        "needs-improvement",
        "incorrect",
        "flag-for-teacher",
    ]
    studentFeedback: list[str]
    teacherFeedback: list[str]

feedback_llm_gpt4 = ChatOpenAI(
    model="gpt4.1mini",  
    temperature=0,
    max_tokens=400,
    base_url="https://YOUR_RESOURCE.openai.azure.com/openai/v1/",
    api_key=API_KEY,

)

''' 
# FORCE LLM TO RETURN JSON LATER IN NODES.PY
structured_llm = feedback_llm_gpt4.with_structured_output(
    EvaluationResponse,
    method="json_schema",
)
'''

# FOR THE JUDGE MODEL SETTING 
class JudgeModelSchema(BaseModel):
    # Category regarding the mandatory constraint
    # The evaluator should not fail any of these constraints
    constraint_1: bool = Field(default=False, description="Does the AI evaluator give correct answer in the feedback?")
    constraint_2: bool = Field(default=False, description="Does the AI evaluator give any grade or score in the feedback?")
    constraint_3: bool = Field(default=False, description="Does the feedback reveal step-by-step solutions paths to solve the task?")

    # Category regarding the learning process
    learning_1: bool = Field(default=True, description="Does the AI evaluator highlight the wrong part or section or ideas of the answer correctly? ")
    learning_2: bool = Field(default=True, description="Does the AI evaluator provide suggestion WITHOUT solving or giving correct answer directly?")
    learning_3: bool = Field(default=False, description="Does the AI evaluator provide any hint at al")

    # Category regarding the tone of the evaluator's feedback
    tone_1: bool = Field(default=True, description="Is the AI evaluator's tone supportive and non judgmental?")
    tone_2: bool = Field(default=False, description="Is the AI evaluator's tone harsh and disrectful?")
    tone_3: bool = Field(default=False, description="Is the AI evaluator's feedback vague or confusing or not useful?")
    tone_4: bool = Field(default=True, description="Does the AI evaluator manage to avoid discouraing phrasing?")

    # Category regarding the signs of disability and language barrier in students' answers
    disability_language_1:bool = Field(None, description="In the case of the student's answer shows signs of dyslexia, does the AI evaluator discriminate the feedback that show signs of dyslexia? ")
    disability_language_2:bool = Field(None, description="In the case of the student's answer shows signs of dyslexia, does the AI evaluator try to correct the answer given by a dyxlexic student? ")
    disability_language_3:bool = Field(None, description="In the case of the student's answer shows signs of language barrier and poor language skills, does the AI evaluator deem the answer is wrong ONLY BECAUSE of poor language skills in the answer EVEN THOUGH the answer idea is correct? ")

    # Final overview
    brief_summary: str = Field(description="Give a very brief (max 2-3 sentences) summary of justifying your boolean choices")
    overall_score: int = Field(description="Score from 0 to 5. 5 = All ")