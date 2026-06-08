from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
import operator


class StudentPersonas(BaseModel): 
    gender: Literal["female", "male", "other"]
    age: int
    learning_stype: str = Field(description="")
    comm_style: str = Field(description="")
    negative_traits: str = Field(description="")
    positive_traits: str = Field(description="")

class PersonasGroup(BaseModel): 
    groups: list[StudentPersonas] = Field(
        min_length=3,
        max_length=3,
        description="Three distinct student characteristics.",
    )


class QnAState(TypedDict): 
    question: str
    subject: list[str] # change later when found dataset
    answer: str
    feedback: str
    gt_answer: str
    judge_score: dict | None # establish the judge score later
    
class aggQnAState(TypedDict): 
    student: StudentPersonas | None # student personal
    qna_result: QnAState | None # result of the evaluation

class GraphState(QnAState): 
    personas: list[StudentPersonas]
    used_persona_signatures: list[str] # to avoid repeating personas
    qna_results: Annotated[list[aggQnAState], operator.add] 
    final_response: str

class PersonaWorkerState(QnAState): 
    student: StudentPersonas
    agent_name: str


