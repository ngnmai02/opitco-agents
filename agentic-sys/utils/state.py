from typing import TypedDict, Annotated
import operator
from config import EvaluationResponse, JudgeModelSchema, StudentPersonaSchema


class QnAState(TypedDict): 
    question: str
    subject: list[str] # change later when found dataset
    answer: str
    feedback: EvaluationResponse | None
    gt_answer: str
    judge_score: JudgeModelSchema | None
    
class aggQnAState(TypedDict): 
    student: StudentPersonaSchema | None # student personal
    qna_result: QnAState | None 

class GraphState(QnAState): 
    personas: list[StudentPersonaSchema]
    used_persona_signatures: list[str] # to avoid repeating personas
    qna_results: Annotated[list[aggQnAState], operator.add] 
    evaluated_qna_results: list[aggQnAState]
    final_response: str

class PersonaWorkerState(QnAState): 
    student: StudentPersonaSchema
    agent_name: str

