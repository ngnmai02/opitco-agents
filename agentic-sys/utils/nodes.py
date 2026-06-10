from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy, Send
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
from pathlib import Path
from config import *
from state import *

def checkpoint(message: str) -> None:
    """Print a flushed CLI checkpoint for long-running graph execution."""

    print(f"[checkpoint] {message}", flush=True)


# TODO: EDIT MODEL API
checkpoint("Initializing LLM clients")
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

# System prompt of the feedback AI sys
# TODO: EDIT BOTH FILE CONTENT
checkpoint("-----------------")
checkpoint("Retrieving system prompts")
FEEDBACK_SYSTEM_PROMPT_PATH = Path(__file__).with_name("feedback_sys_prompt.txt") 
checkpoint(FEEDBACK_SYSTEM_PROMPT_PATH)
JUDGE_SYSTEM_PROMPT_PATH = Path(__file__).with_name("judge_sys_prompt.txt") 

# ----------------------------
# functions

# PERSONAS GENERATOR 
def persona_signature(persona: StudentPersonas) -> str:
    """Create a stable, compact description used to detect repeated personas."""
    
    return "|".join(
        [
            persona.gender.lower().strip(),
            str(persona.age // 5 * 5),
            persona.learning_stype.lower().strip(),
            persona.comm_style.lower().strip(),
            persona.negative_traits.lower().strip(),
            persona.positive_traits.lower().strip(),
        ]
    )

def unique_personas(
    personas: list[StudentPersonas], used_signatures: set[str]
) -> list[StudentPersonas]:
    """Filter out repeated persona types within this run and across prior runs."""

    accepted: list[StudentPersonas] = []
    seen = set(used_signatures)

    for persona in personas:
        signature = persona_signature(persona)
        if signature in seen:
            continue
        accepted.append(persona)
        seen.add(signature)

    return accepted

def generate_student_personas(state: GraphState) -> dict:
    """Generate three personas and avoid previously used persona types."""

    checkpoint("-----------------")
    checkpoint("Node start: generate student personas")
    generator = llm.with_structured_output(PersonasGroup)
    used_signatures = set(state.get("used_persona_signatures", []))
    prior_personas = "\n".join(sorted(used_signatures)) or "None yet."

    accepted: list[StudentPersonas] = []
    number_of_attempts = 5
    for attempt in range(number_of_attempts):
        checkpoint(f"Generating student personas, attempt {attempt + 1}/{number_of_attempts}")
        prompt = f"""
        Generate exactly three student personas that can be encountered at a typical classroom

        Do not generate any persona type matching these previously used persona
        signatures:
        {prior_personas}

        Attempt: {attempt + 1}

        Make each persona distinct by age, learning_stype, comm_style,
        negative_traits, and positive_traits. Avoid generic duplicates like
        three shy visual learners or three highly motivated high achievers.
        """
        generated = generator.invoke(prompt)
        checkpoint(f"Generated {len(generated.groups)} persona candidates")
        accepted = unique_personas(generated.groups, used_signatures)
        checkpoint(f"Accepted {len(accepted)} unique persona candidates")

        if len(accepted) == number_of_attempts:
            break

        used_signatures.update(persona_signature(persona) for persona in accepted)
        prior_personas = "\n".join(sorted(used_signatures))

    if len(accepted) != number_of_attempts:
        raise ValueError(
            "Could not generate three unique student personas after three attempts."
        )

    new_signatures = [persona_signature(persona) for persona in accepted]
    checkpoint("Node complete: generate_student_personas")
    return {
        "personas": accepted,
        "used_persona_signatures": state.get("used_persona_signatures", []) + new_signatures,
    }

# GENERATING REPLIES ANSWERING BY DIFFERENT PERSONAS 
# ROUTER
def route_personas_to_agents(state: GraphState) -> list[Send]:
    """Route each generated persona to a separate student-agent node."""

    if len(state["personas"]) != 3:
        raise ValueError("Expected exactly three generated personas.")

    agent_nodes = ["student_agent_1", "student_agent_2", "student_agent_3"]
    sends: list[Send] = []

    for agent_node, persona in zip(agent_nodes, state["personas"]):
        sends.append(
            Send(
                agent_node,
                {
                    "student": persona,
                    "agent_name": agent_node,
                    "question": state["question"],
                    "subject": state["subject"],
                    "answer": "",
                    "feedback": "",
                    "gt_answer": state.get("gt_answer", ""),
                    "judge_score": None,
                },
            )
        )

    return sends

# AGENT 1, 2, 3
def answer_as_student_persona(state: PersonaWorkerState) -> dict:
    """Answer the question while acting as one student persona."""

    checkpoint("-----------------")
    checkpoint(f"Node start: {state.get('agent_name', 'student_agent')}")

    student = state["student"]
    subjects = ", ".join(state.get("subject", [])) or "general classwork"

    messages = [
        SystemMessage(
            content=f"""
            You are acting as this student persona:
            gender: {student.gender}
            age: {student.age}
            learning_stype: {student.learning_stype}
            comm_style: {student.comm_style}
            negative_traits: {student.negative_traits}
            positive_traits: {student.positive_traits}

            Answer as this student would answer in a typical classroom setting. 
            Reflect both strengths and weaknesses of such student through your words and answering style. 
            Do not mention that you are an AI or that this is a role.
            """
        ),
        HumanMessage(
            # subject and question
            content=f"""
            Subject(s): {subjects}.        
            Question: {state["question"]}
            """
        ),
    ]


    answer = llm.invoke(messages).content
    checkpoint(f"Answer LLM complete for {state.get('agent_name', 'student_agent')}")

    qna_result: QnAState = {
        "question": state["question"],
        "subject": state["subject"],
        "answer": answer,
        "feedback": "",
        "gt_answer": state.get("gt_answer", ""),
        "judge_score": None,
    }

    return {
        "qna_results": [
            {
                "student": student,
                "qna_result": qna_result,
            }
        ]
    }

# Combining all the result
def aggregate_qna_results(state: GraphState) -> dict:
    """Create a readable final response from all persona answers."""

    checkpoint("-----------------")
    checkpoint("Node start: aggregate_qna_results")
    sections = []
    for index, item in enumerate(state["qna_results"], start=1):
        student = item["student"]
        qna = item["qna_result"]

        if student is None or qna is None:
            continue

        sections.append(
            f"## Persona {index}\n"
            f"- Gender: {student.gender}\n"
            f"- Age: {student.age}\n"
            f"- Learning style: {student.learning_stype}\n"
            f"- Communication style: {student.comm_style}\n"
            f"- Negative traits: {student.negative_traits}\n"
            f"- Positive traits: {student.positive_traits}\n\n"
            f"Question: {qna['question']}\n\n"
            f"Answer:\n{qna['answer']}"
        )

    checkpoint("Node complete: aggregate_qna_results")
    return {"final_response": "\n\n".join(sections)}


# GENERATE FEEDBACK 
def load_feedback_system_prompt() -> str:
    """Load the external system prompt used by the feedback LLM."""

    return FEEDBACK_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def evaluate_answers_against_ground_truth(state: GraphState) -> dict:
    """Generate word-only feedback for each answer.

    The feedback LLM receives only the external system prompt plus the student
    answer and ground-truth answer.
    It does not receive persona
    """

    checkpoint("-----------------")
    checkpoint("Node start: evaluate_answers_against_ground_truth")

    system_prompt = load_feedback_system_prompt()
    evaluated_results: list[aggQnAState] = []

    for index, item in enumerate(state["qna_results"], start=1):
        checkpoint(f"Evaluating answer {index}/{len(state['qna_results'])}")
        student = item["student"]
        qna = item["qna_result"]


        if qna is None:
            evaluated_results.append(item)
            continue

        feedback = ""
        if qna.get("gt_answer"):
            feedback = feedback_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"Student answer:\n{qna['answer']}\n\n"
                            f"Ground-truth answer:\n{qna['gt_answer']}"
                            f"Original question:\n{qna['question']}"
                            f"Subject that question belongs to:\n{qna['subject']}"
                        )
                    ),
                ]
            ).content
            checkpoint(f"Feedback LLM complete for answer {index}")

        evaluated_qna: QnAState = {
            "question": qna["question"],
            "subject": qna["subject"],
            "answer": qna["answer"],
            "feedback": feedback,
            "gt_answer": qna["gt_answer"],
            "judge_score": None,
        }
        evaluated_results.append(
            {
                "student": student,
                "qna_result": evaluated_qna,
            }
        )
    checkpoint("Node complete: evaluate_answers_against_ground_truth")

    return {"evaluated_qna_results": evaluated_results}


def judge_feedback_quality(state: GraphState) -> dict:
    """Judge how well the feedback LLM responded, scoring feedback from 1 to 10."""

    checkpoint("-----------------")
    checkpoint("Node start: judge_feedback_quality")

    judge = judge_llm.with_structured_output(FeedbackJudgeResult)
    judged_results: list[aggQnAState] = []
    results = state.get("evaluated_qna_results") or state["qna_results"]

    for index, item in enumerate(results, start=1):
        checkpoint(f"Judging feedback {index}/{len(results)}")
        student = item["student"]
        qna = item["qna_result"]

        if qna is None:
            judged_results.append(item)
            continue

        judge_result = None
        if qna.get("feedback"):
            # TODO: CHANGE TO THE NEWER SYSTEM PROMPT AFTER EVALUATING WHICH ONE 
            judge_result = judge.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are judging educational feedback quality. "
                            "Score how useful, accurate, clear, and actionable "
                            "the feedback is from 1 to 10."
                        )
                    ),
                    HumanMessage(
                        content=f"""
                        Original question: 
                        {qna["question"]}        

                        Subject of the question: 
                        {qna["subject"]}                 

                        Student answer:
                        {qna["answer"]}

                        Ground-truth answer:
                        {qna["gt_answer"]}

                        Feedback to judge:
                        {qna["feedback"]}
                        """
                    ),
                ]
            )
            checkpoint(f"Judge LLM complete for feedback {index}")

        judged_qna: QnAState = {
            "question": qna["question"],
            "subject": qna["subject"],
            "answer": qna["answer"],
            "feedback": qna["feedback"],
            "gt_answer": qna["gt_answer"],
            "judge_score": judge_result.model_dump() if judge_result else None,
        }
        judged_results.append(
            {
                "student": student,
                "qna_result": judged_qna,
            }
        )
    checkpoint("Node complete: judge_feedback_quality")
    return {"evaluated_qna_results": judged_results}


# BUILD GRAPH
def build_graph():
    """Build and compile the full LangGraph workflow."""

    checkpoint("-----------------")
    checkpoint("Building LangGraph workflow")
    workflow = StateGraph(GraphState)

    # step 1: generate the personas
    workflow.add_node("generate_student_personas", generate_student_personas)
    # step 2: route this to different agents to generate answers from those personas
    workflow.add_node("student_agent_1", answer_as_student_persona)
    workflow.add_node("student_agent_2", answer_as_student_persona)
    workflow.add_node("student_agent_3", answer_as_student_persona)
    # step 3: giving feedback for the personas' answers
    workflow.add_node(
        "evaluate_answers_against_ground_truth",
        evaluate_answers_against_ground_truth,
    )
    # step 4: judge the feedback quality if it is helpful
    workflow.add_node("judge_feedback_quality", judge_feedback_quality)
    workflow.add_node("aggregate_qna_results", aggregate_qna_results)

    workflow.add_edge(START, "generate_student_personas")
    workflow.add_conditional_edges(
        "generate_student_personas",
        route_personas_to_agents,
        ["student_agent_1", "student_agent_2", "student_agent_3"],
    )
    workflow.add_edge(
        ["student_agent_1", "student_agent_2", "student_agent_3"],
        "evaluate_answers_against_ground_truth",
    )
    workflow.add_edge("evaluate_answers_against_ground_truth", "judge_feedback_quality")
    workflow.add_edge("judge_feedback_quality", "aggregate_qna_results")
    workflow.add_edge("aggregate_qna_results", END)

    compiled_graph = workflow.compile()
    checkpoint("LangGraph workflow compiled")

    return compiled_graph

student_characteristics_graph = build_graph()

def save_graph_visualization(path: str = "outputs/student_characteristics_graph.png") -> None:
    png_bytes = student_characteristics_graph.get_graph().draw_mermaid_png()
    Path(path).write_bytes(png_bytes)


if __name__ == "__main__":
    checkpoint("CLI run started")
    example_state = {
        "question": "Why is a for loop useful in Python?",
        "subject": ["introductory Python programming"],
        "answer": "",
        "feedback": "",
        "gt_answer": "A for loop repeats a block of code for each item in a sequence.",
        "judge_score": None,
        "used_persona_signatures": [],
        "qna_results": [],
        "evaluated_qna_results": [],
    }

    result = student_characteristics_graph.invoke(example_state)
    print(result["final_response"])

    # Reuse this list in later calls to avoid regenerating the same persona types.
    already_used = result["used_persona_signatures"]
    checkpoint("CLI run finished")