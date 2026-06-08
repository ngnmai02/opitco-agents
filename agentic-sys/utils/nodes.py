from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from config import *
from state import *
from pydantic import BaseModel, Field


llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://aitta-api.csc.fi/openai/v1",
    model="openai/gpt-oss-120b"
)

# ----------------------------
# functions

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

    generator = llm.with_structured_output(PersonasGroup)
    used_signatures = set(state.get("used_persona_signatures", []))
    prior_personas = "\n".join(sorted(used_signatures)) or "None yet."

    accepted: list[StudentPersonas] = []
    number_of_attempts = 5
    for attempt in range(number_of_attempts):
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
        accepted = unique_personas(generated.groups, used_signatures)
        if len(accepted) == number_of_attempts:
            break

        used_signatures.update(persona_signature(persona) for persona in accepted)
        prior_personas = "\n".join(sorted(used_signatures))

    if len(accepted) != number_of_attempts:
        raise ValueError(
            "Could not generate three unique student personas after three attempts."
        )

    new_signatures = [persona_signature(persona) for persona in accepted]
    return {
        "personas": accepted,
        "used_persona_signatures": state.get("used_persona_signatures", []) + new_signatures,
    }
