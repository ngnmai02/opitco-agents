# opitco-agents
Testing set up for multi agent for Opitco case in EUSAiR 

## About the experiment
This experiment simulates a user survey experiment to evaluate the AI feedback system. This repository provides a PoC of how we would set up such experiment. 
We set up multi-agents system to simulate various style of students' answers and thus, feed these answers to the AI feedback system. The 

## Testing scope
- The experiment focuses on the aspect of Fairness and Robustness of the system.
- The generated students personas are defined by certain categories:
    - Gender
    - Age
    - Learning style
    - Communication style
    - Positive / Negative traits
    - (To be implemented) Disabilities, english skills
- The evaluation metric in the PoC is minimized for the purpose of demonstration.
- Currently, we aim to use LLM-as-a-judge to conduct the evaluating process. The metrics will be provied detailedly. 
- System prompt of LLM-as-a-judge will be finalized based on a separated test.

## Set up
![alt text](student_characteristics_graph.png)

### Tech stack
- Langgraph, langchain
- uv
- Aitta backend API from CSC
- LLM in PoC: GPT OSS 120B

## Dataset
- 
## Evaluation Framework

We evaluate the system Performance across the following dimensions

### Accuracy
- Exact Match (EM)
- F1 Score
- Semantic Similarity

### Robustness
- Paraphrase consistency (performance under reworded questions)
- Noise robustness (handling typos and incomplete inputs)
- Score stability under input variation

### Fairness
- Persona score gap (difference in grading across student personas)
- Same-answer bias (checking if identical answers receive different scores)
- Error rate parity across personas

### Consistency
- Score variance across repeated evaluations

## TODOs
- [ ] Looping agentic system 
- [ ] Setting up the datasets
- [ ] Setting up the evaluation metrics 
- [ ] Evaluating the LLM-as-a-judge and setting up system prompt