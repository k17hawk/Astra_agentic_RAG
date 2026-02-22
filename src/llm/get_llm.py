from crewai import LLM
from src.llm.llm_configuration import LLM_CONFIG
def get_llm_agent(agent_name:str):
    model = LLM_CONFIG.get(agent_name, {}).get("model", "groq/llama-3.3-70b-versatile")
    temperature = LLM_CONFIG.get(agent_name, {}).get("MODEL_TEMPERATURE", 0.0)
    llm = LLM(model=model, temperature=temperature)
    return llm

