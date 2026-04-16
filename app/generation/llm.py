from langchain_openai import ChatOpenAI
from config.settings import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a configured OpenAI chat model."""
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,
    )