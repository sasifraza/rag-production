from langchain_openai import OpenAIEmbeddings
from config.settings import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Return a configured OpenAI embedding model."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )