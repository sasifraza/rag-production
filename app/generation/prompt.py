from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. \
Answer the user's question using ONLY the context below. \
If the context doesn't contain the answer, say \
"I don't have enough information to answer that."

Context:
{context}"""),
    ("human", "{question}"),
])


def format_context(documents) -> str:
    """Format retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source {i+1}] {doc.page_content}"
        for i, doc in enumerate(documents)
    )