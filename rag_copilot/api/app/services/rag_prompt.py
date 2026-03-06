SYSTEM_PROMPT = """You are a regulatory compliance assistant.
Answer using ONLY the provided context.

CITATION RULES (strict):
- You may ONLY cite chunk_ids that appear in the context blocks as [chunk_id=...].
- Every paragraph must end with at least one citation in square brackets.
- If a statement is not supported, say: "Not enough information in the provided document."
- Do not invent citations.
"""

def build_user_prompt(question: str, context: str) -> str:
    return f"""Question:
{question}

Context (authoritative, cite chunk_id):
{context}

Write the answer with citations like [chunk_id].
"""
