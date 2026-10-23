
import os

from google import genai

from src.retrieval_gemini import search_policies


def create_chatbot(
    documents_df,
    embedding_client,
    faiss_index,
    api_key=None,
    model_name="gemini-3.5-flash-lite"
):
    """Create a policy-grounded chatbot using hosted embeddings."""

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Gemini API key not found.")

    generation_client = genai.Client(api_key=api_key)

    def answer_question(question, history=None):
        if not question or not question.strip():
            return "Please enter a question."

        results = search_policies(
            question,
            documents_df,
            embedding_client,
            faiss_index,
            top_k=3
        )

        policy_context = []
        for _, row in results.iterrows():
            matching = documents_df.loc[
                documents_df["source"] == row["source"], "text"
            ]
            if not matching.empty:
                policy_context.append(
                    f"Source: {row['source']}\n{matching.iloc[0]}"
                )

        context_text = "\n\n".join(policy_context)
        history_text = ""

        for item in (history or [])[-6:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str):
                    history_text += f"{role.capitalize()}: {content}\n"
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_text, assistant_text = item
                if user_text:
                    history_text += f"User: {user_text}\n"
                if assistant_text:
                    history_text += f"Assistant: {assistant_text}\n"

        prompt = f"""
You are ShopAssist AI, an e-commerce customer support assistant.

Answer using only the policy documents supplied below. Use conversation
history to understand follow-up questions, but do not treat it as policy.

Rules:
1. Do not invent policies, deadlines, exceptions, or guarantees.
2. If the policies do not answer the question, say so clearly.
3. Give concise, helpful next steps when supported by the policies.
4. Cite relevant policy filenames, for example (Source: returns_policy.md).
5. Do not claim to have completed an action or contacted support.
6. Never ask for passwords or full card numbers.

Relevant policy documents:
{context_text if context_text else "No relevant policy documents were retrieved."}

Recent conversation:
{history_text if history_text else "No previous conversation."}

Current customer question:
{question.strip()}
"""

        try:
            response = generation_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            answer = response.text
            return answer if answer else (
                "I couldn't generate an answer. Please try again."
            )
        except Exception as error:
            error_text = str(error).lower()
            if "429" in error_text or "resource_exhausted" in error_text:
                return (
                    "The AI service has reached its current usage limit. "
                    "Please try again later."
                )
            return (
                "Sorry, I couldn't process your question right now. "
                "Please try again later."
            )

    return answer_question
