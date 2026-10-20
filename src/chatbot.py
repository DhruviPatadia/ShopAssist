import os

from google import genai

from src.retrieval import search_policies


def create_chatbot(
    documents_df,
    embedding_model,
    faiss_index,
    api_key=None,
    model_name="gemini-3.5-flash-lite"
):
    """Create a policy-grounded ShopAssist chatbot."""

    api_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Pass api_key explicitly."
        )

    client = genai.Client(api_key=api_key)

    def answer_question(question, history=None):
        if not question or not question.strip():
            return "Please enter a question."

        history = history or []

        # Retrieve relevant policies
        results = search_policies(
            question,
            documents_df,
            embedding_model,
            faiss_index,
            top_k=3
        )

        policy_context = []

        for _, row in results.iterrows():
            policy_text = documents_df.loc[
                documents_df["source"] == row["source"],
                "text"
            ].iloc[0]

            policy_context.append(
                f"Source: {row['source']}\n{policy_text}"
            )

        context_text = "\n\n".join(policy_context)

        # Keep only recent conversation turns
        history_text = ""
        for item in history[-6:]:
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

Answer the customer's question using only the supplied policy documents.
Use the conversation history to understand follow-up questions.

Rules:
1. Do not invent policies, deadlines, exceptions, or guarantees.
2. If the policies do not answer the question, clearly say that the
   available policies do not specify the answer.
3. Give clear, concise, helpful next steps when supported by the policies.
4. Cite the relevant policy filenames in your answer, for example:
   (Source: returns_policy.md).
5. Do not claim an action has been completed or that you contacted support.
6. Never ask the customer to provide passwords or full card numbers.

Relevant policy documents:
{context_text}

Recent conversation:
{history_text if history_text else "No previous conversation."}

Current customer question:
{question.strip()}
"""

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            answer = response.text

            if not answer:
                return "I couldn't generate an answer. Please try again."

            return answer

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
