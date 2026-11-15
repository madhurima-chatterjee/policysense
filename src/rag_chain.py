"""
Step 3 - RAG chain
question (+ chat history) -> retrieve chunks -> Groq LLM -> answer + sources

Run:  python -m src.rag_chain
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, TOP_K, MAX_HISTORY_TURNS, NO_ANSWER_MESSAGE,
)
from src.ingest import get_vector_store

NO_ANSWER_TOKEN = "NO_ANSWER"

SYSTEM_PROMPT = f"""You are PolicySense, an HR policy assistant for Nexora Technologies employees.

RULES:
1. Answer ONLY using the policy excerpts given in the CONTEXT of the user's message.
2. If the context does not contain the answer, reply with exactly: {NO_ANSWER_TOKEN}
   Do not guess and do not use outside knowledge.
3. Be clear and concise. Include exact numbers, limits and timelines from the policy.
4. If the answer has conditions or exceptions (for example manager approval), mention them.
5. Do not mention the words "context" or "excerpt" in your answer, and do not write source
   names or page numbers (they are added separately).
6. Reply in English. Use short bullet points only when listing several items."""


class PolicyAssistant:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
        self.store = get_vector_store()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

    # ---------- memory: turn a follow-up into a standalone question ----------
    def _standalone_question(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        recent = history[-2 * MAX_HISTORY_TURNS:]
        convo = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:500]}" for m in recent
        )
        prompt = (
            "Given the conversation and the follow-up question, rewrite the follow-up "
            "as one standalone question that can be understood without the conversation. "
            "If it is already standalone, return it unchanged. "
            "Return ONLY the question.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up question: {question}\n\nStandalone question:"
        )
        rewritten = self.llm.invoke(prompt).content.strip()
        return rewritten or question

    # ---------- main entry point ----------
    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        """
        history = list of {"role": "user" | "assistant", "content": "..."}
        Returns {"answer": str, "sources": [{"source", "page"}], "found": bool}
        """
        history = history or []
        try:
            standalone = self._standalone_question(question, history)
            docs = self.store.similarity_search(standalone, k=TOP_K)

            context = "\n\n".join(
                f"[{d.metadata['source']}, Page {d.metadata['page']}]\n{d.page_content}"
                for d in docs
            )

            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for m in history[-2 * MAX_HISTORY_TURNS:]:
                cls = HumanMessage if m["role"] == "user" else AIMessage
                messages.append(cls(content=m["content"]))
            messages.append(HumanMessage(
                content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"
            ))

            answer = self.llm.invoke(messages).content.strip()

        except Exception as e:  # network error, rate limit, bad key...
            return {
                "answer": ("Sorry, I could not reach the AI service right now "
                           f"({type(e).__name__}). Please try again in a minute."),
                "sources": [], "found": False, "error": str(e),
            }

        if NO_ANSWER_TOKEN in answer or not answer:
            return {"answer": NO_ANSWER_MESSAGE, "sources": [], "found": False}

        # unique (source, page) pairs, in ranking order, max 3
        seen, sources = set(), []
        for d in docs:
            key = (d.metadata["source"], d.metadata["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"source": key[0], "page": key[1]})
        return {"answer": answer, "sources": sources[:3], "found": True}


if __name__ == "__main__":
    bot = PolicyAssistant()
    history: list[dict] = []

    test_questions = [
        "How many WFH days can I take per month?",
        "Is that allowed during probation?",       # follow-up: tests memory
        "What is the hotel limit in a metro city for grade L3?",
        "Who is the CEO of Nexora?",               # not in documents: tests guardrail
    ]
    for q in test_questions:
        print("\n" + "=" * 60)
        print("Q:", q)
        result = bot.ask(q, history)
        print("A:", result["answer"])
        for s in result["sources"]:
            print(f"   Source: {s['source']} - Page {s['page']}")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"]})