"""
Step 5 - Evaluation
Runs data/eval/test_questions.csv through the bot and prints accuracy.

Run:  python evaluate.py
"""
import re
import time

import pandas as pd
from langchain_groq import ChatGroq

from src.config import EVAL_FILE, GROQ_API_KEY, GROQ_MODEL
from src.rag_chain import PolicyAssistant

RESULTS_FILE = EVAL_FILE.parent / "results.csv"
PAUSE = 2  # seconds between questions (free tier rate limits)


def norm(text: str) -> str:
    """Lowercase letters and digits only, for comparing policy names."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def ask_with_retry(bot: PolicyAssistant, question: str) -> dict:
    """Retry once after a pause if the API was rate limited."""
    result = bot.ask(question, [])
    if "error" in result:
        print("    (API error, waiting 30s and retrying once)")
        time.sleep(30)
        result = bot.ask(question, [])
    return result


def judge(llm: ChatGroq, question: str, expected: str, actual: str) -> bool:
    prompt = (
        "You are grading a policy chatbot.\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Chatbot answer: {actual}\n\n"
        "Does the chatbot answer contain the key facts and numbers of the expected answer, "
        "without contradicting them? Extra correct detail is fine. "
        "Reply with exactly one word: CORRECT or INCORRECT."
    )
    verdict = llm.invoke(prompt).content.strip().upper()
    return verdict.startswith("CORRECT")


def main():
    df = pd.read_csv(EVAL_FILE)
    bot = PolicyAssistant()
    judge_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

    rows = []
    for _, r in df.iterrows():
        q, expected, policy = r["question"], r["expected_answer"], r["source_policy"]
        out_of_scope = "out of scope" in str(policy).lower()

        result = ask_with_retry(bot, q)
        time.sleep(PAUSE)

        if out_of_scope:
            answer_ok = not result["found"]       # bot must refuse
            source_ok = True
        else:
            try:
                answer_ok = result["found"] and judge(judge_llm, q, expected, result["answer"])
            except Exception as e:
                print("    judge error:", type(e).__name__)
                answer_ok = False
            source_ok = any(norm(policy) == norm(s["source"]) for s in result["sources"])
            time.sleep(PAUSE)

        status = "PASS" if answer_ok else "FAIL"
        print(f"[{status}] #{r['id']:>2} {q}")
        rows.append({
            "id": r["id"], "question": q, "expected": expected,
            "bot_answer": result["answer"], "answer_correct": answer_ok,
            "source_correct": source_ok, "type": "out_of_scope" if out_of_scope else "in_scope",
        })

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_FILE, index=False)

    in_scope = res[res["type"] == "in_scope"]
    out_scope = res[res["type"] == "out_of_scope"]
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Answer accuracy (in-scope)     : {in_scope['answer_correct'].sum()}/{len(in_scope)}"
          f"  ({in_scope['answer_correct'].mean():.0%})")
    print(f"Correct source cited (in-scope): {in_scope['source_correct'].sum()}/{len(in_scope)}"
          f"  ({in_scope['source_correct'].mean():.0%})")
    print(f"Refused out-of-scope questions : {out_scope['answer_correct'].sum()}/{len(out_scope)}")
    print(f"Overall                        : {res['answer_correct'].sum()}/{len(res)}"
          f"  ({res['answer_correct'].mean():.0%})")
    print(f"\nDetails saved to: {RESULTS_FILE}")
    failed = res[~res["answer_correct"]]
    if len(failed):
        print("\nFailed questions:")
        for _, f in failed.iterrows():
            print(f"  #{f['id']}: {f['question']}")


if __name__ == "__main__":
    main()