import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


MODEL_NAME = "llama-3.1-8b-instant"

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_documents(folder_path):
    documents = []

    for file_path in folder_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        sections = text.split("\n\n")

        for i, section in enumerate(sections):
            clean_section = section.strip()

            if clean_section:
                documents.append(
                    {
                        "source": file_path.name,
                        "chunk_id": i + 1,
                        "text": clean_section
                    }
                )

    return pd.DataFrame(documents)


def retrieve_context(question, docs_df, top_k=2):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2)
    )

    corpus = docs_df["text"].tolist()
    vectors = vectorizer.fit_transform(corpus + [question])

    doc_vectors = vectors[:-1]
    question_vector = vectors[-1]

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()

    if similarities.max() == 0:
        question_words = set(
            question.lower()
            .replace("?", "")
            .replace(".", "")
            .replace(",", "")
            .split()
        )

        def keyword_overlap(text):
            text_words = set(
                text.lower()
                .replace(":", "")
                .replace(".", "")
                .replace(",", "")
                .split()
            )
            overlap_count = len(question_words.intersection(text_words))
            total_question_words = max(len(question_words), 1)
            return overlap_count / total_question_words

        docs_df = docs_df.copy()
        docs_df["similarity"] = docs_df["text"].apply(keyword_overlap)

    else:
        docs_df = docs_df.copy()
        docs_df["similarity"] = similarities

    top_docs = docs_df.sort_values("similarity", ascending=False).head(top_k)

    context = "\n\n".join(top_docs["text"].tolist())
    retrieval_score = top_docs["similarity"].max()

    return context, top_docs, retrieval_score


def classify_safety(question):
    question_lower = question.lower()

    needs_support_words = [
        "stressed", "stress", "anxious", "anxiety", "sad",
        "overwhelmed", "pressure", "worried", "scared",
        "lonely", "afraid", "confused", "upset"
    ]

    out_of_scope_words = [
        "hack", "steal", "illegal", "gambling", "betting",
        "password of someone", "cheat exam", "fake certificate"
    ]

    if any(word in question_lower for word in out_of_scope_words):
        return "out_of_scope"

    if any(word in question_lower for word in needs_support_words):
        return "needs_support"

    return "safe_general"


def generate_answer(question, context, safety_label):
    system_prompt = """
You are a safe, supportive assistant for adolescents.

Rules:
- Answer using only the approved context.
- Be brief, kind, and age-appropriate.
- Do not claim to be a doctor, therapist, lawyer, or emergency service.
- If the user seems stressed or needs support, encourage them to speak with a trusted adult, teacher, counselor, parent, guardian, or qualified professional.
- If the answer is not in the approved context, say you do not have enough approved information.
- Do not give unsafe, illegal, or harmful instructions.
"""

    user_prompt = f"""
Safety label:
{safety_label}

Approved context:
{context}

User question:
{question}

Write a helpful answer grounded in the approved context.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": user_prompt.strip()
                }
            ],
            temperature=0.2,
            max_tokens=250
        )

        return response.choices[0].message.content

    except Exception as e:
        print("\nGroq API failed. Using local fallback response.")
        print("API error:", e)

        if safety_label == "needs_support":
            return (
                "Exam stress is common, and you do not have to handle it alone. "
                "Based on the approved information, helpful steps include making a revision timetable, "
                "taking breaks, sleeping well, eating regular meals, and asking for help early. "
                "If the stress feels too heavy, speak with a trusted adult, school counselor, parent, "
                "guardian, or qualified professional."
            )

        if safety_label == "out_of_scope":
            return (
                "I do not have approved information to help with that request. "
                "I can only provide safe, supportive guidance based on the approved knowledge base."
            )

        return (
            "Based on the approved information, you can take small positive steps such as preparing well, "
            "asking questions, joining positive peer groups, learning useful digital skills, and speaking "
            "with trusted people such as teachers, mentors, parents, guardians, or counselors when needed."
        )


def run_rag(question):
    docs_df = load_documents(KNOWLEDGE_BASE_DIR)

    if docs_df.empty:
        raise ValueError("No documents found in data/knowledge_base.")

    context, top_docs, retrieval_score = retrieve_context(question, docs_df)
    safety_label = classify_safety(question)
    answer = generate_answer(question, context, safety_label)

    result = {
        "question": question,
        "model_provider": "Groq",
        "model_name": MODEL_NAME,
        "safety_label": safety_label,
        "retrieval_score": round(float(retrieval_score), 3),
        "retrieved_context": context,
        "answer": answer
    }

    results_df = pd.DataFrame([result])
    results_df.to_csv(OUTPUT_DIR / "rag_single_result_groq.csv", index=False)

    print("\nQUESTION:")
    print(question)

    print("\nMODEL PROVIDER:")
    print("Groq")

    print("\nMODEL NAME:")
    print(MODEL_NAME)

    print("\nSAFETY LABEL:")
    print(safety_label)

    print("\nRETRIEVAL SCORE:")
    print(round(float(retrieval_score), 3))

    print("\nRETRIEVED CONTEXT:")
    print(context)

    print("\nANSWER:")
    print(answer)

    print("\nSaved to outputs/rag_single_result_groq.csv")


if __name__ == "__main__":
    sample_question = "I feel stressed because of exams. What can I do?"
    run_rag(sample_question)