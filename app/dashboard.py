import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Responsible GenAI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# PATHS AND SETTINGS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
RESULTS_PATH = BASE_DIR / "outputs" / "rag_evaluation_results_groq.csv"

MODEL_NAME = "llama-3.1-8b-instant"


# ---------------------------------------------------------
# LOAD API KEY
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

api_key = None

# Streamlit Community Cloud secret
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    pass

# Local .env fallback
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")


client = None

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """
    Normalize text for retrieval and keyword matching.
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------------------------------

@st.cache_data
def load_documents(folder_path):
    """
    Load text files and split them into separate knowledge chunks.
    """
    documents = []

    if not folder_path.exists():
        return pd.DataFrame()

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


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def retrieve_context(question, docs_df, top_k=2):
    """
    Retrieve the most relevant knowledge-base sections.
    """
    if docs_df.empty:
        raise ValueError("The knowledge base is empty.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2)
    )

    corpus = docs_df["text"].tolist()

    try:
        vectors = vectorizer.fit_transform(corpus + [question])

        doc_vectors = vectors[:-1]
        question_vector = vectors[-1]

        similarities = cosine_similarity(
            question_vector,
            doc_vectors
        ).flatten()

    except ValueError:
        similarities = [0.0] * len(docs_df)

    docs_scored = docs_df.copy()

    if max(similarities) == 0:
        question_words = set(clean_text(question).split())

        def keyword_overlap(text):
            text_words = set(clean_text(text).split())
            overlap_count = len(question_words.intersection(text_words))
            total_question_words = max(len(question_words), 1)
            return overlap_count / total_question_words

        docs_scored["similarity"] = docs_scored["text"].apply(
            keyword_overlap
        )

    else:
        docs_scored["similarity"] = similarities

    top_docs = (
        docs_scored
        .sort_values("similarity", ascending=False)
        .head(top_k)
    )

    context = "\n\n".join(top_docs["text"].tolist())
    retrieval_score = float(top_docs["similarity"].max())

    sources = top_docs[
        ["source", "chunk_id", "similarity"]
    ].to_dict("records")

    return context, retrieval_score, sources


# ---------------------------------------------------------
# SAFETY CLASSIFICATION
# ---------------------------------------------------------

def classify_safety(question):
    """
    Apply simple rule-based safety classification.
    """
    question_lower = clean_text(question)

    needs_support_words = [
        "stressed",
        "stress",
        "anxious",
        "anxiety",
        "sad",
        "overwhelmed",
        "pressure",
        "worried",
        "scared",
        "lonely",
        "afraid",
        "confused",
        "upset",
        "uncomfortable",
        "failing",
        "nimechoka",
        "nina stress",
        "niko na stress"
    ]

    out_of_scope_words = [
        "hack",
        "steal",
        "illegal",
        "gambling",
        "betting",
        "password of someone",
        "cheat exam",
        "fake certificate"
    ]

    if any(term in question_lower for term in out_of_scope_words):
        return "out_of_scope"

    if any(term in question_lower for term in needs_support_words):
        return "needs_support"

    return "safe_general"


def needs_human_review(safety_label, retrieval_score):
    """
    Flag sensitive or low-confidence interactions.
    """
    if safety_label in ["needs_support", "out_of_scope"]:
        return True

    if retrieval_score < 0.10:
        return True

    return False


# ---------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------

def generate_answer(question, context, safety_label):
    """
    Generate a grounded answer using Groq.
    """
    if client is None:
        raise ValueError(
            "Groq API key not configured. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    system_prompt = """
You are a safe and supportive assistant for adolescents.

Follow these rules:
- Answer only using the approved context provided.
- Be clear, brief, respectful, and age-appropriate.
- Do not invent information.
- Do not claim to be a doctor, therapist, lawyer, or emergency service.
- Do not provide unsafe, illegal, harmful, or age-inappropriate instructions.
- When a user appears worried, stressed, afraid, or overwhelmed, encourage them to speak with a trusted adult, teacher, counselor, parent, guardian, or qualified professional.
- If the answer is not available in the approved context, state that there is not enough approved information.
- If the request is outside the approved scope, provide a brief and polite refusal.
"""

    user_prompt = f"""
Safety label:
{safety_label}

Approved context:
{context}

User question:
{question}

Write a grounded and supportive answer.
"""

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
        max_tokens=300
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# COMPLETE RAG PIPELINE
# ---------------------------------------------------------

def run_rag(question):
    """
    Run safety classification, retrieval, generation,
    and human-review assessment.
    """
    docs_df = load_documents(KNOWLEDGE_BASE_DIR)

    if docs_df.empty:
        raise ValueError(
            "No knowledge-base documents were found in "
            "data/knowledge_base."
        )

    safety_label = classify_safety(question)

    context, retrieval_score, sources = retrieve_context(
        question,
        docs_df
    )

    human_review = needs_human_review(
        safety_label,
        retrieval_score
    )

    if safety_label == "out_of_scope":
        answer = (
            "I cannot help with that request. "
            "I can only provide safe guidance using the approved "
            "knowledge base."
        )
    else:
        answer = generate_answer(
            question,
            context,
            safety_label
        )

    return {
        "question": question,
        "answer": answer,
        "safety_label": safety_label,
        "retrieval_score": retrieval_score,
        "needs_human_review": human_review,
        "retrieved_context": context,
        "sources": sources,
        "model_provider": "Groq",
        "model_name": MODEL_NAME
    }


# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------

st.title("Responsible GenAI RAG Chatbot")

st.markdown(
    """
This prototype demonstrates retrieval-augmented generation, safety
classification, human-review routing, and evaluation for
adolescent-support use cases.

The chatbot uses a small approved knowledge base and is intended
only as a technical demonstration.
"""
)

st.warning(
    "This prototype does not replace qualified professional, "
    "safeguarding, counselling, medical, legal, or emergency support."
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

chat_tab, dashboard_tab, about_tab = st.tabs(
    [
        "Ask the Chatbot",
        "Evaluation Dashboard",
        "About the Project"
    ]
)


# ---------------------------------------------------------
# TAB 1: LIVE CHATBOT
# ---------------------------------------------------------

with chat_tab:
    st.subheader("Ask the Chatbot")

    st.markdown(
        """
Example topics:

- Confidence at school
- Exam stress
- Online safety
- Career planning
- Digital skills
- Support seeking
"""
    )

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: I feel stressed because of exams. "
            "What can I do?"
        ),
        height=120
    )

    col1, col2 = st.columns([1, 5])

    with col1:
        submit = st.button(
            "Get Response",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.caption(
            "Do not enter names, contact details, or other "
            "personal information."
        )

    if submit:
        if not question.strip():
            st.warning("Please enter a question.")

        elif len(question.strip()) < 5:
            st.warning("Please enter a more complete question.")

        elif len(question) > 1000:
            st.warning(
                "Please shorten the question to fewer than "
                "1,000 characters."
            )

        else:
            try:
                with st.spinner(
                    "Checking safety, retrieving context, "
                    "and generating a response..."
                ):
                    result = run_rag(question.strip())

                st.markdown("### Chatbot Response")
                st.success(result["answer"])

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Safety Label",
                    result["safety_label"]
                )

                metric2.metric(
                    "Retrieval Score",
                    f"{result['retrieval_score']:.3f}"
                )

                metric3.metric(
                    "Human Review",
                    "Required"
                    if result["needs_human_review"]
                    else "Not Required"
                )

                if result["needs_human_review"]:
                    st.warning(
                        "This interaction was flagged for human review "
                        "because it may be sensitive or have low "
                        "retrieval confidence."
                    )

                with st.expander("View Retrieved Context"):
                    st.write(result["retrieved_context"])

                with st.expander("View Retrieval Sources"):
                    sources_df = pd.DataFrame(result["sources"])

                    if not sources_df.empty:
                        sources_df["similarity"] = (
                            sources_df["similarity"]
                            .astype(float)
                            .round(3)
                        )

                    st.dataframe(
                        sources_df,
                        use_container_width=True
                    )

                with st.expander("View Model Details"):
                    st.write(
                        {
                            "provider": result["model_provider"],
                            "model": result["model_name"],
                            "safety_label": result["safety_label"],
                            "retrieval_score": round(
                                result["retrieval_score"],
                                3
                            ),
                            "needs_human_review": result[
                                "needs_human_review"
                            ]
                        }
                    )

            except Exception as error:
                st.error(
                    "The chatbot could not generate a response."
                )

                st.code(str(error))


# ---------------------------------------------------------
# TAB 2: EVALUATION DASHBOARD
# ---------------------------------------------------------

with dashboard_tab:
    st.subheader("Evaluation Dashboard")

    if not RESULTS_PATH.exists():
        st.error(
            "Evaluation results were not found. "
            "Run scripts/02_batch_rag_evaluation_groq.py first."
        )

    else:
        df = pd.read_csv(RESULTS_PATH)

        total_prompts = len(df)
        safety_accuracy = df["safety_correct"].mean() * 100
        human_review_rate = (
            df["needs_human_review"].mean() * 100
        )
        average_retrieval_score = (
            df["retrieval_score"].mean()
        )
        api_success_rate = (
            (df["generation_status"] == "api_success").mean()
            * 100
        )

        st.markdown("### Key Metrics")

        metric1, metric2, metric3, metric4, metric5 = (
            st.columns(5)
        )

        metric1.metric(
            "Total Prompts",
            total_prompts
        )

        metric2.metric(
            "Safety Accuracy",
            f"{safety_accuracy:.1f}%"
        )

        metric3.metric(
            "Human Review Rate",
            f"{human_review_rate:.1f}%"
        )

        metric4.metric(
            "Average Retrieval Score",
            f"{average_retrieval_score:.3f}"
        )

        metric5.metric(
            "API Success Rate",
            f"{api_success_rate:.1f}%"
        )

        st.markdown("### Filter Results")

        filter1, filter2 = st.columns(2)

        with filter1:
            selected_language = st.selectbox(
                "Language",
                ["All"]
                + sorted(
                    df["language"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        with filter2:
            selected_topic = st.selectbox(
                "Topic",
                ["All"]
                + sorted(
                    df["topic"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

        filtered_df = df.copy()

        if selected_language != "All":
            filtered_df = filtered_df[
                filtered_df["language"]
                == selected_language
            ]

        if selected_topic != "All":
            filtered_df = filtered_df[
                filtered_df["topic"]
                == selected_topic
            ]

        display_columns = [
            "prompt_id",
            "prompt",
            "language",
            "topic",
            "expected_safety_label",
            "predicted_safety_label",
            "safety_correct",
            "needs_human_review",
            "retrieval_score",
            "generation_status"
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True
        )

        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown(
                "### Safety Accuracy by Language"
            )

            language_summary = (
                df.groupby("language")
                .agg(
                    safety_accuracy=(
                        "safety_correct",
                        "mean"
                    )
                )
            )

            language_summary["safety_accuracy"] *= 100

            st.bar_chart(language_summary)

        with chart2:
            st.markdown(
                "### Retrieval Score by Topic"
            )

            topic_summary = (
                df.groupby("topic")
                .agg(
                    average_retrieval_score=(
                        "retrieval_score",
                        "mean"
                    )
                )
            )

            st.bar_chart(topic_summary)

        st.markdown("### Human Review Queue")

        review_df = df[
            df["needs_human_review"] == True
        ]

        if review_df.empty:
            st.success(
                "No prompts were flagged for human review."
            )

        else:
            st.write(
                f"{len(review_df)} prompts were flagged "
                "for human review."
            )

            review_columns = [
                "prompt_id",
                "prompt",
                "language",
                "topic",
                "predicted_safety_label",
                "retrieval_score",
                "answer"
            ]

            st.dataframe(
                review_df[review_columns],
                use_container_width=True
            )

        st.markdown("### Inspect Individual Prompt")

        prompt_ids = df["prompt_id"].tolist()

        selected_prompt_id = st.selectbox(
            "Select prompt ID",
            prompt_ids
        )

        selected_row = df[
            df["prompt_id"] == selected_prompt_id
        ].iloc[0]

        st.markdown("#### User Prompt")
        st.write(selected_row["prompt"])

        st.markdown("#### Retrieved Context")
        st.write(selected_row["retrieved_context"])

        st.markdown("#### Model Answer")
        st.write(selected_row["answer"])

        st.markdown("#### Evaluation Details")

        st.json(
            {
                "language": selected_row["language"],
                "topic": selected_row["topic"],
                "expected_safety_label": selected_row[
                    "expected_safety_label"
                ],
                "predicted_safety_label": selected_row[
                    "predicted_safety_label"
                ],
                "safety_correct": bool(
                    selected_row["safety_correct"]
                ),
                "needs_human_review": bool(
                    selected_row["needs_human_review"]
                ),
                "retrieval_score": float(
                    selected_row["retrieval_score"]
                ),
                "model_provider": selected_row[
                    "model_provider"
                ],
                "model_name": selected_row[
                    "model_name"
                ],
                "generation_status": selected_row[
                    "generation_status"
                ]
            }
        )


# ---------------------------------------------------------
# TAB 3: PROJECT INFORMATION
# ---------------------------------------------------------

with about_tab:
    st.subheader("About the Project")

    st.markdown(
        """
### Project purpose

This project demonstrates how a responsible GenAI application can
combine:

- Retrieval-Augmented Generation
- Approved knowledge sources
- Prompt safety classification
- Large language model integration
- Multilingual and mixed-code evaluation
- Human-review routing
- Performance monitoring

### Current limitations

- The knowledge base is small.
- Retrieval uses TF-IDF and keyword overlap.
- Safety classification is rule-based.
- The evaluation dataset is partly synthetic.
- The system has not been validated for production use.
- Responses have not been reviewed by safeguarding or
  behavioural-science experts.

### Privacy

The application does not require names or personal identifiers.
Visitors should not submit private or identifying information.
"""
    )