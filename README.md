# Responsible GenAI RAG Chatbot

A responsible Retrieval-Augmented Generation chatbot prototype designed for adolescent-support use cases. The project combines an approved knowledge base, Groq-hosted LLM integration, safety classification, multilingual prompt testing, human-review flagging, and a Streamlit evaluation dashboard.

## Live Demo

[Launch the Responsible GenAI RAG Chatbot](https://responsible-genai-rag-chatbot-mvgxfjvdt5fcsl2eu729rg.streamlit.app/)

> The live application allows users to ask questions and inspect the chatbot response, safety label, retrieval score, retrieved context, and human-review status.

## Project Overview

The chatbot follows a safety-aware RAG workflow:

```text
User Prompt
→ Safety Classification
→ Knowledge Base Retrieval
→ Relevant Context
→ Groq LLM Response Generation
→ Human Review Check
→ Evaluation Dashboard
```

The system is designed to demonstrate how responsible AI controls can be incorporated into chatbot applications intended for young and potentially vulnerable users.

## Key Features

* Interactive RAG chatbot
* Groq API integration using Llama 3.1 8B Instant
* Approved local knowledge base
* TF-IDF and cosine-similarity retrieval
* Keyword-overlap fallback retrieval
* Safety classification
* Human-review routing
* English, Kiswahili, and mixed-code prompt testing
* Batch chatbot evaluation
* Streamlit monitoring dashboard
* API failure handling
* Secure environment-variable management

## Safety Categories

The chatbot classifies prompts into:

* `safe_general` — normal questions covered by the knowledge base
* `needs_support` — questions showing stress, fear, worry, loneliness, or pressure
* `out_of_scope` — unsafe, illegal, harmful, or unsupported requests

Sensitive and low-confidence interactions are flagged for human review.

## Evaluation Dashboard

The Streamlit dashboard displays:

* total prompts tested
* safety-classification accuracy
* human-review rate
* average retrieval score
* API-success rate
* performance by language
* retrieval performance by topic
* human-review queue
* individual prompt inspection

## Knowledge Base Topics

The current approved knowledge base covers:

* confidence at school
* exam stress
* online safety
* career planning
* digital skills
* support seeking

## Technologies

* Python
* Groq API
* Llama 3.1 8B Instant
* Pandas
* Scikit-learn
* TF-IDF
* Cosine Similarity
* Streamlit
* Git and GitHub

## Project Structure

```text
Responsible-GenAI-RAG-Chatbot/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── knowledge_base/
│   │   └── adolescent_support_notes.txt
│   └── test_prompts.csv
│
├── outputs/
│   ├── rag_evaluation_results_groq.csv
│   ├── rag_single_result_groq.csv
│   ├── safety_metrics_summary.csv
│   └── safety_metrics_summary.xlsx
│
├── scripts/
│   ├── 01_basic_rag_groq.py
│   ├── 02_batch_rag_evaluation_groq.py
│   └── 03_generate_metrics_summary.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/craigthompsonotieno/Responsible-GenAI-RAG-Chatbot.git
cd Responsible-GenAI-RAG-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not upload `.env` to GitHub.

### 4. Run the application

```bash
python -m streamlit run app/dashboard.py
```

### 5. Run batch evaluation

```bash
python scripts/02_batch_rag_evaluation_groq.py
```

### 6. Generate evaluation summaries

```bash
python scripts/03_generate_metrics_summary.py
```

## Responsible AI Considerations

This prototype is not intended to replace professional safeguarding, counselling, medical, legal, or emergency support.

The project emphasizes:

* grounded responses based on approved information
* age-appropriate and supportive language
* controlled refusal of unsafe requests
* human review of sensitive interactions
* transparent evaluation metrics
* protection of API credentials
* documentation of system limitations

No real personal data from adolescents was used. The evaluation prompts were created for testing and demonstration purposes.

## Current Limitations

* Small knowledge base
* Rule-based safety classification
* Limited Kiswahili and mixed-code coverage
* TF-IDF retrieval rather than semantic embeddings
* Small and partly synthetic evaluation dataset
* No expert safeguarding validation
* Prototype-level deployment

## Future Improvements

* Add multilingual sentence embeddings
* Introduce FAISS or ChromaDB
* Expand the approved knowledge base
* Add automated faithfulness and relevance evaluation
* Track latency, token usage, and API costs
* Compare multiple LLM providers
* Improve multilingual safety classification
* Add prompt and model version tracking
* Conduct expert-led safeguarding evaluation

## Author

**Craig Thompson Otieno**

* Portfolio: https://craigthompsonotieno.github.io/portfolio/
* LinkedIn: https://www.linkedin.com/in/craigthompsonotieno/
* GitHub: https://github.com/craigthompsonotieno
