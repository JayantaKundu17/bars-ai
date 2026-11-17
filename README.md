# BARS AI — AI-Powered Legal Information Assistant

**Understand legal issues. Organise cases. Prepare documents. Practise legal arguments.**


###<a href="https://bars-ai-legal.streamlit.app/">Live Demo</a> ·


## Overview

BARS AI is an AI-powered legal information assistant designed to make Indian legal information more accessible and help users organise their legal matters.

Built using Python, Streamlit and Google's Gemini API, BARS AI combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide context-aware responses grounded in a collection of Indian legal documents.

The platform brings legal research, case organisation, evidence preparation, document drafting and argument practice together in one application.

## Key Features

### 1. Legal Chat
- Ask questions about Indian laws and legal procedures.
- Retrieve relevant passages from legal documents.
- Receive structured, context-aware explanations.

### 2. Build the Case
- Convert unstructured case descriptions into organised case records.
- Identify parties, facts, chronology and legal issues.
- Distinguish stated facts from allegations and disputed information.
- Identify missing information and desired outcomes.

### 3. Case Analysis
- Analyse legal problems based on user-provided facts.
- Identify potential legal issues and possible remedies.
- Examine strengths, weaknesses and missing information.
- Highlight relevant legal provisions from retrieved documents.

### 4. Evidence Checklist
- Organise documents according to dispute categories.
- Identify relevant supporting evidence.
- Track collected and missing documents.

### 5. Document Drafting
- Generate structured first drafts of legal documents.
- Organise case facts into formal document formats.
- Include placeholders for missing information.
- Generate downloadable PDF documents.

### 6. Help Them Argue
- Practise responding to opposing arguments.
- Explore potential objections and counterarguments.
- Receive feedback on clarity, relevance, evidence usage and persuasiveness.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Generative AI | Google Gemini API |
| Embeddings | Gemini Embedding API |
| Semantic Retrieval | Vector-based search |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion (RRF) |
| PDF Processing | pypdf |
| PDF Generation | ReportLab |
| Image Processing | Pillow |
| Numerical Processing | NumPy |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## System Architecture

BARS AI uses a Retrieval-Augmented Generation pipeline to provide responses informed by relevant legal documents.

```text
              USER QUERY
                   |
                   v
           STREAMLIT INTERFACE
                   |
                   v
            QUERY PROCESSING
                   |
          +--------+--------+
          |                 |
          v                 v
     SEMANTIC SEARCH    BM25 SEARCH
     (Vector Search)   (Keyword Search)
          |                 |
          +--------+--------+
                   |
                   v
            HYBRID RETRIEVAL
                   |
                   v
          RECIPROCAL RANK FUSION
                   |
                   v
          RELEVANT LEGAL PASSAGES
                   |
                   v
             GEMINI LLM
                   |
                   v
          CONTEXTUAL RESPONSE
                   |
                   v
               USER
```

### Retrieval Workflow

1. Legal documents are processed and divided into manageable text chunks.
2. Embeddings are generated to support semantic retrieval.
3. BM25 indexes enable keyword-based retrieval.
4. Hybrid retrieval combines results using Reciprocal Rank Fusion (RRF).
5. Relevant legal passages are provided as context to the Gemini model.
6. Gemini generates a response using the retrieved context and user's query.

## Project Structure

```text
bars-ai/
│
├── app.py
│
├── backend/
│   ├── config.py
│   ├── gemini_service.py
│   ├── document_processor.py
│   ├── new_document_processor.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── hybrid_retrieval_service.py
│   ├── legal_assistant.py
│   ├── case_builder.py
│   └── argument_practice.py
│
├── data/
│   ├── legal_documents/
│   ├── new_legal_documents/
│   └── processed/
│
├── assets/
│   └── fonts/
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

- Python
- Git
- Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/JayantaKundu17/bars-ai.git
cd bars-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Keep this file private. Never commit API keys or credentials to GitHub.

For cloud deployment, configure `GEMINI_API_KEY` through Streamlit Community Cloud Secrets.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Deployment

BARS AI is deployed using Streamlit Community Cloud.

**Live Application:**  
https://bars-ai-legal.streamlit.app/

Deployment configuration:
- Repository: `JayantaKundu17/bars-ai`
- Branch: `main`
- Main application file: `app.py`
- API credentials: Streamlit Secrets

## Security and Responsible Use

- API credentials are managed through environment variables or Streamlit Secrets.
- Sensitive configuration files are excluded using `.gitignore`.
- Legal information and AI-generated outputs should be independently verified.
- User-provided allegations should not automatically be treated as established facts.

## Limitations

- Response quality depends on the coverage, accuracy and currency of the available legal documents.
- Legal applicability can vary according to jurisdiction and individual circumstances.
- AI-generated responses may contain errors, omissions or incomplete interpretations.
- A retrieved legal provision does not automatically establish that it applies to a particular case.
- Generated documents are drafts and should be reviewed before formal use.

**BARS AI is an informational and document-preparation tool, not a substitute for advice from a qualified legal professional.**

## Author

**Jayanta Kundu**

GitHub: [@JayantaKundu17](https://github.com/JayantaKundu17)


---

<p align="center">
  <strong>BARS AI</strong><br>
  AI-Powered Legal Information Assistant
</p>
