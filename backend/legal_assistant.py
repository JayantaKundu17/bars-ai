
from google import genai
from google.genai import types

from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.hybrid_retrieval_service import search_legal_documents


client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """
You are BARS AI, an Indian legal information assistant.
Help ordinary citizens understand legal issues using retrieved legal documents.

You are not a lawyer and must never claim to provide professional legal representation.

STRICT LEGAL ACCURACY AND CITATION RULES:

1. Use retrieved legal passages as the primary evidence for statutory claims.

2. Every statement describing a specific legal provision, statutory right, duty, penalty, deadline, or procedure must cite the source that directly supports it, using [S1], [S2], etc.

3. A retrieved source is not automatically evidence for the user's question. Cite it only when its actual text supports the specific statement.

4. Never cite a source merely because it appeared in the retrieved context.

5. Never attach multiple source labels to a claim unless each cited source independently supports that claim.

6. Do not use a source to prove that a legal right, remedy, rule, or procedure does not exist.

7. If the retrieved documents do not address an issue, say:
   "The documents available in my current knowledge base do not establish a specific rule on this point."

8. Make clear that this is a limitation of the available sources, not proof that no such law exists.

9. Never invent legal sections, judgments, regulations, remedies, deadlines, or court procedures.

10. Never present general legal knowledge as though it came from retrieved documents.

11. Clearly distinguish between explicit law, possible interpretation, and unverified information.

12. Ask for the user's state or city when local laws may materially affect the answer.

13. Never claim that the retrieved documents contain all applicable or currently valid laws.

14. Use only source labels supplied in the retrieved context.

15. In "Sources used", list only sources actually cited in the answer.

16. If no retrieved source directly supports a legal claim, do not attach a citation to it.

17. Recommend consulting a qualified Indian advocate when the issue requires legal interpretation, urgent action, or professional representation.

RESPONSE FORMAT:

Use this structure when appropriate:

1. Understanding your situation
2. What the available law says
3. What this means for your situation
4. Information still needed
5. Practical next steps
6. Sources used

Use simple English and explain legal terminology.
Keep answers structured and concise.
"""


def format_legal_context(search_results):
    """
    Convert retrieved legal passages into labelled context.
    Include retrieval metadata for transparency.
    """

    if not search_results:
        return "No relevant legal documents were retrieved."

    context_parts = []

    for index, result in enumerate(search_results, start=1):
        source_label = f"S{index}"

        similarity = result.get("similarity")
        bm25_score = result.get("bm25_score")
        rrf_score = result.get("rrf_score")
        retrieval_method = result.get("retrieval_method", "unknown")
        retrieval_sources = result.get("retrieval_sources", [])

        context_parts.append(
            f"""
[{source_label}]
Document: {result.get('document', 'Unknown document')}
Page: {result.get('page', 'Unknown page')}
Retrieval method: {retrieval_method}
Retrieval sources: {retrieval_sources}
Semantic similarity: {similarity if similarity is not None else 'Not available'}
BM25 score: {bm25_score if bm25_score is not None else 'Not available'}
RRF score: {rrf_score if rrf_score is not None else 'Not available'}

Legal text:
{result.get('text', '')}
"""
        )

    return "\n".join(context_parts)


def ask_legal_question(question, chat_history=None):
    """
    Retrieve relevant legal passages and generate
    a grounded response using Gemini.
    """

    if not question or not question.strip():
        return {
            "answer": "Please describe your legal problem.",
            "sources": []
        }

    try:
        # Step 1: Retrieve relevant legal passages
        search_results = search_legal_documents(
            question=question,
            top_k=5
        )

        # Step 2: Format retrieved context
        legal_context = format_legal_context(search_results)

        # Step 3: Prepare conversation history
        history_text = chat_history or "No previous conversation."

        # Step 4: Build the prompt
        prompt = f"""
CONVERSATION HISTORY:

{history_text}


USER'S LEGAL QUESTION:

{question}


RETRIEVED LEGAL DOCUMENTS:

{legal_context}


INSTRUCTIONS:

Answer the user's question using the retrieved documents
and the legal accuracy rules in your system instructions.

Do not assume that the retrieved documents cover every
applicable law.

If the available sources do not establish the answer,
clearly explain what is missing.

Do not fabricate legal provisions, remedies, deadlines,
judgments, or source references.
"""

        # Step 5: Generate response
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2
            )
        )

        answer = response.text or (
            "I could not generate a response. Please try again."
        )

        # Step 6: Prepare source metadata
        sources = []

        for index, result in enumerate(search_results, start=1):
            sources.append({
                "label": f"S{index}",
                "document": result["document"],
                "page": result["page"],
                "similarity": round(result.get("similarity", 0.0), 4),
                "retrieval_method": result.get("retrieval_method", "unknown"),
                "retrieval_sources": result.get("retrieval_sources", []),
                "rrf_score": round(result.get("rrf_score", 0.0), 6),
                "text": result["text"]
            })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as error:
        print("Legal assistant error:", error)

        return {
            "answer": (
                "BARS AI could not process your question "
                "because of a technical error. Please try again."
            ),
            "sources": [],
            "error": str(error)
        }


if __name__ == "__main__":

    test_question = """
    My landlord is refusing to return my security deposit
    of Rs. 15,000 after I vacated the rented apartment.
    What can I do?
    """

    result = ask_legal_question(test_question)

    print("\n========== BARS AI RESPONSE ==========\n")
    print(result["answer"])

    print("\n========== RETRIEVED SOURCES ==========\n")

    for source in result["sources"]:
        print(f"[{source['label']}]")
        print("Document:", source["document"])
        print("Page:", source["page"])
        print("Similarity:", source["similarity"])
        print("-" * 60)
