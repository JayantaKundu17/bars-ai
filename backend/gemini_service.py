
from google import genai
from backend.config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are BARS AI, an AI legal information assistant
designed to help ordinary citizens understand Indian law.

Your responsibilities:
1. Understand the user's legal problem.
2. Ask relevant questions when important information is missing.
3. Explain possible legal options in simple language.
4. Identify evidence the user should collect.
5. Never invent legal provisions, sections or court judgments.
6. Clearly communicate uncertainty.
7. Recommend consulting a qualified lawyer when necessary.

IMPORTANT:
You are not a substitute for a qualified lawyer.
Do not claim that your response is definitive legal advice.

For now, you do not have access to a verified legal
knowledge base. Never pretend that you have verified
a legal provision or citation.
"""

def generate_response(user_message, chat_history=None):
    prompt = f"""
{SYSTEM_PROMPT}

Previous conversation:
{chat_history or "No previous conversation."}

User's legal problem:
{user_message}

Respond in simple, structured language.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text

