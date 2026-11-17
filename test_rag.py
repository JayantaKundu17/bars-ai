
from backend.legal_assistant import ask_legal_question

question = """
My landlord is refusing to return my security
deposit of Rs. 15000 after I vacated the apartment.

What legal options might be available to me?
"""

print("\n========== BARS AI LEGAL ASSISTANT ==========\n")

answer = ask_legal_question(question)

print(answer)

print("\n=============================================")

