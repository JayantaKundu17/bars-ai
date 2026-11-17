
from backend.gemini_service import generate_response

question = """
My landlord is refusing to return my security deposit
of Rs. 15000 after I vacated the apartment.
What can I do?
"""

answer = generate_response(question)

print("\n========== BARS AI RESPONSE ==========\n")
print(answer)
print("\n======================================")

