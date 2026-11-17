"""
BARS AI retrieval evaluation
----------------------------
Run from the project root:
    python test_retrieval.py

This evaluates more than whether the expected Act appears. It checks:
1. Whether the expected Act appears in the top 5.
2. Whether at least one expected statutory provision appears in the top 5.
3. Whether a relevant provision appears in the top 3 / at rank 1.

This is a retrieval diagnostic, not a legal-accuracy certification.
Some questions can involve more than one provision. The expected section
patterns below are deliberately limited to the specific question asked.
"""

import re
from backend.hybrid_retrieval_service import search_legal_documents


TESTS = [
    {
        "act": "Consumer Protection Act",
        "question": "How can a consumer file a complaint before the District Consumer Commission?",
        "expected_document": "consumer protection act",
        "expected_sections": [r"\bsection\s*35\b", r"\b35\s*\.\s"],
        "notes": "Section 35: manner in which complaint shall be made.",
    },
    {
        "act": "Indian Contract Act",
        "question": "When is an agreement void under the Indian Contract Act?",
        "expected_document": "indian contract act",
        "expected_sections": [
            r"\bsection\s*(?:2\s*\(\s*g\s*\)|24|25|26|27|28|29|30)\b",
            r"\b(?:24|25|26|27|28|29|30)\s*\.\s",
        ],
        "notes": "Void agreement definition / provisions on agreements declared void.",
    },
    {
        "act": "Transfer of Property Act",
        "question": "What are the rights and liabilities of a lessor and lessee?",
        "expected_document": "transfer of property act",
        "expected_sections": [r"\bsection\s*108\b", r"\b108\s*\.\s"],
        "notes": "Section 108: rights and liabilities of lessor and lessee.",
    },
    {
        "act": "Bharatiya Nyaya Sanhita",
        "question": "What is the punishment for theft under Bharatiya Nyaya Sanhita?",
        "expected_document": "bharatiya nyaya sanhita",
        "expected_sections": [r"\bsection\s*303\b", r"\b303\s*\.\s"],
        "notes": "Section 303: theft and punishment.",
    },
    {
        "act": "Bharatiya Nagarik Suraksha Sanhita",
        "question": "What is the procedure for granting bail in a bailable offence?",
        "expected_document": "bharatiya nagarik suraksha sanhita",
        "expected_sections": [r"\bsection\s*478\b", r"\b478\s*\.\s"],
        "notes": "Section 478: bail in bailable cases.",
    },
    {
        "act": "Bharatiya Sakshya Adhiniyam",
        "question": "When is an electronic record admissible as evidence?",
        "expected_document": "bharatiya sakshya adhiniyam",
        "expected_sections": [r"\bsection\s*63\b", r"\b63\s*\.\s"],
        "notes": "Section 63: admissibility of electronic records.",
    },
    {
        "act": "Motor Vehicles Act",
        "question": "What documents must a driver produce when asked by a police officer?",
        "expected_document": "motor vehicles act",
        "expected_sections": [r"\bsection\s*130\b", r"\b130\s*\.\s"],
        "notes": "Section 130: production of licence and certificate of registration.",
    },
    {
        "act": "Information Technology Act",
        "question": "What is the punishment for identity theft?",
        "expected_document": "information technology act",
        "expected_sections": [r"\bsection\s*66c\b", r"\b66c\s*\.\s"],
        "notes": "Section 66C: punishment for identity theft.",
    },
    {
        "act": "Limitation Act",
        "question": "What is the limitation period for filing a suit for recovery of money?",
        "expected_document": "limitation act",
        "expected_sections": [
            r"\barticle\s*19\b",
            r"\barticle\s*55\b",
            r"\barticle\s*113\b",
        ],
        "notes": (
            "Money-recovery limitation depends on the cause of action and the "
            "applicable Schedule article. This test accepts several potentially "
            "relevant articles and does not assume one universal limitation period."
        ),
    },
    {
        "act": "Specific Relief Act",
        "question": "When can specific performance of a contract be enforced?",
        "expected_document": "specific relief act",
        "expected_sections": [r"\bsection\s*10\b", r"\b10\s*\.\s"],
        "notes": "Section 10: specific performance in accordance with the Act.",
    },
    {
        "act": "Rights of Persons with Disabilities Act",
        "question": "What protections does the law provide against discrimination on grounds of disability?",
        "expected_document": "rights of persons with disabilities act",
        "expected_sections": [r"\bsection\s*3\b", r"\b3\s*\.\s"],
        "notes": "Section 3: equality, non-discrimination and reasonable accommodation.",
    },
    {
        "act": "Arbitration and Conciliation Act",
        "question": "How can an arbitral award be set aside?",
        "expected_document": "arbitration and conciliation act",
        "expected_sections": [r"\bsection\s*34\b", r"\b34\s*\.\s"],
        "notes": "Section 34: application for setting aside arbitral award.",
    },
    {
        "act": "Companies Act",
        "question": "What duties must a company director follow?",
        "expected_document": "companies act",
        "expected_sections": [r"\bsection\s*166\b", r"\b166\s*\.\s"],
        "notes": "Section 166: duties of directors.",
    },
    {
        "act": "Hindu Marriage Act",
        "question": "What are the grounds for divorce under the Hindu Marriage Act?",
        "expected_document": "hindu marriage act",
        "expected_sections": [r"\bsection\s*13\b", r"\b13\s*\.\s"],
        "notes": "Section 13: grounds for divorce; Section 13B concerns mutual consent.",
    },
    {
        "act": "Hindu Succession Act",
        "question": "Who are the Class I heirs under the Hindu Succession Act?",
        "expected_document": "hindu succession act",
        "expected_sections": [
            r"\bclass\s*i\b",
            r"\bschedule\b",
            r"\bsection\s*8\b",
            r"\bsection\s*10\b",
        ],
        "notes": "Section 8 and the Schedule identify heirs; Section 10 gives distribution rules.",
    },
    {
        "act": "Negotiable Instruments Act",
        "question": "What happens when a cheque bounces?",
        "expected_document": "negotiable instruments act",
        "expected_sections": [r"\bsection\s*138\b", r"\b138\s*\.\s"],
        "notes": "Section 138: dishonour of cheque; check the notice and 15-day payment conditions.",
    },
    {
        "act": "Prevention of Money Laundering Act",
        "question": "What constitutes the offence of money laundering?",
        "expected_document": "prevention of money laundering act",
        "expected_sections": [r"\bsection\s*3\b", r"\b3\s*\.\s"],
        "notes": "Section 3: offence of money-laundering.",
    },
    {
        "act": "Real Estate (Regulation and Development) Act",
        "question": "How can an allottee file a complaint against a real estate promoter?",
        "expected_document": "real estate",
        "expected_sections": [r"\bsection\s*31\b", r"\b31\s*\.\s"],
        "notes": "Section 31: filing complaints with the Authority or adjudicating officer.",
    },
]


TOP_K = 5


def normalise_document_name(value):
    value = (value or "").lower()
    value = re.sub(r"[_\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def section_match(text, patterns):
    text = text or ""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def main():
    act_passed = 0
    section_passed = 0
    top3_section_passed = 0
    first_rank_section_passed = 0
    errors = 0

    print("\nBARS AI — RETRIEVAL QUALITY EVALUATION")
    print("Top-k:", TOP_K)
    print("Tests:", len(TESTS))

    for number, test in enumerate(TESTS, 1):
        print("\n" + "=" * 96)
        print(f"TEST {number}/{len(TESTS)}: {test['act']}")
        print("QUESTION:", test["question"])
        print("EXPECTED:", test["notes"])

        try:
            results = search_legal_documents(test["question"], top_k=TOP_K)
            if not results:
                print("ACT: FAIL | SECTION: FAIL | TOP 3: FAIL")
                print("No passages retrieved.")
                errors += 1
                continue

            act_ranks = []
            section_ranks = []

            for rank, result in enumerate(results, 1):
                document = normalise_document_name(result.get("document", ""))
                text = result.get("text", "")
                is_expected_act = test["expected_document"] in document
                has_expected_section = section_match(
                    text, test["expected_sections"]
                )

                if is_expected_act:
                    act_ranks.append(rank)
                if is_expected_act and has_expected_section:
                    section_ranks.append(rank)

                flags = []
                if is_expected_act:
                    flags.append("EXPECTED ACT")
                if has_expected_section:
                    flags.append("EXPECTED SECTION / TERM")

                print(
                    f"\nRank: {rank}"
                    f"\nDocument: {result.get('document', '')}"
                    f"\nPage: {result.get('page')}"
                    f"\nChunk: {result.get('chunk')}"
                    f"\nMethod: {result.get('retrieval_method')}"
                    f"\nFlags: {', '.join(flags) if flags else '-'}"
                    f"\nText: {text[:500]}"
                )

            act_ok = bool(act_ranks)
            section_ok = bool(section_ranks)
            top3_ok = any(rank <= 3 for rank in section_ranks)
            rank1_ok = 1 in section_ranks

            act_passed += int(act_ok)
            section_passed += int(section_ok)
            top3_section_passed += int(top3_ok)
            first_rank_section_passed += int(rank1_ok)

            print("\nACT IN TOP 5:", "PASS" if act_ok else "FAIL",
                  f"(ranks: {act_ranks or 'none'})")
            print("EXPECTED SECTION IN TOP 5:", "PASS" if section_ok else "FAIL",
                  f"(ranks: {section_ranks or 'none'})")
            print("EXPECTED SECTION IN TOP 3:", "PASS" if top3_ok else "FAIL")
            print("EXPECTED SECTION AT RANK 1:", "PASS" if rank1_ok else "FAIL")

        except Exception as error:
            print("ERROR:", repr(error))
            errors += 1

    total = len(TESTS)
    print("\n" + "=" * 96)
    print("FINAL RETRIEVAL SCORE")
    print(f"Expected Act in top 5:       {act_passed}/{total}")
    print(f"Expected section in top 5:   {section_passed}/{total}")
    print(f"Expected section in top 3:   {top3_section_passed}/{total}")
    print(f"Expected section at rank 1:  {first_rank_section_passed}/{total}")
    print(f"Errors / no-result tests:    {errors}/{total}")
    print("=" * 96)
    print(
        "\nNote: These are automated retrieval checks. Inspect the returned "
        "passages manually before treating any result as legally sufficient."
    )


if __name__ == "__main__":
    main()
