import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from backend.retrieval_service import (
    load_knowledge_base,
    embed_query,
    is_useful_chunk,
)


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "legal_chunks.json"
)

NEW_CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "new_legal_chunks.json"
)

DEFAULT_TOP_K = 5
CANDIDATE_K = 50
RRF_K = 60

# Separate cached BM25 indexes for the original and newer legal corpora.
# These indexes are built locally and do not modify the source files.
_BM25_INDEX_CACHE = {}


# ==================================================
# TOKENIZATION
# ==================================================

def tokenize(text):
    """Extract lowercase words and numeric legal references."""
    return re.findall(
        r"[a-z]+(?:'[a-z]+)?|\d+[a-z]*",
        text.lower()
    )


# ==================================================
# TABLE OF CONTENTS DETECTION
# ==================================================

def is_table_of_contents(text):
    """
    Detect table-of-contents chunks containing lists of section headings,
    while preserving substantive statutory provisions.
    """

    text_lower = text.lower()

    toc_markers = (
        "arrangement of sections",
        "table of contents",
        "list of sections",
    )

    if any(marker in text_lower for marker in toc_markers):
        return True

    # Count lines that look like section headings.
    section_entries = re.findall(
        r"(?m)^\s*\d+[a-z]?\.\s+[A-Z][^\n]{2,120}$",
        text
    )

    has_substantive_provision = bool(
        re.search(
            r"\b(shall|means|provided that|where any|"
            r"subject to|notwithstanding)\b",
            text_lower
        )
    )

    # TOCs usually contain many headings but no operative provisions.
    return (
        len(section_entries) >= 5
        and not has_substantive_provision
    )


# ==================================================
# LEGAL QUERY EXPANSION
# ==================================================

def expand_legal_query(question):
    """
    Expand common user expressions into statutory terminology.

    This is rule-based and does not call Gemini.
    """

    query = question.lower()
    expanded_terms = [question]

    # ----------------------------------------------
    # CHEQUE DISHONOUR - SECTION 138
    # ----------------------------------------------

    cheque_related = bool(
        re.search(
            r"\b(cheque|check)\b.*\b"
            r"(bounce\w*|dishonou?r\w*|unpaid|return\w*)\b"
            r"|\b(bounce\w*|dishonou?r\w*|unpaid|return\w*)\b.*\b"
            r"(cheque|check)\b",
            query
        )
    )

    explicit_section_138 = bool(
        re.search(r"\bsection\s*138\b", query)
    )

    if cheque_related or explicit_section_138:
        expanded_terms.extend([
            "Negotiable Instruments Act",
            "Section 138",
            "dishonour of cheque",
            "dishonour of cheque for insufficiency of funds",
            "cheque returned unpaid",
            "insufficiency of funds in the account",
            "drawer of cheque",
            "notice of demand",
            "payment of cheque",
            "fails to make payment within fifteen days",
            "within fifteen days of receipt of the said notice",
            "drawer fails to pay payee",
        ])

    # ----------------------------------------------
    # COMPANY DIRECTOR DUTIES - SECTION 166
    # ----------------------------------------------

    if (
        re.search(r"\bdirector\w*\b", query)
        and re.search(
            r"\b(duties|responsibilit\w*|obligation\w*)\b",
            query
        )
    ):
        expanded_terms.extend([
            "Companies Act Section 166",
            "duties of directors",
            "good faith",
            "care skill diligence",
            "independent judgment",
            "conflict of interest",
        ])

    # ----------------------------------------------
    # ARBITRATION - INTENT-SPECIFIC EXPANSION
    # ----------------------------------------------

    arbitration_related = bool(
        re.search(r"\barbitrat\w*\b", query)
        or re.search(r"\barbitral\b", query)
        or re.search(r"\barbitrator\b", query)
    )

    if arbitration_related:
        expanded_terms.extend([
            "Arbitration and Conciliation Act",
            "arbitral tribunal",
            "arbitration proceedings",
            "arbitration agreement",
        ])

        # Appointment of arbitrator - Section 11
        if re.search(
            r"\b(appoint\w*|appointment|selection|choose)\b",
            query
        ) and re.search(
            r"\b(arbitrator|arbitrators|tribunal)\b",
            query
        ):
            expanded_terms.extend([
                "Section 11",
                "appointment of arbitrators",
                "procedure for appointment",
                "number of arbitrators",
            ])

        # Challenge to an arbitrator - Section 13
        elif (
            re.search(r"\bchallenge\w*\b", query)
            and re.search(r"\b(arbitrator|arbitrators)\b", query)
            and not re.search(
                r"\b(award|decision|judgment)\b",
                query
            )
        ):
            expanded_terms.extend([
                "Section 13",
                "challenge procedure",
                "challenge to arbitrator",
                "Section 12",
                "grounds for challenge",
            ])

        # Challenge or setting aside an arbitral award - Section 34
        elif (
            re.search(
                r"\b(challenge\w*|set aside|setting aside|"
                r"quash|annul)\b",
                query
            )
            and re.search(
                r"\b(award|decision|arbitral award)\b",
                query
            )
        ):
            expanded_terms.extend([
                "Section 34",
                "application for setting aside arbitral award",
                "grounds for setting aside an award",
                "recourse against arbitral award",
                "Section 34 subsection 3 three months",
                "further period of thirty days but not thereafter",
                "date on which party received arbitral award",
            ])

        # Party default or non-participation - Section 25
        elif re.search(
            r"\b(refus\w*|fail\w*|default|absent|"
            r"non-participat\w*|not participat\w*)\b",
            query
        ):
            expanded_terms.extend([
                "Section 25",
                "default of a party",
                "failure to communicate statement of claim",
                "failure to communicate statement of defence",
                "failure to appear at oral hearing",
                "failure to produce documentary evidence",
            ])

        # Time limit for arbitral award - Section 29A
        elif re.search(
            r"\b(time limit|deadline|duration|"
            r"how long|period|within how long)\b",
            query
        ) and re.search(
            r"\b(award|arbitral|arbitration|tribunal)\b",
            query
        ):
            expanded_terms.extend([
                "Section 29A",
                "time limit for arbitral award",
                "period for making arbitral award",
                "completion of pleadings",
            ])

        # Commencement of proceedings - Section 21
        elif re.search(
            r"\b(commenc\w*|start\w*|begin\w*)\b",
            query
        ):
            expanded_terms.extend([
                "Section 21",
                "commencement of arbitral proceedings",
                "date on which request is received",
            ])

        # General arbitration procedure
        else:
            expanded_terms.extend([
                "Section 7 arbitration agreement",
                "Section 11 appointment of arbitrators",
                "Section 21 commencement of arbitral proceedings",
                "Section 23 statements of claim and defence",
                "Section 24 hearings and written proceedings",
                "Section 25 default of a party",
                "Section 29 decision making",
                "Section 29A time limit for arbitral award",
                "Section 31 form and contents of arbitral award",
            ])

    # ----------------------------------------------
    # REAL ESTATE / RERA
    # ----------------------------------------------

    if re.search(
        r"\b(real estate|builder|promoter|rera|allottee)\b",
        query
    ):
        expanded_terms.extend([
            "Real Estate Regulation and Development Act",
            "Section 31",
            "complaint to Authority",
            "adjudicating officer",
            "allottee",
        ])

    # ----------------------------------------------
    # CONSUMER PROTECTION - COMPLAINTS AND REMEDIES
    # ----------------------------------------------

    consumer_related = bool(
        re.search(
            r"\b(consumer|defective product|defective goods|"
            r"faulty product|faulty goods|product liability)\b",
            query
        )
    )

    if consumer_related:
        expanded_terms.extend([
            "Consumer Protection Act 2019",
            "District Commission",
            "consumer complaint",
            "Section 35 manner in which complaint shall be made",
            "Section 39 findings of District Commission",
        ])

        if re.search(
            r"\b(file|filing|submit|lodge|make|register|how|procedure|complaint)\b",
            query
        ):
            expanded_terms.extend([
                "consumer may file complaint",
                "manner in which complaint shall be made",
                "complaint may be filed with a District Commission",
            ])

        if re.search(
            r"\b(remed\w*|refund|replace|replacement|compensation|repair|order)\b",
            query
        ):
            expanded_terms.extend([
                "remove defect replace goods return price refund",
                "compensation loss injury",
                "orders of District Commission",
            ])

    # ----------------------------------------------
    # PERSONS WITH DISABILITIES
    # ----------------------------------------------

    if re.search(
        r"\b(disabilit\w*|disabled|divyang)\b",
        query
    ):
        expanded_terms.extend([
            "Rights of Persons with Disabilities Act",
            "rights and entitlements",
            "equality and non-discrimination",
            "accessibility",
            "equal opportunity",
        ])

        if re.search(
            r"\b(abuse|violence|exploitation|assault|mistreat\w*)\b",
            query
        ):
            expanded_terms.extend([
                "Section 7 protection from abuse violence and exploitation",
                "legal remedies incidents of abuse violence exploitation",
                "rescue protect and rehabilitate victims",
                "procedure for reporting abuse",
            ])

    # ----------------------------------------------
    # HINDU SUCCESSION
    # ----------------------------------------------

    if re.search(
        r"\b(succession|inheritance|inherit|property share)\b",
        query
    ):
        expanded_terms.extend([
            "Hindu Succession Act",
            "intestate succession",
            "Class I heirs",
            "coparcenary property",
            "inheritance rights",
        ])


    # ----------------------------------------------
    # PREVENTION OF MONEY-LAUNDERING ACT (PMLA)
    # ----------------------------------------------

    pmla_related = bool(
        re.search(
            r"\b(money laundering|proceeds of crime|"
            r"anti money laundering|pmla|enforcement directorate|"
            r"ed attachment)\b",
            query
        )
    )
    if pmla_related:
        expanded_terms.extend([
            "Prevention of Money Laundering Act 2002",
            "Section 3 offence of money laundering",
            "Section 4 punishment for money laundering",
            "proceeds of crime property derived or obtained",
            "Section 5 provisional attachment of property",
            "Section 8 adjudication confiscation",
            "Section 17 search and seizure",
            "Section 19 power to arrest",
            "Section 24 burden of proof",
            "Section 45 offences cognizable and non-bailable",
        ])

    # ----------------------------------------------
    # HINDU MARRIAGE ACT
    # ----------------------------------------------

    hindu_marriage_related = bool(
        re.search(
            r"\b(hindu marriage|divorce|marital|spouse|"
            r"judicial separation|cruelty|desertion|adultery)\b",
            query
        )
    )
    if hindu_marriage_related:
        expanded_terms.extend([
            "Hindu Marriage Act 1955",
            "Section 13 divorce grounds",
            "Section 13A alternative relief",
            "Section 13B divorce by mutual consent",
            "Section 10 judicial separation",
            "Section 9 restitution of conjugal rights",
            "cruelty desertion adultery conversion mental disorder",
        ])

    # ----------------------------------------------
    # HINDU SUCCESSION ACT
    # ----------------------------------------------

    succession_related = bool(
        re.search(
            r"\b(succession|inheritance|inherit|inheritance share|"
            r"class i heirs|coparcenary|ancestral property)\b",
            query
        )
    )
    if succession_related:
        expanded_terms.extend([
            "Hindu Succession Act 1956",
            "Section 8 general rules of succession in case of males",
            "Section 9 order of succession among heirs",
            "Section 10 distribution of property among Class I heirs",
            "Schedule Class I heirs",
            "daughter son widow mother share",
            "coparcenary property daughter by birth",
        ])

    # ----------------------------------------------
    # MOTOR VEHICLES ACT
    # ----------------------------------------------

    motor_vehicle_related = bool(
        re.search(
            r"\b(motor vehicle|driving licence|driver.s licence|"
            r"driving license|vehicle licence|vehicle license|"
            r"helmet|protective headgear|traffic offence)\b",
            query
        )
    )
    if motor_vehicle_related:
        expanded_terms.extend([
            "Motor Vehicles Act 1988",
            "Section 3 necessity for driving licence",
            "Section 4 age limit for driving",
            "Section 181 driving without licence",
            "Section 129 wearing protective headgear",
            "Section 130 production of licence",
        ])

    # ----------------------------------------------
    # BHARATIYA NYAYA SANHITA - THEFT
    # ----------------------------------------------

    if re.search(r"\b(theft|steal|stolen|stealing|thief)\b", query):
        expanded_terms.extend([
            "Bharatiya Nyaya Sanhita 2023",
            "Section 303 theft",
            "dishonestly takes movable property",
            "without consent out of possession",
            "punishment for theft",
        ])

    return " ".join(dict.fromkeys(expanded_terms))


# ==================================================
# LOAD BM25 KNOWLEDGE BASES
# ==================================================

def _load_bm25_index(cache_key, chunks_file):
    """
    Load a chunk JSON file and build a cached local BM25 index.
    Unusable and table-of-contents chunks are excluded.
    """

    if not chunks_file.exists():
        print("BM25 corpus not found:", chunks_file)
        return [], None

    current_mtime = chunks_file.stat().st_mtime
    cached = _BM25_INDEX_CACHE.get(cache_key)

    if cached and cached.get("mtime") == current_mtime:
        return cached["chunks"], cached["bm25"]

    with open(chunks_file, "r", encoding="utf-8") as file:
        raw_chunks = json.load(file)

    useful_chunks = []
    toc_removed = 0

    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text")
        if not isinstance(text, str) or not is_useful_chunk(text):
            continue

        if is_table_of_contents(text):
            toc_removed += 1
            continue

        useful_chunks.append(chunk)

    if not useful_chunks:
        _BM25_INDEX_CACHE[cache_key] = {
            "mtime": current_mtime,
            "chunks": [],
            "bm25": None,
        }
        return [], None

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in useful_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    _BM25_INDEX_CACHE[cache_key] = {
        "mtime": current_mtime,
        "chunks": useful_chunks,
        "bm25": bm25,
    }

    print(
        "BM25 index loaded for",
        cache_key + ":",
        len(useful_chunks),
        "useful chunks;",
        toc_removed,
        "table-of-contents chunks excluded",
    )

    return useful_chunks, bm25


def load_original_bm25_knowledge_base():
    """Load the original ten-Act corpus for local BM25 retrieval."""
    return _load_bm25_index("original", ORIGINAL_CHUNKS_FILE)


def load_bm25_knowledge_base():
    """Load the eight newer Acts for local BM25 retrieval."""
    return _load_bm25_index("new", NEW_CHUNKS_FILE)


# ==================================================
# STATUTORY RELEVANCE PRIORITY
# ==================================================

def statutory_priority(question, result):
    """
    Apply a modest ranking bonus to directly relevant
    statutory provisions.

    This affects retrieval ranking only.
    """

    query = question.lower()
    text = result.get("text", "").lower()
    document = result.get("document", "").lower()

    bonus = 0.0

    # Section 138 - cheque dishonour
    cheque_query = (
        (
            re.search(r"\b(cheque|check)\b", query)
            and re.search(
                r"\b(bounce\w*|dishonou?r\w*|unpaid|return\w*)\b",
                query
            )
        )
        or re.search(r"\bsection\s*138\b", query)
    )

    if cheque_query and "negotiable instruments act" in document:

        # Main Section 138 heading
        if (
            re.search(r"\b138\s*[.\-]", text)
            and "dishonour of cheque" in text
        ):
            bonus += 0.08

        # Continuation of Section 138: offence, punishment and conditions
        elif (
            "insufficient to honour the cheque" in text
            or (
                "twice the amount of the cheque" in text
                and "imprisonment" in text
            )
            or (
                "payee or the holder in due course" in text
                and "notice in writing" in text
            )
        ):
            bonus += 0.06

        # Supporting provisions
        elif re.search(r"\b(139|140|141)\s*[.\-]", text):
            bonus += 0.03

    # Section 166 - director duties
    director_query = (
        re.search(r"\bdirector\w*\b", query)
        and re.search(
            r"\b(duties|responsibilit\w*|obligation\w*)\b",
            query
        )
    )

    if (
        director_query
        and "companies act" in document
        and re.search(r"\b166\s*[.\-]", text)
    ):
        bonus += 0.02

    # Arbitration - intent-specific statutory priority
    if "arbitration and conciliation act" in document:
        target_section = None

        if (
            re.search(
                r"\b(challenge\w*|set aside|setting aside|"
                r"quash|annul)\b",
                query
            )
            and re.search(r"\b(award|decision)\b", query)
        ):
            target_section = "34"

        elif (
            re.search(
                r"\b(refus\w*|fail\w*|default|absent|"
                r"non-participat\w*)\b",
                query
            )
        ):
            target_section = "25"

        elif (
            re.search(
                r"\b(appoint\w*|appointment|selection)\b",
                query
            )
            and re.search(r"\b(arbitrator|tribunal)\b", query)
        ):
            target_section = "11"

        elif re.search(
            r"\b(time limit|deadline|duration|how long|period)\b",
            query
        ):
            target_section = "29a"

        elif re.search(
            r"\b(commenc\w*|start\w*|begin\w*)\b",
            query
        ):
            target_section = "21"

        if target_section:
            section_pattern = (
                r"\b" + re.escape(target_section) + r"\s*[.\-]"
            )

            if re.search(section_pattern, text):
                if target_section == "34":
                    bonus += 0.08
                else:
                    bonus += 0.04

        # For award challenges, Section 13 concerns a different issue:
        # challenging the arbitrator rather than challenging the award.
        if target_section == "34":
            if (
                re.search(r"\b13\s*[.\-]", text)
                and "challenge procedure" in text
            ):
                bonus -= 0.02

            elif (
                "setting aside" in text
                and "arbitral award" in text
            ):
                bonus += 0.04

            elif (
                re.search(r"\b37\s*[.\-]", text)
                and "section 34" in text
            ):
                bonus += 0.03

    # Disability rights - prioritize Chapter II (Sections 3-12).
    disability_query = bool(
        re.search(
            r"\b(disability|disabled|persons with disabilities|"
            r"person with a disability)\b",
            query
        )
        and re.search(
            r"\b(rights|entitlements|accessibility|protection|"
            r"discrimination|equality)\b",
            query
        )
    )

    if (
        disability_query
        and "rights of persons with disabilities act" in document
    ):
        if re.search(
            r"\b(3|4|5|6|7|8|9|10|11|12)\s*[.\-]",
            text
        ):
            bonus += 0.06

    # Consumer Protection Act: filing procedure (Section 35) and
    # remedies (Section 39) are different intents.
    consumer_query = bool(
        re.search(
            r"\b(consumer|defective product|defective goods|"
            r"faulty product|faulty goods|product liability)\b",
            query
        )
    )
    normalized_document = re.sub(r"[^a-z0-9]+", " ", document)
    if consumer_query and "consumer protection act" in normalized_document:
        filing_intent = bool(
            re.search(
                r"\b(file|filing|submit|lodge|register|procedure|complaint|how)\b",
                query
            )
        )
        remedy_intent = bool(
            re.search(
                r"\b(remed\w*|refund|replace|replacement|compensation|repair|order)\b",
                query
            )
        )

        if filing_intent and re.search(r"\b35\s*[.\-]", text):
            bonus += 0.12
        if remedy_intent and re.search(r"\b39\s*[.\-]", text):
            bonus += 0.12

    # A deadline question about setting aside an award needs Section 34(3).
    award_deadline_query = (
        "arbitration and conciliation act" in normalized_document
        and re.search(r"\b(award|arbitral)\b", query)
        and re.search(
            r"\b(deadline|how long|time limit|limitation|within|"
            r"number of days|months|period for filing)\b",
            query
        )
    )
    if award_deadline_query:
        has_section_34_3 = bool(
            re.search(r"\b34\s*\(\s*3\s*\)", text)
            or (
                "three months" in text
                and "thirty days" in text
                and "arbitral award" in text
            )
        )
        if has_section_34_3:
            bonus += 0.18

    # Abuse/violence questions should prioritize Section 7 specifically.
    abuse_query = (
        re.search(r"\b(disabilit\w*|disabled|divyang)\b", query)
        and re.search(
            r"\b(abuse|violence|exploitation|assault|mistreat\w*)\b",
            query
        )
    )
    if abuse_query and "rights of persons with disabilities act" in normalized_document:
        if (
            re.search(r"\b7\s*[.\-]", text)
            and re.search(r"abuse|violence|exploitation", text)
        ):
            bonus += 0.18


    # PMLA: distinguish the offence, punishment and procedural powers.
    if "prevention of money laundering act" in normalized_document:
        pmla_query = bool(
            re.search(
                r"\b(money laundering|proceeds of crime|pmla|"
                r"enforcement directorate|ed attachment)\b",
                query
            )
        )
        if pmla_query:
            if re.search(r"\b3\s*[.\-]", text) and (
                "money laundering" in text or "proceeds of crime" in text
            ):
                bonus += 0.20
            if re.search(r"\b4\s*[.\-]", text) and "punishment" in text:
                bonus += 0.12
            if re.search(r"\b5\s*[.\-]", text) and "attachment" in text:
                bonus += 0.10
            if re.search(r"\b8\s*[.\-]", text) and (
                "adjudicating authority" in text or "confiscation" in text
            ):
                bonus += 0.10
            if re.search(r"\b24\s*[.\-]", text) and "burden of proof" in text:
                bonus += 0.08
            if re.search(r"\b45\s*[.\-]", text) and "bail" in text:
                bonus += 0.08

    # RERA: route by the legal issue rather than matching the Act title alone.
    if "real estate regulation and development act" in normalized_document:
        rera_query = bool(
            re.search(
                r"\b(real estate|builder|promoter|rera|allottee|"
                r"flat|apartment|housing project)\b",
                query
            )
        )
        if rera_query:
            if re.search(r"\b31\s*[.\-]", text) and (
                "complaint" in text or "authority" in text
            ):
                bonus += 0.18
            if re.search(r"\b18\s*[.\-]", text) and (
                "refund" in text or "interest" in text
                or "compensation" in text or "delay" in text
            ):
                bonus += 0.18
            if re.search(r"\b19\s*[.\-]", text) and "allottee" in text:
                bonus += 0.10
            if re.search(r"\b43\s*[.\-]", text) and "authority" in text:
                bonus += 0.08
            if re.search(r"\b44\s*[.\-]", text) and "appeal" in text:
                bonus += 0.08
            if re.search(r"\b71\s*[.\-]", text) and "adjudicating officer" in text:
                bonus += 0.10

    # Hindu Marriage Act: prioritize divorce grounds over unrelated provisions.
    if "hindu marriage act" in normalized_document:
        marriage_query = bool(
            re.search(
                r"\b(hindu marriage|divorce|cruelty|desertion|"
                r"adultery|judicial separation|mutual consent)\b",
                query
            )
        )
        if marriage_query:
            if re.search(r"\b13\s*[.\-]", text) and (
                "divorce" in text or "marriage" in text
            ):
                bonus += 0.22
            if re.search(r"\b13\s*\(\s*b\s*\)", text) or (
                "mutual consent" in text and "divorce" in text
            ):
                bonus += 0.14
            if re.search(r"\b10\s*[.\-]", text) and "judicial separation" in text:
                bonus += 0.12
            if re.search(r"\b9\s*[.\-]", text) and "restitution of conjugal rights" in text:
                bonus += 0.08

    # Hindu Succession Act: distinguish general succession from Class I distribution.
    if "hindu succession act" in normalized_document:
        succession_query = bool(
            re.search(
                r"\b(succession|inheritance|inherit|heir|"
                r"coparcenary|ancestral property|property share)\b",
                query
            )
        )
        if succession_query:
            if re.search(r"\b10\s*[.\-]", text) and (
                "class i" in text or "distribution" in text or "heirs" in text
            ):
                bonus += 0.20
            if re.search(r"\b8\s*[.\-]", text) and "succession" in text:
                bonus += 0.16
            if re.search(r"\b9\s*[.\-]", text) and "order of succession" in text:
                bonus += 0.12
            if "class i heirs" in text or "class i" in text and "schedule" in text:
                bonus += 0.14
            if "daughter" in text and "coparcener" in text:
                bonus += 0.10

    # Motor Vehicles Act: distinguish licence queries from helmet requirements.
    if "motor vehicles act" in normalized_document:
        if re.search(
            r"\b(licen[cs]e|driving|drive without|unlicensed)\b",
            query
        ):
            if re.search(r"\b3\s*[.\-]", text) and "driving licence" in text:
                bonus += 0.18
            if re.search(r"\b4\s*[.\-]", text) and "age" in text:
                bonus += 0.10
            if re.search(r"\b181\s*[.\-]", text) and (
                "without licence" in text or "without a licence" in text
            ):
                bonus += 0.20
        if re.search(r"\b(helmet|protective headgear)\b", query):
            if re.search(r"\b129\s*[.\-]", text) and "headgear" in text:
                bonus += 0.20

    # BNS theft: prefer Section 303 and its operative definition/punishment.
    if "bharatiya nyaya sanhita" in normalized_document and re.search(
        r"\b(theft|steal|stolen|stealing|thief)\b", query
    ):
        if re.search(r"\b303\s*[.\-]", text) and "theft" in text:
            bonus += 0.22
        elif "dishonestly takes" in text and "movable property" in text:
            bonus += 0.16

    # A generic award challenge should prioritize the grounds in Section 34,
    # not the separate Section 34(3) filing limitation.
    award_challenge_query = (
        "arbitration and conciliation act" in normalized_document
        and re.search(r"\b(award|arbitral)\b", query)
        and re.search(r"\b(challenge|set aside|setting aside|quash|annul)\b", query)
        and not re.search(
            r"\b(deadline|how long|time limit|limitation|within|"
            r"number of days|months|period for filing)\b",
            query
        )
    )
    if award_challenge_query:
        if re.search(r"\b34\s*[.\-]", text) and (
            "public policy" in text or "patent illegality" in text
            or "unable to present" in text
            or "invalid" in text and "arbitration agreement" in text
            or "grounds for setting aside" in text
        ):
            bonus += 0.24
        if (
            "public policy" in text or "patent illegality" in text
            or "unable to present" in text
        ):
            bonus += 0.12

    # Cheque-bounce deadline clause: ensure the 15-day continuation is surfaced.
    if cheque_query and "negotiable instruments act" in document:
        if (
            "within fifteen days" in text
            or "fifteen days of the receipt" in text
            or "fifteen days of receipt" in text
        ):
            bonus += 0.14

    return bonus


# ==================================================
# BM25 SEARCH - LOCAL ONLY
# ==================================================

def _search_corpus_bm25(question, chunks, bm25, top_k):
    """Run BM25 over an already-loaded corpus index."""
    if not chunks or bm25 is None:
        return []

    query_tokens = tokenize(expand_legal_query(question))
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    results = []

    for index, score in enumerate(scores):
        score = float(score)
        if score <= 0:
            continue

        result = chunks[index].copy()
        result["similarity"] = 0.0
        result["bm25_score"] = score
        result["retrieval_method"] = "bm25"
        result["legal_priority"] = statutory_priority(question, result)
        result["ranking_score"] = score + result["legal_priority"]
        results.append(result)

    results.sort(
        key=lambda item: item["ranking_score"],
        reverse=True
    )
    return results[:top_k]


def search_existing_documents_bm25(
    question,
    top_k=DEFAULT_TOP_K
):
    """Search the original ten-Act corpus using local BM25 only."""
    if not question or not question.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    chunks, bm25 = load_original_bm25_knowledge_base()
    return _search_corpus_bm25(question, chunks, bm25, top_k)


def search_new_documents_bm25(
    question,
    top_k=DEFAULT_TOP_K
):
    """Search the eight newer Acts using local BM25 only."""
    if not question or not question.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    chunks, bm25 = load_bm25_knowledge_base()
    return _search_corpus_bm25(question, chunks, bm25, top_k)


# ==================================================
# SEMANTIC SEARCH - EXISTING CORPUS
# ==================================================

def search_existing_semantic(
    question,
    candidate_k=CANDIDATE_K
):
    """
    Search the original corpus using existing stored
    embeddings and one Gemini query embedding.
    """

    chunks, embeddings = load_knowledge_base()

    query_vector = embed_query(question)

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        raise ValueError("Query embedding is zero.")

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    denominator = embedding_norms * query_norm

    similarities = np.divide(
        np.dot(embeddings, query_vector),
        denominator,
        out=np.zeros(
            len(embeddings),
            dtype=np.float32
        ),
        where=denominator != 0
    )

    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices:
        chunk = chunks[int(index)]

        if not is_useful_chunk(chunk["text"]):
            continue

        result = chunk.copy()

        result["similarity"] = float(similarities[index])
        result["retrieval_method"] = "semantic"

        results.append(result)

        if len(results) >= candidate_k:
            break

    return results


def preferred_documents(question):
    """
    Route clearly identifiable legal issues to their most relevant Act.
    Routing is deliberately conservative to avoid excluding useful Acts
    when a question spans more than one legal subject.
    """
    query = question.lower()

    if (
        re.search(r"\b(cheque|check)\b", query)
        and re.search(
            r"\b(bounce\w*|dishonou?r\w*|unpaid|return\w*)\b",
            query
        )
    ):
        return ("negotiable instruments act",)

    if re.search(
        r"\b(defective product|defective goods|faulty product|"
        r"faulty goods|product liability|consumer protection|"
        r"consumer complaint)\b",
        query
    ):
        return ("consumer protection act",)

    if (
        re.search(r"\b(arbitrat\w*|arbitral|arbitrator)\b", query)
        and re.search(
            r"\b(award|challenge|set aside|setting aside|"
            r"annul|quash|appoint\w*|appointment|proceedings)\b",
            query
        )
    ):
        return ("arbitration and conciliation act",)

    if (
        re.search(r"\bdirector\w*\b", query)
        and re.search(
            r"\b(duties|responsibilit\w*|obligation\w*)\b",
            query
        )
    ):
        return ("companies act",)

    if (
        re.search(r"\b(disabilit\w*|disabled|divyang)\b", query)
        and re.search(
            r"\b(rights|protection|abuse|violence|exploitation|"
            r"discrimination|equality|accessibility)\b",
            query
        )
    ):
        return ("rights of persons with disabilities act",)

    if re.search(
        r"\b(money laundering|proceeds of crime|pmla|"
        r"anti money laundering|enforcement directorate)\b",
        query
    ):
        return ("prevention of money laundering act",)

    if re.search(
        r"\b(real estate|rera|builder|promoter|allottee|"
        r"flat|apartment|housing project)\b",
        query
    ):
        return ("real estate regulation and development act",)

    if re.search(
        r"\b(hindu marriage|divorce|judicial separation|"
        r"cruelty|desertion|adultery|mutual consent)\b",
        query
    ):
        return ("hindu marriage act",)

    if re.search(
        r"\b(succession|inheritance|inherit|class i heirs|"
        r"coparcenary|ancestral property)\b",
        query
    ):
        return ("hindu succession act",)

    if re.search(r"\b(theft|steal|stolen|stealing|thief)\b", query):
        return ("bharatiya nyaya sanhita",)

    if re.search(
        r"\b(motor vehicle|driving licence|driver.s licence|"
        r"driving license|vehicle licence|vehicle license|"
        r"helmet|protective headgear|traffic offence)\b",
        query
    ):
        return ("motor vehicles act",)

    return None


def _matches_preferred_document(result, preferred):
    if preferred is None:
        return True

    document = str(result.get("document", "")).lower()

    # Normalize underscores, hyphens and punctuation in filenames.
    document = re.sub(r"[^a-z0-9]+", " ", document)
    document = " ".join(document.split())

    return any(
        name in document
        for name in preferred
    )


def _append_adjacent_continuations(results):
    """
    Add the immediately following raw source chunk when a retrieved
    passage ends mid-sentence. Read the original JSON files directly:
    the BM25 index may exclude a valid continuation as "not useful".
    """
    if not results:
        return results

    chunk_lookup = {}

    for chunks_file in (ORIGINAL_CHUNKS_FILE, NEW_CHUNKS_FILE):
        if not chunks_file.exists():
            continue

        try:
            with open(chunks_file, "r", encoding="utf-8") as file:
                raw_chunks = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            print("Could not load chunks for continuation lookup:", error)
            continue

        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            try:
                chunk_number = int(chunk.get("chunk"))
            except (TypeError, ValueError):
                continue

            identity = (
                str(chunk.get("document", "")),
                str(chunk.get("page", "")),
                chunk_number,
            )
            chunk_lookup[identity] = chunk

    expanded = list(results)
    included = {
        (
            str(item.get("document", "")),
            str(item.get("page", "")),
            str(item.get("chunk", "")),
        )
        for item in results
    }

    for result in results:
        text = str(result.get("text", "")).rstrip()
        if not text or re.search(r"[.!?][\"')\\]]?\\s*$", text):
            continue

        try:
            next_number = int(result.get("chunk")) + 1
        except (TypeError, ValueError):
            continue

        identity = (
            str(result.get("document", "")),
            str(result.get("page", "")),
            next_number,
        )
        next_chunk = chunk_lookup.get(identity)
        if not next_chunk:
            continue

        next_text = str(next_chunk.get("text", "")).lstrip()
        if not next_text:
            continue

        # The next chunk should continue the unfinished sentence or clause.
        if not (
            next_text[0].islower()
            or re.match(
                r"(?i)^(or|and|but|provided that|within|where|which|"
                r"as the case may be|in such case|the receipt)\\b",
                next_text,
            )
        ):
            continue

        next_key = (
            str(next_chunk.get("document", "")),
            str(next_chunk.get("page", "")),
            str(next_chunk.get("chunk", "")),
        )
        if next_key in included:
            continue

        continuation = next_chunk.copy()
        continuation["similarity"] = 0.0
        continuation["bm25_score"] = 0.0
        continuation["rrf_score"] = 0.0
        continuation["legal_priority"] = 0.0
        continuation["ranking_score"] = 0.0
        continuation["retrieval_method"] = "adjacent_continuation"
        continuation["retrieval_sources"] = ["adjacent_continuation"]
        continuation["retrieval_methods"] = ["adjacent_continuation"]
        continuation["is_adjacent_continuation"] = True

        expanded.append(continuation)
        included.add(next_key)

    return expanded



# ==================================================
# RELEVANCE FILTERING + RECIPROCAL RANK FUSION
# ==================================================

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "could", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "may", "me",
    "my", "of", "on", "or", "should", "the", "their", "them", "there",
    "these", "they", "this", "to", "under", "was", "we", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "your",
    "act", "acts", "law", "legal", "section", "sections"
}


def _normalise_term(term):
    term = term.lower().strip()

    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]

    return term


def _meaningful_query_terms(question):
    return {
        _normalise_term(term)
        for term in tokenize(question)
        if len(term) > 2 and term not in _STOPWORDS
    }


def _passage_query_overlap(question, result):
    query_terms = _meaningful_query_terms(question)

    if not query_terms:
        return 0

    passage = (
        str(result.get("text", "")) + " " +
        str(result.get("document", ""))
    )

    passage_terms = {
        _normalise_term(term)
        for term in tokenize(passage)
    }

    return len(query_terms.intersection(passage_terms))


def _is_relevant_candidate(question, result, method):
    overlap = _passage_query_overlap(question, result)
    priority = statutory_priority(question, result)

    if method == "bm25":
        return (
            float(result.get("bm25_score", 0.0)) > 0
            and (overlap >= 2 or priority > 0)
        )

    similarity = float(result.get("similarity", 0.0))

    # Semantic results must demonstrate stronger relevance.
    # A very strong semantic match can pass without lexical overlap.
    return (
        (overlap >= 2 and similarity >= 0.62)
        or similarity >= 0.70
    )


def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    top_k=DEFAULT_TOP_K,
    question=""
):
    """Filter weak candidates before combining the ranked result lists."""

    fused = {}

    preferred = preferred_documents(question)

    semantic_results = [
        result for result in semantic_results
        if _matches_preferred_document(result, preferred)
    ]

    bm25_results = [
        result for result in bm25_results
        if _matches_preferred_document(result, preferred)
    ]

    def get_identity(result):
        return (
            result.get("document"),
            result.get("page"),
            result.get("chunk")
        )

    def add_results(results, method):
        accepted_rank = 0

        for result in results:
            if not _is_relevant_candidate(question, result, method):
                continue

            accepted_rank += 1
            identity = get_identity(result)

            if identity not in fused:
                fused[identity] = result.copy()
                fused[identity]["rrf_score"] = 0.0
                fused[identity]["retrieval_methods"] = []
                fused[identity]["similarity"] = float(
                    result.get("similarity", 0.0)
                )

            entry = fused[identity]

            entry["rrf_score"] += 1.0 / (RRF_K + accepted_rank)

            if method not in entry["retrieval_methods"]:
                entry["retrieval_methods"].append(method)

            if method == "semantic":
                entry["similarity"] = float(
                    result.get("similarity", 0.0)
                )
            elif method == "bm25":
                entry["bm25_score"] = float(
                    result.get("bm25_score", 0.0)
                )

    add_results(semantic_results, "semantic")
    add_results(bm25_results, "bm25")

    final_results = list(fused.values())

    for result in final_results:
        methods = result["retrieval_methods"]

        # Report the actual retrieval source(s), not the overall pipeline.
        result["retrieval_method"] = (
            methods[0] if len(methods) == 1 else "both"
        )
        result["retrieval_sources"] = list(methods)

        result["legal_priority"] = statutory_priority(
            question,
            result
        )

        result["ranking_score"] = (
            result["rrf_score"] + result["legal_priority"]
        )

    final_results.sort(
        key=lambda item: item["ranking_score"],
        reverse=True
    )

    selected_results = final_results[:top_k]
    return _append_adjacent_continuations(selected_results)


# ==================================================
# MAIN HYBRID SEARCH
# ==================================================

def search_legal_documents(
    question,
    top_k=DEFAULT_TOP_K
):
    """
    Search both legal corpora and combine results.

    Original corpus:
        Semantic retrieval using stored embeddings plus local BM25.

    New corpus:
        Local BM25 keyword retrieval.

    Fusion:
        Reciprocal Rank Fusion with statutory priority.
    """

    if not question or not question.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    semantic_results = []
    bm25_results = []
    semantic_error = None

    # Search original semantic corpus
    try:
        semantic_results = search_existing_semantic(
            question,
            candidate_k=max(
                CANDIDATE_K,
                top_k * 5
            )
        )

    except Exception as error:
        semantic_error = error
        print(
            "Semantic retrieval unavailable:",
            error
        )

    # Search both corpora with local BM25. This adds keyword retrieval
    # for the original Acts without regenerating or changing embeddings.
    bm25_candidate_k = max(CANDIDATE_K, top_k * 5)

    try:
        original_bm25_results = search_existing_documents_bm25(
            question,
            top_k=bm25_candidate_k
        )
    except Exception as error:
        original_bm25_results = []
        print("Original-corpus BM25 unavailable:", error)

    try:
        new_bm25_results = search_new_documents_bm25(
            question,
            top_k=bm25_candidate_k
        )
    except Exception as error:
        new_bm25_results = []
        print("New-corpus BM25 unavailable:", error)

    bm25_results = original_bm25_results + new_bm25_results

    if not semantic_results and not bm25_results:
        if semantic_error is not None:
            raise semantic_error

        return []

    return reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        top_k=top_k,
        question=question
    )


# ==================================================
# LOCAL BM25 TEST - NO GEMINI API CALL
# ==================================================

if __name__ == "__main__":

    test_questions = [
        "What happens when a cheque is dishonoured?",
        "What happens when a cheque bounces?",
        "What is Section 138 of the Negotiable Instruments Act?",
        "What are the duties of a company director?",
        "What are the rights of persons with disabilities?",
        "What is the procedure for arbitration?",
        "How is an arbitrator appointed?",
        "What happens when a party refuses to participate in arbitration?",
        "What is the time limit for making an arbitral award?",
        "How can an arbitral award be challenged?",
        "How long do I have to challenge an arbitral award?",
        "How can I file a consumer complaint for a defective product?",
        "What remedies can the Consumer Commission order for defective goods?",
        "What protection is available to a person with disability facing abuse?",
        "What is the time limit for filing a real estate complaint?",
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        results = (
            search_existing_documents_bm25(question, top_k=5)
            + search_new_documents_bm25(question, top_k=5)
        )
        results.sort(
            key=lambda item: item.get("ranking_score", 0.0),
            reverse=True
        )
        results = results[:5]

        if not results:
            print("No useful BM25 passages found.")
            continue

        for index, result in enumerate(results, start=1):
            print("\nRESULT", index)
            print("Document:", result["document"])
            print("Page:", result["page"])
            print("Chunk:", result["chunk"])
            print("BM25 score:", round(result["bm25_score"], 4))
            print("Legal priority:", result["legal_priority"])
            print("Text:")
            print(result["text"][:600])
            print("-" * 70)
