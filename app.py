import re
import json
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import streamlit as st
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    HRFlowable,
    Flowable,
)

from backend.legal_assistant import ask_legal_question
from backend.case_builder import build_case_record
from backend.argument_practice import (
    start_practice_session,
    submit_practice_turn,
    evaluate_practice_session,
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="BARS AI | Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
STAMP_IMAGE = BASE_DIR / "assets" / "indian_non_judicial_stamp_sample.png"


# ==================================================
# 3. CUSTOM CSS
# ==================================================

st.markdown(
    """
<style>
:root {
    --bars-primary: #2563EB;
    --bars-primary-hover: #1D4ED8;
    --bars-ink: #1F2937;
    --bars-muted: #77798B;
    --bars-border: #E6E8EF;
    --bars-surface: #FFFFFF;
    --bars-background: #F7F8FA;
}

/* Clean, light application canvas */
.stApp,
[data-testid="stAppViewContainer"],
section.main {
    background: var(--bars-background) !important;
    color: var(--bars-ink) !important;
}

.main .block-container {
    max-width: 1440px;
    padding-top: 2.25rem;
    padding-bottom: 3.25rem;
}

.main-title {
    color: #1F2937 !important;
    font-size: 2rem;
    line-height: 1.2;
    font-weight: 750;
    letter-spacing: -0.035em;
    margin-bottom: 0.4rem;
}

.subtitle {
    color: var(--bars-muted) !important;
    font-size: 0.96rem;
    line-height: 1.55;
    margin-bottom: 1.6rem;
}

.main .stMarkdown p,
.main .stMarkdown li,
.main .stMarkdown strong,
.main .stMarkdown h1,
.main .stMarkdown h2,
.main .stMarkdown h3,
.main [data-testid="stCaptionContainer"] p {
    color: var(--bars-ink);
}

/* Light navigation rail */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #FFFFFF !important;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid var(--bars-border);
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label * {
    color: #333449 !important;
    -webkit-text-fill-color: #333449 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    color: #8A8C9B !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0.28rem;
}

/* Permanently hide only the sidebar radio indicators; labels remain clickable. */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div[data-baseweb="radio"],
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label input[type="radio"] {
    display: none !important;
}

/* Keep the selected navigation item visibly blue even when not hovered. */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has([aria-checked="true"]) {
    background: #EFF6FF !important;
    box-shadow: inset 3px 0 0 var(--bars-primary) !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has([aria-checked="true"]) p,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has([aria-checked="true"]) span {
    color: var(--bars-primary) !important;
    -webkit-text-fill-color: var(--bars-primary) !important;
    font-weight: 650 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    border-radius: 9px;
    padding: 0.48rem 0.62rem;
    margin: 0;
    transition: background-color 120ms ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: #F3F7FF;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: #EFF6FF;
    box-shadow: inset 3px 0 0 var(--bars-primary);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {
    color: var(--bars-primary) !important;
    -webkit-text-fill-color: var(--bars-primary) !important;
    font-weight: 650;
}

/* Cards and bordered sections */
.info-card {
    background: var(--bars-surface);
    border: 1px solid var(--bars-border);
    border-radius: 13px;
    padding: 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 8px rgba(31, 35, 60, 0.025);
}

.card-title {
    color: #1F2937 !important;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.card-description {
    color: var(--bars-muted) !important;
    font-size: 0.9rem;
    line-height: 1.55;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--bars-border) !important;
    border-radius: 12px !important;
    background: #FFFFFF;
}

/* Buttons */
.stButton button,
.stDownloadButton button,
.stFormSubmitButton button,
.stLinkButton a {
    background: var(--bars-primary) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--bars-primary) !important;
    border-radius: 9px !important;
    min-height: 2.55rem;
    padding: 0.48rem 1.05rem;
    font-weight: 650;
    box-shadow: none !important;
}

.stButton button:hover,
.stDownloadButton button:hover,
.stFormSubmitButton button:hover,
.stLinkButton a:hover {
    background: var(--bars-primary-hover) !important;
    border-color: var(--bars-primary-hover) !important;
    color: #FFFFFF !important;
}

.stButton button p,
.stDownloadButton button p,
.stFormSubmitButton button p,
.stLinkButton a p {
    color: #FFFFFF !important;
}

/* Inputs and selectors */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: var(--bars-ink) !important;
    -webkit-text-fill-color: var(--bars-ink) !important;
    border: 1px solid #DDE0E9 !important;
    border-radius: 9px !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color: #9294A2 !important;
    -webkit-text-fill-color: #9294A2 !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #FFFFFF !important;
    border: 1px solid #DDE0E9 !important;
    border-radius: 9px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: var(--bars-ink) !important;
    -webkit-text-fill-color: var(--bars-ink) !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background: #FFFFFF !important;
    color: var(--bars-ink) !important;
}

div[data-baseweb="popover"] li * {
    color: var(--bars-ink) !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    color: #37364A !important;
}

[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label span {
    color: #37364A !important;
    -webkit-text-fill-color: #37364A !important;
}

/* Chat and retrieved-source panels */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid var(--bars-border) !important;
    border-radius: 13px !important;
    padding: 1rem !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] strong {
    color: var(--bars-ink) !important;
}

[data-testid="stChatInput"] {
    background: var(--bars-background) !important;
}

/* Expanders, progress and metrics */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--bars-border) !important;
    border-radius: 11px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #374151 !important;
}

[data-testid="stProgressBar"] {
    background: #E8EEF7 !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--bars-muted) !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--bars-border);
    border-radius: 11px;
    padding: 0.8rem;
}

/* Existing document preview and footer */
.draft-preview {
    background: #FFFFFF;
    border: 1px solid var(--bars-border);
    border-radius: 12px;
    padding: 1.75rem;
    color: var(--bars-ink);
    line-height: 1.8;
}

.draft-preview h1,
.draft-preview h2,
.draft-preview h3 {
    color: #1F2937 !important;
}

.draft-preview p,
.draft-preview li {
    color: var(--bars-ink) !important;
}

.footer {
    text-align: center;
    color: #9294A2 !important;
    font-size: 0.78rem;
    padding: 0.8rem;
}

/* Keep Streamlit status and alert text legible */
[data-testid="stAlert"] {
    border-radius: 10px;
}

@media (max-width: 768px) {
    .main .block-container {
        padding-top: 1.25rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .main-title {
        font-size: 1.65rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# 4. PDF GENERATION
# ==================================================

class StampHeader(Flowable):
    """Displays the supplied stamp image as a clearly marked sample."""

    def __init__(self, image_path, width=175 * mm):
        super().__init__()
        self.image_path = str(image_path)
        self.width = width

        with PILImage.open(self.image_path) as image:
            image_width, image_height = image.size

        self.height = width * image_height / image_width

    def wrap(self, available_width, available_height):
        actual_width = min(self.width, available_width)
        scale = actual_width / self.width
        self.width = actual_width
        self.height *= scale
        return self.width, self.height

    def draw(self):
        self.canv.drawImage(
            self.image_path,
            0,
            0,
            width=self.width,
            height=self.height,
            preserveAspectRatio=True,
            mask="auto",
        )

        # A prominent overlay prevents the sample from appearing genuine.
        self.canv.saveState()
        self.canv.setFillColor(colors.Color(0.65, 0.05, 0.15, alpha=0.78))
        self.canv.setFont("Helvetica-Bold", 15)
        self.canv.translate(self.width / 2, self.height / 2)
        self.canv.rotate(18)
        self.canv.drawCentredString(
            0, 0, "SAMPLE - NOT VALID STAMP PAPER"
        )
        self.canv.restoreState()


def markdown_to_pdf_elements(markdown_text, styles):
    """Convert a legal notice into clean PDF paragraphs and lists."""

    elements = []
    paragraph_buffer = []
    list_buffer = []

    section_heading = re.compile(
        r"^\s*\d+\.\s*(sender details|recipient details|date and place|"
        r"subject|brief introduction|chronological statement of facts|"
        r"relevant legal grounds|specific request or relief sought|"
        r"closing and signature placeholders)\s*:\s*(.*)$",
        re.IGNORECASE,
    )

    def paragraph_markup(text):
        text = escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
        return text

    def flush_paragraph():
        if paragraph_buffer:
            text = " ".join(paragraph_buffer).strip()
            if text:
                # Keep the document title centred and visually distinct.
                if re.fullmatch(r"DRAFT\s+LEGAL\s+NOTICE", text, re.IGNORECASE):
                    elements.append(
                        Paragraph("DRAFT LEGAL NOTICE", styles["DocumentTitle"])
                    )
                else:
                    elements.append(
                        Paragraph(paragraph_markup(text), styles["BodyText"])
                    )
            paragraph_buffer.clear()

    def flush_list():
        if list_buffer:
            for item, is_numbered in list_buffer:
                style_name = "NumberedItem" if is_numbered else "BulletItem"
                elements.append(
                    Paragraph(paragraph_markup(item), styles[style_name])
                )
            list_buffer.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            flush_list()
            continue

        # Remove numbered section labels such as "1. Sender Details:"
        # while retaining any content that follows the label.
        heading_match = section_heading.match(line)
        if heading_match:
            flush_paragraph()
            flush_list()
            line = heading_match.group(2).strip()
            if not line:
                continue

        # Convert sender/recipient details into simple address blocks:
        # name on one line and address on the next, without field labels.
        if re.match(r"^(?:Name\s*:)", line, re.IGNORECASE):
            line = re.sub(r"^Name\s*:\s*", "", line, flags=re.IGNORECASE)
            line = re.sub(
                r"\s+Address\s*:\s*",
                "\n",
                line,
                flags=re.IGNORECASE,
            )
            for part in [piece.strip() for piece in line.split("\n") if piece.strip()]:
                flush_list()
                paragraph_buffer.append(part)
                flush_paragraph()
            continue

        if line in ("---", "***", "___"):
            flush_paragraph()
            flush_list()
            elements.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=colors.HexColor("#DCE5F2"),
                    spaceBefore=8,
                    spaceAfter=10,
                )
            )
            continue

        if line.startswith("# "):
            flush_paragraph()
            flush_list()
            title_text = line[2:].strip()
            if re.fullmatch(r"DRAFT\s+LEGAL\s+NOTICE", title_text, re.IGNORECASE):
                elements.append(
                    Paragraph("DRAFT LEGAL NOTICE", styles["DocumentTitle"])
                )
            else:
                elements.append(
                    Paragraph(paragraph_markup(title_text), styles["Heading1"])
                )
            continue

        if line.startswith("## "):
            flush_paragraph()
            flush_list()
            elements.append(
                Paragraph(paragraph_markup(line[3:]), styles["Heading2"])
            )
            continue

        if line.startswith("### "):
            flush_paragraph()
            flush_list()
            elements.append(
                Paragraph(paragraph_markup(line[4:]), styles["Heading3"])
            )
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)", line)
        if bullet_match:
            flush_paragraph()
            list_buffer.append(("• " + bullet_match.group(1), False))
            continue

        # Retain numbered factual points, but not numbered section headings.
        numbered_match = re.match(r"^(\d+)[.)]\s+(.*)", line)
        if numbered_match:
            flush_paragraph()
            list_buffer.append(
                (f"{numbered_match.group(1)}.  {numbered_match.group(2)}", True)
            )
            continue

        flush_list()
        paragraph_buffer.append(line)

    flush_paragraph()
    flush_list()
    return elements


def generate_legal_pdf(document_text):
    """Build the final downloadable PDF with the sample stamp image."""

    if not STAMP_IMAGE.exists():
        raise FileNotFoundError(
            f"Stamp image not found at: {STAMP_IMAGE}. "
            "Place the image in assets/indian_non_judicial_stamp_sample.png."
        )

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title="BARS AI - Legal Document Draft",
        author="BARS AI",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="DocumentTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=21,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=12,
        )
    )

    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 11.5
    styles["BodyText"].leading = 17
    styles["BodyText"].textColor = colors.HexColor("#374151")
    styles["BodyText"].alignment = TA_LEFT
    styles["BodyText"].spaceAfter = 8
    styles["BodyText"].leftIndent = 0
    styles["BodyText"].rightIndent = 0
    styles["BodyText"].firstLineIndent = 0

    styles.add(
        ParagraphStyle(
            name="NumberedItem",
            parent=styles["BodyText"],
            leftIndent=18,
            firstLineIndent=-18,
            rightIndent=4,
            spaceBefore=2,
            spaceAfter=9,
            leading=17,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletItem",
            parent=styles["BodyText"],
            leftIndent=18,
            firstLineIndent=-10,
            rightIndent=4,
            spaceBefore=1,
            spaceAfter=8,
            leading=17,
        )
    )

    styles["Heading1"].textColor = colors.HexColor("#1F2937")
    styles["Heading1"].spaceBefore = 8
    styles["Heading1"].spaceAfter = 9
    styles["Heading2"].spaceBefore = 7
    styles["Heading2"].spaceAfter = 7
    styles["Heading3"].spaceBefore = 6
    styles["Heading3"].spaceAfter = 6
    styles["Heading2"].textColor = colors.HexColor("#374151")
    styles["Heading3"].textColor = colors.HexColor("#374151")

    story = []

    # Sample image at the top of the first page.
    story.append(StampHeader(STAMP_IMAGE))
    story.append(Spacer(1, 10))

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#DCE5F2"),
            spaceBefore=5,
            spaceAfter=15,
        )
    )

    # Omit the opening AI/legal disclaimer from the PDF body.
    # The disclaimer can be placed in the separate project report instead.
    pdf_body = document_text.strip()
    pdf_body = re.sub(
        r"^\s*\*{0,2}Disclaimer:\*{0,2}.*?(?=\n\s*(?:---|#{1,3}\s|[A-Z][^\n]{0,80}:)|\Z)",
        "",
        pdf_body,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    story.extend(markdown_to_pdf_elements(pdf_body, styles))

    def draw_pdf_page(canvas, document):
        # Warm off-white background requested by the user:
        # RGB(255, 251, 242), hex #FFFBF2.
        canvas.saveState()
        canvas.setFillColor(colors.Color(1.0, 251 / 255, 242 / 255))
        canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)

        # Keep the page number minimal: 1, 2, 3, ...
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#514735"))
        canvas.drawCentredString(
            A4[0] / 2,
            9 * mm,
            str(document.page),
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=draw_pdf_page,
        onLaterPages=draw_pdf_page,
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ==================================================
# 5. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "case_result" not in st.session_state:
    st.session_state.case_result = None

if "draft_result" not in st.session_state:
    st.session_state.draft_result = None

if "draft_editor" not in st.session_state:
    st.session_state.draft_editor = ""

if "argument_practice" not in st.session_state:
    st.session_state.argument_practice = None

if "case_record" not in st.session_state:
    st.session_state.case_record = None


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def display_sources(sources):
    if not sources:
        st.info("No source passages were retrieved.")
        return

    st.markdown("### Retrieved legal sources")

    for source in sources:
        label = source.get("label", "Source")
        document = source.get("document", "Unknown document")
        page_number = source.get("page", "Unknown")
        similarity = source.get("similarity", 0)
        source_text = source.get("text", "")

        with st.expander(
            f"{label} | {document} | Page {page_number}"
        ):
            st.caption(f"Similarity score: {similarity}")
            st.markdown(source_text)


def build_chat_history():
    history = []
    previous_messages = st.session_state.messages[:-1]

    for message in previous_messages[-8:]:
        role = message["role"].upper()
        history.append(f"{role}: {message['content']}")

    return "\n\n".join(history)


def generate_legal_response(question, history=None):
    with st.spinner("BARS AI is analysing your request..."):
        return ask_legal_question(
            question=question,
            chat_history=history,
        )


def format_case_context(record):
    """Create a factual context block from the active structured case."""
    if not record:
        return ""
    return json.dumps(
        {
            "case_title": record.get("case_title", ""),
            "case_type": record.get("case_type", ""),
            "jurisdiction": record.get("jurisdiction", ""),
            "user_role": record.get("user_role", ""),
            "case_summary": record.get("case_summary", ""),
            "parties": record.get("parties", []),
            "timeline": record.get("timeline", []),
            "key_facts": record.get("key_facts", []),
            "allegations_or_claims": record.get("allegations_or_claims", []),
            "disputed_or_unclear_facts": record.get("disputed_or_unclear_facts", []),
            "evidence_available": record.get("evidence_available", []),
            "missing_evidence": record.get("missing_evidence", []),
            "questions_to_clarify": record.get("questions_to_clarify", []),
            "desired_outcome": record.get("desired_outcome", ""),
            "issues_to_review": record.get("issues_to_review", []),
        },
        ensure_ascii=False,
        indent=2,
    )


def is_temporary_gemini_error(error_or_result):
    """Identify temporary Gemini capacity/availability errors."""
    if isinstance(error_or_result, dict):
        parts = [
            str(error_or_result.get("error", "")),
            str(error_or_result.get("answer", "")),
            str(error_or_result.get("message", "")),
        ]
        message = " ".join(parts).lower()
    else:
        message = str(error_or_result).lower()

    indicators = (
        "503",
        "unavailable",
        "high demand",
        "overloaded",
        "service unavailable",
        "temporarily unavailable",
    )
    return any(indicator in message for indicator in indicators)


def display_generation_error(error_or_result, context="request"):
    """Show an accurate message for Gemini outages or other failures."""
    if is_temporary_gemini_error(error_or_result):
        st.warning(
            "**External AI Service Temporarily Unavailable**\n\n"
            "Gemini API is currently experiencing high demand. "
            "Spikes in demand are usually temporary. Please try again later.\n\n"
            "BARS AI application is running, but the external AI service "
            "could not process this request."
        )
    else:
        st.error(
            f"BARS AI could not complete this {context} because of a "
            "technical error. Please try again."
        )


def result_has_error(result):
    """Return True when the backend marks a response as failed."""
    if not isinstance(result, dict):
        return True
    error_value = result.get("error")
    if error_value:
        return True
    answer = str(result.get("answer", "")).lower()
    return (
        not answer.strip()
        or "could not process your question because of a technical error" in answer
        or "unable to generate a response" in answer
    )


def render_assistant_result(result):
    answer = result.get("answer", "Unable to generate a response.")
    sources = result.get("sources", [])

    st.markdown(answer)

    if result_has_error(result):
        display_generation_error(result, context="response")
        return

    with st.expander("View retrieved legal sources"):
        display_sources(sources)


# ==================================================
# 7. SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 25px 0;">
            <div style="font-size: 31px; font-weight: 800;">
                BARS AI
            </div>
            <div style="font-size: 12px;">
                YOUR AI LEGAL ASSISTANT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Legal Chat",
            "Build the Case",
            "Case Analysis",
            "Evidence Checklist",
            "Document Drafting",
            "Help Them Argue",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    if st.session_state.case_record:
        st.caption("Active case file")
        st.write(st.session_state.case_record.get("case_title", "Untitled case"))



# ==================================================
# 8. DASHBOARD
# ==================================================

if page == "Dashboard":
    st.markdown(
        '<div class="main-title">Welcome to BARS AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Understand your legal situation, '
        'explore relevant information and prepare your documents.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Legal Chat</div>
                <div class="card-description">
                    Ask questions about Indian law and receive
                    structured answers based on retrieved legal documents.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Evidence Checklist</div>
                <div class="card-description">
                    Identify and organise documents that may help
                    support your case.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Case Analysis</div>
                <div class="card-description">
                    Describe your legal problem and identify important
                    facts and missing information.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">Document Drafting</div>
                <div class="card-description">
                    Prepare editable first drafts of legal notices
                    and complaints.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Legal Acts & Reference Library")
    st.markdown(
        "Browse and download the official Act PDFs. Once the knowledge base "
        "has been rebuilt, BARS AI can retrieve passages from all ten Acts."
    )

    legal_acts = [
        (
            "Consumer Protection Act, 2019",
            "consumer_protection_act_2019.pdf",
            "Consumer rights, product defects and deficient services.",
            "https://www.indiacode.nic.in/indiacode/bitstream/123456789/15256/1/eng201935.pdf",
        ),
        (
            "Indian Contract Act, 1872",
            "indian_contract_act_1872.pdf",
            "Contracts, promises, performance and breach.",
            "https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf",
        ),
        (
            "Transfer of Property Act, 1882",
            "transfer_of_property_act_1882.pdf",
            "Transfer of property and certain landlord–tenant matters.",
            "https://www.indiacode.nic.in/bitstream/123456789/14648/1/tpa.pdf",
        ),
        (
            "Bharatiya Nyaya Sanhita, 2023",
            "bharatiya_nyaya_sanhita_2023.pdf",
            "Criminal offences and punishments.",
            "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf",
        ),
        (
            "Bharatiya Nagarik Suraksha Sanhita, 2023",
            "bharatiya_nagarik_suraksha_sanhita_2023.pdf",
            "Criminal procedure and investigation process.",
            "https://www.indiacode.nic.in/bitstream/123456789/20335/1/a2023-46.pdf",
        ),
        (
            "Bharatiya Sakshya Adhiniyam, 2023",
            "bharatiya_sakshya_adhiniyam_2023.pdf",
            "Rules relating to relevance and proof of evidence.",
            "https://www.indiacode.nic.in/indiacode/bitstream/123456789/20063/1/aa202347.pdf",
        ),
        (
            "Motor Vehicles Act, 1988",
            "motor_vehicles_act_1988.pdf",
            "Motor vehicles, registration, licensing and road transport.",
            "https://www.indiacode.nic.in/bitstream/123456789/1798/1/eng.pdf",
        ),
        (
            "Information Technology Act, 2000",
            "information_technology_act_2000.pdf",
            "Electronic records, electronic transactions and cyber-related provisions.",
            "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
        ),
        (
            "Limitation Act, 1963",
            "limitation_act_1963.pdf",
            "Limitation periods for suits, appeals and applications.",
            "https://www.indiacode.nic.in/bitstream/123456789/1565/5/A1963-36.pdf",
        ),
        (
            "Specific Relief Act, 1963",
            "specific_relief_act_1963.pdf",
            "Certain forms of specific relief and contractual remedies.",
            "https://www.indiacode.nic.in/bitstream/123456789/1583/7/A1963-47.pdf",
        ),
    ]

    library_dir = BASE_DIR / "data" / "legal_documents"
    for act_name, filename, description, source_url in legal_acts:
        file_path = library_dir / filename
        with st.container(border=True):
            text_col, download_col, source_col = st.columns(
                [7, 1.3, 1.2], vertical_alignment="center"
            )
            with text_col:
                st.markdown(f"**{act_name}**")
                st.caption(description)
                if file_path.exists():
                    st.caption("PDF available locally")
                else:
                    st.caption("PDF not downloaded yet")
            with download_col:
                if file_path.exists():
                    st.download_button(
                        "↓ PDF",
                        data=file_path.read_bytes(),
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_{filename}",
                        help=f"Download {act_name}",
                    )
                else:
                    st.link_button(
                        "Get PDF",
                        source_url,
                        help=f"Open official India Code PDF for {act_name}",
                    )
            with source_col:
                st.link_button(
                    "Source",
                    source_url,
                    help=f"View official India Code source for {act_name}",
                )



# ==================================================
# 9. LEGAL CHAT
# ==================================================

elif page == "Legal Chat":
    st.markdown(
        '<div class="main-title">Legal Chat</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Describe your legal problem in your own words.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    active_case = st.session_state.case_record
    use_case_in_chat = False
    if active_case:
        use_case_in_chat = st.checkbox(
            f"Use current case file: {active_case.get('case_title', 'Untitled case')}",
            value=True,
            key="legal_chat_use_case",
        )
        if use_case_in_chat:
            st.caption("Your questions will include the structured case facts as context.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                with st.expander("View legal sources"):
                    display_sources(message.get("sources", []))

    user_question = st.chat_input("Describe your legal problem...")

    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        history = build_chat_history()

        try:
            chat_question = user_question
            if active_case and use_case_in_chat:
                chat_question = (
                    "Answer the user's question using the following case file as factual context. "
                    "Do not treat disputed allegations as established facts. If the user asks "
                    "about a fact not in the case file, say it is not specified.\n\n"
                    "STRUCTURED CASE FILE:\n"
                    + format_case_context(active_case)
                    + "\n\nUSER QUESTION:\n"
                    + user_question
                )
            result = generate_legal_response(chat_question, history)

            with st.chat_message("assistant"):
                if result_has_error(result):
                    display_generation_error(result, context="question")
                else:
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                    render_assistant_result(result)

        except Exception as error:
            display_generation_error(error, context="question")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ==================================================
# 10. CASE ANALYSIS
# ==================================================

elif page == "Case Analysis":
    st.markdown(
        '<div class="main-title">Case Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Organise the facts of your legal problem.</div>',
        unsafe_allow_html=True,
    )


    active_case = st.session_state.case_record
    use_case_for_analysis = False
    if active_case:
        use_case_for_analysis = st.checkbox(
            f"Use current case file: {active_case.get('case_title', 'Untitled case')}",
            value=True,
            key="case_analysis_use_case",
        )

    with st.form("case_analysis_form"):
        case_type = st.selectbox(
            "Select the type of legal problem",
            [
                "Landlord and Tenant Dispute",
                "Consumer Complaint",
                "Employment Dispute",
                "Contract Dispute",
                "Family and Matrimonial Dispute",
                "Inheritance and Succession Dispute",
                "Cheque Bounce / Dishonour",
                "Real Estate / Builder-Buyer Dispute",
                "Company and Corporate Dispute",
                "Arbitration Dispute",
                "Disability Rights Dispute",
                "Money Laundering / Financial Crime",
                "Other Legal Dispute",
            ],
        )

        location = st.text_input(
            "State and city",
            placeholder="Example: Kozhikode, Kerala",
        )

        case_description = st.text_area(
            "Describe what happened / add any new details",
            value=(
                (active_case.get("original_description", "") if active_case else "")
                if use_case_for_analysis else ""
            ),
            placeholder=(
                "Explain the events, dates, people involved "
                "and the outcome you are seeking..."
            ),
            height=200,
        )

        submitted = st.form_submit_button("Analyse My Case")

    if submitted:
        if not case_description.strip():
            st.warning("Please describe your legal problem.")
        else:
            prompt = f"""
Analyse the following legal situation.

Type of dispute: {case_type}
Location: {location or (active_case.get("jurisdiction", "Not provided") if active_case and use_case_for_analysis else "Not provided")}

User's description:
{case_description}

Structured case context (if supplied; treat disputed facts as disputed):
{format_case_context(active_case) if active_case and use_case_for_analysis else "No saved case file selected."}

Provide:
1. Summary of the situation.
2. Main legal issues that may arise.
3. Relevant retrieved legal provisions.
4. Facts and documents that are missing.
5. Questions the user should answer.
6. Safe practical next steps.

Distinguish verified legal provisions from possible interpretations.
Do not invent legal sections or remedies.
"""
            try:
                case_result = generate_legal_response(prompt)
                if result_has_error(case_result):
                    st.session_state.case_result = None
                    display_generation_error(case_result, context="case analysis")
                else:
                    st.session_state.case_result = case_result
            except Exception as error:
                display_generation_error(error, context="case analysis")

    if st.session_state.case_result:
        st.markdown("---")
        st.markdown("### Case Analysis Result")
        render_assistant_result(st.session_state.case_result)


# ==================================================
# 11. BUILD THE CASE
# ==================================================

elif page == "Build the Case":
    st.markdown(
        '<div class="main-title">Build the Case</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Turn your account into an organised case file that you can review and reuse.</div>',
        unsafe_allow_html=True,
    )

    with st.form("build_case_form"):
        build_case_type = st.selectbox(
            "Type of dispute",
            [
                "Landlord and Tenant Dispute",
                "Consumer Complaint",
                "Employment Dispute",
                "Contract Dispute",
                "Property Dispute",
                "Family Dispute",
                "Other",
            ],
        )
        build_jurisdiction = st.text_input(
            "Jurisdiction (state/city, if known)",
            value="India",
            placeholder="Example: Kozhikode, Kerala",
        )
        build_user_role = st.text_input(
            "Your role in the dispute",
            placeholder="Example: tenant, consumer, employee, buyer",
        )
        build_desired_outcome = st.text_input(
            "What outcome are you seeking?",
            placeholder="Example: return of security deposit",
        )
        build_description = st.text_area(
            "Describe everything that happened",
            placeholder=(
                "Include people involved, dates, amounts, agreements, "
                "communications, what each side says, and documents you have. "
                "Do not worry about organising it first."
            ),
            height=230,
        )
        build_submitted = st.form_submit_button("Create Structured Case File", type="primary")

    if build_submitted:
        if len(build_description.strip()) < 20:
            st.warning("Please provide a little more detail about the dispute.")
        else:
            try:
                with st.spinner("Organising the facts, timeline and evidence..."):
                    st.session_state.case_record = build_case_record(
                        description=build_description,
                        case_type=build_case_type,
                        jurisdiction=build_jurisdiction or "India",
                        user_role=build_user_role or "Not specified",
                        desired_outcome=build_desired_outcome,
                    )
                st.success("Your structured case file has been created. Review it carefully.")
            except Exception as error:
                display_generation_error(error, context="case file creation")

    record = st.session_state.case_record
    if record:
        st.markdown("---")
        st.markdown(f"### {record.get('case_title', 'Case file')}")
        st.caption(
            f"Case type: {record.get('case_type', 'Not specified')} | "
            f"Jurisdiction: {record.get('jurisdiction', 'Not specified')} | "
            f"Your role: {record.get('user_role', 'Not specified')}"
        )
        st.markdown("#### Case summary")
        st.write(record.get("case_summary") or "No summary was generated.")

        left, right = st.columns(2)
        with left:
            with st.expander("Parties involved", expanded=True):
                parties = record.get("parties", [])
                if parties:
                    for party in parties:
                        if isinstance(party, dict):
                            st.markdown(f"**{party.get('name_or_role', 'Party')}**")
                            st.write(party.get("position", ""))
                            if party.get("known_details"):
                                st.caption(party["known_details"])
                        else:
                            st.write(party)
                else:
                    st.write("No parties were identified.")
            with st.expander("Key facts", expanded=True):
                for item in record.get("key_facts", []):
                    st.markdown(f"- {item}")
            with st.expander("Claims and allegations"):
                for item in record.get("allegations_or_claims", []):
                    st.markdown(f"- {item}")
            with st.expander("Disputed or unclear facts"):
                for item in record.get("disputed_or_unclear_facts", []):
                    st.markdown(f"- {item}")
        with right:
            with st.expander("Timeline", expanded=True):
                timeline = record.get("timeline", [])
                if timeline:
                    for event in timeline:
                        if isinstance(event, dict):
                            st.markdown(f"**{event.get('date_or_period', 'Date not specified')}**")
                            st.write(event.get("event", ""))
                            st.caption(f"Status: {event.get('status', 'Unspecified')}")
                        else:
                            st.write(event)
                else:
                    st.write("No dated events were identified.")
            with st.expander("Evidence mentioned", expanded=True):
                for evidence in record.get("evidence_available", []):
                    if isinstance(evidence, dict):
                        st.markdown(f"**{evidence.get('item', 'Evidence item')}**")
                        st.write(evidence.get("what_it_may_support", ""))
                        st.caption(f"Status: {evidence.get('provided_or_mentioned', 'Not specified')}")
                    else:
                        st.write(evidence)
            with st.expander("Evidence still needed"):
                for item in record.get("missing_evidence", []):
                    st.markdown(f"- {item}")
            with st.expander("Questions to clarify", expanded=True):
                for item in record.get("questions_to_clarify", []):
                    st.markdown(f"- {item}")

        st.markdown("#### Issues to review")
        for item in record.get("issues_to_review", []):
            st.markdown(f"- {item}")
        st.markdown("**Desired outcome:** " + (record.get("desired_outcome") or "Not specified"))

        st.download_button(
            "Download Case File (JSON)",
            data=json.dumps(record, ensure_ascii=False, indent=2),
            file_name="BARS_AI_Case_File.json",
            mime="application/json",
        )

        with st.expander("View original description"):
            st.write(record.get("original_description", ""))

        st.markdown("#### Continue with this case")
        practice_mode_label = st.radio(
            "Choose the opposing role for argument practice",
            ["Realistic opposing party", "Opposing lawyer"],
            horizontal=True,
            key="case_to_practice_mode",
        )
        if st.button("Use this case in Help Them Argue", type="primary"):
            mode = "realistic_party" if practice_mode_label == "Realistic opposing party" else "opposing_lawyer"
            practice_summary = (
                record.get("case_summary", "")
                + "\n\nKey facts:\n- "
                + "\n- ".join(record.get("key_facts", []))
                + "\n\nDisputed facts:\n- "
                + "\n- ".join(record.get("disputed_or_unclear_facts", []))
            )
            try:
                with st.spinner("Starting argument practice with this case..."):
                    st.session_state.argument_practice = start_practice_session(
                        case_summary=practice_summary,
                        mode=mode,
                        user_side=record.get("user_role", "claimant"),
                        opponent_name="the opposing party",
                        jurisdiction=record.get("jurisdiction", "India"),
                        user_goal=record.get("desired_outcome", ""),
                    )
                st.success("Practice session started. Select Help Them Argue in the sidebar.")
            except Exception as error:
                display_generation_error(error, context="argument-practice session")

        if st.button("Clear Current Case File"):
            st.session_state.case_record = None
            st.rerun()


# ==================================================
# 12. EVIDENCE CHECKLIST
# ==================================================

elif page == "Evidence Checklist":
    st.markdown(
        '<div class="main-title">Evidence Checklist</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Organise the documents related to your legal problem.</div>',
        unsafe_allow_html=True,
    )

    # General organisational prompts, not exhaustive or legally mandatory requirements.
    checklists = {
        "Landlord and Tenant Dispute": [
            "Rental or lease agreement and renewals",
            "Security deposit payment proof",
            "Monthly rent payment records",
            "Written communication with landlord or tenant",
            "Notice or messages about termination or vacating",
            "Photographs or videos of property condition",
            "Repair requests and maintenance records",
            "Utility bill payment records",
            "Handover, possession or inspection records",
        ],
        "Consumer Complaint": [
            "Purchase invoice, bill or receipt",
            "Order confirmation and product/service description",
            "Warranty or guarantee document",
            "Photographs or videos showing the defect or deficiency",
            "Payment proof and transaction reference",
            "Delivery, installation or service records",
            "Communication with seller, manufacturer or service provider",
            "Repair, replacement or refund requests and responses",
            "Any complaint already filed and its acknowledgement",
        ],
        "Employment Dispute": [
            "Appointment letter and employment agreement",
            "Job description and applicable company policies",
            "Salary slips and bank salary records",
            "Attendance, leave and working-hour records",
            "Emails and messages with employer or HR",
            "Performance reviews or disciplinary communications",
            "Termination, resignation or suspension letter",
            "Full and final settlement or dues calculation",
            "Relevant witness details and other supporting records",
        ],
        "Contract Dispute": [
            "Signed contract, agreement and amendments",
            "Invoices, bills and payment records",
            "Emails, letters and messages between the parties",
            "Proof of obligations each party agreed to perform",
            "Delivery, completion or performance records",
            "Evidence of alleged breach or non-performance",
            "Loss, damage or expense records, if claimed",
            "Notices exchanged and proof of service",
            "Dispute resolution or arbitration clause, if any",
        ],
        "Family and Matrimonial Dispute": [
            "Marriage certificate or other available proof of marriage",
            "Relevant identity and address records of the parties",
            "Records of children, if relevant to the dispute",
            "Relevant communications between the parties",
            "Financial records, income details and expense records",
            "Documents relating to shared assets, loans or liabilities",
            "Medical or other supporting records, where relevant",
            "Police complaints, protection orders or prior court orders, if any",
            "Names and contact details of relevant witnesses",
        ],
        "Inheritance and Succession Dispute": [
            "Death certificate of the deceased",
            "Will or codicil, if one exists",
            "Documents showing family relationships",
            "Property title deeds and ownership records",
            "Revenue, mutation and tax records, where available",
            "Bank, investment and other asset records relevant to the estate",
            "Existing probate, succession or court documents, if any",
            "Communications among heirs or claimants",
            "Details of debts, liabilities and disputed transactions",
        ],
        "Cheque Bounce / Dishonour": [
            "Copy or image of the cheque",
            "Bank return memo showing the reason for dishonour",
            "Relevant loan, sale, service or transaction agreement",
            "Invoices, receipts and proof of the underlying transaction",
            "Bank records showing relevant payments",
            "Copy of the statutory demand notice, if issued",
            "Proof of dispatch and delivery or service of the notice",
            "Replies or communications between the parties",
            "Relevant dates and a record of events",
        ],
        "Real Estate / Builder-Buyer Dispute": [
            "Builder-buyer agreement or allotment letter",
            "Booking form and payment receipts",
            "Bank statements or loan disbursement records",
            "Project advertisements and written promises relied upon",
            "Construction progress updates and site photographs",
            "Possession offer, completion or occupancy documents, if any",
            "Correspondence with the developer or agent",
            "Delay, defect or maintenance complaint records",
            "RERA registration details and any complaint/order already filed",
        ],
        "Company and Corporate Dispute": [
            "Company incorporation and constitutional documents",
            "Shareholding records and share certificates, if applicable",
            "Board and shareholder meeting notices and minutes",
            "Relevant resolutions and statutory filings",
            "Shareholder, director or investment agreements",
            "Company accounts and relevant financial statements",
            "Contracts, invoices and transaction records in dispute",
            "Emails and other communications among the parties",
            "Existing notices, regulatory correspondence or court orders",
        ],
        "Arbitration Dispute": [
            "Contract containing the arbitration clause or agreement",
            "Relevant amendments and related contracts",
            "Notice invoking arbitration and proof of service",
            "Correspondence about appointment of the arbitrator",
            "Statement of claim, defence and counterclaim, if filed",
            "Documents supporting the claims and responses",
            "Payment, loss and damages calculations, if relevant",
            "Procedural orders, hearing notices and minutes",
            "Interim orders, award or enforcement records, if any",
        ],
        "Disability Rights Dispute": [
            "Disability certificate or other relevant supporting record",
            "Application, request or communication seeking accommodation",
            "Written response or refusal from the institution or authority",
            "Relevant employment, education or service records",
            "Policies or rules relied upon by the institution or authority",
            "Medical or accessibility-related records, where relevant",
            "Details and dates of the incident or alleged denial of rights",
            "Names of witnesses and relevant communications",
            "Prior complaints, acknowledgements or orders, if any",
        ],
        "Money Laundering / Financial Crime": [
            "Bank statements and relevant transaction records",
            "Records showing the source and purpose of funds, if available",
            "Relevant contracts, invoices and receipts",
            "Company, ownership or account records relevant to the matter",
            "Notices, summons or official communications received",
            "Seizure, attachment or search documents, if applicable",
            "Copies of statements or submissions already made",
            "Chronology of relevant transactions and events",
            "Details of legal representation and any existing court orders",
        ],
        "Other Legal Dispute": [
            "Relevant agreements, notices or official documents",
            "Payment and transaction records, if applicable",
            "Emails, letters and messages relating to the dispute",
            "Photographs, videos or other supporting material",
            "A dated timeline of important events",
            "Names and contact details of relevant witnesses",
            "Records of complaints or proceedings already initiated",
            "Documents showing loss, damage or requested relief, if relevant",
        ],
    }

    case_type = st.selectbox(
        "Select your dispute type",
        list(checklists.keys()),
        key="evidence_dispute_type",
    )

    st.markdown("### Documents to collect")
    selected_items = []

    for index, item in enumerate(checklists[case_type]):
        checked = st.checkbox(
            item,
            key=f"evidence_{case_type}_{index}",
        )
        if checked:
            selected_items.append(item)

    total_items = len(checklists[case_type])
    completed = len(selected_items)

    st.progress(completed / total_items)
    st.caption(
        f"{completed} of {total_items} documents marked as collected."
    )

    if st.button("Show Missing Documents", key="show_missing_evidence"):
        missing = [
            item for item in checklists[case_type]
            if item not in selected_items
        ]

        st.markdown("### Documents still to collect")

        if missing:
            for item in missing:
                st.markdown(f"- {item}")
        else:
            st.success("All checklist items are marked as collected.")



# 13. DOCUMENT DRAFTING
# ==================================================

elif page == "Document Drafting":
    st.markdown(
        '<div class="main-title">Document Drafting</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Prepare a first draft using the facts you provide.</div>',
        unsafe_allow_html=True,
    )


    active_case = st.session_state.case_record
    use_case_for_drafting = False
    if active_case:
        use_case_for_drafting = st.checkbox(
            f"Use current case file: {active_case.get('case_title', 'Untitled case')}",
            value=True,
            key="document_drafting_use_case",
        )

    with st.form("document_drafting_form"):
        document_type = st.selectbox(
            "Select document type",
            [
                "Legal Notice",
                "Consumer Complaint",
                "Reply to Legal Notice",
                "Formal Request Letter",
            ],
        )

        sender_name = st.text_input(
            "Your name",
            placeholder="Enter your name",
        )

        recipient_name = st.text_input(
            "Recipient name",
            placeholder="Landlord, seller, employer, etc.",
        )

        location = st.text_input(
            "State and city",
            value=(active_case.get("jurisdiction", "") if active_case and use_case_for_drafting else ""),
            placeholder="Example: Kozhikode, Kerala",
        )

        case_facts_default = ""
        if active_case and use_case_for_drafting:
            case_facts_default = (
                active_case.get("original_description", "")
                + "\n\nStructured case summary: "
                + active_case.get("case_summary", "")
            )

        facts = st.text_area(
            "Describe the facts of your case",
            value=case_facts_default,
            placeholder=(
                "Include dates, amounts, agreements, "
                "communications and the outcome you want."
            ),
            height=200,
        )

        submitted = st.form_submit_button("Generate First Draft")

    if submitted:
        if not facts.strip():
            st.warning("Please provide the facts of your case.")
        else:
            prompt = f"""
Prepare a first draft of a {document_type}.

Sender: {sender_name or "[Sender name]"}
Recipient: {recipient_name or "[Recipient name]"}
Location: {location or "[Location]"}

Facts supplied by the user:
{facts}

Structured case file (if selected; do not treat disputed allegations as established facts):
{format_case_context(active_case) if active_case and use_case_for_drafting else "No saved case file selected."}

DRAFTING INSTRUCTIONS:

Create a complete, reasonably detailed first draft of the requested legal notice.
The previous draft was too short. Do not reduce the notice to only two generic
sentences. Aim for approximately 6 to 9 substantive paragraphs, with enough
detail to make it a useful editable draft. Do not add empty filler.

Create only the legal notice itself. Do not produce a case-analysis report,
checklist, explanation, source summary, or numbered outline.

LAYOUT:
- Put "DRAFT LEGAL NOTICE" alone at the top.
- Show the sender's name, then a separate placeholder line for the sender's
  full postal address, city, state and PIN code.
- Show "To," on its own line, followed by the recipient's name and a separate
  placeholder line for the recipient's full postal address, city, state and PIN.
- Put the date and place on separate lines.
- Include one concise "Subject:" line.
- Do not use numbered headings or labels such as "Sender Details",
  "Recipient Details", "Date and Place", "Brief Introduction",
  "Chronological Statement of Facts", "Relevant Legal Grounds",
  "Specific Request", or "Closing and Signature".
- Write the body as formal, connected paragraphs. Use a short list only if
  necessary to make specific requests clear.
- End with "Yours faithfully,", a signature line, and the sender's name alone.
  Do not append "(Sender)" or any role label after the sender's name.

CONTENT REQUIREMENTS FOR A LOST-BIKE NOTICE:
- State that the sender reports the loss of a bike, using only that fact as
  established if no further facts have been supplied.
- Include clearly marked placeholders for the date and approximate time of
  loss, location, bike make/model, colour, registration number, chassis/engine
  number if available, and a concise description of the circumstances.
- Include a paragraph with a placeholder for explaining the recipient's
  actual connection to the bike or incident. Do not accuse the recipient of
  taking, possessing, concealing, damaging, or selling the bike unless the
  user has expressly supplied that fact. If no connection is known, phrase
  this neutrally and request clarification.
- Include a paragraph recording prior attempts to contact the recipient,
  with placeholders for dates, method of communication and responses. Do not
  state that contact was attempted unless the user confirms it.
- Include a paragraph with placeholders for any police complaint/FIR or
  other report number and date, but do not claim one has been filed.
- State the sender's concern and the practical outcome sought.
- Make clear, in polite but firm language, that the recipient is requested to
  provide accurate information about the bike's whereabouts and, if the bike
  is in their custody or control, arrange its safe return. Keep the request
  conditional where custody or control has not been established.
- Include a reasonable response period as a placeholder, not an invented
  number of days.
- Include a final paragraph reserving the sender's ability to seek appropriate
  advice and remedies after the facts and applicable law are verified. Do not
  threaten criminal or civil proceedings as an established entitlement.

LEGAL ACCURACY:
- Use only facts supplied by the user. Use square-bracket placeholders for
  missing details instead of inventing them.
- Do not invent dates, amounts, events, addresses, evidence, relationships,
  police reports, statutes, legal sections, cases, or legal remedies.
- Do not imply that the recipient is legally liable merely because the bike
  is described as lost.
- Include legal provisions only if the retrieved sources directly support
  their application to these facts. If they do not, omit legal citations.
- Do not claim that the notice is legally compliant or ready to send.
- Use professional, formal language. This is a draft for review by a
  qualified Indian advocate before it is sent."""
            try:
                result = generate_legal_response(prompt)
                if result_has_error(result):
                    st.session_state.draft_result = None
                    st.session_state.draft_editor = ""
                    display_generation_error(result, context="document draft")
                else:
                    st.session_state.draft_result = result
                    st.session_state.draft_editor = result.get("answer", "")
            except Exception as error:
                st.session_state.draft_result = None
                st.session_state.draft_editor = ""
                display_generation_error(error, context="document draft")

    if st.session_state.draft_result:
        st.markdown("---")
        st.markdown("### Generated Document Draft")
        st.markdown("#### Document Preview")

        # The editor key is updated when a new draft is generated.
        edited_text = st.text_area(
            "Edit your document before downloading",
            key="draft_editor",
            height=420,
        )

        # Formatted preview of the same editable text.
        with st.container(border=True):
            st.markdown(edited_text)

        # PDF download.
        try:
            pdf_data = generate_legal_pdf(edited_text)

            st.download_button(
                label="Download Final Draft as PDF",
                data=pdf_data,
                file_name="BARS_AI_Legal_Draft.pdf",
                mime="application/pdf",
            )

        except Exception as error:
            st.error(f"PDF generation failed: {error}")

        with st.expander("View Retrieved Legal Sources"):
            display_sources(
                st.session_state.draft_result.get("sources", [])
            )




# ==================================================
# 14. HELP THEM ARGUE
# ==================================================

elif page == "Help Them Argue":
    st.markdown(
        '<div class="main-title">Help Them Argue</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Practise presenting your position and responding to the other side.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This is an educational roleplay exercise. The opposing side's "
        "statements are simulated arguments, not verified facts or legal advice."
    )

    session = st.session_state.argument_practice
    saved_case = st.session_state.case_record

    if session is None:
        st.markdown("### Set up a practice session")

        # Automatically carry the structured Build the Case record into
        # argument practice, so the user does not have to enter the dispute again.
        saved_case_summary = ""
        if saved_case:
            case_details = {
                "parties": saved_case.get("parties", []),
                "timeline": saved_case.get("timeline", []),
                "key_facts": saved_case.get("key_facts", []),
                "allegations_or_claims": saved_case.get("allegations_or_claims", []),
                "disputed_or_unclear_facts": saved_case.get("disputed_or_unclear_facts", []),
                "evidence_available": saved_case.get("evidence_available", []),
                "missing_evidence": saved_case.get("missing_evidence", []),
                "questions_to_clarify": saved_case.get("questions_to_clarify", []),
                "issues_to_review": saved_case.get("issues_to_review", []),
            }
            saved_case_summary = (
                f"Case title: {saved_case.get('case_title', 'Untitled case')}\n"
                f"Case type: {saved_case.get('case_type', 'Not specified')}\n"
                f"Jurisdiction: {saved_case.get('jurisdiction', 'India')}\n"
                f"User's role: {saved_case.get('user_role', 'Not specified')}\n"
                f"Case summary: {saved_case.get('case_summary', '')}\n"
                f"Desired outcome: {saved_case.get('desired_outcome', 'Not specified')}\n\n"
                "Structured case details (user-provided facts, allegations and "
                "unresolved points must remain distinct):\n"
                + json.dumps(case_details, ensure_ascii=False, indent=2)
            )
            st.info(
                f"The current case file **{saved_case.get('case_title', 'Untitled case')}** "
                "has been loaded automatically. You can edit the details before starting practice."
            )

        with st.form("argument_practice_setup"):
            case_summary = st.text_area(
                "Describe your dispute",
                value=saved_case_summary,
                placeholder=(
                    "Explain what happened, important dates, people involved, "
                    "what you have already tried and what outcome you want."
                ),
                height=260 if saved_case else 180,
            )
            mode_label = st.radio(
                "Choose the opposing role",
                [
                    "Realistic opposing party",
                    "Opposing lawyer",
                ],
                help=(
                    "The realistic party speaks as the landlord, employer, seller, "
                    "etc. The opposing lawyer challenges your case as an advocate."
                ),
            )
            col_a, col_b = st.columns(2)
            with col_a:
                user_side = st.text_input(
                    "Your side",
                    value=(saved_case.get("user_role", "Claimant") if saved_case else "Claimant"),
                    placeholder="Tenant, employee, consumer, claimant...",
                )
                opponent_name = st.text_input(
                    "Opposing party",
                    value="the opposing party",
                    placeholder="Landlord, employer, seller...",
                )
            with col_b:
                jurisdiction = st.text_input(
                    "Jurisdiction",
                    value=(saved_case.get("jurisdiction", "India") if saved_case else "India"),
                    placeholder="State and city, if relevant",
                )
                user_goal = st.text_input(
                    "What do you want to achieve?",
                    value=(saved_case.get("desired_outcome", "") if saved_case else ""),
                    placeholder="For example: return of deposit",
                )
            start_clicked = st.form_submit_button(
                "Start Argument Practice",
                type="primary",
            )

        if start_clicked:
            if len(case_summary.strip()) < 20:
                st.warning("Please provide at least a short description of the dispute.")
            else:
                mode = (
                    "realistic_party"
                    if mode_label == "Realistic opposing party"
                    else "opposing_lawyer"
                )
                try:
                    with st.spinner("Preparing the opposing side's opening position..."):
                        st.session_state.argument_practice = start_practice_session(
                            case_summary=case_summary,
                            mode=mode,
                            user_side=user_side or "claimant",
                            opponent_name=opponent_name or "the opposing party",
                            jurisdiction=jurisdiction or "India",
                            user_goal=user_goal,
                        )
                    st.rerun()
                except Exception as error:
                    display_generation_error(error, context="argument-practice session")
    else:
        session = st.session_state.argument_practice
        mode_name = (
            "Realistic opposing party"
            if session.get("mode") == "realistic_party"
            else "Opposing lawyer"
        )
        st.markdown("### Current practice session")
        st.caption(
            f"Mode: {mode_name} | Your side: {session.get('user_side', 'Not specified')} "
            f"| Opponent: {session.get('opponent_name', 'Not specified')}"
        )
        with st.expander("Case summary", expanded=False):
            st.write(session.get("case_summary", ""))

        for item in session.get("history", []):
            role = item.get("role", "assistant")
            speaker = "You" if role == "user" else "Opposing side"
            with st.chat_message("user" if role == "user" else "assistant"):
                st.caption(speaker)
                st.markdown(item.get("content", ""))

        if session.get("status") == "active":
            with st.form("argument_practice_turn", clear_on_submit=True):
                user_argument = st.text_area(
                    "Your response",
                    placeholder="Present your argument, explain the facts and respond to the objection...",
                    height=120,
                )
                submit_turn = st.form_submit_button("Submit Argument", type="primary")

            if submit_turn:
                if not user_argument.strip():
                    st.warning("Enter your argument before submitting.")
                else:
                    try:
                        with st.spinner("The opposing side is preparing its response..."):
                            st.session_state.argument_practice = submit_practice_turn(
                                session,
                                user_argument,
                            )
                        st.rerun()
                    except Exception as error:
                        display_generation_error(error, context="practice turn")

            st.markdown("---")
            if st.button("Finish and Evaluate My Arguments", type="primary"):
                try:
                    with st.spinner("Evaluating your arguments..."):
                        evaluate_practice_session(session)
                    st.rerun()
                except Exception as error:
                    display_generation_error(error, context="practice evaluation")

        if session.get("status") == "completed":
            evaluation = session.get("evaluation", {})
            st.markdown("---")
            st.markdown("### Your advocacy feedback")
            st.markdown(evaluation.get("overall_assessment", "No overall assessment was returned."))

            scores = evaluation.get("scores", {})
            if scores:
                score_cols = st.columns(5)
                score_labels = [
                    ("clarity", "Clarity"),
                    ("relevance", "Relevance"),
                    ("evidence_use", "Evidence use"),
                    ("response_to_objections", "Objections"),
                    ("persuasiveness", "Persuasiveness"),
                ]
                for col, (key, label) in zip(score_cols, score_labels):
                    with col:
                        st.metric(label, f"{scores.get(key, '—')}/5")

            feedback_sections = [
                ("Strengths", "strengths"),
                ("Areas to improve", "weaknesses"),
                ("Missing evidence", "missing_evidence"),
                ("Stronger arguments to consider", "stronger_arguments"),
                ("Arguments to prepare for", "opposing_arguments_to_prepare_for"),
                ("Legal points to verify", "legal_points_to_verify"),
                ("Recommended next steps", "recommended_next_steps"),
            ]
            for heading, key in feedback_sections:
                items = evaluation.get(key, [])
                if items:
                    with st.expander(heading, expanded=(key in ("weaknesses", "stronger_arguments"))):
                        if isinstance(items, list):
                            for item in items:
                                st.markdown(f"- {item}")
                        else:
                            st.write(items)

        sources = session.get("sources", [])
        if sources:
            with st.expander("View legal passages retrieved for this session"):
                display_sources(sources)

        if st.button("Start a New Practice Session"):
            st.session_state.argument_practice = None
            st.rerun()


# ==================================================
# 15. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        BARS AI | AI-powered legal information assistant
        <br>
        Prototype - Verify legal information before relying on it.
    </div>
    """,
    unsafe_allow_html=True,
)
