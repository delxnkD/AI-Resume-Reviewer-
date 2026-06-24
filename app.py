import streamlit as st
import pdfplumber
import os
import time
import re
import io
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

MAX_REQUESTS_PER_WINDOW = 5
TIME_WINDOW_SECONDS     = 300

def check_rate_limit() -> bool:
    now = time.time()
    if "request_log" not in st.session_state:
        st.session_state.request_log = []
    st.session_state.request_log = [
        t for t in st.session_state.request_log
        if now - t < TIME_WINDOW_SECONDS
    ]
    if len(st.session_state.request_log) >= MAX_REQUESTS_PER_WINDOW:
        return False
    st.session_state.request_log.append(now)
    return True

MAX_INPUT_LENGTH   = 8000
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024

def validate_input(text: str, field_name: str, min_length: int = 10) -> tuple[bool, str]:
    if not isinstance(text, str):
        return False, f"{field_name} must be plain text."
    text = text.strip()
    if len(text) < min_length:
        return False, f"{field_name} is too short (minimum {min_length} characters)."
    if len(text) > MAX_INPUT_LENGTH:
        return False, f"{field_name} is too long (maximum {MAX_INPUT_LENGTH} characters)."
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text):
        return False, f"{field_name} contains invalid characters."
    return True, ""

def sanitize_input(text: str) -> str:
    return text.strip().replace("\x00", "")

def extract_text_from_pdf(uploaded_file) -> tuple[bool, str]:
    file_bytes = uploaded_file.read()
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        return False, f"PDF file is too large (max {MAX_PDF_SIZE_BYTES // (1024*1024)} MB)."
    if uploaded_file.type != "application/pdf":
        return False, "Invalid file type. Please upload a PDF file only."
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages[:20]).strip()
        if not text:
            return False, "Could not extract text. File may be scanned — try pasting text instead."
        return True, text[:MAX_INPUT_LENGTH]
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return False, "Failed to read the PDF. Try pasting text instead."

def analyze_resume(job_description: str, resume: str) -> str:
    prompt = f"""
You are an expert HR analyst, career coach, and ATS specialist with deep knowledge of hiring standards across different countries.

Today's date is {time.strftime("%B %Y")}. When reviewing dates on the resume:
- If any dates appear to be in the future, ignore this and treat them as valid current or past experience. Do NOT penalize or flag future dates.
- Focus entirely on the content, skills, and experience — not the formatting of dates.

Analyze the resume against the job description and return ONLY the following structured analysis in clean markdown. Do not use any emojis. Do not add any preamble outside this structure.

---

## Overall Match Score
Give a single score out of 100 with a two-sentence explanation.
Display it visually: ████████░░ 72/100

---

## Sub-scores
Rate each out of 10 with one sentence:
- Skills Match: /10
- Experience Match: /10
- Language Match: /10 (does the candidate's language match the job location — e.g. German for Germany)
- ATS Score: /10 (keyword density, standard headers, absence of tables/graphics that hurt ATS parsing)

---

## Country-Specific Standards
Identify the country from the job description.
Evaluate against that country's hiring norms.
Note: Do NOT comment on whether a photo is included or missing, and do NOT mention nationality or marital status — these are not relevant criteria.
- Germany: formal tone, chronological order, Lebenslauf format, date of birth
- UK/US: no personal details, achievement-focused, concise
- France: formal tone, personal details common
Flag only what is relevant and observable from the text content of the resume.

---

## ATS Friendliness
- Top 5 keywords from job description: state Yes/No if each appears in resume
- Section headers: are standard headers used? Yes/No with notes
- Format risks: tables, columns, graphics, or special characters that could break ATS parsing
- Resume length: appropriate for the role level?

---

## Keyword Density
List 5 key terms from the job description.
For each: present (Yes/No) and count of appearances.

---

## Matching Strengths
3-5 bullet points of what aligns well.

---

## Missing Skills / Gaps
3-5 bullet points of what is missing or weak.

---

## Language & Tone
- Appropriate language for the job location?
- Formal and professional tone?
- Strong action verbs or passive language? Give 2-3 examples.
- Resume length appropriate for seniority?

---

## Suggestions to Improve
5 specific, actionable bullet points for this exact role and country.

---

## Verdict
Two short paragraphs: (1) should this person apply and why, (2) the single most important fix before applying.

---

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}
"""
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return response.text

def render_results(markdown_text: str):
    sections = re.split(r'\n## ', markdown_text)
    
    colors = {
        "Overall Match Score": "#1a1a2e",
        "Sub-scores": "#16213e",
        "Country-Specific Standards": "#0f3460",
        "ATS Friendliness": "#1a1a2e",
        "Keyword Density": "#16213e",
        "Matching Strengths": "#1b4332",
        "Missing Skills / Gaps": "#3b1a1a",
        "Language & Tone": "#1a2a3a",
        "Suggestions to Improve": "#2a1a3e",
        "Verdict": "#1a1a1a",
    }

    for i, section in enumerate(sections):
        if not section.strip():
            continue
        if i == 0:
            section = section.lstrip('#').strip()
            if not section:
                continue
            lines = section.split('\n', 1)
            title = lines[0].strip()
            body  = lines[1].strip() if len(lines) > 1 else ""
        else:
            lines = section.split('\n', 1)
            title = lines[0].strip()
            body  = lines[1].strip() if len(lines) > 1 else ""

        bg = colors.get(title, "#1a1a1a")

        st.markdown(f"""
        <div style="
            background: {bg};
            border-radius: 12px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.06);
        ">
            <p style="font-size:11px; font-weight:600; letter-spacing:0.08em;
                      text-transform:uppercase; color:rgba(255,255,255,0.4);
                      margin:0 0 8px 0;">{title}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown(body)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

st.set_page_config(page_title="AI Resume Screener", page_icon="◆", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 3rem 4rem 3rem 4rem !important; max-width: 1200px !important; }

    h1 { font-size: 26px !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
    h1 a, h2 a, h3 a { display: none !important; }

    [data-testid="stTextArea"] textarea {
        border: 1.5px solid #e8e8e8 !important;
        border-radius: 12px !important;
        font-size: 13.5px !important;
        line-height: 1.6 !important;
        background: #fff !important;
        color: #1a1a1a !important;
        padding: 14px 16px !important;
        resize: none !important;
    }
    [data-testid="stTextArea"] textarea:focus { border-color: #1a1a1a !important; box-shadow: none !important; }
    [data-testid="stTextArea"] textarea::placeholder { color: #ccc !important; }
    [data-testid="stTextArea"] label { display: none !important; }

    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #ddd !important;
        border-radius: 12px !important;
        padding: 1.4rem 1rem !important;
        background: #fafafa !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #aaa !important; }

    [data-testid="stButton"] > button[kind="primary"] {
        background: #1a3a6b !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        height: 50px !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover { background: #15305a !important; }

    hr { display: none !important; }

    [data-testid="stRadio"] label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered Resume Screener")
st.markdown(
    "Provide a **job description** and your **resume** below. "
    "The AI will score your match and suggest improvements."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    jd_mode = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True, key="jd_mode")
    job_desc_input = ""
    if jd_mode == "Paste text":
        st.caption("Paste the full job posting. Supports plain text or Markdown format.")
        job_desc_input = st.text_area("Job Description", placeholder="Paste the job description here...", height=320, label_visibility="collapsed")
    else:
        st.caption("Upload the job description as a PDF. Max 20 MB, first 20 pages used.")
        jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd_upload", label_visibility="collapsed")
        if jd_file:
            success, result = extract_text_from_pdf(jd_file)
            if success:
                job_desc_input = result
                st.success(f"Extracted {len(result):,} characters from PDF.")
                with st.expander("Preview extracted text"):
                    st.text(result[:500] + ("..." if len(result) > 500 else ""))
            else:
                st.error(result)

with col2:
    st.subheader("Your Resume")
    rv_mode = st.radio("Input method", ["Paste text", "Upload PDF"], horizontal=True, key="rv_mode")
    resume_input = ""
    if rv_mode == "Paste text":
        st.caption("Paste your resume. Supports plain text or Markdown format.")
        resume_input = st.text_area("Resume", placeholder="Paste your resume here...", height=320, label_visibility="collapsed")
    else:
        st.caption("Upload your resume as a PDF. Max 20 MB, first 20 pages used.")
        rv_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="rv_upload", label_visibility="collapsed")
        if rv_file:
            success, result = extract_text_from_pdf(rv_file)
            if success:
                resume_input = result
                st.success(f"Extracted {len(result):,} characters from PDF.")
                with st.expander("Preview extracted text"):
                    st.text(result[:500] + ("..." if len(result) > 500 else ""))
            else:
                st.error(result)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

if st.button("Analyze My Resume", type="primary", use_container_width=True):
    if not check_rate_limit():
        st.warning(f"Rate limit reached: {MAX_REQUESTS_PER_WINDOW} analyses allowed every {TIME_WINDOW_SECONDS // 60} minutes.")
        st.stop()

    jd_valid, jd_error = validate_input(job_desc_input, "Job Description", min_length=20)
    rv_valid, rv_error = validate_input(resume_input, "Resume", min_length=50)

    if not jd_valid:
        st.error(f"Job Description: {jd_error}")
        st.stop()
    if not rv_valid:
        st.error(f"Resume: {rv_error}")
        st.stop()

    clean_jd     = sanitize_input(job_desc_input)
    clean_resume = sanitize_input(resume_input)

    with st.spinner("Analyzing your resume..."):
        try:
            result = analyze_resume(clean_jd, clean_resume)
            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
            render_results(result)
        except Exception as e:
            st.error("Something went wrong. Please check your API key and try again.")
            print(f"[ERROR] Gemini API call failed: {e}")

st.markdown("<p style='font-size:12px;color:#aaa;text-align:center;margin-top:2rem;'>AI-generated suggestions only — always review with a human recruiter.</p>", unsafe_allow_html=True)