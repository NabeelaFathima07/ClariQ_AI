"""ClariQ — Enterprise Company Internal-Document Knowledge-Base Chatbot.
Built with Python, Streamlit, and Voice Assistance on the Truth Layer architecture.
"""

import io
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Ensure database path resolves to the project directory if not overridden
if not os.getenv("TL_DB_PATH"):
    os.environ["TL_DB_PATH"] = str(PROJECT_DIR / "truthlayer.db")

import gtts
import speech_recognition as sr
import streamlit as st
from app import demo, gaps, ingest, owner_loop, qa, reasoning, reviews
from app.config import DEFAULT_AUTHORITY, cfg
from app.db import DB
from app.textutil import today

# -----------------------------------------------------------------------------
# Streamlit Page Config & Professional Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ClariQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-brand {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #0F172A;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.2rem;
    }
    .app-brand span {
        color: #2563EB;
    }

    .persona-bar {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 8px 14px;
        margin-bottom: 14px;
    }

    .citation-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.90rem;
    }

    .citation-badge {
        background: #DBEAFE;
        color: #1E40AF;
        padding: 2px 7px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        margin-right: 6px;
    }

    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 9px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 2px 9px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 9px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    
    .approval-card {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .notification-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }

    .merged-search-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Database & Tenant Management
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db():
    database = DB(cfg.db_path)
    database.ensure_tenant("Acme Corp", expert="admin@acme.com")
    if not database.one("SELECT id FROM documents LIMIT 1"):
        demo.seed(database)
    return database


db = get_db()
tenant = db.one("SELECT * FROM tenants ORDER BY id LIMIT 1") or db.ensure_tenant()


# -----------------------------------------------------------------------------
# Multilingual Support & Voice Helpers
# -----------------------------------------------------------------------------
LANGUAGE_MAP = {
    "Auto-Detect / English": ("auto", "en", "en"),
    "Spanish (Español)": ("es-ES", "es", "es"),
    "French (Français)": ("fr-FR", "fr", "fr"),
    "German (Deutsch)": ("de-DE", "de", "de"),
    "Hindi (हिन्दी)": ("hi-IN", "hi", "hi"),
    "Japanese (日本語)": ("ja-JP", "ja", "ja"),
    "Chinese (中文)": ("zh-CN", "zh-CN", "zh-CN"),
    "Italian (Italiano)": ("it-IT", "it", "it"),
    "Portuguese (Português)": ("pt-PT", "pt", "pt"),
    "Arabic (العربية)": ("ar-SA", "ar", "ar"),
}


def translate_to_english(text: str, lang_name: str) -> str:
    """Translate incoming user query to English for optimal knowledge-base retrieval."""
    if not text or lang_name == "Auto-Detect / English":
        return text
    src_loc, target_code, gtts_code = LANGUAGE_MAP.get(lang_name, ("auto", "en", "en"))
    try:
        from deep_translator import MyMemoryTranslator
        return MyMemoryTranslator(source=src_loc, target="en-GB").translate(text)
    except Exception:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            return text


def translate_from_english(text: str, lang_name: str) -> str:
    """Translate verified English answer back into user's preferred language."""
    if not text or lang_name == "Auto-Detect / English":
        return text
    src_loc, target_code, gtts_code = LANGUAGE_MAP.get(lang_name, ("auto", "en", "en"))
    try:
        from deep_translator import MyMemoryTranslator
        res = MyMemoryTranslator(source="en-GB", target=src_loc).translate(text)
        return res if res else text
    except Exception:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source="en", target=gtts_code).translate(text)
        except Exception:
            return text


def clean_for_speech(text: str) -> str:
    """Strip bracketed citations and markdown for clear voice synthesis."""
    t = re.sub(r"\[\d+\]", "", text)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[*_#`~>]", "", t)
    return t.strip()


def text_to_speech_bytes(text: str, lang_name: str = "Auto-Detect / English") -> io.BytesIO:
    """Generate audio bytes for speech playback in the user's chosen language."""
    clean = clean_for_speech(text)
    _, _, gtts_code = LANGUAGE_MAP.get(lang_name, ("auto", "en", "en"))
    fp = io.BytesIO()
    try:
        tts = gtts.gTTS(text=clean[:600], lang=gtts_code)
        tts.write_to_fp(fp)
    except Exception:
        tts = gtts.gTTS(text=clean[:600], lang="en")
        tts.write_to_fp(fp)
    fp.seek(0)
    return fp


def transcribe_microphone_audio(audio_bytes: bytes, lang_name: str = "Auto-Detect / English") -> str:
    """Transcribe user microphone audio to text."""
    r = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    src_loc, _, _ = LANGUAGE_MAP.get(lang_name, ("en-US", "en", "en"))
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        try:
            # Try transcribing in user's specified language locale
            return r.recognize_google(audio_data, language=src_loc if src_loc != "auto" else "en-US")
        except Exception:
            return r.recognize_google(audio_data)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_user_email" not in st.session_state:
    st.session_state.active_user_email = "bob@acme.com"
if "selected_doc_filter" not in st.session_state:
    st.session_state.selected_doc_filter = "All Documents (Global)"
if "auto_speech" not in st.session_state:
    st.session_state.auto_speech = False
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "Auto-Detect / English"
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

# -----------------------------------------------------------------------------
# ClariQ Login / Person Selection (demo mode)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ClariQ Login / Person Selection (demo mode)
# -----------------------------------------------------------------------------
if "login_complete" not in st.session_state:
    st.session_state.login_complete = False

if not st.session_state.login_complete:
    st.markdown(
        """
        <div style="max-width:650px;margin:70px auto 20px;text-align:center;">
            <div style="font-size:48px;font-weight:800;color:#2563EB;">
                ClariQ
            </div>
            <h2>Welcome to your company knowledge hub</h2>
            <p style="color:#64748b;">Sign in to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_users = db.all(
        "SELECT email, name, role, department FROM users "
        "WHERE tenant_id=? ORDER BY id",
        (tenant["id"],),
    )

    # DEMO ONLY: replace these with secure authentication before company use.
    demo_passwords = {
        "admin@acme.com": "Admin@123",
        "hr@acme.com": "HR@12345",
        "finance@acme.com": "Finance@123",
        "asha@acme.com": "Asha@123",
        "bob@acme.com": "Bob@123",
        "carol@acme.com": "Carol@123",
    }

    if not login_users:
        st.error("No users found. Run the demo seed script first.")
        st.stop()

    user_by_label = {
        f"{u['name'] or u['email']} — {u['role'].upper()} ({u['email']})": u
        for u in login_users
    }

    with st.form("clariq_demo_login"):
        selected_label = st.selectbox(
            "Choose your profile",
            list(user_by_label.keys()),
        )
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Sign in",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        selected_user = user_by_label[selected_label]
        expected_password = demo_passwords.get(selected_user["email"])

        if expected_password and password == expected_password:
            st.session_state.active_user_email = selected_user["email"]
            st.session_state.login_complete = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()

# -----------------------------------------------------------------------------
# Top-Left Corner: System Information & Features Guide (Professional & Uncluttered)
# -----------------------------------------------------------------------------
with st.sidebar:
    with st.expander("ℹ️ How ClariQ Runs & System Features", expanded=False):
        st.markdown(
            """
            **ClariQ Architecture & Features:**
            
            * ⚡ **Pre-Retrieval ACL Enforcement:** Access controls are filtered *before* semantic scoring to prevent permission leaks.
            * 📑 **Page & Line Citations:** Every answer pinpoints the exact page and line range in the source document.
            * ⚖️ **Authority Precedence:** Signed Policies (`100`) override Handbooks (`80`) and Team Docs (`50`).
            * 🔔 **HR Approval Workflow:** Employee uploads require HR acceptance before becoming company-wide knowledge.
            * 🌐 **Multilingual Voice Engine:** Dictate or type in any language; ClariQ queries the documents and answers in your native language.
            * 🛡️ **Zero Hallucination:** Computes confidence scores and routes weak evidence to human experts.
            """
        )
    
    if st.button("🔓 Log out / Switch person", use_container_width=True):
        st.session_state.login_complete = False
        st.session_state.messages = []
        st.rerun()
    st.caption("Active Workspace")
    current_user = db.one("SELECT * FROM users WHERE tenant_id=? AND email=?", (tenant["id"], st.session_state.active_user_email))
    if not current_user:
        current_user = db.one("SELECT * FROM users WHERE tenant_id=? ORDER BY id LIMIT 1", (tenant["id"],))
        st.session_state.active_user_email = current_user["email"]

    st.markdown(
        f"**Logged in as:** `{current_user['name']}`  \n"
        f"**Role:** `{current_user['role'].upper()}` | **Dept:** `{current_user.get('department') or 'Global'}`  \n"
        f"**Groups:** `{', '.join(current_user['groups']) if current_user['groups'] else 'None'}`"
    )

    st.divider()

    st.subheader("📁 Document Scope")
    all_docs = db.all("SELECT id, title, version, status FROM documents WHERE tenant_id=? AND status='active' ORDER BY id DESC", (tenant["id"],))
    doc_filter_options = ["All Documents (Global)"] + [f"#{d['id']} — {d['title']} (v{d['version']})" for d in all_docs]

    current_filter_idx = 0
    if st.session_state.selected_doc_filter in doc_filter_options:
        current_filter_idx = doc_filter_options.index(st.session_state.selected_doc_filter)

    selected_doc_filter = st.selectbox(
        "Search Knowledge From:",
        options=doc_filter_options,
        index=current_filter_idx,
        help="Select a specific document to restrict answers strictly to that file.",
    )
    st.session_state.selected_doc_filter = selected_doc_filter
    target_doc_id = None
    if selected_doc_filter != "All Documents (Global)":
        target_doc_id = int(selected_doc_filter.split(" — ")[0].replace("#", ""))
        st.info(f"Targeting Document #{target_doc_id}")

    st.divider()

    st.subheader("⏱️ Historical Policy ('As Of')")
    enable_as_of = st.checkbox("Query historical date", value=False)
    as_of_date = None
    if enable_as_of:
        as_of_val = st.date_input("Policy in effect on:", value=date(2025, 6, 1))
        as_of_date = as_of_val.isoformat()
        st.caption(f"Testing policy as of: `{as_of_date}`")

    st.divider()

    with st.expander("🛠️ Maintenance"):
        if st.button("🔄 Reset Demo Database", use_container_width=True):
            for tbl in ["documents", "chunks", "claims", "conflicts", "question_logs", "owner_requests", "gap_tasks", "notifications", "audit_log"]:
                db.run(f"DELETE FROM {tbl}")
            demo.seed(db)
            st.session_state.messages = []
            st.session_state.selected_doc_filter = "All Documents (Global)"
            st.success("Acme Corp database re-seeded!")
            st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# -----------------------------------------------------------------------------
# Clean Header: Just App Name & 1-Click Persona Switcher
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([4, 6])
with col_h1:
    st.markdown('<div class="app-brand">Clari<span>Q</span></div>', unsafe_allow_html=True)

with col_h2:
    # 1-Click Persona Switcher (Seamless toggle between Admin, HR, Employee)
    st.markdown("**Switch Persona (1-Click):**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        is_admin_active = current_user["role"] == "admin"
        if st.button("🛡️ Admin", key="btn_switch_admin", type="primary" if is_admin_active else "secondary", use_container_width=True):
            st.session_state.active_user_email = "admin@acme.com"
            st.rerun()

    with p_col2:
        is_hr_active = current_user["role"] == "expert" and current_user.get("department") == "hr"
        if st.button("👥 HR", key="btn_switch_hr", type="primary" if is_hr_active else "secondary", use_container_width=True):
            st.session_state.active_user_email = "hr@acme.com"
            st.rerun()

    with p_col3:
        is_emp_bob = current_user["email"] == "bob@acme.com"
        if st.button("💼 Bob (US Eng)", key="btn_switch_bob", type="primary" if is_emp_bob else "secondary", use_container_width=True):
            st.session_state.active_user_email = "bob@acme.com"
            st.rerun()

    with p_col4:
        is_emp_carol = current_user["email"] == "carol@acme.com"
        if st.button("💰 Carol (Finance)", key="btn_switch_carol", type="primary" if is_emp_carol else "secondary", use_container_width=True):
            st.session_state.active_user_email = "carol@acme.com"
            st.rerun()


# -----------------------------------------------------------------------------
# Live Inbox Badge Calculation
# -----------------------------------------------------------------------------
pending_approvals = db.all("SELECT * FROM documents WHERE tenant_id=? AND status='pending_approval' ORDER BY id DESC", (tenant["id"],))
unread_notifications = db.all("SELECT * FROM notifications WHERE tenant_id=? AND email=? AND read=0 ORDER BY id DESC", (tenant["id"], current_user["email"]))

inbox_badge_text = "🔔 Inbox"
if current_user["role"] in ("admin", "expert") and pending_approvals:
    inbox_badge_text = f"🔔 Inbox ({len(pending_approvals)} Approval{'s' if len(pending_approvals) > 1 else ''})"
elif unread_notifications:
    inbox_badge_text = f"🔔 Inbox ({len(unread_notifications)})"


# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_upload, tab_inbox, tab_kb, tab_conflicts, tab_audit = st.tabs([
    "💬 Chat",
    "📤 Upload",
    inbox_badge_text,
    "📚 Knowledge Base",
    "⚖️ Policy Conflicts",
    "📈 Audit & Analytics",
])


# =============================================================================
# TAB 1: Chat with ClariQ (Unified Search & Multilingual Voice Assistance)
# =============================================================================
with tab_chat:
    scope_str = "All Documents" if target_doc_id is None else f"Document #{target_doc_id}"
    
    # Preset question pills
    if target_doc_id is None and not st.session_state.messages:
        st.markdown("**💡 Quick prompts:**")
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            if st.button("🏖️ How many vacation days do I get?", use_container_width=True):
                st.session_state["staged_query"] = "How many vacation days do I get?"
        with p_c2:
            if st.button("🍽️ What is the meal allowance when travelling?", use_container_width=True):
                st.session_state["staged_query"] = "What is the meal allowance when travelling?"
        with p_c3:
            if st.button("💰 What is the CEO bonus target?", use_container_width=True):
                st.session_state["staged_query"] = "What is the CEO bonus target?"

    # Display Chat History
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]

                # Confidence badge
                score = meta["confidence"]["score"]
                level = meta["confidence"]["level"]
                badge_class = f"badge-{level}"
                st.markdown(
                    f'<span class="{badge_class}">Confidence: {level.upper()} ({score:.0%})</span>',
                    unsafe_allow_html=True,
                )

                # Show translated note if applicable
                if meta.get("translated_lang") and meta["translated_lang"] != "Auto-Detect / English":
                    st.caption(f"🌐 Translated for {meta['translated_lang']} | English source: *\"{meta.get('raw_english_answer', '')}\"*")

                # Voice audio playback
                v_col1, _ = st.columns([3, 7])
                with v_col1:
                    if st.button("🔊 Listen to Answer", key=f"speak_ans_{msg_idx}"):
                        try:
                            audio_io = text_to_speech_bytes(msg["content"], meta.get("translated_lang", "Auto-Detect / English"))
                            st.audio(audio_io, format="audio/mp3", autoplay=True)
                        except Exception as ex:
                            st.caption(f"Speech error: {ex}")

                # Citations with Page & Line Numbers
                if meta.get("sources"):
                    with st.expander(f"📖 View {len(meta['sources'])} Citation(s) with Page & Line Numbers"):
                        for i, src in enumerate(meta["sources"], 1):
                            auth = src.get("authority", 50)
                            doc_title = src.get("document") or "Unknown Document"
                            page_num = src.get("page", 1)
                            lines_str = src.get("lines", "Line 1")
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <span class="citation-badge">[{i}]</span>
                                    <strong>{doc_title}</strong> (v{src.get('version', 1)}) — 
                                    <span class="citation-badge">Page {page_num}</span> 
                                    <span class="citation-badge">{lines_str}</span> 
                                    (Authority: <code>{auth}/100</code>)<br/>
                                    <em>"{src.get('text')}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Precedence / Conflict Notes
                if meta.get("notes"):
                    for n in meta["notes"]:
                        if "takes precedence" in n:
                            st.info(f"📌 **Precedence Rule:** {n}")
                        elif "disagree" in n:
                            st.warning(f"⚠️ **Conflict:** {n}")

                # Feedback widget
                if meta.get("question_id"):
                    qid = meta["question_id"]
                    fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                    with fb_col1:
                        if st.button("👍 Helpful", key=f"thumb_up_{qid}"):
                            qa.feedback(db, tenant, current_user, qid, True, "Helpful")
                            st.toast("Feedback recorded: Helpful! Thank you.")
                    with fb_col2:
                        if st.button("👎 Inaccurate", key=f"thumb_down_{qid}"):
                            qa.feedback(db, tenant, current_user, qid, False, "Not helpful")
                            st.toast("Feedback recorded: Inaccurate.")

    # -------------------------------------------------------------------------
    # Merged Searchbar & Multilingual Voice Assistance Console
    # -------------------------------------------------------------------------
    st.markdown('<div class="merged-search-box">', unsafe_allow_html=True)
    st.caption("🎙️ Unified Search & Multilingual Voice Assistant (Speak or type in any language)")
    
    col_v1, col_v2 = st.columns([3, 7])
    with col_v1:
        lang_selection = st.selectbox(
            "Language",
            options=list(LANGUAGE_MAP.keys()),
            index=list(LANGUAGE_MAP.keys()).index(st.session_state.selected_lang),
            label_visibility="collapsed",
            help="Select any language. Voice and text questions will be translated and answered in this language.",
        )
        st.session_state.selected_lang = lang_selection
    
    with col_v2:
        voice_rec = st.audio_input("Record Voice Question", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    voice_query_text = None
    if voice_rec is not None:
        rec_bytes = voice_rec.read()
        try:
            with st.spinner("Transcribing your voice..."):
                transcribed = transcribe_microphone_audio(rec_bytes, lang_selection)
                if transcribed and transcribed != st.session_state.last_transcription:
                    st.session_state.last_transcription = transcribed
                    st.success(f"🗣️ Transcribed ({lang_selection}): \"{transcribed}\"")
                    voice_query_text = transcribed
        except Exception as e:
            st.caption(f"Speech note: {e}")

    # Standard chat text input
    chat_prompt = st.chat_input("Ask ClariQ any question from company documents (or speak via microphone above)...")
    staged_query = st.session_state.pop("staged_query", None)
    
    final_query = voice_query_text or staged_query or chat_prompt

    if final_query:
        # Add user query to history
        st.session_state.messages.append({"role": "user", "content": final_query})

        # 1. Translate question to English if user spoke/typed in another language
        with st.spinner("Processing & translating query..."):
            query_in_english = translate_to_english(final_query, lang_selection)

        # 2. Query knowledge base using English question
        with st.spinner("Searching verified claims, pages & lines..."):
            result = qa.ask(
                db=db,
                tenant=tenant,
                user=current_user,
                question=query_in_english,
                as_of=as_of_date,
                document_id=target_doc_id,
            )

        # 3. Translate answer back to user's native language if needed
        english_answer = result["answer"]
        final_answer = english_answer
        if lang_selection != "Auto-Detect / English":
            with st.spinner(f"Translating response to {lang_selection}..."):
                final_answer = translate_from_english(english_answer, lang_selection)

        result["translated_lang"] = lang_selection
        result["raw_english_answer"] = english_answer

        # Append assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "meta": result,
        })
        st.rerun()


# =============================================================================
# TAB 2: Ingest & Upload Documents (with Employee Approval Flow)
# =============================================================================
with tab_upload:
    is_employee = current_user["role"] == "employee"
    if is_employee:
        st.info("ℹ️ **Employee Upload Policy:** Documents uploaded by employees must be reviewed and accepted by HR before becoming accessible to everyone.")

    ingest_mode = st.radio(
        "Ingestion Method:",
        ["📁 Upload Document File", "✍️ Type or Paste Policy Text", "📋 Load Pre-made Sample Document"],
        horizontal=True,
    )

    sample_dir = PROJECT_DIR / "sample_docs"
    sample_files = sorted([f.name for f in sample_dir.glob("*.txt")]) if sample_dir.exists() else []

    with st.form("doc_ingest_form", clear_on_submit=False):
        uploaded_file = None
        raw_text_input = ""
        chosen_sample = None

        if ingest_mode == "📁 Upload Document File":
            uploaded_file = st.file_uploader(
                "Choose a document file",
                type=["txt", "pdf", "docx", "csv", "html", "htm"],
                help="Supported: PDF, Microsoft Word DOCX, Plain Text, CSV tables, and HTML pages.",
            )
        elif ingest_mode == "✍️ Type or Paste Policy Text":
            raw_text_input = st.text_area(
                "Policy Content",
                placeholder="Example:\nRemote Work Equipment Policy 2026.\nEmployees receive a $500 home office equipment stipend every two years.\nMonitors must be requested via IT Helpdesk.",
                height=150,
            )
        else:
            chosen_sample = st.selectbox("Select Sample Document", options=sample_files)
            if chosen_sample:
                sample_preview_path = sample_dir / chosen_sample
                if sample_preview_path.exists():
                    st.caption("Preview:")
                    st.code(sample_preview_path.read_text(encoding="utf-8")[:300] + "...")

        col_u1, col_u2 = st.columns(2)

        with col_u1:
            doc_title = st.text_input("Document Title", placeholder="Defaults to filename")
            source_type_options = list(tenant.get("authority_map", DEFAULT_AUTHORITY).keys())
            source_type_labels = {
                "signed_policy": "Signed Policy (Authority 100)",
                "expert_verified": "Expert Verified (Authority 90)",
                "official_wiki": "Official Wiki / Handbook (Authority 80)",
                "team_doc": "Team Doc / Notes (Authority 50)",
                "slack_export": "Slack / Chat Export (Authority 20)",
            }
            source_type = st.selectbox(
                "Source Type & Authority Ranking",
                options=source_type_options,
                format_func=lambda x: source_type_labels.get(x, x),
                index=source_type_options.index("team_doc") if "team_doc" in source_type_options else 0,
            )

        with col_u2:
            owner_email = st.text_input("Owner Email", value=current_user["email"])
            allowed_groups = st.text_input("Access Control Groups (Optional)", placeholder="e.g. finance, hr")
            effective_from = st.date_input("Effective From Date", value=date.today())

        upload_submitted = st.form_submit_button("🚀 Ingest Document", use_container_width=True)

    if upload_submitted:
        file_bytes = None
        extracted_text = None
        filename_to_use = "document.txt"

        if ingest_mode == "📁 Upload Document File":
            if not uploaded_file:
                st.error("Please select a file.")
            else:
                file_bytes = uploaded_file.read()
                filename_to_use = uploaded_file.name
        elif ingest_mode == "✍️ Type or Paste Policy Text":
            if not raw_text_input.strip():
                st.error("Please provide text.")
            else:
                extracted_text = raw_text_input.strip()
                filename_to_use = "custom_policy.txt"
        else:
            if chosen_sample:
                sample_path = sample_dir / chosen_sample
                extracted_text = sample_path.read_text(encoding="utf-8")
                filename_to_use = chosen_sample

        if file_bytes or extracted_text:
            final_title = doc_title.strip() or os.path.splitext(filename_to_use)[0].replace("_", " ").title()
            groups_list = [g.strip() for g in allowed_groups.split(",") if g.strip()]
            
            # Determine initial status: employees require HR approval!
            init_status = "pending_approval" if is_employee else "active"

            with st.spinner(f"Processing '{final_title}'..."):
                try:
                    doc_id = ingest.create_document(
                        db=db,
                        tenant=tenant,
                        uploader_email=current_user["email"],
                        title=final_title,
                        filename=filename_to_use,
                        data=file_bytes,
                        text=extracted_text,
                        source_type=source_type,
                        owner_email=owner_email.strip() or current_user["email"],
                        allowed_groups=groups_list,
                        effective_from=effective_from.isoformat() if effective_from else None,
                    )

                    summary = ingest.process_document(db, tenant, doc_id, initial_status=init_status)

                    if is_employee:
                        # Send notification to HR / Admin
                        hr_user = db.one("SELECT email FROM users WHERE tenant_id=? AND role='expert' AND department='hr' LIMIT 1", (tenant["id"],))
                        hr_target = hr_user["email"] if hr_user else "admin@acme.com"
                        db.insert("notifications", tenant_id=tenant["id"], email=hr_target,
                                  message=f"🔔 Document #{doc_id} ('{final_title}') uploaded by {current_user['name']} requires HR approval.")
                        db.insert("notifications", tenant_id=tenant["id"], email=current_user["email"],
                                  message=f"⏳ Your document '{final_title}' has been submitted for HR review.")
                        
                        st.warning(f"⏳ **Submitted for HR Review!** Document #{doc_id}: **{final_title}** is awaiting HR approval before becoming accessible to everyone.")
                    else:
                        st.success(f"✅ Successfully ingested document #{doc_id}: **{final_title}** (Status: ACTIVE)")
                        st.session_state["just_uploaded_doc_id"] = doc_id
                        st.session_state["just_uploaded_doc_title"] = final_title

                    c_m1, c_m2, c_m3 = st.columns(3)
                    with c_m1:
                        st.metric("Extracted Claims", summary.get("claims", 0))
                    with c_m2:
                        st.metric("Conflicts Detected", summary.get("conflicts", 0))
                    with c_m3:
                        st.metric("Approval Status", "Pending HR" if is_employee else "Active")

                except ValueError as ve:
                    st.error(f"Upload rejected: {str(ve)}")
                except Exception as ex:
                    st.error(f"Error processing document: {str(ex)}")

    # Ask directly from uploaded document card
    if "just_uploaded_doc_id" in st.session_state:
        recent_doc_id = st.session_state["just_uploaded_doc_id"]
        recent_doc_title = st.session_state.get("just_uploaded_doc_title", "Document")
        st.divider()
        st.markdown(f"**🎯 Ask Directly From `{recent_doc_title}`:**")
        q_doc_text = st.text_input("Question:", key="q_doc_direct_input")
        if st.button("Query Document", key="btn_q_doc_direct"):
            if q_doc_text.strip():
                ans_doc = qa.ask(db, tenant, current_user, q_doc_text.strip(), document_id=recent_doc_id)
                st.markdown(f"**Answer:** {ans_doc['answer']}")
                if ans_doc.get("sources"):
                    src = ans_doc["sources"][0]
                    st.caption(f"📍 Cited from Page {src.get('page', 1)}, {src.get('lines', 'Line 1')}")


# =============================================================================
# TAB 3: Inbox (HR Approval & Notification Center)
# =============================================================================
with tab_inbox:
    st.subheader("Inbox & Notifications")

    # SECTION 1: HR Pending Document Approvals
    if current_user["role"] in ("admin", "expert"):
        st.markdown("### 📥 Documents Awaiting HR Approval")
        pending_docs = db.all("SELECT * FROM documents WHERE tenant_id=? AND status='pending_approval' ORDER BY id DESC", (tenant["id"],))

        if not pending_docs:
            st.success("✨ No documents pending HR approval.")
        else:
            for p_doc in pending_docs:
                with st.expander(f"⚠️ Document #{p_doc['id']}: '{p_doc['title']}' (Uploaded by {p_doc['owner_email']})", expanded=True):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.write(f"**Source Type:** `{p_doc['source_type']}` (Authority: `{p_doc['authority']}`)")
                        st.write(f"**Filename:** `{p_doc['filename']}`")
                        st.write(f"**Uploaded Date:** `{p_doc['created_at']}`")
                    with col_p2:
                        p_claims = db.all("SELECT * FROM claims WHERE document_id=?", (p_doc["id"],))
                        st.write(f"**Extracted Claims ({len(p_claims)}):**")
                        for pc in p_claims[:3]:
                            st.caption(f"- {pc['text']}")

                    btn_app_col1, btn_app_col2, _ = st.columns([2, 2, 6])
                    with btn_app_col1:
                        if st.button(f"✅ Accept & Publish", key=f"appr_{p_doc['id']}"):
                            db.update("documents", p_doc["id"], status="active")
                            db.run("UPDATE claims SET status='active' WHERE document_id=?", (p_doc["id"],))
                            db.insert("notifications", tenant_id=tenant["id"], email=p_doc["owner_email"],
                                      message=f"🎉 HR has accepted and published your document '{p_doc['title']}'! It is now accessible to everyone.")
                            db.audit(tenant["id"], current_user["email"], "document.approved", f"Doc #{p_doc['id']} {p_doc['title']}")
                            st.success(f"Document #{p_doc['id']} accepted! Now active for all employees.")
                            st.rerun()

                    with btn_app_col2:
                        if st.button(f"❌ Reject", key=f"rej_{p_doc['id']}"):
                            db.update("documents", p_doc["id"], status="rejected")
                            db.run("UPDATE claims SET status='rejected' WHERE document_id=?", (p_doc["id"],))
                            db.insert("notifications", tenant_id=tenant["id"], email=p_doc["owner_email"],
                                      message=f"❌ HR did not approve your document '{p_doc['title']}'.")
                            st.warning(f"Document #{p_doc['id']} rejected.")
                            st.rerun()

        st.divider()

    # SECTION 2: General User Notifications
    st.markdown("### 🔔 Recent Notifications")
    user_notifs = db.all("SELECT * FROM notifications WHERE tenant_id=? AND email=? ORDER BY id DESC LIMIT 30", (tenant["id"], current_user["email"]))

    if not user_notifs:
        st.info("No notifications in your inbox.")
    else:
        for notif in user_notifs:
            read_badge = "⚪" if notif["read"] else "🔵 NEW"
            st.markdown(
                f"""
                <div class="notification-item">
                    <small><strong>{read_badge}</strong> • {notif['created_at']}</small><br/>
                    {notif['message']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Mark All Notifications as Read"):
            db.run("UPDATE notifications SET read=1 WHERE tenant_id=? AND email=?", (tenant["id"], current_user["email"]))
            st.rerun()

    # SECTION 3: Knowledge Gap Owner Requests
    if current_user["role"] in ("admin", "expert"):
        st.divider()
        st.markdown("### ❓ Unanswered Questions (Owner Inbox)")
        owner_reqs = db.all("SELECT * FROM owner_requests WHERE tenant_id=? AND status='pending' ORDER BY id DESC", (tenant["id"],))
        if not owner_reqs:
            st.caption("No pending owner requests.")
        else:
            for o_req in owner_reqs:
                with st.expander(f"Question: \"{o_req['question']}\" (Asked {o_req['ask_count']}x)"):
                    ans_text = st.text_area("Provide Expert Verified Answer:", key=f"ans_req_{o_req['id']}")
                    if st.button("Submit Verified Answer", key=f"btn_sub_ans_{o_req['id']}"):
                        if ans_text.strip():
                            cid = owner_loop.answer_request(db, tenant, o_req["id"], current_user, ans_text.strip())
                            st.success(f"Verified claim #{cid} created! ClariQ will now answer this question.")
                            st.rerun()


# =============================================================================
# TAB 4: Knowledge Base Explorer
# =============================================================================
with tab_kb:
    st.subheader("Knowledge Base Documents & Claims")
    all_kb_docs = db.all("SELECT * FROM documents WHERE tenant_id=? ORDER BY id DESC", (tenant["id"],))

    for d in all_kb_docs:
        status_icon = "🟢 ACTIVE" if d["status"] == "active" else "⏳ PENDING HR" if d["status"] == "pending_approval" else "🔴 REJECTED"
        with st.expander(f"{d['title']} (v{d['version']}) — {status_icon}"):
            st.write(f"**Authority:** `{d['authority']}/100` | **Owner:** `{d['owner_email']}` | **Status:** `{d['status'].upper()}`")
            doc_c = db.all("SELECT * FROM claims WHERE document_id=?", (d["id"],))
            st.write(f"**Extracted Claims ({len(doc_c)}):**")
            for c in doc_c:
                st.markdown(f"- **{c['subject'] or 'General'}:** {c['text']}")


# =============================================================================
# TAB 5: Policy Conflicts & Contradiction Resolution
# =============================================================================
with tab_conflicts:
    st.subheader("Policy Contradiction Detection")
    conflicts = db.all("SELECT * FROM conflicts WHERE tenant_id=? ORDER BY id DESC", (tenant["id"],))
    if not conflicts:
        st.success("No contradictions detected.")
    else:
        for cf in conflicts:
            ca = db.get("claims", cf["claim_a"])
            cb = db.get("claims", cf["claim_b"])
            is_res = cf["status"] == "resolved"
            with st.expander(f"{'✅ RESOLVED' if is_res else '⚠️ OPEN'} #{cf['id']} — {cf.get('reason') or 'Conflict'}"):
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    st.write(f"**Claim A ({ca.get('source_label')}):** {ca.get('text') if ca else 'Deleted'}")
                with c_c2:
                    st.write(f"**Claim B ({cb.get('source_label')}):** {cb.get('text') if cb else 'Deleted'}")
                if is_res:
                    winner = db.get("claims", cf["winner_claim"])
                    st.success(f"Winner: Claim #{cf['winner_claim']} ({winner.get('source_label') if winner else ''}) — {cf.get('resolution')}")
                elif current_user["role"] in ("admin", "expert"):
                    choice = st.radio(f"Pick winning claim for #{cf['id']}:", [cf["claim_a"], cf["claim_b"]], key=f"conf_choice_{cf['id']}")
                    note = st.text_input("Resolution note:", "HR policy takes precedence.", key=f"conf_note_{cf['id']}")
                    if st.button("Resolve Conflict", key=f"btn_res_conf_{cf['id']}"):
                        reasoning.resolve_conflict(db, cf["id"], choice, current_user["email"], note)
                        st.success("Resolved!")
                        st.rerun()


# =============================================================================
# TAB 6: Audit Trail & Analytics
# =============================================================================
with tab_audit:
    st.subheader("Audit Trail & Query Activity")
    q_logs = db.all("SELECT q.*, u.name, u.email FROM question_logs q LEFT JOIN users u ON u.id=q.user_id WHERE q.tenant_id=? ORDER BY q.id DESC LIMIT 40", (tenant["id"],))

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Queries", len(q_logs))
    with m_col2:
        ans_rate = (sum(1 for q in q_logs if q["answered"]) / len(q_logs) * 100) if q_logs else 100
        st.metric("Answer Rate", f"{ans_rate:.0f}%")
    with m_col3:
        avg_conf = (sum(q["confidence"] or 0 for q in q_logs) / len(q_logs)) if q_logs else 0
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

    if q_logs:
        table_rows = []
        for q in q_logs:
            table_rows.append({
                "User": q["email"] or "Anonymous",
                "Question": q["question"],
                "Confidence": f"{(q['confidence'] or 0):.0%}",
                "Answered": "Yes" if q["answered"] else "No",
                "Date": q["created_at"],
            })
        st.dataframe(table_rows, use_container_width=True)
