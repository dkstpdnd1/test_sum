from pathlib import Path
import sys
import streamlit as st

# -----------------------------------------------------------------------------
# Robust multipage entrypoint
# Works both when Streamlit Cloud runs /app.py and when it is configured to run
# /pages/app.py. Streamlit 1.37 resolves st.Page file paths relative to the
# entrypoint file, so the page path must match the actual entrypoint location.
# -----------------------------------------------------------------------------
ENTRY_DIR = Path(__file__).resolve().parent
RUNNING_FROM_PAGES = ENTRY_DIR.name.lower() == "pages"

if RUNNING_FROM_PAGES:
    PAGE_1 = "01_app_1.py"
    PAGE_2 = "02_app_2.py"
    PAGE_3 = "03_app_3.py"
else:
    PAGE_1 = "pages/01_app_1.py"
    PAGE_2 = "pages/02_app_2.py"
    PAGE_3 = "pages/03_app_3.py"

# Fail early with a readable message if deployment structure is incomplete.
ROOT_DIR = ENTRY_DIR.parent if RUNNING_FROM_PAGES else ENTRY_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

required_files = [
    ROOT_DIR / "pages" / "01_app_1.py",
    ROOT_DIR / "pages" / "02_app_2.py",
    ROOT_DIR / "pages" / "03_app_3.py",
]
missing = [p for p in required_files if not p.exists()]

st.set_page_config(
    page_title="ICN T2 통합 운영 시스템",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
<style>
:root {
  --nav-bg:#081421;
  --nav-panel:#0D1B2A;
  --nav-line:#263A52;
  --nav-ink:#E6EDF3;
  --nav-muted:#91A4B7;
  --nav-accent:#4DA3FF;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:#07111F !important;
  color:var(--nav-ink) !important;
}
[data-testid="stHeader"] {
  background:rgba(7,17,31,.94) !important;
  border-bottom:1px solid rgba(38,58,82,.72) !important;
}
[data-testid="stSidebar"] {
  background:var(--nav-bg) !important;
  border-right:1px solid var(--nav-line) !important;
}
[data-testid="stSidebar"] * { color:var(--nav-ink); }
[data-testid="stSidebarNav"] { padding:10px 8px 8px 8px; }
[data-testid="stSidebarNav"] ul { gap:7px !important; }
[data-testid="stSidebarNav"] a {
  border:1px solid transparent !important;
  border-radius:9px !important;
  padding:8px 10px !important;
  transition:background .15s ease,border-color .15s ease,transform .15s ease;
}
[data-testid="stSidebarNav"] a:hover {
  background:#12243A !important;
  border-color:#314C67 !important;
  transform:translateX(1px);
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background:linear-gradient(180deg,#173B5D,#112D48) !important;
  border-color:#4D8DC5 !important;
  box-shadow:inset 3px 0 0 var(--nav-accent),0 5px 14px rgba(0,0,0,.16);
}
[data-testid="stSidebarNav"] a[aria-current="page"] * {
  color:#F5FAFF !important;
  font-weight:700 !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
  color:var(--nav-ink) !important;
  background:#102238 !important;
  border:1px solid var(--nav-line) !important;
}


/* ------------------------------------------------------------------
   Unified dark widget finish (Streamlit 1.37.x)
   ------------------------------------------------------------------ */
html, body, .stApp { color-scheme: dark !important; }

/* Inputs / selects / dates - including disabled fields */
[data-baseweb="select"] > div,
[data-baseweb="base-input"],
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stTextArea"] textarea {
  background:#0F2033 !important;
  color:#E6EDF3 !important;
  border-color:#304A64 !important;
  box-shadow:none !important;
}
input:disabled,
textarea:disabled,
[data-testid="stDateInput"] input:disabled,
[data-testid="stTextInput"] input:disabled,
[data-testid="stNumberInput"] input:disabled {
  background:#0D1B2A !important;
  color:#9EB1C4 !important;
  -webkit-text-fill-color:#9EB1C4 !important;
  border-color:#263A52 !important;
  opacity:1 !important;
}
[data-baseweb="select"] *, [data-baseweb="base-input"] * {
  color:#E6EDF3 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background:#0F2033 !important;
  color:#E6EDF3 !important;
  border-color:#263A52 !important;
}
[role="option"] {background:#0F2033 !important;color:#E6EDF3 !important;}
[role="option"]:hover, [role="option"][aria-selected="true"] {background:#17304C !important;color:#FFF !important;}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background:#12243A !important;
  color:#E6EDF3 !important;
  border:1px solid #35506C !important;
  border-radius:9px !important;
  box-shadow:none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background:#17304C !important;
  border-color:#5EA9EE !important;
  color:#FFF !important;
}

/* Sliders: dark track, blue handle, and plain min/max/value labels */
[data-testid="stSlider"] [role="slider"] {
  background:#4DA3FF !important;
  border:2px solid #8CC8FF !important;
  box-shadow:0 0 0 2px rgba(77,163,255,.12) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
  background:#38536F !important;
}
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
[data-testid="stSlider"] label {
  color:#DCE8F3 !important;
  background:transparent !important;
  box-shadow:none !important;
  border-color:transparent !important;
}
/* Streamlit 1.37 tick/value wrappers may receive a filled background. Strip it. */
[data-testid="stSlider"] [data-testid*="TickBar"],
[data-testid="stSlider"] [data-testid*="tickBar"],
[data-testid="stSlider"] [class*="TickBar"],
[data-testid="stSlider"] [class*="tickBar"],
[data-testid="stSlider"] [class*="tick-bar"] {
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
}
[data-testid="stSlider"] [data-testid*="TickBar"] *,
[data-testid="stSlider"] [data-testid*="tickBar"] *,
[data-testid="stSlider"] [class*="TickBar"] *,
[data-testid="stSlider"] [class*="tickBar"] *,
[data-testid="stSlider"] [class*="tick-bar"] * {
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
  color:#DCE8F3 !important;
}
/* Fallback for the tiny min/max label boxes used by 1.37.x. */
[data-testid="stSlider"] div[style*="background-color"]:not([role="slider"]),
[data-testid="stSlider"] span[style*="background-color"] {
  background-color:transparent !important;
  box-shadow:none !important;
  border-color:transparent !important;
}

/* Dataframes / editors: dark outer shell and toolbar. Cell colors are also
   styled through Pandas Styler in the individual pages. */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  background:#0D1B2A !important;
  border:1px solid #263A52 !important;
  border-radius:12px !important;
  overflow:hidden !important;
  box-shadow:0 8px 22px rgba(0,0,0,.12) !important;
}
[data-testid="stDataFrame"] > div,
[data-testid="stDataEditor"] > div,
[data-testid="stDataFrame"] [data-testid="stElementToolbar"],
[data-testid="stDataEditor"] [data-testid="stElementToolbar"] {
  background:#0D1B2A !important;
}
[data-testid="stDataFrame"] button,
[data-testid="stDataEditor"] button {
  color:#CFE0EF !important;
  background:#112338 !important;
}

/* Alerts, expanders, code blocks */
[data-testid="stAlert"] {
  background:#102238 !important;
  border:1px solid #2E4965 !important;
  color:#E6EDF3 !important;
  border-radius:10px !important;
}
[data-testid="stExpander"] {
  background:#0D1B2A !important;
  border:1px solid #263A52 !important;
  border-radius:10px !important;
}
pre, code, [data-testid="stJson"] {
  background:#091827 !important;
  color:#CFE5F7 !important;
  border-color:#263A52 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if missing:
    st.error("필수 페이지 파일을 찾을 수 없습니다.")
    st.code("\n".join(str(p) for p in missing))
    st.caption(f"현재 진입 파일: {Path(__file__).resolve()}")
    st.stop()

pg = st.navigation(
    [
        st.Page(PAGE_1, title="통합 운영 상황판", icon="🏠", default=True),
        st.Page(PAGE_2, title="T2 운영 최적화 수정 시스템", icon="📈"),
        st.Page(PAGE_3, title="가상 운영 시나리오 & AI 의사결정 지원", icon="⚙️"),
    ]
)
pg.run()
