from pathlib import Path
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
