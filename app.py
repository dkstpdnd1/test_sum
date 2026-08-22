import streamlit as st

# Streamlit page configuration must be called exactly once in the navigation entrypoint.
st.set_page_config(
    page_title="ICN T2 통합 운영 시스템",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation shell shared by all pages. Page-specific visual systems remain in each page.
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
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:#07111F !important;
}
[data-testid="stHeader"] {
  background:rgba(7,17,31,.94) !important;
  border-bottom:1px solid rgba(38,58,82,.72) !important;
}
[data-testid="stSidebar"] {
  background:var(--nav-bg) !important;
  border-right:1px solid var(--nav-line) !important;
}
[data-testid="stSidebar"] * {
  color:var(--nav-ink);
}
/* st.navigation sidebar: card-like separation without relying on the app page CSS. */
[data-testid="stSidebarNav"] {
  padding:10px 8px 8px 8px;
}
[data-testid="stSidebarNav"] ul {
  gap:7px !important;
}
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
/* Make collapsed/expanded sidebar edge unobtrusive on all pages. */
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

pg = st.navigation(
    [
        st.Page("pages/01_app_1.py", title="통합 운영 상황판", icon="🏠", default=True),
        st.Page("pages/02_app_2.py", title="T2 운영 최적화 수정 시스템", icon="📈"),
        st.Page("pages/03_app_3.py", title="가상 운영 시나리오 & AI 의사결정 지원", icon="⚙️"),
    ]
)
pg.run()
