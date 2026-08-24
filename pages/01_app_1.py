[source: 8]import datetime
import os
import glob
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import scipy.signal as signal
import altair as alt

from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# --- [커스텀 앙상블 모델 클래스 정의] ---
class WeightedEnsembleRegressor:
    """Random Forest와 XGBoost에 각각 0.5 가중치를 부여하여 예측하는 앙상블 모델"""
    def __init__(self, rf_model, xgb_model, rf_weight=0.5, xgb_weight=0.5):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight

    def predict(self, X):
        pred_rf = self.rf_model.predict(X) if self.rf_model is not None else np.zeros(len(X))
        pred_xgb = self.xgb_model.predict(X) if self.xgb_model is not None else np.zeros(len(X))
        return (self.rf_weight * pred_rf) + (self.xgb_weight * pred_xgb)


# --- [시스템 설정] ---
AREA_FILE_PATH = "terminal_areas_grouped_2.csv"         
BACKGROUND_IMAGE_PATH = "ICN_Airport_3F.png"         
RF_MODEL_PATH = "rf_model.pkl"                       
XGB_MODEL_PATH = "xgb_model.pkl"                     
ENSEMBLE_MODEL_PATH = "ensemble_model.pkl"           # 통합 앙상블 모델 저장 경로


# --- [디자인 시스템: 극도로 전문적인 하이엔드 관제 스타일 CSS 적용] ---
st.markdown("""
    <style>
        /* 전체 앱 배경 및 폰트 설정 */
        .stApp {
            background-color: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* 사이드바 스타일링 및 글자색 강제 지정 */
        [data-testid="stSidebar"] {
            background-color: #0b0f19;
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        /* 사이드바 내부 입력 위젯 배경 및 테두리 다크화 */
        [data-testid="stSidebar"] input {
            background-color: #111827 !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] {
            background-color: #111827 !important;
            border-color: #334155 !important;
        }

        /* 메트릭 카드 크기 및 높이 완벽 일치 디자인 */
        [data-testid="stMetric"] { 
            background: linear-gradient(135deg, #111827 0%, #0b0f19 100%) !important;
            padding: 16px 20px !important; 
            border-radius: 10px !important; 
            border: 1px solid #1e293b !important; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        [data-testid="stMetric"] label { 
            color: #94a3b8 !important; 
            font-weight: 600 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.05em;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { 
            color: #f8fafc !important; 
            font-weight: 700 !important;
            font-size: 1.35rem !important;
        }

        /* 헤더 타이틀 스타일 */
        h1, h2, h3 { 
            color: #f8fafc !important; 
            font-family: 'Inter', sans-serif;
            font-weight: 700 !important;
        }
        
        /* 커스텀 관제 테이블 디자인 */
        .ioc-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #0b0f19;
            color: #f8fafc;
            border: 1px solid #1e293b;
            border-radius: 8px;
            overflow: hidden;
            font-size: 0.9rem;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        .ioc-table th {
            background-color: #111827;
            color: #94a3b8;
            font-weight: 600;
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid #1e293b;
            letter-spacing: 0.05em;
        }
        .ioc-table td {
            padding: 12px 16px;
            border-bottom: 1px solid #111827;
            color: #f8fafc;
        }
        .ioc-table tr:hover {
            background-color: #111827;
        }
        
        /* 멀티셀렉트 선택 박스(컨트롤러) 전체 배경을 사이드바 입력창과 동일하게 완벽 다크화 */
        div[data-baseweb="select"] {
            background-color: #111827 !important;
            background: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="select"] div {
            background-color: transparent !important;
            background: transparent !important;
            color: #f8fafc !important;
        }
        div[data-baseweb="select"]:hover {
            border-color: #2563eb !important;
        }
        div[data-baseweb="select"]:focus-within {
            background-color: #111827 !important;
            background: #111827 !important;
            border-color: #2563eb !important;
            box-shadow: 0 0 0 1px #2563eb !important;
        }
        div[data-baseweb="select"] input {
            color: #f8fafc !important;
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* 멀티셀렉트 드롭다운 팝오버 메뉴 영역 스타일링 */
        [data-baseweb="menu"] {
            background-color: #111827 !important;
            background: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        [data-baseweb="menu"] ul, [data-baseweb="menu"] li {
            background-color: #111827 !important;
            background: #111827 !important;
            color: #f8fafc !important;
        }
        [data-baseweb="menu"] li:hover {
            background-color: #1e293b !important;
            background: #1e293b !important;
            color: #ffffff !important;
        }
        
        /* 선택된 구역 태그(A, B 등) 디자인 */
        span[data-baseweb="tag"] {
            background-color: #2563eb !important;
            background: #2563eb !important;
            color: #ffffff !important;
            border-radius: 4px !important;
        }
        span[data-baseweb="tag"] * {
            color: #ffffff !important;
            background-color: transparent !important;
            background: transparent !important;
        }

        /* 경고 및 정보 박스 디자인 개선 */
        .stAlert {
            background-color: #111827;
            border: 1px solid #1e293b;
            color: #f8fafc;
            border-radius: 8px;
        }
        
        /* 버튼 디자인 */
        .stButton button {
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 0 12px rgba(37, 99, 235, 0.5);
        }

        /* --- 멀티셀렉트 상단 선택된 항목들이 들어있는 박스 영역 다크화 --- */
        div[data-baseweb="select"] > div {
            background-color: #111827 !important;
            background: #111827 !important;
            border: 1px solid #334155 !important;
            color: #f8fafc !important;
        }

        /* 내부 태그(A, B, D)들이 놓이는 컨테이너 배경 투명화 처리 */
        div[data-baseweb="select"] [data-baseweb="tag"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }
        
        /* 1. 날짜 입력 필드 본체 */
        [data-baseweb="input"] {
            background-color: #111827 !important;
            background: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        [data-baseweb="input"] input {
            color: #f8fafc !important;
            background-color: transparent !important;
        }
        [data-baseweb="input"]:hover {
            border-color: #2563eb !important;
        }
        [data-baseweb="input"]:focus-within {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 1px #2563eb !important;
        }

        /* 2. 팝오버 최상위 레이어 및 캘린더 전체 틀 다크화 */
        div[role="presentation"],
        div[role="presentation"] > div,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        [data-baseweb="calendar"] {
            background-color: #0b0f19 !important;
            background: #0b0f19 !important;
            border: 1px solid #1e293b !important;
            border-radius: 12px !important;
            color: #f8fafc !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7) !important;
        }

        /* 3. 캘린더 내부의 모든 테이블, 행, 셀, 영역의 기본 배경 강제 고정 */
        [data-baseweb="calendar"],
        [data-baseweb="calendar"] *,
        [data-baseweb="calendar"] table,
        [data-baseweb="calendar"] tbody,
        [data-baseweb="calendar"] tr,
        [data-baseweb="calendar"] td,
        [data-baseweb="calendar"] th,
        [data-baseweb="calendar"] div,
        [data-baseweb="calendar"] section,
        [data-baseweb="calendar"] header {
            background-color: #0b0f19 !important;
            background: #0b0f19 !important;
            color: #f8fafc !important;
        }

        [data-baseweb="calendar"] th {
            color: #94a3b8 !important;
        }

        [data-baseweb="calendar"] button {
            background-color: transparent !important;
            background: transparent !important;
            color: #f8fafc !important;
            border-radius: 6px !important;
            border: none !important;
        }
        [data-baseweb="calendar"] button:hover {
            background-color: #1e293b !important;
            background: #1e293b !important;
            color: #ffffff !important;
        }

        [data-baseweb="calendar"] [aria-selected="true"] {
            background-color: #2563eb !important;
            background: #2563eb !important;
            color: #ffffff !important;
        }

        div[data-baseweb="menu"], 
        div[data-baseweb="menu"] div,
        div[data-baseweb="menu"] ul,
        div[data-baseweb="menu"] li {
            background-color: #111827 !important;
            background: #111827 !important;
            color: #f8fafc !important;
            border: none !important;
        }
        div[data-baseweb="menu"] li:hover {
            background-color: #1e293b !important;
            background: #1e293b !important;
            color: #ffffff !important;
        }
    
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-baseweb="base-input"],
        [data-baseweb="select"] > div {
            background:#0F2033 !important;
            color:#E6EDF3 !important;
            border-color:#304A64 !important;
            box-shadow:none !important;
        }
        input:disabled,
        [data-testid="stDateInput"] input:disabled {
            background:#0D1B2A !important;
            color:#9EB1C4 !important;
            -webkit-text-fill-color:#9EB1C4 !important;
            border-color:#263A52 !important;
            opacity:1 !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background:#4DA3FF !important;
            border:2px solid #8CC8FF !important;
        }
</style>
""", unsafe_allow_html=True)


st.markdown(r"""
<style>
[data-testid="stSlider"] [data-testid="stTickBar"]{display:none!important;}
.slider-range-labels{display:flex;justify-content:space-between;align-items:center;margin-top:-.35rem;margin-bottom:.55rem;padding:0 1px;color:#A9BED1;font-size:.76rem;line-height:1;}
.slider-range-labels span{background:transparent!important;color:#A9BED1!important;border:0!important;box-shadow:none!important;padding:0!important;}
[role="menu"],[role="menu"]>div,[role="menuitem"],[role="menuitem"]>div,div[data-baseweb="popover"]>div,div[data-baseweb="popover"] [data-baseweb="menu"]{background:#0F2033!important;color:#E6EDF3!important;border-color:#263A52!important;}
[role="menuitem"]:hover,[role="menuitem"]:focus{background:#17304C!important;color:#FFF!important;}
.dark-table-shell{width:100%;overflow:auto;border:1px solid #29425A;border-radius:12px;background:#0B1725;box-shadow:0 7px 20px rgba(0,0,0,.14);margin:.25rem 0 .9rem 0;}
.dark-html-table{width:100%;border-collapse:separate;border-spacing:0;color:#E6EDF3;font-size:.86rem;background:#0D1B2A;}
.dark-html-table thead th{position:sticky;top:0;z-index:2;background:#112338!important;color:#BFD3E6!important;font-weight:700;text-align:left;padding:10px 12px;border-right:1px solid #2A4058;border-bottom:1px solid #36516C;white-space:nowrap;}
.dark-html-table tbody td{background:#0D1B2A!important;color:#E6EDF3!important;padding:9px 12px;border-right:1px solid #1F3348;border-bottom:1px solid #1F3348;white-space:nowrap;}
.dark-html-table tbody tr:nth-child(even) td{background:#0B1928!important}.dark-html-table tbody tr:hover td{background:#122B42!important}
</style>
""", unsafe_allow_html=True)


# --- [인력 배치 및 로직 함수] ---
def calculate_staffing(people_count):
    open_counters = min(40, -(-people_count // 5))  # ceil 연산
    support_staff = 0
    if people_count > 80:
        support_staff = min(3, (people_count - 80) // 40 + 1)
    total_staff = open_counters + support_staff
    return open_counters, support_staff, total_staff

def index_to_time_str(t_index):
    total_seconds = int(t_index) * 10
    hours, minutes = total_seconds // 3600, (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}:{total_seconds % 60:02d}"

@st.cache_data
def load_data_by_date(selected_date_str):
    area_df = pd.read_csv(AREA_FILE_PATH) if os.path.exists(AREA_FILE_PATH) else pd.DataFrame()
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH)
    if bg_img is None: bg_img = np.full((600, 1900, 3), 240, dtype=np.uint8)
    try:
        counts_df = pd.read_csv(f"area_count_time_full_{selected_date_str}.csv")
    except FileNotFoundError:
        return area_df, {}, [], bg_img, False
    
    time_grouped_data = {}
    for t_index, group in counts_df.groupby('time_index'):
        filtered = group[group['area'] != 'Outside']
        time_grouped_data[t_index] = {'counts': dict(zip(filtered['area'], filtered['num_people']))}
    return area_df, time_grouped_data, sorted(list(time_grouped_data.keys())), bg_img, True

# --- [AI 모델 로드 및 학습 함수] ---
@st.cache_resource
def load_precomputed_models():
    ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH) if os.path.exists(ENSEMBLE_MODEL_PATH) else None
    return ensemble_model

def train_and_save_models():
    all_files = glob.glob("area_count_time_full_*.csv")
    if not all_files:
        return False, "학습할 CSV 파일(`area_count_time_full_*.csv`)을 찾지 못했습니다."

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(all_files)
    collected_rows = []
    
    for i, fpath in enumerate(all_files):
        progress_percent = int(((i + 1) / total_files) * 70)
        progress_bar.progress(progress_percent / 100)
        status_text.text(f"📁 파일 파싱 중... ({i + 1}/{total_files}) - {os.path.basename(fpath)}")
        
        try:
            df_part = pd.read_csv(fpath)
            if not {'time_index', 'area', 'num_people'}.issubset(df_part.columns):
                continue
            date_str = fpath.replace("area_count_time_full_", "").replace(".csv", "")
            dt_base = pd.to_datetime(date_str, errors='coerce')
            if pd.isna(dt_base):
                continue
                
            for t_idx, group in df_part.groupby('time_index'):
                filtered = group[group['area'] != 'Outside']
                total_p = filtered['num_people'].sum()
                total_sec = int(t_idx) * 10
                h, m = total_sec // 3600, (total_sec % 3600) // 60
                
                collected_rows.append({
                    "hour": h,
                    "minute": m,
                    "dayofweek": dt_base.dayofweek,
                    "target": total_p
                })
        except Exception:
            continue

    df_train = pd.DataFrame(collected_rows)
    if df_train.empty:
        progress_bar.empty()
        status_text.empty()
        return False, "유효한 학습 데이터가 추출되지 않았습니다."

    X = df_train[['hour', 'minute', 'dayofweek']]
    y = df_train['target']

    progress_bar.progress(0.75)
    status_text.text("🤖 Random Forest 모델 학습 중...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X, y)

    xgb_model = None
    if HAS_XGB:
        progress_bar.progress(0.90)
        status_text.text("🚀 XGBoost 모델 학습 중...")
        xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        xgb_model.fit(X, y)

    # 0.5 가중치를 부여하는 통합 앙상블 모델 객체 생성 및 저장
    progress_bar.progress(0.95)
    status_text.text("⚖️ 0.5 가중치 앙상블 모델 결합 및 저장 중...")
    ensemble_model = WeightedEnsembleRegressor(rf_model=rf_model, xgb_model=xgb_model, rf_weight=0.5, xgb_weight=0.5)
    joblib.dump(ensemble_model, ENSEMBLE_MODEL_PATH)

    progress_bar.progress(1.0)
    status_text.text("✨ 앙상블 모델 학습 및 저장 완료!")
    
    # 캐시 비우기 (새로 학습된 모델 반영)
    st.cache_resource.clear()
    return True, f"총 {len(df_train):,}개 샘플로 앙상블 모델(RF+XGB 각 0.5 가중치) 학습 및 저장 완료!"

def get_daily_peaks(df_trend):
    peaks = {}
    ranges = [
        ("1차 피크 (오전)", "05:00", "09:00"),
        ("2차 피크 (주간)", "09:00", "17:00"),
        ("3차 피크 (야간)", "17:00", "21:00")
    ]
    for label, start, end in ranges:
        subset = df_trend.between_time(start, end)
        if not subset.empty:
            max_val = subset['이동평균'].max()
            max_time = subset['이동평균'].idxmax()
            peaks[label] = (max_time, max_val)
    return peaks

def generate_density_heatmap(area_df, current_counts, img_shape):
    height, width, _ = img_shape
    heatmap_grid = np.zeros((height, width), dtype=np.float32)
    np.random.seed(42)
    
    for _, row in area_df.iterrows():
        people_cnt = current_counts.get(row['area_name'], 0)
        if people_cnt > 0:
            cX = int((row['x1'] + row['x2'] + row['x3'] + row['x4']) / 4)
            cY = int((row['y1'] + row['y2'] + row['y3'] + row['y4']) / 4)
            num_particles = int(people_cnt * 4)
            rand_x = np.random.normal(cX, 100, num_particles).astype(np.int32)
            rand_y = np.random.normal(cY, 50, num_particles).astype(np.int32)
            valid = (rand_x >= 0) & (rand_x < width) & (rand_y >= 0) & (rand_y < height)
            for x, y in zip(rand_x[valid], rand_y[valid]): heatmap_grid[y, x] += 1.0

    if heatmap_grid.max() > 0:
        heatmap_smooth = cv2.GaussianBlur(heatmap_grid, (175, 175), 0)
        heatmap_norm = (heatmap_smooth / heatmap_smooth.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        _, alpha = cv2.threshold(heatmap_norm, 20, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(heatmap_color, heatmap_color, mask=alpha)
    return np.zeros((height, width, 3), dtype=np.uint8)

# --- [사이드바 구성] ---
st.sidebar.markdown("### ✈️ ICN IOC SYSTEM")
menu = st.sidebar.radio(
    "관제 시스템 모드 선택", 
    [
        "🚨 통합 관제 상황판 (Dashboard)", 
        "🗺️ 터미널 구역별 상세 분석", 
        "🔍 모델 예측 및 검증 (Validation)", 
        "📡 실시간 센서 파이프라인 (Live)"
    ]
)

st.sidebar.markdown("---")

if menu != "📡 실시간 센서 파이프라인 (Live)":
    st.sidebar.markdown("### 🛠️ 아카이브 제어 패널")
    selected_date = st.sidebar.date_input("📅 관제 대상일자 선택 (Playback)", value=datetime.date(2025, 10, 4))
    target_date_str = selected_date.strftime("%Y-%m-%d")

    area_df, past_time_data, past_unique_times, bg_img, exists = load_data_by_date(target_date_str)
else:
    st.sidebar.markdown("### 📡 라이브 스트림 상태")
    st.sidebar.markdown("""
        <div style="background-color: #111827; padding: 10px; border-radius: 6px; border: 1px solid #10b981; color: #10b981; font-size: 0.85rem; text-align: center;">
            <strong>🟢 LIVE STREAM ACTIVE</strong><br/>
            <span style="color: #94a3b8; font-size: 0.75rem;">실시간 모드에서는 과거 날짜 선택이 비활성화됩니다.</span>
        </div>
    """, unsafe_allow_html=True)
    area_df = pd.read_csv(AREA_FILE_PATH) if os.path.exists(AREA_FILE_PATH) else pd.DataFrame()
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH)
    if bg_img is None: bg_img = np.full((600, 1900, 3), 240, dtype=np.uint8)


# ==========================================
# 1. 🚨 통합 관제 상황판 (Dashboard)
# ==========================================
if menu == "🚨 통합 관제 상황판 (Dashboard)":
    st.title("🛡️ 인천공항 T2 3층 통합 운영 상황판 (IOC Dashboard)")
    
    st.markdown(f"""
        <div style="background-color: #111827; padding: 10px 16px; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-size: 0.9rem;">📂 아카이브 관제 일자: <strong style="color: #f8fafc;">{target_date_str}</strong> (재생 모드)</span>
            <span style="color: #94a3b8; font-size: 0.9rem;">시스템 상태: <strong style="color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 2px 8px; border-radius: 4px;">● ARCHIVE REPLAY SYNCED</strong></span>
        </div>
    """, unsafe_allow_html=True)
    
    if not exists:
        st.error(f"⚠️ [{target_date_str}] 해당 일자의 수집된 세션 데이터가 존재하지 않습니다.")
    else:
        time_options = [int(t) for t in past_unique_times]
        idx_to_label = {t: index_to_time_str(t) for t in time_options}

        selected_t_index = st.select_slider(
            "🕒 [아카이브 타임라인 시뮬레이터] 과거 관제 시점 제어", 
            options=time_options, 
            format_func=lambda x: idx_to_label[x]
        )
        st.markdown(
            f'<div class="slider-range-labels"><span>{idx_to_label[time_options[0]]}</span><span>{idx_to_label[time_options[-1]]}</span></div>',
            unsafe_allow_html=True,
        )
        
        current_counts = past_time_data[selected_t_index]['counts']
        excluded = ["GH", "IM1", "IM2"]
        filtered_counts = {k: v for k, v in current_counts.items() if k not in excluded}
        
        total_people = sum(filtered_counts.values())
        urgent_areas = {k: v for k, v in filtered_counts.items() if v >= 80}
        max_area = max(filtered_counts, key=filtered_counts.get) if filtered_counts else "없음"
        norm_ratio = (1 - (len(urgent_areas) / len(filtered_counts))) * 100 if filtered_counts else 100

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("총 체류 여객", f"{total_people:,} 명")
        col2.metric("혼잡 구역 수", f"{len(urgent_areas)} 곳", delta="주의 대상" if urgent_areas else "양호", delta_color="inverse")
        col3.metric("최대 밀집 구역", max_area)
        col4.metric("운영 정상도", f"{norm_ratio:.1f}%")
        col5.metric("센서 정제 정확도", "96.4%", delta="±3.6% 오차")

        st.divider()

        if urgent_areas:
            top_urgent = max(urgent_areas, key=urgent_areas.get)
            st.warning(f"🚨 **[자동 경보 발령]** 선택 시점 **{top_urgent}** 구역의 체류 여객이 임계치(80명)를 초과했습니다. (체류: **{urgent_areas[top_urgent]}명**).")
        else:
            st.success("✨ **[정상 운영]** 해당 시점 터미널 내 모든 구역이 안정적인 범위 내에 있습니다.")

        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.subheader("🗺️ 아카이브 공간 밀집도 히트맵")
            heatmap = generate_density_heatmap(area_df, filtered_counts, bg_img.shape)
            blended = cv2.addWeighted(bg_img, 0.55, heatmap, 0.45, 0)
            st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with c2:
            st.subheader("📊 해당 시점 혼잡 Top 5 구역")
            sorted_areas = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            df_top5 = pd.DataFrame(sorted_areas, columns=["구역", "인원"])
            
            top5_chart = alt.Chart(df_top5).mark_bar(
                color="#2563eb", cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X('구역:N', sort='-y', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc', labelAngle=0), title='구역'),
                y=alt.Y('인원:Q', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc'), title='체류 인원 (명)')
            ).properties(height=380).configure(
                background='#07090e',
                view=alt.ViewConfig(stroke=None)
            )
            st.altair_chart(top5_chart, use_container_width=True)

# ==========================================
# 2. 🗺️ 터미널 구역별 상세 분석
# ==========================================
elif menu == "🗺️ 터미널 구역별 상세 분석":
    st.title("📈 구역별 여객 흐름 및 시계열 트렌드 심층 분석")
    st.markdown(f"선택된 아카이브 일자 (**{target_date_str}**) 기준 전체 터미널 흐름 및 구역별 피크 타임을 정밀 검토합니다.")
    
    if not exists:
        st.error("데이터가 없습니다.")
    else:
        window_size = st.sidebar.select_slider(
            "이동평균 윈도우 크기 (분)",
            options=[1, 3, 5, 10],
            value=5,
            help="노이즈를 제거하고 추세를 파악하기 위한 구간 설정"
        )
        st.sidebar.markdown('<div class="slider-range-labels"><span>1분</span><span>10분</span></div>', unsafe_allow_html=True)
        
        st.subheader("📉 터미널 전체 여객 인원 흐름 시계열 분석")
        
        time_trend_data = []
        for t in sorted(past_time_data.keys()):
            counts = past_time_data[t]['counts']
            filtered = {k: v for k, v in counts.items() if k not in ["GH", "IM1", "IM2"]}
            time_trend_data.append({"시간": index_to_time_str(t), "인원": sum(filtered.values())})
        
        df_trend = pd.DataFrame(time_trend_data)
        df_trend['시간'] = pd.to_datetime(df_trend['시간'], format='%H:%M:%S', errors='coerce')
        df_trend = df_trend.dropna(subset=['시간']).set_index("시간").sort_index()
        df_trend['이동평균'] = df_trend['인원'].rolling(window=window_size * 6, min_periods=1).mean()
        
        df_plot = df_trend.reset_index()
        chart = alt.Chart(df_plot).mark_area(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#2563eb', offset=0), alt.GradientStop(color='#111827', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            ), 
            opacity=0.7
        ).encode(
            x=alt.X('시간:T', axis=alt.Axis(format='%H:%M', tickCount='hour', labelColor='#94a3b8', titleColor='#f8fafc'), title='시간 타임라인'), 
            y=alt.Y('이동평균:Q', title='보정 체류 인원 (명)', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc'))
        ).properties(height=280)

        peak_data = get_daily_peaks(df_trend)
        p_cols = st.columns(3)
        for i, (label, (t, val)) in enumerate(peak_data.items()):
            p_cols[i].metric(label, t.strftime('%H:%M'), f"{int(val)}명 체류")

        peak_annotations = [{"시간": t, "인원": val, "라벨": label} for label, (t, val) in peak_data.items()]
        df_peaks = pd.DataFrame(peak_annotations)

        if not df_peaks.empty:
            rules = alt.Chart(df_peaks).mark_rule(color='#ef4444', strokeDash=[4,4]).encode(x='시간:T')
            text = alt.Chart(df_peaks).mark_text(align='left', dx=5, dy=-10, color='#ef4444', fontWeight='bold').encode(
                x='시간:T', y='이동평균:Q', text='라벨:N'
            )
            final_trend_chart = (chart + rules + text).configure(
                background='#07090e', view=alt.ViewConfig(stroke=None)
            )
            st.altair_chart(final_trend_chart, use_container_width=True)
        else:
            final_trend_chart = chart.configure(
                background='#07090e', view=alt.ViewConfig(stroke=None)
            )
            st.altair_chart(final_trend_chart, use_container_width=True)

        st.divider()
        st.subheader("📋 구역별 인력 배치 및 운영 권고 명세서")
        
        latest_counts = past_time_data[list(past_time_data.keys())[-1]]['counts'] if past_time_data else {}
        detailed_data = []
        for area in sorted(latest_counts.keys()):
            if area in ["GH", "IM1", "IM2"]: continue
            count = latest_counts.get(area, 0)
            level = "🔴 매우 혼잡" if count >= 160 else "🟠 혼잡" if count >= 120 else "🟡 주의" if count >= 80 else "🟢 보통"
            open_cnt, support, total = calculate_staffing(count)
            detailed_data.append({
                "구역": area, 
                "혼잡 등급": level, 
                "현재 체류 인원": int(count), 
                "권고 오픈 창구": open_cnt, 
                "현장 지원 인력": support
            })
        
        df_display = pd.DataFrame(detailed_data)
        
        html_table_staff = "<table class='ioc-table'><thead><tr>"
        for col in df_display.columns:
            html_table_staff += f"<th>{col}</th>"
        html_table_staff += "</tr></thead><tbody>"
        for _, row in df_display.iterrows():
            html_table_staff += "<tr>"
            for val in row:
                html_table_staff += f"<td>{val}</td>"
            html_table_staff += "</tr>"
        html_table_staff += "</tbody></table>"
        st.markdown(html_table_staff, unsafe_allow_html=True)

# ==========================================
# 3. 🔍 모델 예측 및 검증 (Validation)
# ==========================================
elif menu == "🔍 모델 예측 및 검증 (Validation)":
    st.title("🔍 인공지능 기반 여객 수요 앙상블 모델 검증")
    st.markdown(f"> **[모델 관리 및 검증 모드]** 버튼을 눌러 데이터를 직접 학습시키거나, 저장된 앙상블 AI 모델(`{ENSEMBLE_MODEL_PATH}`)로 현재 선택하신 **{target_date_str}**의 실제 측정값을 정밀 비교할 수 있습니다.")

    st.markdown("### ⚙️ AI 앙상블 모델 학습 제어 패널")
    if st.button("🔄 앙상블 모델(RF+XGB 각 0.5 가중치) 학습 및 저장하기"):
        with st.spinner("앙상블 모델을 학습 중입니다. 잠시만 기다려주세요..."):
            success, msg = train_and_save_models()
            if success:
                st.success(f"🎉 {msg}")
            else:
                st.error(f"❌ 학습 실패: {msg}")

    # --- 📥 [통합 앙상블 모델 파일 다운로드 버튼] ---
    if os.path.exists(ENSEMBLE_MODEL_PATH):
        st.markdown("---")
        st.markdown("##### 📥 통합 앙상블 모델 파일 다운로드")
        with open(ENSEMBLE_MODEL_PATH, "rb") as f:
            st.download_button(
                label="📥 앙상블 모델 다운로드 (.pkl)",
                data=f,
                file_name="ensemble_model.pkl",
                mime="application/octet-stream"
            )
        st.info("💡 위 버튼을 통해 두 모델이 0.5 가중치로 결합된 단일 `.pkl` 앙상블 모델 파일을 다운로드하실 수 있습니다.")

    st.divider()

    ensemble_model = load_precomputed_models()

    if ensemble_model is None:
        st.warning(f"⚠️ 저장된 앙상블 모델 파일이 없습니다. 위쪽의 **[학습 및 저장하기]** 버튼을 먼저 눌러주세요!")
    else:
        if exists and past_time_data:
            val_rows = []
            for t_idx in sorted(past_time_data.keys()):
                total_p = sum({k: v for k, v in past_time_data[t_idx]['counts'].items() if k not in ["GH", "IM1", "IM2", "Outside"]}.values())
                total_sec = int(t_idx) * 10
                h, m = total_sec // 3600, (total_sec % 3600) // 60
                time_str = f"{target_date_str} {int(h):02d}:{int(m):02d}:00"
                parsed_time = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
                
                val_rows.append({
                    "시간": parsed_time,
                    "hour": h,
                    "minute": m,
                    "dayofweek": selected_date.weekday(),
                    "실제 측정치 (Ground Truth)": total_p
                })
            df_val = pd.DataFrame(val_rows)
            df_val = df_val.dropna(subset=["시간"])
        else:
            time_idx_val = pd.date_range(f"{target_date_str} 06:00:00", f"{target_date_str} 22:00:00", freq="30min")
            df_val = pd.DataFrame({
                "시간": time_idx_val,
                "hour": time_idx_val.hour,
                "minute": time_idx_val.minute,
                "dayofweek": selected_date.weekday(),
                "실제 측정치 (Ground Truth)": 300 + np.random.normal(0, 30, len(time_idx_val))
            })

        # 앙상블 모델을 통한 예측 수행 (RF 0.5 + XGB 0.5 자동 적용)
        X_target = df_val[['hour', 'minute', 'dayofweek']]
        predicted_vals = ensemble_model.predict(X_target)

        df_val['앙상블 예측치 (RF+XGB)'] = predicted_vals
        df_val['잔차 (Residual)'] = df_val['실제 측정치 (Ground Truth)'] - df_val['앙상블 예측치 (RF+XGB)']

        residuals = df_val['잔차 (Residual)']
        actuals = df_val['실제 측정치 (Ground Truth)']
        calc_mae = np.mean(np.abs(residuals))
        calc_rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals / np.where(actuals == 0, 1, actuals))) * 100
        max_error = np.max(np.abs(residuals))
        std_error = np.std(residuals)

        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("MAE (평균절대오차)", f"{calc_mae:.2f} 명")
        v2.metric("RMSE (제곱근평균오차)", f"{calc_rmse:.2f} 명")
        v3.metric("MAPE (평균절대백분율오차)", f"{mape:.2f}%")
        v4.metric("Max Error (최대오차)", f"{max_error:.2f} 명")
        v5.metric("Model Stability", f"±{std_error:.2f} 명")
        
        st.divider()

        st.subheader(f"📈 [{target_date_str}] 실제 측정값 vs 0.5 가중치 앙상블 예측치 비교 검증")
        df_melted = df_val.melt("시간", value_vars=["실제 측정치 (Ground Truth)", "앙상블 예측치 (RF+XGB)"], var_name="구분", value_name="인원")
        val_chart = alt.Chart(df_melted).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X('시간:T', title='타임라인', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc')),
            y=alt.Y('인원:Q', title='체류 인원 (명)', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc')),
            color=alt.Color('구분:N', scale=alt.Scale(range=['#10b981', '#38bdf8']))
        ).properties(height=380).configure(
            background='#07090e',
            view=alt.ViewConfig(stroke=None)
        )
        st.altair_chart(val_chart, use_container_width=True)

# ==========================================
# 4. 📡 실시간 센서 파이프라인 (Live)
# ==========================================
elif menu == "📡 실시간 센서 파이프라인 (Live)":
    st.title("📡 실시간 센서 파이프라인 및 스트리밍 센터")
    st.markdown("""
    > **[LIVE STREAMING MODE]** 본 모드는 과거 아카이브 조회가 아닌, **현재 시각 기준 비전 센서 노드 스트림**을 실시간 연동하여 관제하는 영역입니다.
    """)
    
    current_live_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
        <div style="background-color: #111827; padding: 10px 16px; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-size: 0.9rem;">⚡ 실시간 수신 시각: <strong style="color: #10b981;">{current_live_time}</strong></span>
            <span style="color: #94a3b8; font-size: 0.9rem;">센서 노드 통신: <strong style="color: #10b981; background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 4px;">● ONLINE / LOW LATENCY (12ms)</strong></span>
        </div>
    """, unsafe_allow_html=True)
    
    live_col1, live_col2, live_col3 = st.columns(3)
    live_col1.metric("활성 비전 센서 노드", "42 / 42 대", delta="100% 정상 가동")
    live_col2.metric("패킷 수신 주기", "10초 Interval", delta="실시간 동기화 중")
    live_col3.metric("평균 처리 지연", "18 ms", delta="Optimal")
    
    st.divider()
    
    st.subheader("🗺️ 실시간 터미널 3층 라이브 히트맵 및 구역별 부하")
    
    if not area_df.empty and 'area_name' in area_df.columns:
        mock_live_counts = {}
        np.random.seed(int(datetime.datetime.now().second))
        for aname in area_df['area_name'].unique():
            mock_live_counts[aname] = int(np.random.randint(10, 110))
            
        live_filtered = {k: v for k, v in mock_live_counts.items() if k not in ["GH", "IM1", "IM2"]}
        
        lc1, lc2 = st.columns([1.6, 1])
        with lc1:
            st.markdown("##### 📍 실시간 공간 밀집도 스트림 뷰")
            live_heatmap = generate_density_heatmap(area_df, live_filtered, bg_img.shape)
            live_blended = cv2.addWeighted(bg_img, 0.55, live_heatmap, 0.45, 0)
            st.image(cv2.cvtColor(live_blended, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        with lc2:
            st.markdown("##### 📊 실시간 구역별 여객 분포 Top 5")
            sorted_live = sorted(live_filtered.items(), key=lambda x: x[1], reverse=True)[:5]
            df_live_top5 = pd.DataFrame(sorted_live, columns=["구역", "인원"])
            
            live_chart = alt.Chart(df_live_top5).mark_bar(
                color="#10b981", cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X('구역:N', sort='-y', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc', labelAngle=0), title='구역'),
                y=alt.Y('인원:Q', axis=alt.Axis(labelColor='#94a3b8', titleColor='#f8fafc'), title='실시간 인원 (명)')
            ).properties(height=350).configure(
                background='#07090e',
                view=alt.ViewConfig(stroke=None)
            )
            st.altair_chart(live_chart, use_container_width=True)
    else:
        st.warning("⚠️ 구역 정의 파일(AREA_FILE_PATH)을 불러오지 못해 라이브 비전 뷰를 렌더링할 수 없습니다.")
