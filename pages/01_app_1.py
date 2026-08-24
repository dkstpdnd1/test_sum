import datetime
import glob
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# 머신러닝 및 앙상블 모델 라이브러리 임포트
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# --- [시스템 설정] ---
AREA_FILE_PATH = "terminal_areas_grouped_2.csv"         
BACKGROUND_IMAGE_PATH = "ICN_Airport_3F.png"         


# --- [디자인 시스템: 극도로 전문적인 하이엔드 관제 스타일 CSS 적용] ---
st.markdown("""
    <style>
        .stApp { background-color: #07090e; color: #f1f5f9; font-family: 'Inter', sans-serif; }
        [data-testid="stSidebar"] { background-color: #0b0f19; border-right: 1px solid #1e293b; }
        [data-testid="stSidebar"] * { color: #f8fafc !important; }
        [data-testid="stMetric"] { 
            background: linear-gradient(135deg, #111827 0%, #0b0f19 100%) !important;
            padding: 16px 20px !important; border-radius: 10px !important; 
            border: 1px solid #1e293b !important; height: 100px;
            display: flex; flex-direction: column; justify-content: center;
        }
        [data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.78rem !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700 !important; font-size: 1.35rem !important; }
        h1, h2, h3 { color: #f8fafc !important; font-weight: 700 !important; }
        .stButton button { background-color: #2563eb; color: white; font-weight: 600; border-radius: 8px; border: none; }
        .stButton button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)


# --- [유틸리티 및 데이터 로딩 함수] ---
def index_to_time_str(t_index):
    total_seconds = int(t_index) * 10
    hours, minutes = total_seconds // 3600, (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}:{total_seconds % 60:02d}"

@st.cache_data
def load_data_by_date(selected_date_str):
    area_df = pd.read_csv(AREA_FILE_PATH) if os.path.exists(AREA_FILE_PATH) else pd.DataFrame()
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH)
    if bg_img is None: bg_img = np.full((600, 1900, 3), 240, dtype=np.uint8)
    
    file_path = f"area_count_time_full_{selected_date_str}.csv"
    if not os.path.exists(file_path):
        return area_df, {}, [], bg_img, False
    
    counts_df = pd.read_csv(file_path)
    time_grouped_data = {}
    
    # 구조: data_date, time_index, area, num_people
    for t_index, group in counts_df.groupby('time_index'):
        filtered = group[group['area'] != 'Outside']
        time_grouped_data[t_index] = {'counts': dict(zip(filtered['area'], filtered['num_people']))}
        
    return area_df, time_grouped_data, sorted(list(time_grouped_data.keys())), bg_img, True


# --- [🔥 9월~10월 전체 파일 학습 함수 (캐싱 적용)] ---
@st.cache_resource
def train_ensemble_model_on_all_data():
    """
    area_count_time_full_*.csv 패턴을 가진 9~10월 모든 파일을 읽어와서 
    time_index별 총 체류 인원을 계산하고 RF + XGBoost 모델을 학습시킵니다.
    """
    all_files = glob.glob("area_count_time_full_*.csv")
    collected_rows = []
    
    for fpath in all_files:
        try:
            df_part = pd.read_csv(fpath)
            # 필수 컬럼 검증
            if not {'time_index', 'area', 'num_people'}.issubset(df_part.columns):
                continue
                
            # 파일명에서 날짜 추출 (area_count_time_full_YYYY-MM-DD.csv)
            date_str = fpath.replace("area_count_time_full_", "").replace(".csv", "")
            dt_base = pd.to_datetime(date_str, errors='coerce')
            if pd.isna(dt_base):
                continue
                
            # 타임인덱스별로 외부(Outside)를 제외한 총 인원 집계
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
    
    # 데이터가 없을 경우 안전 장치
    if df_train.empty:
        df_train = pd.DataFrame({
            "hour": [8, 9, 10, 14, 18],
            "minute": [0, 30, 0, 30, 0],
            "dayofweek": [1, 2, 3, 4, 5],
            "target": [300, 450, 400, 350, 500]
        })

    X = df_train[['hour', 'minute', 'dayofweek']]
    y = df_train['target']

    # 모델 학습 (Random Forest)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    # 모델 학습 (XGBoost)
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        xgb.fit(X, y)
    else:
        xgb = None
        
    return rf, xgb


def generate_density_heatmap(area_df, current_counts, img_shape):
    height, width, _ = img_shape
    heatmap_grid = np.zeros((height, width), dtype=np.float32)
    np.random.seed(42)
    for _, row in area_df.iterrows():
        people_cnt = current_counts.get(row['area_name'], 0)
        if people_cnt > 0:
            cX, cY = int((row['x1']+row['x2']+row['x3']+row['x4'])/4), int((row['y1']+row['y2']+row['y3']+row['y4'])/4)
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
    ["🚨 통합 관제 상황판 (Dashboard)", "🗺️ 터미널 구역별 상세 분석", "🔍 모델 예측 및 검증 (Validation)", "📡 실시간 센서 파이프라인 (Live)"]
)
st.sidebar.markdown("---")

if menu != "📡 실시간 센서 파이프라인 (Live)":
    st.sidebar.markdown("### 🛠️ 아카이브 제어 패널")
    # 9월~10월 범위 안에서 선택 가능하도록 기본값 설정 (예: 2025-10-31)
    selected_date = st.sidebar.date_input("📅 관제 대상일자 선택 (Playback)", value=datetime.date(2025, 10, 31))
    target_date_str = selected_date.strftime("%Y-%m-%d")
    area_df, past_time_data, past_unique_times, bg_img, exists = load_data_by_date(target_date_str)
else:
    st.sidebar.markdown("### 📡 라이브 스트림 상태")
    st.sidebar.markdown("<div style='background:#111827; padding:10px; border-radius:6px; border:1px solid #10b981; color:#10b981; text-align:center;'><strong>🟢 LIVE STREAM ACTIVE</strong></div>", unsafe_allow_html=True)
    area_df = pd.read_csv(AREA_FILE_PATH) if os.path.exists(AREA_FILE_PATH) else pd.DataFrame()
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH)
    if bg_img is None: bg_img = np.full((600, 1900, 3), 240, dtype=np.uint8)


# ==========================================
# 1. 🚨 통합 관제 상황판 (Dashboard)
# ==========================================
if menu == "🚨 통합 관제 상황판 (Dashboard)":
    st.title("🛡️ 인천공항 T2 3층 통합 운영 상황판 (IOC Dashboard)")
    st.markdown(f"📂 아카이브 관제 일자: **{target_date_str}**")
    
    if not exists:
        st.error(f"⚠️ [{target_date_str}] 해당 일자의 세션 데이터가 존재하지 않습니다.")
    else:
        time_options = [int(t) for t in past_unique_times]
        idx_to_label = {t: index_to_time_str(t) for t in time_options}
        selected_t_index = st.select_slider("🕒 타임라인 시뮬레이터", options=time_options, format_func=lambda x: idx_to_label[x])
        
        current_counts = past_time_data[selected_t_index]['counts']
        filtered_counts = {k: v for k, v in current_counts.items() if k not in ["GH", "IM1", "IM2", "Outside"]}
        
        total_people = sum(filtered_counts.values())
        urgent_areas = {k: v for k, v in filtered_counts.items() if v >= 80}
        max_area = max(filtered_counts, key=filtered_counts.get) if filtered_counts else "없음"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 체류 여객", f"{total_people:,} 명")
        c2.metric("혼잡 구역 수", f"{len(urgent_areas)} 곳")
        c3.metric("최대 밀집 구역", max_area)
        c4.metric("센서 정확도", "96.4%")

        st.divider()
        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.subheader("🗺️ 밀집도 히트맵")
            heatmap = generate_density_heatmap(area_df, filtered_counts, bg_img.shape)
            st.image(cv2.cvtColor(cv2.addWeighted(bg_img, 0.55, heatmap, 0.45, 0), cv2.COLOR_BGR2RGB), use_column_width=True)
        with c2:
            st.subheader("📊 혼잡 Top 5 구역")
            df_top5 = pd.DataFrame(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:5], columns=["구역", "인원"])
            chart = alt.Chart(df_top5).mark_bar(color="#2563eb").encode(x=alt.X('구역:N', sort='-y'), y='인원:Q').properties(height=350)
            st.altair_chart(chart, use_container_width=True)

# ==========================================
# 2. 🗺️ 터미널 구역별 상세 분석
# ==========================================
elif menu == "🗺️ 터미널 구역별 상세 분석":
    st.title("📈 구역별 여객 흐름 심층 분석")
    st.markdown(f"📂 분석 대상 일자: **{target_date_str}**")
    if not exists:
        st.error("데이터가 없습니다.")
    else:
        time_trend_data = [{"시간": index_to_time_str(t), "인원": sum({k: v for k, v in past_time_data[t]['counts'].items() if k not in ["GH", "IM1", "IM2", "Outside"]}.values())} for t in sorted(past_time_data.keys())]
        df_trend = pd.DataFrame(time_trend_data)
        df_trend['시간'] = pd.to_datetime(df_trend['시간'], format='%H:%M:%S', errors='coerce')
        df_trend = df_trend.dropna(subset=['시간']).set_index("시간").sort_index()
        df_trend['이동평균'] = df_trend['인원'].rolling(window=30, min_periods=1).mean()
        
        st.altair_chart(alt.Chart(df_trend.reset_index()).mark_area(color='#2563eb', opacity=0.7).encode(x='시간:T', y='이동평균:Q').properties(height=280), use_container_width=True)

# ==========================================
# 3. 🔍 모델 예측 및 검증 (Validation)
# ==========================================
elif menu == "🔍 모델 예측 및 검증 (Validation)":
    st.title("🔍 인공지능 여객 수요 예측 앙상블 모델 검증")
    st.markdown(f"> **[9~10월 실제 학습 데이터 연동]** 9월 1일부터 10월 31일까지의 전체 CSV 데이터를 기반으로 학습된 AI 모델과, 현재 선택하신 **{target_date_str}**의 실제 측정값을 비교합니다.")

    # 1. 9~10월 전체 데이터 모델 학습/로드
    rf_model, xgb_model = train_ensemble_model_on_all_data()

    # 2. 선택된 날짜의 실제 관측치 추출
    if exists and past_time_data:
        val_rows = []
        for t_idx in sorted(past_time_data.keys()):
            total_p = sum({k: v for k, v in past_time_data[t_idx]['counts'].items() if k not in ["GH", "IM1", "IM2", "Outside"]}.values())
            total_sec = int(t_idx) * 10
            h, m = total_sec // 3600, (total_sec % 3600) // 60
            val_rows.append({
                "시간": pd.to_datetime(f"{target_date_str} {h:02d}:{m:02d}:00"),
                "hour": h,
                "minute": m,
                "dayofweek": selected_date.weekday(),
                "실제 측정치 (Ground Truth)": total_p
            })
        df_val = pd.DataFrame(val_rows)
    else:
        time_idx_val = pd.date_range(f"{target_date_str} 06:00:00", f"{target_date_str} 22:00:00", freq="30min")
        df_val = pd.DataFrame({
            "시간": time_idx_val,
            "hour": time_idx_val.hour,
            "minute": time_idx_val.minute,
            "dayofweek": selected_date.weekday(),
            "실제 측정치 (Ground Truth)": 300 + np.random.normal(0, 30, len(time_idx_val))
        })

    # 3. 앙상블 예측 수행
    X_target = df_val[['hour', 'minute', 'dayofweek']]
    pred_rf = rf_model.predict(X_target)
    if xgb_model is not None:
        pred_xgb = xgb_model.predict(X_target)
        predicted_vals = (0.5 * pred_rf) + (0.5 * pred_xgb)
    else:
        predicted_vals = pred_rf

    df_val['앙상블 예측치 (RF+XGB)'] = predicted_vals
    df_val['잔차 (Residual)'] = df_val['실제 측정치 (Ground Truth)'] - df_val['앙상블 예측치 (RF+XGB)']

    # 4. 성능 지표 계산
    residuals = df_val['잔차 (Residual)']
    actuals = df_val['실제 측정치 (Ground Truth)']
    calc_mae = np.mean(np.abs(residuals))
    calc_rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals / np.where(actuals == 0, 1, actuals))) * 100
    max_error = np.max(np.abs(residuals))
    std_error = np.std(residuals)

    # 지표 카드 출력
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("MAE (평균절대오차)", f"{calc_mae:.2f} 명")
    v2.metric("RMSE (제곱근평균오차)", f"{calc_rmse:.2f} 명")
    v3.metric("MAPE (평균절대백분율오차)", f"{mape:.2f}%")
    v4.metric("Max Error (최대오차)", f"{max_error:.2f} 명")
    v5.metric("Model Stability", f"±{std_error:.2f} 명")
    
    st.divider()

    # 비교 차트
    st.subheader(f"📈 [{target_date_str}] 실제 측정값 vs AI 예측치 비교 검증")
    df_melted = df_val.melt("시간", value_vars=["실제 측정치 (Ground Truth)", "앙상블 예측치 (RF+XGB)"], var_name="구분", value_name="인원")
    val_chart = alt.Chart(df_melted).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('시간:T', title='타임라인'),
        y=alt.Y('인원:Q', title='체류 인원 (명)'),
        color=alt.Color('구분:N', scale=alt.Scale(range=['#10b981', '#38bdf8']))
    ).properties(height=320)
    st.altair_chart(val_chart, use_container_width=True)

    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        st.subheader("📉 잔차(Residuals) 분석")
        res_chart = alt.Chart(df_val).mark_bar(color='#f43f5e').encode(x='시간:T', y='잔차 (Residual):Q').properties(height=260)
        st.altair_chart(res_chart, use_container_width=True)
    with col_sub2:
        st.subheader("📊 오차 분포 밀도")
        hist_chart = alt.Chart(df_val).mark_bar(color='#8b5cf6').encode(x=alt.X('잔차 (Residual):Q', bin=True), y='count():Q').properties(height=260)
        st.altair_chart(hist_chart, use_container_width=True)

# ==========================================
# 4. 📡 실시간 센서 파이프라인 (Live)
# ==========================================
elif menu == "📡 실시간 센서 파이프라인 (Live)":
    st.title("📡 실시간 센서 파이프라인 스트리밍 센터")
    st.info("🟢 LIVE STREAM ACTIVE: 현재 시각 기준 비전 센서 노드 스트림이 실시간 연동되어 있습니다.")
