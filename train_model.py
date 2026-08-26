import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_and_save_models():
    print("🚀 데이터 로딩 및 전처리 시작...")
    
    # 1. 9월 1일 ~ 10월 31일 파일명 패턴에 맞는 파일들 불러오기
    # 파일 이름 형식이 area_count_time_full_2025-09-09.csv 라고 하셨으므로 패턴 지정
    file_list = glob.glob("area_count_time_full_2025-*.csv")
    
    if not file_list:
        print("❌ 경고: 조건에 맞는 CSV 파일이 없습니다! 파일 경로를 확인해주세요.")
        return False, "CSV 파일이 없습니다."

    df_list = []
    for file in file_list:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 ({file}): {e}")

    if not df_list:
        return False, "읽어온 데이터가 없습니다."

   # 모든 데이터 합치기
    df_all = pd.concat(df_list, ignore_index=True)
    
    df_all['data_date'] = pd.to_datetime(df_all['data_date'], errors='coerce')
    df_all = df_all.dropna(subset=['data_date'])
    
    total_seconds = df_all['time_index'] * 10  
    df_all['hour'] = (total_seconds // 3600) % 24
    df_all['minute'] = (total_seconds % 3600) // 60
    df_all['dayofweek'] = df_all['data_date'].dt.weekday

    # ⭐ [핵심 수정] 구역(area) 중 Outside 등 제외하고, 같은 날짜 + 같은 시간대(time_index)끼리 묶어서 '전체 총인원'을 구합니다!
    df_filtered_area = df_all[~df_all['area'].isin(["GH", "IM1", "IM2", "Outside"])]
    
    # 날짜와 time_index(또는 hour, minute, dayofweek)를 기준으로 그룹화하여 해당 시점의 총인원 합계 계산
    df_grouped = df_filtered_area.groupby(['data_date', 'time_index', 'hour', 'minute', 'dayofweek'])['num_people'].sum().reset_index()
    df_grouped.rename(columns={'num_people': 'target_total'}, inplace=True)

    # 3. X(입력 피처)와 y(정답 레이블: 전체 총인원) 설정
    features = ['hour', 'minute', 'dayofweek']
    
    X = df_grouped[features]
    y = df_grouped['target_total']  # 이제 정답이 전체 인원 총합이 됩니다!

    print(f"🤖 Random Forest 모델 학습 중... (데이터 크기: {len(X)}행)")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)

    print("🤖 XGBoost 모델 학습 중...")
    xgb_model = XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1, learning_rate=0.1)
    xgb_model.fit(X, y)

    # 4. 0.5 가중치 앙상블 패키지 묶기
    model_package = {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "weights": [0.5, 0.5], # 0.5씩 가중치 부여 명시
        "features": features
    }

    # 5. 폴더 생성 및 저장
    os.makedirs("pages", exist_ok=True)
    save_path = "pages/ensemble_traffic_model.pkl"
    joblib.dump(model_package, save_path, compress=3)
    
    print(f"🎉 성공! 모델이 안전하게 저장되었습니다: {save_path}")
    return True, "학습 및 저장 완료"

if __name__ == "__main__":
    train_and_save_models()