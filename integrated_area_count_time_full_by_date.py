"""
인천공항 S-WARD 전처리 통합 파이프라인 - 메모리 절약형 날짜별 출력 버전

핵심 변경점
- 여러 날짜 CSV를 한 번에 concat하지 않음
- 날짜 파일 하나씩 처리하고 바로 area_count_time_full_날짜.csv 저장
- 중간 CSV 저장 없음
- 최종 출력물만 날짜별 저장

기본 실행:
    python integrated_area_count_time_full_by_date.py

현재 폴더에 다음과 같은 파일이 있으면 자동 처리:
    2025-09-01.csv
    2025-09-02.csv
    2025-09-03.csv

출력:
    area_count_time_full_2025-09-01.csv
    area_count_time_full_2025-09-02.csv
    area_count_time_full_2025-09-03.csv
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import re
from pathlib import Path as FilePath
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from matplotlib.path import Path


RAW_REQUIRED_COLUMNS = {"time_index", "mac_address", "sward_name", "rssi"}
SWARD_REQUIRED_COLUMNS = {"sward_id", "pos_x", "pos_y"}
AREA_REQUIRED_COLUMNS = {
    "area_name",
    "x1", "y1",
    "x2", "y2",
    "x3", "y3",
    "x4", "y4",
}

DATE_COLUMN_CANDIDATES = [
    "data_date",
    "date",
    "Date",
    "DATE",
    "날짜",
    "일자",
    "기준일",
    "base_date",
    "record_date",
]

DEFAULT_INPUT_GLOBS = [
    "20??-??-??.csv",
    "20??_??_??.csv",
    "icn_20??_??_??.csv",
    "icn_20??-??-??.csv",
]


def validate_columns(df: pd.DataFrame, required: set[str], file_label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"[{file_label}] 필수 컬럼 누락: {missing}")


def sanitize_filename_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_date"


def normalize_date_value(value: object) -> str:
    if pd.isna(value):
        return "unknown_date"

    text = str(value).strip()
    if not text:
        return "unknown_date"

    match = re.search(r"(20\d{2})[-_/\.](\d{1,2})[-_/\.](\d{1,2})", text)
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

    match = re.search(r"(20\d{2})(\d{2})(\d{2})", text)
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{mm}-{dd}"

    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%Y-%m-%d")

    return sanitize_filename_part(text)


def infer_data_date_from_filename(path: str) -> str:
    stem = FilePath(path).stem
    return normalize_date_value(stem)


def normalize_input_files(
    input_files: Sequence[str] | None,
    input_glob: str | None,
) -> List[str]:
    paths: list[str] = []

    if input_files:
        for item in input_files:
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    paths.append(part)

    if input_glob:
        paths.extend(glob.glob(input_glob))
    elif not paths:
        for pattern in DEFAULT_INPUT_GLOBS:
            paths.extend(glob.glob(pattern))

    unique_paths = sorted(list(dict.fromkeys(paths)))

    if not unique_paths:
        raise FileNotFoundError(
            "입력 CSV를 찾지 못했습니다. 예: --input_glob \"2025-09-*.csv\""
        )

    missing = [p for p in unique_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"존재하지 않는 입력 파일: {missing}")

    return unique_paths


def pick_date_column(df: pd.DataFrame, requested_date_column: str | None) -> str | None:
    if requested_date_column:
        if requested_date_column not in df.columns:
            raise ValueError(f"지정한 날짜 컬럼이 원본 CSV에 없습니다: {requested_date_column}")
        return requested_date_column

    for col in DATE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    return None


def load_one_raw_detection_file(path: str, date_column: str | None = None) -> pd.DataFrame:
    """
    CSV 하나만 읽어서 data_date를 부여한다.
    여러 파일을 concat하지 않는다.
    """
    df = pd.read_csv(
        path,
        usecols=lambda c: c in RAW_REQUIRED_COLUMNS or c in DATE_COLUMN_CANDIDATES or c == date_column,
    )

    validate_columns(df, RAW_REQUIRED_COLUMNS, path)

    selected_date_column = pick_date_column(df, date_column)

    if selected_date_column:
        df["data_date"] = df[selected_date_column].apply(normalize_date_value)
    else:
        df["data_date"] = infer_data_date_from_filename(path)

    # 메모리 절약용 타입 정리
    df["time_index"] = pd.to_numeric(df["time_index"], errors="coerce").astype("Int64")
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce").astype("float32")
    df["sward_name"] = df["sward_name"].astype(str)
    df["mac_address"] = df["mac_address"].astype(str)
    df["data_date"] = df["data_date"].astype(str)

    df = df.dropna(subset=["time_index", "rssi"])
    df["time_index"] = df["time_index"].astype("int32")

    return df[["data_date", "time_index", "mac_address", "sward_name", "rssi"]]


def estimate_person_locations(
    raw_df: pd.DataFrame,
    sward_df: pd.DataFrame,
    strongest_k: int = 5,
) -> pd.DataFrame:
    """
    RSSI median filter + strongest K 센서 + RSSI 가중평균 위치 추정

    반환:
        data_date, time_index, mac_address, x, y
    """
    validate_columns(raw_df, RAW_REQUIRED_COLUMNS | {"data_date"}, "raw detection data")
    validate_columns(sward_df, SWARD_REQUIRED_COLUMNS, "sward location data")

    if strongest_k <= 0:
        raise ValueError("strongest_k는 1 이상의 정수여야 합니다.")

    df = raw_df.merge(
        sward_df[["sward_id", "pos_x", "pos_y"]],
        left_on="sward_name",
        right_on="sward_id",
        how="left",
    )

    before_drop = len(df)
    df = df.dropna(subset=["pos_x", "pos_y", "rssi"])
    dropped = before_drop - len(df)

    if dropped:
        print(f"        [경고] sward 위치 또는 rssi가 없어 제외된 행: {dropped:,}")

    group_cols = ["data_date", "time_index", "mac_address", "sward_name"]

    df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            rssi=("rssi", "median"),
            pos_x=("pos_x", "first"),
            pos_y=("pos_y", "first"),
        )
    )

    df = df.sort_values(
        ["data_date", "time_index", "mac_address", "rssi"],
        ascending=[True, True, True, False],
    )

    df = df.groupby(
        ["data_date", "time_index", "mac_address"],
        as_index=False,
    ).head(strongest_k)

    df["weight"] = np.power(10.0, df["rssi"].astype("float32") / 10.0).astype("float32")
    df["wx"] = df["weight"] * df["pos_x"].astype("float32")
    df["wy"] = df["weight"] * df["pos_y"].astype("float32")

    result = (
        df.groupby(["data_date", "time_index", "mac_address"], as_index=False)
        .agg(
            wx=("wx", "sum"),
            wy=("wy", "sum"),
            weight=("weight", "sum"),
        )
    )

    result = result[result["weight"] > 0].copy()
    result["x"] = result["wx"] / result["weight"]
    result["y"] = result["wy"] / result["weight"]

    return result[["data_date", "time_index", "mac_address", "x", "y"]]


def build_area_paths(area_df: pd.DataFrame) -> dict[str, Path]:
    validate_columns(area_df, AREA_REQUIRED_COLUMNS, "terminal area data")

    paths: dict[str, Path] = {}

    for _, row in area_df.iterrows():
        coords = [
            (row["x1"], row["y1"]),
            (row["x2"], row["y2"]),
            (row["x3"], row["y3"]),
            (row["x4"], row["y4"]),
        ]
        paths[str(row["area_name"])] = Path(coords)

    return paths


def assign_areas(person_location_df: pd.DataFrame, area_df: pd.DataFrame) -> pd.DataFrame:
    required = {"data_date", "time_index", "mac_address", "x", "y"}
    validate_columns(person_location_df, required, "person location data")

    paths = build_area_paths(area_df)
    df = person_location_df.copy()

    points = df[["x", "y"]].to_numpy(dtype=float)

    assigned = np.full(len(df), "Outside", dtype=object)
    unassigned_mask = np.ones(len(df), dtype=bool)

    for area_name, path in paths.items():
        if not unassigned_mask.any():
            break

        contained = path.contains_points(points[unassigned_mask])
        original_idx = np.where(unassigned_mask)[0][contained]

        assigned[original_idx] = area_name
        unassigned_mask[original_idx] = False

    df["area"] = assigned

    return df


def aggregate_area_count_time_full(
    person_area_df: pd.DataFrame,
    area_df: pd.DataFrame,
    include_outside: bool = True,
    full_day_time_index: bool = False,
    time_index_start: int = 0,
    time_index_end: int = 8639,
) -> pd.DataFrame:
    """
    날짜·시간·구역별 고유 MAC 수 집계.
    0명 구역도 포함.
    """
    required = {"data_date", "time_index", "mac_address", "area"}
    validate_columns(person_area_df, required, "person area data")
    validate_columns(area_df, AREA_REQUIRED_COLUMNS, "terminal area data")

    dates = sorted(person_area_df["data_date"].dropna().astype(str).unique())

    areas = area_df["area_name"].astype(str).tolist()

    if include_outside and "Outside" not in areas:
        areas.append("Outside")

    output_frames: list[pd.DataFrame] = []

    for data_date in dates:
        day_df = person_area_df[person_area_df["data_date"].astype(str) == data_date]

        if full_day_time_index:
            time_values = list(range(time_index_start, time_index_end + 1))
        else:
            time_values = sorted(day_df["time_index"].dropna().unique())

        full_index = pd.MultiIndex.from_product(
            [[data_date], time_values, areas],
            names=["data_date", "time_index", "area"],
        )

        area_count = (
            day_df.groupby(["data_date", "time_index", "area"])["mac_address"]
            .nunique()
            .rename("num_people")
        )

        day_full = (
            area_count.reindex(full_index, fill_value=0)
            .reset_index()
            .astype({"num_people": "int64"})
        )

        output_frames.append(day_full)

    if not output_frames:
        return pd.DataFrame(columns=["data_date", "time_index", "area", "num_people"])

    return pd.concat(output_frames, ignore_index=True)


def build_dated_output_path(output_path: str, data_date: str) -> str:
    path = FilePath(output_path)
    suffix = path.suffix or ".csv"
    stem = path.stem or "area_count_time_full"
    safe_date = sanitize_filename_part(data_date)
    return str(path.with_name(f"{stem}_{safe_date}{suffix}"))


def save_one_date_output(
    area_count_time_full: pd.DataFrame,
    output_path: str,
    data_date: str,
    encoding: str = "utf-8-sig",
    drop_date_column: bool = False,
) -> str:
    day_output_path = build_dated_output_path(output_path, data_date)

    output_df = area_count_time_full.copy()

    if drop_date_column and "data_date" in output_df.columns:
        output_df = output_df.drop(columns=["data_date"])

    output_df.to_csv(day_output_path, index=False, encoding=encoding)

    return day_output_path


def run_pipeline(args: argparse.Namespace) -> None:
    input_paths = normalize_input_files(args.input_files, args.input_glob)

    print(f"[준비] 처리할 원본 감지 CSV: {len(input_paths)}개")

    print("[준비] S-WARD 위치 CSV 로드")
    sward_df = pd.read_csv(args.sward_path)
    validate_columns(sward_df, SWARD_REQUIRED_COLUMNS, args.sward_path)

    sward_df = sward_df[["sward_id", "pos_x", "pos_y"]].copy()
    sward_df["sward_id"] = sward_df["sward_id"].astype(str)
    sward_df["pos_x"] = pd.to_numeric(sward_df["pos_x"], errors="coerce").astype("float32")
    sward_df["pos_y"] = pd.to_numeric(sward_df["pos_y"], errors="coerce").astype("float32")

    print("[준비] 구역 polygon CSV 로드")
    area_df = pd.read_csv(args.area_path)
    validate_columns(area_df, AREA_REQUIRED_COLUMNS, args.area_path)

    saved_paths: list[str] = []

    for idx, input_path in enumerate(input_paths, start=1):
        print("=" * 70)
        print(f"[{idx}/{len(input_paths)}] 처리 시작: {input_path}")

        raw_df = load_one_raw_detection_file(input_path, date_column=args.date_column)

        detected_dates = sorted(raw_df["data_date"].dropna().astype(str).unique())
        print(f"    감지된 날짜: {', '.join(detected_dates)}")
        print(f"    원본 행 수: {len(raw_df):,}")

        print("    [1/4] RSSI 가중평균 위치 추정")
        person_location = estimate_person_locations(
            raw_df=raw_df,
            sward_df=sward_df,
            strongest_k=args.strongest_k,
        )
        print(f"        추정 위치 수: {len(person_location):,}")

        del raw_df
        gc.collect()

        print("    [2/4] 좌표 → 구역 매핑")
        person_area = assign_areas(person_location, area_df)

        del person_location
        gc.collect()

        print("    [3/4] 시간·구역별 인원 집계")
        area_count_time_full = aggregate_area_count_time_full(
            person_area_df=person_area,
            area_df=area_df,
            include_outside=not args.exclude_outside,
            full_day_time_index=args.full_day_time_index,
            time_index_start=args.time_index_start,
            time_index_end=args.time_index_end,
        )

        del person_area
        gc.collect()

        print("    [4/4] 날짜별 최종 CSV 저장")

        for data_date, day_df in area_count_time_full.groupby("data_date", sort=True):
            saved_path = save_one_date_output(
                area_count_time_full=day_df,
                output_path=args.output_path,
                data_date=str(data_date),
                encoding=args.encoding,
                drop_date_column=args.drop_date_column_in_split,
            )
            saved_paths.append(saved_path)
            print(f"        저장 완료: {saved_path} / 행 수: {len(day_df):,}")

        del area_count_time_full
        gc.collect()

    print("=" * 70)
    print("전체 처리 완료")
    print("생성된 파일:")
    for path in saved_paths:
        print(f"  - {path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S-WARD 위치추정부터 날짜별 area_count_time_full 생성까지 수행합니다."
    )

    parser.add_argument(
        "--input_files",
        nargs="*",
        default=None,
        help="원본 감지 CSV 파일 목록. 예: --input_files 2025-09-01.csv 2025-09-02.csv",
    )

    parser.add_argument(
        "--input_glob",
        default=None,
        help="여러 원본 CSV를 찾는 glob 패턴. 예: 2025-09-*.csv",
    )

    parser.add_argument(
        "--sward_path",
        default="sward_locations.csv",
        help="S-WARD 센서 위치 CSV 경로. 기본값: sward_locations.csv",
    )

    parser.add_argument(
        "--area_path",
        default="terminal_areas_grouped_2.csv",
        help="구역 polygon CSV 경로. 기본값: terminal_areas_grouped_2.csv",
    )

    parser.add_argument(
        "--output_path",
        default="area_count_time_full.csv",
        help="최종 출력 CSV 템플릿 경로. 기본값: area_count_time_full.csv",
    )

    parser.add_argument(
        "--strongest_k",
        type=int,
        default=5,
        help="위치 추정에 사용할 strongest 센서 개수. 기본값: 5",
    )

    parser.add_argument(
        "--date_column",
        default=None,
        help="원본 CSV 안의 날짜 컬럼명. 없으면 파일명에서 날짜 추출.",
    )

    parser.add_argument(
        "--full_day_time_index",
        action="store_true",
        help="관측된 time_index만 쓰지 않고 전체 time_index 범위를 0명 조합까지 생성.",
    )

    parser.add_argument(
        "--time_index_start",
        type=int,
        default=0,
        help="전체 time_index 시작값. 기본값: 0",
    )

    parser.add_argument(
        "--time_index_end",
        type=int,
        default=8639,
        help="전체 time_index 종료값. 10초 단위 24시간이면 8639.",
    )

    parser.add_argument(
        "--exclude_outside",
        action="store_true",
        help="Outside 구역을 최종 집계에서 제외.",
    )

    parser.add_argument(
        "--drop_date_column_in_split",
        action="store_true",
        help="날짜별 저장 CSV에서 data_date 컬럼 제거.",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV 저장 인코딩. 기본값: utf-8-sig",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    run_pipeline(parse_args())
