from pathlib import Path
from modules.core import load_operation_data, load_flight_data, data_quality_report
from modules.simulation import ALL_AREAS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "pages" / "data"
op = load_operation_data(DATA / "operation_dashboard_oct2025.csv.gz")
fl = load_flight_data(DATA / "flight_counter_oct2025.csv")
report = data_quality_report(op, fl)

print("=== DATA QA ===")
for k, v in report.items():
    print(f"{k}: {v}")

assert report["operation_dates"] == 61, "2025-09-01~2025-10-31 61일치 운영 데이터가 아닙니다."
assert report["operation_duplicates"] == 0, "일자/분/구역 중복 행이 있습니다."
assert set(report["areas"]) == set(ALL_AREAS), "예상 구역 목록과 실제 데이터가 다릅니다."
print("\n검수 통과")
