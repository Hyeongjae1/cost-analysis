"""
신규 엑셀 양식(AI단기과제_더미데이터.xlsx) → 손익분석 앱 표준 스키마 변환.

시트:
  - 매출-원가자료: 거래(연도·공장·품목·납품처) 단위 매출·판관·물류
  - 품목별제조원가: 연도·공장·품목 단위 제조원가(재료·노무·경비) — 판매수량 비율로 배분
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SALES_SHEET = "매출-원가자료"
COST_SHEET = "품목별제조원가"

# 매출-원가자료: 0-based 열 (헤더 5행 이후 데이터)
S_COL_YEAR = 1
S_COL_ITEM = 2
S_COL_FACTORY = 4
S_COL_CUSTOMER = 11
S_COL_QTY_KG = 16
S_COL_SALES = 18
S_COL_VALVE_CONTAINER = 23
S_COL_CONTAINER_DEPR_UNIT = 27
S_COL_ADJ_BEGIN_END = 40
S_COL_ADJ_PRODUCT = 41
S_COL_ADJ_GOODS = 42
S_COL_SGNA = 44
S_COL_LABOR_PRODUCT = 46
S_COL_FIXED_DIRECT = 47
S_COL_LOG_SAS = 54
S_COL_LOG_EXPORT = 55
S_COL_LOG_IMPORT = 56
S_COL_LOG_RESIDUAL = 57

# 품목별제조원가
C_COL_YEAR = 1
C_COL_FACTORY = 2
C_COL_ITEM = 3
C_COL_MATERIAL = 27
C_COL_LABOR_VAR = 28
C_COL_LABOR_FIX = 29
C_COL_EXPENSE_START = 30
C_COL_EXPENSE_END = 39
C_COL_PRODUCTION_QTY = 40

OUTPUT_COLUMNS = [
    "연도",
    "공장",
    "매출처",
    "품목",
    "매출수량",
    "매출액",
    "재료비",
    "노무비",
    "경비",
    "용기상각비",
    "공통판관비",
    "물류비",
    "총원가",
    "영업이익",
    "영업이익율",
]


def resolve_data_path(base_dir: Path | None = None) -> Path:
    """신규 양식 → 동일 폴더 → 레거시 순으로 데이터 파일을 찾습니다."""
    root = base_dir or Path(__file__).resolve().parent
    candidates = [
        root.parent / "AI단기과제_더미데이터.xlsx",
        root / "AI단기과제_더미데이터.xlsx",
        root / "profit_loss_analysis_data.xlsx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _mfg_totals(row: pd.Series) -> tuple[float, float, float, float]:
    material = float(pd.to_numeric(row[C_COL_MATERIAL], errors="coerce") or 0.0)
    labor = sum(
        float(pd.to_numeric(row[i], errors="coerce") or 0.0)
        for i in (C_COL_LABOR_VAR, C_COL_LABOR_FIX)
    )
    expense = sum(
        float(pd.to_numeric(row[i], errors="coerce") or 0.0)
        for i in range(C_COL_EXPENSE_START, C_COL_EXPENSE_END)
    )
    production = float(pd.to_numeric(row[C_COL_PRODUCTION_QTY], errors="coerce") or 0.0)
    return material, labor, expense, production


def load_pl_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else resolve_data_path()
    if not path.is_file():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    # 레거시 단일 시트 파일
    try:
        xl = pd.ExcelFile(path)
        if "손익데이터" in xl.sheet_names:
            df = pd.read_excel(path, sheet_name="손익데이터")
            df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
            return df
    except Exception:
        pass

    raw_sales = pd.read_excel(path, sheet_name=SALES_SHEET, header=None)
    raw_cost = pd.read_excel(path, sheet_name=COST_SHEET, header=None)

    sales = raw_sales.iloc[5:].copy()
    sales.columns = range(sales.shape[1])
    cost = raw_cost.iloc[3:].copy()
    cost.columns = range(cost.shape[1])

    cost_tbl = cost[[C_COL_YEAR, C_COL_FACTORY, C_COL_ITEM]].copy()
    cost_tbl.columns = ["연도", "공장", "품목번호"]
    mfg = cost.apply(_mfg_totals, axis=1, result_type="expand")
    cost_tbl["재료비_품목"] = mfg[0]
    cost_tbl["노무비_품목"] = mfg[1]
    cost_tbl["경비_품목"] = mfg[2]
    cost_tbl["생산량"] = mfg[3]

    out = sales[
        [
            S_COL_YEAR,
            S_COL_FACTORY,
            S_COL_ITEM,
            S_COL_CUSTOMER,
            S_COL_QTY_KG,
            S_COL_SALES,
            S_COL_VALVE_CONTAINER,
            S_COL_CONTAINER_DEPR_UNIT,
            S_COL_ADJ_BEGIN_END,
            S_COL_ADJ_PRODUCT,
            S_COL_ADJ_GOODS,
            S_COL_SGNA,
            S_COL_LABOR_PRODUCT,
            S_COL_FIXED_DIRECT,
            S_COL_LOG_SAS,
            S_COL_LOG_EXPORT,
            S_COL_LOG_IMPORT,
            S_COL_LOG_RESIDUAL,
        ]
    ].copy()
    out.columns = [
        "연도",
        "공장",
        "품목번호",
        "매출처",
        "매출수량",
        "매출액",
        "_valve",
        "_depr_unit",
        "_adj0",
        "_adj1",
        "_adj2",
        "공통판관비",
        "_노무_상품",
        "_고정_직접",
        "_log0",
        "_log1",
        "_log2",
        "_log3",
    ]

    out = out.merge(cost_tbl, on=["연도", "공장", "품목번호"], how="left")
    out["매출수량"] = _num(out["매출수량"])
    out["매출액"] = _num(out["매출액"])
    out["생산량"] = _num(out["생산량"])

    ratio = out["매출수량"] / out["생산량"].replace(0, pd.NA)
    ratio = ratio.fillna(0.0)

    out["재료비"] = out["재료비_품목"] * ratio
    out["노무비"] = out["노무비_품목"] * ratio + _num(out["_노무_상품"])
    out["경비"] = (
        out["경비_품목"] * ratio
        + _num(out["_고정_직접"])
        + _num(out["_adj0"])
        + _num(out["_adj1"])
        + _num(out["_adj2"])
    )
    out["용기상각비"] = _num(out["_valve"]) + _num(out["_depr_unit"]) * out["매출수량"]
    out["공통판관비"] = _num(out["공통판관비"])
    out["물류비"] = _num(out["_log0"]) + _num(out["_log1"]) + _num(out["_log2"]) + _num(out["_log3"])

    out["총원가"] = (
        out["재료비"] + out["노무비"] + out["경비"] + out["용기상각비"] + out["공통판관비"] + out["물류비"]
    )
    out["영업이익"] = out["매출액"] - out["총원가"]
    out["영업이익율"] = out["영업이익"] / out["매출액"].replace(0, pd.NA)

    out["품목"] = out["품목번호"].astype(str)
    out["연도"] = pd.to_numeric(out["연도"], errors="coerce").astype("Int64")
    out["매출수량"] = out["매출수량"].round().astype("Int64")

    return out[OUTPUT_COLUMNS].reset_index(drop=True)
