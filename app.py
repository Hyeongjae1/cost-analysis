"""
손익분석 시스템 — Streamlit 대시보드
데이터: profit_loss_analysis_data.xlsx (시트: 손익데이터)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "profit_loss_analysis_data.xlsx"
SHEET_NAME = "손익데이터"
EOK = 100_000_000  # 억 원 환산

YEAR_COL = "연도"
FACTORY_COL = "공장"
CUSTOMER_COL = "매출처"
ITEM_COL = "품목"
AMOUNT_COLS = ["매출액", "총원가", "영업이익"]
COST_DETAIL_COLS = ["재료비", "노무비", "경비", "용기상각비", "공통판관비", "물류비"]

# Wise (getdesign `wise`) — 차트 시리즈 색 (브랜드 라임은 CTA 전용, 차트는 시맨틱/보조색)
WISE_CHART_SALES = "#38c8ff"
WISE_CHART_PROFIT = "#2ead4b"
WISE_CHART_MARGIN = "#163300"
WISE_PLOT_COLORWAY = (
    "#38c8ff",
    "#2ead4b",
    "#454745",
    "#ffc091",
    "#c5edab",
    "#b86700",
    "#9fe870",
)

# 공장별 탭: 상단 차트·선택 표 행 순서 (매출과 무관)
FACTORY_TAB_ORDER: tuple[str, ...] = ("공장A", "공장B", "공장C")


def factories_for_factory_tab(names) -> list[str]:
    """FACTORY_TAB_ORDER에 있는 공장을 그 순서로, 그 외는 사전순으로 뒤에 붙입니다."""
    name_set = {str(x) for x in names if pd.notna(x)}
    ordered = [n for n in FACTORY_TAB_ORDER if n in name_set]
    rest = sorted(name_set - set(ordered))
    return ordered + rest


def get_gemini_api_key() -> str | None:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    try:
        sec = st.secrets
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if k in sec:
                return str(sec[k]).strip()
    except (FileNotFoundError, KeyError, RuntimeError, AttributeError):
        pass
    return None


def _secret_or_env(*names: str) -> str | None:
    for name in names:
        v = os.environ.get(name, "").strip()
        if v:
            return v
        try:
            if name in st.secrets:
                s = str(st.secrets[name]).strip()
                if s:
                    return s
        except (FileNotFoundError, KeyError, RuntimeError, AttributeError, TypeError):
            pass
    return None


def get_vlm_config() -> tuple[str | None, str | None, str | None]:
    """내부(OpenAI 호환) LLM: (base_url, model_name, api_key)."""
    base = _secret_or_env("VLM_BASE_URL", "VLM_SERVER", "OPENAI_BASE_URL")
    model = _secret_or_env("VLM_MODEL_NAME", "OPENAI_MODEL")
    api_key = _secret_or_env("VLM_API_KEY", "OPENAI_API_KEY") or "EMPTY"
    if base and model:
        return base.rstrip("/"), model, api_key
    return None, None, None


def llm_system_prompt() -> str:
    return (
        "당신은 기업 손익 분석 도우미입니다. 제공된 CSV 형태 요약만 사실로 다루고, "
        "없는 수치를 지어내지 마세요. 답은 간결한 한국어로 작성합니다."
    )


def llm_system_prompt_chitchat() -> str:
    return (
        "당신은 손익분석 대시보드의 안내 도우미입니다. 한국어로 짧고 친절하게 답합니다. "
        "인사·감사에는 자연스럽게 응답하고, 구체적 수치는 아래에 주어진 요약에 근거할 때만 언급합니다."
    )


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def use_compact_llm_prompt(last_user: str) -> bool:
    """인사 등 짧은 대화는 대용량 CSV 없이 경량 프롬프트로 전송 (응답 지연·타임아웃 완화)."""
    t = last_user.strip()
    if not t or len(t) > 120:
        return False
    if re.search(r"[0-9]{2,4}|년|공장|매출|품목|이익|원가|고객|매출처|억|%", t):
        return False
    if re.match(r"^(안녕|반가워|하이|hi|hello|고마워|감사|ㅎㅇ|헬로|좋은|반갑)", t, re.I):
        return True
    return len(t) <= 12


def build_llm_one_line_stats(df: pd.DataFrame) -> str:
    ys = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
    tot_sales = float(df["매출액"].sum()) / EOK
    return (
        f"데이터 {len(df)}행, 연도 {ys[0]}~{ys[-1]}, 공장 {df[FACTORY_COL].nunique()}곳, "
        f"매출처 {df[CUSTOMER_COL].nunique()}곳, 품목 {df[ITEM_COL].nunique()}개, "
        f"전 기간 매출 합계 약 {tot_sales:.2f}억 원(원화÷1억)."
    )


def build_llm_user_prompt(df: pd.DataFrame, messages: list[dict], *, compact: bool = False) -> str:
    """데이터 요약 + 최근 대화 + 마지막 사용자 질문."""
    history_lines: list[str] = []
    for m in messages[:-1][-12:]:
        role = "사용자" if m.get("role") == "user" else "어시스턴트"
        history_lines.append(f"{role}: {m.get('content', '')}")
    history_text = "\n".join(history_lines) if history_lines else "(없음)"
    last_user = _last_user_text(messages)

    if compact:
        stats = build_llm_one_line_stats(df)
        return f"""[데이터 한 줄 요약]
{stats}

[이전 대화]
{history_text}

[현재 메시지]
{last_user}

위 메시지에 맞게 한국어로 짧게 답하세요. 인사·감사에는 자연스럽게 응하고, 손익을 물으면 위 요약 범위 안에서만 답하거나 구체 질문을 유도하세요."""

    ctx = build_llm_data_context(df)
    return f"""아래 [데이터 요약]의 수치만 근거로 한국어로 답하세요. 데이터에 없으면 추측하지 말고 '데이터에서 확인할 수 없습니다'라고 하세요.
금액은 필요하면 억 원으로 환산해 표현해도 됩니다 (1억 원 = 100,000,000원).

[데이터 요약]
{ctx}

[이전 대화]
{history_text}

[현재 질문]
{last_user}
"""


def openai_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("openai") is not None
    except Exception:
        return False


def generate_vlm_reply(df: pd.DataFrame, messages: list[dict]) -> str:
    """OpenAI 호환 서버(vLLM 등) Chat Completions."""
    from openai import OpenAI

    base, model, api_key = get_vlm_config()
    if not base or not model:
        raise RuntimeError("VLM_BASE_URL / VLM_MODEL_NAME 미설정")

    read_timeout = float(os.environ.get("VLM_READ_TIMEOUT", "90") or "90")
    connect_timeout = float(os.environ.get("VLM_CONNECT_TIMEOUT", "15") or "15")
    try:
        import httpx

        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30.0, pool=10.0)
    except Exception:
        timeout = read_timeout + connect_timeout

    client = OpenAI(base_url=base, api_key=api_key, timeout=timeout)

    last_u = _last_user_text(messages)
    compact = use_compact_llm_prompt(last_u)
    user_content = build_llm_user_prompt(df, messages, compact=compact)
    sys_msg = llm_system_prompt_chitchat() if compact else llm_system_prompt()

    max_tokens = int(os.environ.get("VLM_MAX_TOKENS", "900") or "900")
    max_tokens = max(64, min(max_tokens, 4096))

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3 if compact else 0.2,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0].message
    out = (choice.content or "").strip()
    if not out:
        refusal = getattr(choice, "refusal", None) or getattr(choice, "reason", None)
        if refusal:
            out = f"(모델 응답 제한: {refusal})"
        else:
            out = "모델이 빈 텍스트를 반환했습니다. 질문을 조금 더 구체적으로 해 보시거나, 잠시 후 다시 시도해 주세요."
    return out


def build_llm_data_context(df: pd.DataFrame, *, max_table_rows: int = 80) -> str:
    """LLM에 넣을 데이터 요약(토큰 절약을 위해 집계 위주). 금액 단위: 원."""
    chunks: list[str] = []
    chunks.append(
        f"행 수: {len(df)}, 연도: {sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())}, "
        f"공장: {sorted(df[FACTORY_COL].dropna().unique().tolist())}, "
        f"매출처 수: {df[CUSTOMER_COL].nunique()}, 품목 수: {df[ITEM_COL].nunique()}."
    )
    fy = (
        df.groupby([FACTORY_COL, YEAR_COL], as_index=False)[["매출액", "총원가", "영업이익"]]
        .sum(numeric_only=True)
        .sort_values([FACTORY_COL, YEAR_COL])
    )
    chunks.append("\n[공장×연도 합계 원]\n" + fy.head(max_table_rows).to_csv(index=False))
    iy = (
        df.groupby([ITEM_COL, YEAR_COL], as_index=False)[["매출액", "영업이익"]]
        .sum(numeric_only=True)
        .sort_values([ITEM_COL, YEAR_COL])
    )
    chunks.append("\n[품목×연도 합계 원]\n" + iy.head(max_table_rows).to_csv(index=False))
    cy = (
        df.groupby([CUSTOMER_COL, YEAR_COL], as_index=False)[["매출액", "영업이익"]]
        .sum(numeric_only=True)
        .sort_values([CUSTOMER_COL, YEAR_COL])
    )
    chunks.append("\n[매출처×연도 합계 원]\n" + cy.head(max_table_rows).to_csv(index=False))
    text = "\n".join(chunks)
    max_len = 14000
    if len(text) > max_len:
        return text[:max_len] + "\n...(이하 생략)"
    return text


def generate_gemini_reply(df: pd.DataFrame, messages: list[dict]) -> str:
    """최근 대화 + 데이터 요약을 바탕으로 Gemini 답변 생성."""
    import google.generativeai as genai

    key = get_gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY 없음")

    genai.configure(api_key=key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"
    user_content = build_llm_user_prompt(df, messages)

    model = genai.GenerativeModel(
        model_name,
        system_instruction=llm_system_prompt(),
    )
    resp = model.generate_content(user_content)
    if not resp.candidates:
        raise RuntimeError("Gemini 응답 없음(차단 또는 빈 결과)")
    out = (resp.text or "").strip()
    if not out:
        raise RuntimeError("Gemini 빈 텍스트")
    return out


def init_session_state() -> None:
    defaults = {
        "nav_tab": "factory",
        "f_factory": None,
        "f_item": None,
        "chat_messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_raw_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    return df


@st.cache_data(show_spinner=False)
def cached_data() -> pd.DataFrame:
    return load_raw_data()


def df_to_eok_display(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (pd.to_numeric(out[c], errors="coerce") / EOK).round(2)
    return out


def format_margin_pct(series: pd.Series) -> pd.Series:
    """영업이익율(비율)을 퍼센트 숫자로 표시, 소수 둘째 자리."""
    return series.apply(lambda x: round(float(x) * 100, 2) if pd.notna(x) else None)


def wise_plotly_layout() -> dict:
    """getdesign wise: 흰 카드 위 잉크 텍스트, Inter 계열."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#ffffff",
        "font": dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color="#0e0f0c", size=13),
        "title_font": dict(
            size=17,
            color="#0e0f0c",
            family="Inter, 'Segoe UI', system-ui, sans-serif",
        ),
    }


def wise_streamlit_css() -> str:
    """Streamlit 전역 스타일 — DESIGN.md (Wise) 토큰 반영."""
    return """
<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --w-primary: #9fe870;
    --w-on-primary: #0e0f0c;
    --w-primary-active: #cdffad;
    --w-ink: #0e0f0c;
    --w-body: #454745;
    --w-mute: #868685;
    --w-canvas: #ffffff;
    --w-canvas-soft: #e8ebe6;
    --w-positive: #2ead4b;
    --w-radius-xl: 24px;
    --w-radius-md: 12px;
    --w-radius-sm: 8px;
  }
  html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: Inter, "Segoe UI", system-ui, sans-serif !important;
    color: var(--w-ink);
  }
  .stApp {
    background: var(--w-canvas-soft) !important;
  }
  .block-container {
    padding-top: 0.75rem !important;
    max-width: 100% !important;
  }
  h1, [data-testid="stHeader"] { color: var(--w-ink) !important; }
  h1 {
    font-weight: 900 !important;
    font-size: clamp(1.5rem, 2.2vw, 2rem) !important;
    letter-spacing: -0.02em !important;
  }
  h2, h3 { font-weight: 700 !important; color: var(--w-ink) !important; }
  [data-testid="stCaption"], .stCaption { color: var(--w-mute) !important; }
  p, span, label { color: var(--w-body); }
  hr {
    border: none;
    border-top: 1px solid color-mix(in srgb, var(--w-ink) 8%, transparent);
    margin: 1rem 0;
  }
  div[data-testid="column"] {
    background: transparent !important;
  }
  div[data-testid="stButton"] > button {
    border-radius: var(--w-radius-xl) !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    border: 1px solid transparent !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }
  div[data-testid="stButton"] > button[kind="primary"] {
    background-color: var(--w-primary) !important;
    color: var(--w-on-primary) !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: var(--w-primary-active) !important;
    color: var(--w-on-primary) !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"] {
    background-color: var(--w-canvas-soft) !important;
    color: var(--w-ink) !important;
    border-color: color-mix(in srgb, var(--w-ink) 12%, transparent) !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background-color: color-mix(in srgb, var(--w-primary) 35%, var(--w-canvas-soft)) !important;
  }
  [data-baseweb="textarea"], [data-baseweb="input"] {
    border-radius: var(--w-radius-md) !important;
  }
  [data-testid="stChatInput"] {
    border-radius: var(--w-radius-xl) !important;
    background: var(--w-canvas) !important;
  }
  [data-testid="stChatMessage"] {
    background: var(--w-canvas) !important;
    border-radius: var(--w-radius-xl) !important;
    border: 1px solid color-mix(in srgb, var(--w-ink) 6%, transparent);
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
  }
  [data-testid="stExpander"] details {
    border-radius: var(--w-radius-xl) !important;
    border: 1px solid color-mix(in srgb, var(--w-ink) 8%, transparent);
    background: var(--w-canvas) !important;
  }
  div[data-testid="stDataFrame"] { border-radius: var(--w-radius-md); overflow: hidden; }
  [data-testid="stAlert"] {
    border-radius: var(--w-radius-xl) !important;
  }
  iframe[title="streamlit_plotly_chart"] { border-radius: var(--w-radius-md); }
</style>
"""


def profit_loss_figure(by_year: pd.DataFrame, title: str, height: int = 360) -> go.Figure:
    """연도별 매출·영업이익(막대) + 영업이익율(꺾은선, 보조축). by_year: 연도, 매출액, 영업이익 (원)."""
    by_year = by_year.sort_values(YEAR_COL).copy()
    x = by_year[YEAR_COL].astype(int).tolist()
    s = (by_year["매출액"].astype(float) / EOK).round(2)
    p = (by_year["영업이익"].astype(float) / EOK).round(2)
    denom = by_year["매출액"].replace(0, pd.NA).astype(float)
    m = ((by_year["영업이익"].astype(float) / denom) * 100).round(2).fillna(0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="매출액", x=x, y=s, marker_color=WISE_CHART_SALES))
    fig.add_trace(go.Bar(name="영업이익", x=x, y=p, marker_color=WISE_CHART_PROFIT))
    fig.add_trace(
        go.Scatter(
            name="영업이익율(%)",
            x=x,
            y=m,
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=WISE_CHART_MARGIN, width=2.5),
            marker=dict(size=7),
        ),
    )
    fig.update_layout(
        title=title,
        barmode="group",
        height=height,
        margin=dict(t=44, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        yaxis=dict(title="억 원", gridcolor="rgba(14,15,12,0.08)", zerolinecolor="rgba(14,15,12,0.12)"),
        yaxis2=dict(title="영업이익율 (%)", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="연도", tickmode="linear", dtick=1),
        **wise_plotly_layout(),
    )
    return fig


def clear_dataframe_selection(key: str) -> None:
    st.session_state[key] = {"selection": {"rows": [], "columns": [], "cells": []}}


def dataframe_selection_rows(state, session_key: str | None = None) -> list[int]:
    """st.dataframe(on_select=...) 반환값 또는 session_state에서 선택 행 인덱스 추출."""
    candidates: list = []
    if state is not None:
        candidates.append(state)
    if session_key and session_key in st.session_state:
        candidates.append(st.session_state[session_key])
    for obj in candidates:
        try:
            sel = obj.selection
            rows = sel.rows if hasattr(sel, "rows") else sel.get("rows", [])
            if rows:
                return list(rows)
        except Exception:
            pass
        try:
            rows = obj["selection"]["rows"]
            if rows:
                return list(rows)
        except Exception:
            pass
    return []


def resolve_item_code(df: pd.DataFrame, question: str) -> str | None:
    """질문 텍스트에서 품목 코드(예: 품목_18) 추출."""
    items = set(df[ITEM_COL].dropna().astype(str).unique())
    patterns = [
        r"품목[_\s\-]*(\d+)",
        r"품목\s*(\d+)\s*번",
        r"(\d+)\s*번\s*품목",
        r"(?<![\d])(\d{1,2})\s*번(?!\s*년)",
    ]
    for pat in patterns:
        m = re.search(pat, question)
        if not m:
            continue
        num = m.group(1)
        if not num.isdigit():
            continue
        n = int(num)
        for candidate in (f"품목_{n}", f"품목_{num}", f"품목_{n:02d}"):
            if candidate in items:
                return candidate
        for it in items:
            if it == f"품목_{num.zfill(2)}":
                return it
        for it in sorted(items):
            if it.endswith(f"_{n}") or it.endswith(f"_{num}") or it.endswith(f"_{n:02d}"):
                return it
    return None


def resolve_customer_code(df: pd.DataFrame, question: str) -> str | None:
    """질문에 매출처 이름이 포함된 경우."""
    for c in sorted(df[CUSTOMER_COL].dropna().astype(str).unique(), key=len, reverse=True):
        if c and c in question:
            return c
    return None


def resolve_factory_code(df: pd.DataFrame, question: str) -> str | None:
    for f in sorted(df[FACTORY_COL].dropna().astype(str).unique(), key=len, reverse=True):
        if f and f in question:
            return f
    m = re.search(r"공장\s*([ABC])", question, re.I)
    if m:
        cand = f"공장{m.group(1).upper()}"
        if cand in set(df[FACTORY_COL].unique()):
            return cand
    m = re.search(r"([ABC])\s*공장", question, re.I)
    if m:
        cand = f"공장{m.group(1).upper()}"
        if cand in set(df[FACTORY_COL].unique()):
            return cand
    return None


def _wants_factory_item_operating_insight(q: str) -> bool:
    """공장 + 연도 + 품목 단위 영업이익/증가 질문."""
    if not re.search(r"영업이익", q):
        return False
    return bool(re.search(r"(품목|공헌|향상|기여|증가|늘|많이|가장|크)", q, re.I))


def format_factory_item_operating_summary(df: pd.DataFrame, fac: str, y_target: int) -> str | None:
    """예: 2025년 공장A에서 영업이익 최대 품목 + 전년 대비 증가 최대 품목."""
    dft = df[(df[FACTORY_COL] == fac) & (df[YEAR_COL].astype(int) == y_target)]
    if dft.empty:
        return f"데이터 없음: {y_target}년 **{fac}**에 해당하는 데이터가 없습니다."
    by_item = dft.groupby(ITEM_COL, as_index=True)["영업이익"].sum()
    it_max = str(by_item.idxmax())
    v_max = float(by_item.max())

    y_prev = y_target - 1
    dfp = df[(df[FACTORY_COL] == fac) & (df[YEAR_COL].astype(int) == y_prev)]
    if dfp.empty:
        return (
            f"{y_target}년 **{fac}**에서 가장 큰 영업이익을 낸 품목은 **{it_max}**으로 "
            f"**{v_max / EOK:.2f}억 원**이 발생하였습니다. "
            f"**{y_prev}년** 데이터가 없어 전년 대비 증가 품목은 확인할 수 없습니다."
        )
    cur = dft.groupby(ITEM_COL)["영업이익"].sum()
    prv = dfp.groupby(ITEM_COL)["영업이익"].sum()
    delta = cur.sub(prv, fill_value=0.0)
    it_inc = str(delta.idxmax())
    v_inc = float(delta.loc[it_inc])
    if v_inc >= 0:
        inc_phrase = f"**{v_inc / EOK:.2f}억 원** 증가하였습니다."
    else:
        inc_phrase = f"**{v_inc / EOK:.2f}억 원**으로 연도 대비 변동이 가장 컸습니다(감소)."
    return (
        f"{y_target}년 **{fac}**에서 가장 큰 영업이익을 낸 품목은 **{it_max}**으로 "
        f"**{v_max / EOK:.2f}억 원**이 발생하였으며, "
        f"{y_prev}년 대비 영업이익이 가장 많이 증가한 품목은 **{it_inc}**으로 {inc_phrase}"
    )


def _wants_margin_yoy_drop_item(q: str) -> bool:
    """직전 연도 대비 영업이익율 하락이 큰 품목 질문."""
    if not re.search(r"영업이익율", q):
        return False
    if not re.search(r"(떨어|하락|감소|낮아|하향)", q):
        return False
    if not re.search(r"(전년|직전|작년|전\s*년도|대비)", q):
        return False
    return bool(re.search(r"품목", q))


def format_margin_yoy_worst_item(df: pd.DataFrame, y_target: int, fac: str | None) -> str | None:
    """연도·품목별 합산 매출/이익으로 영업이익율을 만들고, 전년 대비 %p 하락이 가장 큰 품목."""
    y_prev = y_target - 1
    sub = df.copy()
    if fac:
        sub = sub[sub[FACTORY_COL] == fac]
    gt = sub[sub[YEAR_COL].astype(int) == y_target]
    gp = sub[sub[YEAR_COL].astype(int) == y_prev]
    if gt.empty:
        return f"데이터 없음: {y_target}년 데이터가 없습니다."
    if gp.empty:
        return (
            f"데이터 없음: 직전연도({y_prev}년) 데이터가 없어 영업이익율 변화를 계산할 수 없습니다."
        )

    def _margin_by_item(d: pd.DataFrame) -> pd.Series:
        s = d.groupby(ITEM_COL, as_index=True)[["매출액", "영업이익"]].sum(numeric_only=True)
        return (s["영업이익"] / s["매출액"].replace(0, pd.NA)).astype(float)

    mt = _margin_by_item(gt)
    mp = _margin_by_item(gp)
    common = mt.index.intersection(mp.index)
    if len(common) == 0:
        return "데이터 없음: 전년·당년 모두에 존재하는 품목이 없습니다."
    delta_pp = ((mt.loc[common] - mp.loc[common]) * 100).dropna()
    if delta_pp.empty:
        return "확인 불가: 영업이익율 변화를 계산할 수 없습니다."
    neg = delta_pp[delta_pp < 0]
    if neg.empty:
        worst = str(delta_pp.idxmin())
        val = float(delta_pp.min())
        if fac:
            return (
                f"{y_target}년 **{fac}** 기준, 직전연도 대비 영업이익율이 **하락한** 품목은 없습니다. "
                f"이익율 **상승폭이 가장 작은**(상대적으로 가장 부진한) 품목은 **{worst}**입니다 (전년 대비 **{val:+.2f}%p**)."
            )
        return (
            f"{y_target}년에 직전연도 대비 영업이익율이 **하락한** 품목은 없습니다. "
            f"이익율 **상승폭이 가장 작은** 품목은 **{worst}**입니다 (전년 대비 **{val:+.2f}%p**)."
        )

    worst = str(neg.idxmin())
    val = float(neg.min())
    if fac:
        return (
            f"{y_target}년 **{fac}**에서 직전연도 대비 영업이익율이 가장 많이 떨어진 품목은 **{worst}**입니다. "
            f"(약 **{val:.2f}%p** 하락)"
        )
    return (
        f"{y_target}년에 직전연도 대비 영업이익율이 가장 많이 떨어진 품목은 **{worst}**입니다. "
        f"(약 **{val:.2f}%p** 하락)"
    )


def _year_from_question(q: str) -> int | None:
    m = re.search(r"(20[12]\d)\s*년?", q)
    if m:
        return int(m.group(1))
    m = re.search(r"(?<![0-9])(\d{2})\s*년", q)
    if m:
        yy = int(m.group(1))
        y = 2000 + yy if yy < 100 else yy
        if 2000 <= y <= 2100:
            return y
    return None


def answer_from_pl_data(df: pd.DataFrame, question: str) -> str:
    """손익 데이터만으로 답변 가능한 질문에 한해 한국어로 응답. 불가 시 확인 불가/데이터 없음 안내."""
    q = (question or "").strip()
    if not q:
        return "확인 불가: 질문이 비어 있습니다."

    years_sorted = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
    if not years_sorted:
        return "데이터 없음: 연도 정보가 없습니다."
    y_min, y_max = years_sorted[0], years_sorted[-1]

    # --- 요약 / 개요 ---
    if re.search(r"(요약|개요|어떤 데이터|데이터 설명)", q):
        tot_sales = float(df["매출액"].sum()) / EOK
        return (
            f"손익 데이터는 **{y_min}~{y_max}년**, "
            f"공장 **{df[FACTORY_COL].nunique()}**곳, 고객 **{df[CUSTOMER_COL].nunique()}**곳, "
            f"품목 **{df[ITEM_COL].nunique()}**개, 총 **{len(df)}**행입니다. "
            f"전 기간 합산 매출액은 약 **{tot_sales:.2f}억 원**입니다."
        )

    item = resolve_item_code(df, q)
    cust = resolve_customer_code(df, q)
    fac = resolve_factory_code(df, q)
    y_ask = _year_from_question(q)

    # --- 공장 + 연도 + 품목 영업이익(최대·전년비 증가) ---
    if fac and y_ask is not None and _wants_factory_item_operating_insight(q):
        msg = format_factory_item_operating_summary(df, fac, y_ask)
        if msg:
            return msg

    # --- 연도 + 전년 대비 영업이익율 하락 폭이 가장 큰 품목 ---
    if y_ask is not None and _wants_margin_yoy_drop_item(q):
        msg = format_margin_yoy_worst_item(df, y_ask, fac)
        if msg:
            return msg

    # --- 품목 + 원가 세부: 기간 중 증가액이 가장 큰 항목 ---
    inc_kw = bool(re.search(r"(증가|늘었|늘어난|상승)", q)) and bool(
        re.search(r"(원가|비용|재료비|노무비|경비|용기|공통|물류)", q)
    )
    if inc_kw and item:
        sub = df[df[ITEM_COL].astype(str) == item]
        if sub.empty:
            return "데이터 없음: 해당 품목이 데이터에 없습니다."
        by_y = sub.groupby(YEAR_COL, as_index=False)[COST_DETAIL_COLS].sum(numeric_only=True).sort_values(YEAR_COL)
        if len(by_y) < 2:
            return "데이터 없음: 연도별 원가를 비교하기에 데이터가 부족합니다."
        first = by_y.iloc[0]
        last = by_y.iloc[-1]
        deltas = {c: float(last[c]) - float(first[c]) for c in COST_DETAIL_COLS}
        positive = {k: v for k, v in deltas.items() if v > 0}
        if positive:
            best = max(positive, key=positive.get)
            dval = positive[best]
            return (
                f"품목 **{item}**의 원가 세부 항목 중, **{int(first[YEAR_COL])}년 대비 {int(last[YEAR_COL])}년** "
                f"순증가액이 가장 큰 항목은 **{best}**이며, 증가액은 **{dval / EOK:.2f}억 원**입니다."
            )
        # 기간 합계는 감소·동일이나, 연도 사이 증가 폭이 있는지 검사
        best_y1 = best_y2 = best_c = None
        best_step = 0.0
        for i in range(1, len(by_y)):
            row_prev = by_y.iloc[i - 1]
            row_cur = by_y.iloc[i]
            for c in COST_DETAIL_COLS:
                step = float(row_cur[c]) - float(row_prev[c])
                if step > best_step:
                    best_step = step
                    best_c = c
                    best_y1, best_y2 = int(row_prev[YEAR_COL]), int(row_cur[YEAR_COL])
        if best_c is not None and best_step > 0:
            return (
                f"품목 **{item}** 기준, 연도 사이 **원가 세부 증가 폭**이 가장 큰 경우는 "
                f"**{best_y1}년→{best_y2}년**의 **{best_c}**이며, 증가액은 **{best_step / EOK:.2f}억 원**입니다."
            )
        return (
            f"확인 불가: 품목 **{item}**은 제공된 연도 구간에서 원가 세부 항목의 **유의미한 증가(+) 추세를** "
            "찾지 못했습니다."
        )

    # --- 품목 + 특정 연도 가장 큰 원가 항목 ---
    if item and (y_ask is not None) and re.search(r"(가장\s*큰|최대|많은)\s*원가|원가.*(가장\s*큰|최대)", q):
        sub = df[(df[ITEM_COL].astype(str) == item) & (df[YEAR_COL].astype(int) == y_ask)]
        if sub.empty:
            return f"데이터 없음: {item}, {y_ask}년 조건에 맞는 행이 없습니다."
        s = sub[COST_DETAIL_COLS].sum(numeric_only=True)
        best = s.idxmax()
        return f"품목 **{item}**의 **{y_ask}년** 합산 기준 원가 세부 중 금액이 가장 큰 항목은 **{best}** (**{float(s[best]) / EOK:.2f}억 원**)입니다."

    # --- 특정 연도 전체 또는 공장 매출·영업이익 ---
    if y_ask is not None and re.search(r"매출", q) and not re.search(r"품목|고객|매출처", q):
        sub = df[df[YEAR_COL].astype(int) == y_ask]
        if fac:
            sub = sub[sub[FACTORY_COL] == fac]
        if sub.empty:
            return f"데이터 없음: {y_ask}년(지정 조건)에 해당하는 데이터가 없습니다."
        v = float(sub["매출액"].sum()) / EOK
        scope = f"**{fac}** " if fac else "전체 "
        return f"{scope}**{y_ask}년** 매출액 합계는 **{v:.2f}억 원**입니다."

    if y_ask is not None and re.search(r"영업이익(?!율)", q) and not re.search(r"품목|고객|매출처", q):
        sub = df[df[YEAR_COL].astype(int) == y_ask]
        if fac:
            sub = sub[sub[FACTORY_COL] == fac]
        if sub.empty:
            return f"데이터 없음: {y_ask}년(지정 조건)에 해당하는 데이터가 없습니다."
        v = float(sub["영업이익"].sum()) / EOK
        scope = f"**{fac}** " if fac else "전체 "
        return f"{scope}**{y_ask}년** 영업이익 합계는 **{v:.2f}억 원**입니다."

    # --- 고객 + 연도 매출 ---
    if cust and y_ask and re.search(r"매출", q):
        sub = df[(df[CUSTOMER_COL] == cust) & (df[YEAR_COL].astype(int) == y_ask)]
        if sub.empty:
            return f"데이터 없음: **{cust}**, {y_ask}년 매출 데이터가 없습니다."
        v = float(sub["매출액"].sum()) / EOK
        return f"**{cust}**의 **{y_ask}년** 매출액 합계는 **{v:.2f}억 원**입니다."

    if cust and re.search(r"매출", q) and y_ask is None and not re.search(r"영업이익율", q):
        sub = df[df[CUSTOMER_COL] == cust]
        if sub.empty:
            return "데이터 없음: 해당 매출처가 없습니다."
        v = float(sub["매출액"].sum()) / EOK
        return f"**{cust}** 전 기간({y_min}~{y_max}년) 매출액 합계는 **{v:.2f}억 원**입니다."

    # --- 품목 매출/이익 (연도 미지정 시 전체 합) ---
    if item and re.search(r"매출", q):
        sub = df[df[ITEM_COL].astype(str) == item]
        if sub.empty:
            return "데이터 없음: 해당 품목이 없습니다."
        if y_ask:
            sub = sub[sub[YEAR_COL].astype(int) == y_ask]
            if sub.empty:
                return f"데이터 없음: {item}, {y_ask}년 데이터가 없습니다."
        v = float(sub["매출액"].sum()) / EOK
        span = f"{y_ask}년 " if y_ask else "전 기간 "
        return f"품목 **{item}**의 {span}매출액 합계는 **{v:.2f}억 원**입니다."

    if item and re.search(r"영업이익(?!율)", q):
        sub = df[df[ITEM_COL].astype(str) == item]
        if sub.empty:
            return "데이터 없음: 해당 품목이 없습니다."
        if y_ask:
            sub = sub[sub[YEAR_COL].astype(int) == y_ask]
        v = float(sub["영업이익"].sum()) / EOK
        span = f"{y_ask}년 " if y_ask else "전 기간 "
        return f"품목 **{item}**의 {span}영업이익 합계는 **{v:.2f}억 원**입니다."

    # --- 증가 질문인데 품목을 못 찾은 경우 ---
    if inc_kw and not item:
        return "확인 불가: 어떤 품목인지(예: 품목_18, 18번) 데이터에 맞게 알려주시면 원가 증가 분석이 가능합니다."

    return (
        "확인 불가: 제공된 손익 데이터와 질문 형식만으로는 답변을 도출할 수 없습니다. "
        "연도·공장·매출처·품목 코드(예: 품목_18)를 포함해 질문해 주세요."
    )


def render_top_nav() -> None:
    st.title("손익분석 시스템")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        factory_type = "primary" if st.session_state.nav_tab == "factory" else "secondary"
        if st.button("공장별 손익", key="btn_nav_factory", type=factory_type, use_container_width=True):
            st.session_state.nav_tab = "factory"
            st.rerun()
    with c2:
        cust_type = "primary" if st.session_state.nav_tab == "customer" else "secondary"
        if st.button("고객별 손익", key="btn_nav_customer", type=cust_type, use_container_width=True):
            st.session_state.nav_tab = "customer"
            st.rerun()


def factory_dashboard(df: pd.DataFrame) -> None:
    st.subheader("공장별 손익")
    st.caption("금액: 억 원 / 영업이익율(%) = 해당 연도 합산 영업이익 ÷ 합산 매출")

    years = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
    y_last = years[-1]

    agg_f = (
        df.groupby([FACTORY_COL, YEAR_COL], as_index=False)[["매출액", "영업이익"]]
        .sum(numeric_only=True)
        .sort_values([FACTORY_COL, YEAR_COL])
    )

    last_year_facs = agg_f.loc[agg_f[YEAR_COL] == y_last, FACTORY_COL].dropna().unique()
    factories_sorted = factories_for_factory_tab(last_year_facs)

    st.markdown("**전체 공장 — 연도별 매출·영업이익(막대) 및 영업이익율(꺾은선)**")
    fcols = st.columns(len(factories_sorted))
    for i, fac in enumerate(factories_sorted):
        sub_y = agg_f[agg_f[FACTORY_COL] == fac]
        with fcols[i]:
            st.plotly_chart(
                profit_loss_figure(sub_y, f"{fac}", height=320),
                use_container_width=True,
                key=f"pl_factory_head_{i}",
            )

    sum_rows = []
    for fac in factories_sorted:
        ly = agg_f[(agg_f[FACTORY_COL] == fac) & (agg_f[YEAR_COL] == y_last)]
        ssum = float(ly["매출액"].sum()) if len(ly) else 0.0
        psum = float(ly["영업이익"].sum()) if len(ly) else 0.0
        rte = (psum / ssum * 100) if ssum else None
        sum_rows.append(
            {
                FACTORY_COL: fac,
                f"{y_last}년 매출(억)": round(ssum / EOK, 2),
                f"{y_last}년 영업이익(억)": round(psum / EOK, 2),
                f"{y_last}년 영업이익율(%)": round(rte, 2) if rte is not None else None,
            }
        )
    sum_df = pd.DataFrame(sum_rows)
    st.caption("아래 표에서 **공장 행을 선택**하면 해당 공장의 품목별 세부가 표시됩니다.")
    fac_pick_state = st.dataframe(
        sum_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="factory_pick_table",
    )
    pr = dataframe_selection_rows(fac_pick_state, "factory_pick_table")
    if pr:
        idx = int(pr[0])
        fac_new = str(sum_df.iloc[idx][FACTORY_COL])
        if st.session_state.f_factory != fac_new:
            st.session_state.f_item = None
            if "item_pick_table" in st.session_state:
                clear_dataframe_selection("item_pick_table")
        st.session_state.f_factory = fac_new

    fac_sel = st.session_state.f_factory
    if fac_sel is None:
        st.info("**공장** 한 곳을 위 표에서 선택해 주세요.")
        return

    st.divider()
    st.markdown(f"**선택: `{fac_sel}`**")
    if st.button("공장 선택 해제", key="factory_clear_sel"):
        st.session_state.f_factory = None
        st.session_state.f_item = None
        if "factory_pick_table" in st.session_state:
            clear_dataframe_selection("factory_pick_table")
        if "item_pick_table" in st.session_state:
            clear_dataframe_selection("item_pick_table")
        st.rerun()

    sub = df[df[FACTORY_COL] == fac_sel]
    agg_item = sub.groupby([ITEM_COL, YEAR_COL], as_index=False)[["매출액", "영업이익"]].sum(numeric_only=True)
    fac_year = agg_f[agg_f[FACTORY_COL] == fac_sel].sort_values(YEAR_COL)
    st.plotly_chart(
        profit_loss_figure(fac_year, f"{fac_sel} · 연도별 매출·영업이익·이익율", height=380),
        use_container_width=True,
        key="pl_factory_sel_year",
    )

    agg_last = (
        agg_item[agg_item[YEAR_COL] == y_last].sort_values("매출액", ascending=False).copy()
    )
    agg_last["매출(억)"] = (agg_last["매출액"] / EOK).round(2)
    agg_last["영업이익(억)"] = (agg_last["영업이익"] / EOK).round(2)
    agg_last["영업이익율(%)"] = (
        (agg_last["영업이익"] / agg_last["매출액"].replace(0, pd.NA) * 100).round(2)
    )
    item_pick_df = agg_last[[ITEM_COL, "매출(억)", "영업이익(억)", "영업이익율(%)"]].reset_index(drop=True)

    st.markdown(f"**품목별 ({y_last}년 매출 기준 내림차순)**")
    st.caption("표에서 **품목** 행을 선택하면 해당 품목의 연도별 손익 그래프와 원가 세부가 표시됩니다.")
    ip_state = st.dataframe(
        item_pick_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="item_pick_table",
    )
    ir = dataframe_selection_rows(ip_state, "item_pick_table")
    if ir:
        st.session_state.f_item = str(item_pick_df.iloc[int(ir[0])][ITEM_COL])

    item_sel = st.session_state.f_item
    if item_sel is None:
        st.info("품목 표에서 **한 줄을 선택**해 주세요.")
        return

    st.divider()
    st.markdown(f"**`{fac_sel}` / `{item_sel}`**")
    item_y = agg_item[agg_item[ITEM_COL] == item_sel].sort_values(YEAR_COL)
    st.plotly_chart(
        profit_loss_figure(item_y, f"{item_sel} · 연도별 매출·영업이익·이익율", height=380),
        use_container_width=True,
        key="pl_item_sel_year",
    )

    st.markdown("**원가 세부항목 (표 + 추이)**")
    sub_i = df[(df[FACTORY_COL] == fac_sel) & (df[ITEM_COL] == item_sel)]
    by_year = (
        sub_i.groupby(YEAR_COL, as_index=False)[COST_DETAIL_COLS]
        .sum(numeric_only=True)
        .sort_values(YEAR_COL)
    )
    disp = df_to_eok_display(by_year, COST_DETAIL_COLS)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    long = by_year.melt(id_vars=[YEAR_COL], var_name="항목", value_name="금액_원")
    long["금액_억원"] = (long["금액_원"].astype(float) / EOK).round(2)
    fig = go.Figure()
    for name, gl in long.groupby("항목"):
        fig.add_trace(
            go.Scatter(
                x=gl[YEAR_COL].astype(int),
                y=gl["금액_억원"],
                mode="lines+markers",
                name=name,
            )
        )
    fig.update_layout(
        title="원가 세부항목 추이 (억 원)",
        xaxis_title="연도",
        yaxis_title="억 원",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=56),
        height=420,
        colorway=list(WISE_PLOT_COLORWAY),
        **wise_plotly_layout(),
    )
    st.plotly_chart(fig, use_container_width=True, key="pl_cost_detail")

    if st.button("뒤로 (품목만 해제)", key="back_clear_item"):
        st.session_state.f_item = None
        if "item_pick_table" in st.session_state:
            clear_dataframe_selection("item_pick_table")
        st.rerun()


def customer_dashboard(df: pd.DataFrame) -> None:
    st.subheader("고객별 손익")

    g = df.groupby([CUSTOMER_COL, YEAR_COL], as_index=False).agg(
        매출액=("매출액", "sum"),
        영업이익=("영업이익", "sum"),
    )
    g["영업이익율"] = (g["영업이익"] / g["매출액"].replace(0, pd.NA)).round(4)
    years = sorted(g[YEAR_COL].dropna().astype(int).unique().tolist())
    y_last = years[-1]

    rows = []
    for cust in sorted(g[CUSTOMER_COL].dropna().unique().tolist()):
        row = {CUSTOMER_COL: cust}
        sub = g[g[CUSTOMER_COL] == cust]
        for y in years:
            r = sub[sub[YEAR_COL] == y]
            if len(r):
                row[f"{y}_매출액"] = r["매출액"].iloc[0]
                row[f"{y}_영업이익"] = r["영업이익"].iloc[0]
                row[f"{y}_영업이익율"] = r["영업이익율"].iloc[0]
            else:
                row[f"{y}_매출액"] = 0
                row[f"{y}_영업이익"] = 0
                row[f"{y}_영업이익율"] = pd.NA
        rows.append(row)
    list_df = pd.DataFrame(rows)

    col_last_rev = f"{y_last}_매출액"
    list_df = list_df.sort_values(col_last_rev, ascending=False).reset_index(drop=True)

    disp_full = pd.DataFrame({CUSTOMER_COL: list_df[CUSTOMER_COL]})
    for y in years:
        disp_full[f"{y}년 매출(억)"] = (list_df[f"{y}_매출액"].astype(float) / EOK).round(2)
        disp_full[f"{y}년 영업이익(억)"] = (list_df[f"{y}_영업이익"].astype(float) / EOK).round(2)
        disp_full[f"{y}년 영업이익율(%)"] = format_margin_pct(list_df[f"{y}_영업이익율"])

    st.markdown(f"**고객사 목록 ({y_last}년 매출액 기준 내림차순, {years[0]}~{y_last}년)**")
    st.caption("금액: 억 원 / 이익율: % — 아래 그래프는 각 매출처의 연도별 추이입니다. 품목별 상세는 표에서 행을 선택하세요.")
    df_state = st.dataframe(
        disp_full,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="customer_list_sf",
    )
    pick_r = dataframe_selection_rows(df_state, "customer_list_sf")
    sel: str | None = None
    if pick_r:
        sel = str(disp_full.iloc[int(pick_r[0])][CUSTOMER_COL])

    st.markdown("**고객별 연도별 매출·영업이익(막대) 및 영업이익율(꺾은선)**")
    ncols = 4
    cust_list = list_df[CUSTOMER_COL].tolist()
    for r0 in range(0, len(cust_list), ncols):
        chunk = cust_list[r0 : r0 + ncols]
        cols = st.columns(len(chunk))
        for ci, cname in enumerate(chunk):
            cy = g[g[CUSTOMER_COL] == cname][[YEAR_COL, "매출액", "영업이익"]].sort_values(YEAR_COL)
            with cols[ci]:
                st.plotly_chart(
                    profit_loss_figure(cy, str(cname), height=260),
                    use_container_width=True,
                    key=f"pl_cust_grid_{r0}_{ci}",
                )

    if sel is None:
        st.info("품목별 상세를 보려면 위 **고객사 목록** 표에서 매출처 한 줄을 선택해 주세요.")
        return

    st.divider()
    st.markdown(f"**선택: `{sel}`**")

    cust_year = g[g[CUSTOMER_COL] == sel][[YEAR_COL, "매출액", "영업이익"]].sort_values(YEAR_COL)
    st.plotly_chart(
        profit_loss_figure(cust_year, f"{sel} · 연도별 (확대)", height=400),
        use_container_width=True,
        key="pl_customer_year",
    )

    st.markdown(f"**품목별 — {y_last}년 매출·영업이익 (막대) 및 영업이익율 (꺾은선, 연도별)**")
    sub = df[df[CUSTOMER_COL] == sel]
    agg = sub.groupby([ITEM_COL, YEAR_COL], as_index=False).agg(
        매출액=("매출액", "sum"),
        영업이익=("영업이익", "sum"),
    )
    agg["영업이익율"] = (agg["영업이익"] / agg["매출액"].replace(0, pd.NA)).round(4)

    top_items = (
        agg[agg[YEAR_COL] == y_last]
        .sort_values("매출액", ascending=False)[ITEM_COL]
        .astype(str)
        .unique()
        .tolist()
    )
    agg_top = agg[agg[ITEM_COL].astype(str).isin(top_items)].copy()

    fig_m = go.Figure()
    for it in top_items:
        chunk = agg_top[agg_top[ITEM_COL] == it].sort_values(YEAR_COL)
        fig_m.add_trace(
            go.Bar(
                name=str(it),
                x=chunk[YEAR_COL].astype(int),
                y=(chunk["매출액"].astype(float) / EOK).round(2),
                legendgroup=str(it),
            )
        )
    fig_m.update_layout(
        title=f"{sel} · 연도별 매출액 (억 원, 품목별)",
        barmode="group",
        height=360,
        xaxis=dict(title="연도", dtick=1),
        yaxis=dict(title="억 원"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        colorway=list(WISE_PLOT_COLORWAY),
        **wise_plotly_layout(),
    )
    st.plotly_chart(fig_m, use_container_width=True, key="pl_cust_item_sales")

    fig_p = go.Figure()
    for it in top_items:
        chunk = agg_top[agg_top[ITEM_COL] == it].sort_values(YEAR_COL)
        fig_p.add_trace(
            go.Bar(
                name=str(it),
                x=chunk[YEAR_COL].astype(int),
                y=(chunk["영업이익"].astype(float) / EOK).round(2),
                legendgroup=str(it),
                showlegend=False,
            )
        )
    fig_p.update_layout(
        title="연도별 영업이익 (억 원)",
        barmode="group",
        height=360,
        xaxis=dict(title="연도", dtick=1),
        yaxis=dict(title="억 원"),
        colorway=list(WISE_PLOT_COLORWAY),
        **wise_plotly_layout(),
    )
    st.plotly_chart(fig_p, use_container_width=True, key="pl_cust_item_profit")

    fig_l = go.Figure()
    for it in top_items:
        chunk = agg_top[agg_top[ITEM_COL] == it].sort_values(YEAR_COL)
        pct = (chunk["영업이익"].astype(float) / chunk["매출액"].replace(0, pd.NA).astype(float) * 100).round(2)
        pct = pct.fillna(0.0)
        fig_l.add_trace(
            go.Scatter(
                name=str(it),
                x=chunk[YEAR_COL].astype(int),
                y=pct,
                mode="lines+markers",
                legendgroup=str(it),
            )
        )
    fig_l.update_layout(
        title="연도별 영업이익율 (%)",
        height=360,
        xaxis=dict(title="연도", dtick=1),
        yaxis=dict(title="영업이익율 (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08),
        colorway=list(WISE_PLOT_COLORWAY),
        **wise_plotly_layout(),
    )
    st.plotly_chart(fig_l, use_container_width=True, key="pl_cust_item_margin")

    st.caption(
        f"위 품목 차트는 **{y_last}년 매출 기준으로 정렬된 전체 품목**의 연도별 추이입니다. (범례가 길면 그래프 상단에서 확인할 수 있습니다.)"
    )


def chat_panel(df: pd.DataFrame) -> None:
    st.subheader("분석 챗봇")
    vlm_base, vlm_model, _vk = get_vlm_config()
    gem_key = get_gemini_api_key()

    if vlm_base and vlm_model:
        st.caption(f"내부 LLM 사용 중 · `{vlm_model}` @ `{vlm_base}`")
    elif gem_key:
        st.caption("Gemini API가 설정되어 있습니다. 답변은 데이터 요약 + LLM이 생성합니다.")
    else:
        with st.expander("LLM(내부 서버 또는 Gemini) 연결 방법", expanded=False):
            st.markdown(
                """
**1) 내부 OpenAI 호환 API (우선 사용)**  
환경변수 또는 `.streamlit/secrets.toml`:

```toml
VLM_BASE_URL = "http://192.168.106.63:8000/v1"
VLM_MODEL_NAME = "My_Model"
# 서버가 키를 요구하면:
# VLM_API_KEY = "your-key"
```

- `VLM_BASE_URL`: `/v1`까지 포함 (예: `http://호스트:8000/v1`)  
- `VLM_MODEL_NAME`: 서버에 등록된 모델 이름  
- 키가 필요 없으면 `VLM_API_KEY` 생략 시 내부용 더미 값으로 호출합니다.

**2) Google Gemini (내부 미설정 시)**  
`GEMINI_API_KEY` + (선택) `GEMINI_MODEL`

**3) 위가 모두 없으면** 규칙 기반 답변만 사용합니다.
                """
            )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        vlm_base, vlm_model, _vk = get_vlm_config()
        gem_key = get_gemini_api_key()

        if vlm_base and vlm_model:
            if not openai_available():
                reply = (
                    "내부 LLM(OpenAI 호환)을 사용하려면 **`openai`** 패키지가 필요합니다.\n\n"
                    f"이 앱과 **같은 Python**으로 설치하세요:\n```\n{sys.executable} -m pip install openai\n```\n\n"
                    "Streamlit을 다른 가상환경에서 실행 중이면, 그 환경에서 위 명령을 실행한 뒤 앱을 다시 시작하세요."
                )
            else:
                try:
                    with st.spinner("내부 LLM 응답 생성 중…"):
                        reply = generate_vlm_reply(df, st.session_state.chat_messages)
                except Exception as e:
                    reply = f"_(내부 LLM 호출 실패: {e})_\n\n---\n{answer_from_pl_data(df, prompt)}"
        elif gem_key:
            try:
                with st.spinner("Gemini 응답 생성 중…"):
                    reply = generate_gemini_reply(df, st.session_state.chat_messages)
            except ImportError:
                reply = (
                    answer_from_pl_data(df, prompt)
                    + "\n\n_(패키지 설치: `pip install google-generativeai`)_"
                )
            except Exception as e:
                reply = f"_(Gemini 호출 실패: {e})_\n\n---\n{answer_from_pl_data(df, prompt)}"
        else:
            reply = answer_from_pl_data(df, prompt)

        reply = (reply or "").strip() or "응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="손익분석 시스템", layout="wide", initial_sidebar_state="collapsed")
    init_session_state()
    # 좌·우 분할 영역만 고정 높이 안에서 각각 스크롤 (Streamlit 내장 스크롤)
    _split_h = min(920, max(520, int(st.session_state.get("_split_px", 820))))
    st.markdown(wise_streamlit_css(), unsafe_allow_html=True)

    if not DATA_PATH.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        st.stop()

    df = cached_data()

    vb, vm, _ = get_vlm_config()
    if vb and vm and not openai_available():
        st.warning(
            f"VLM이 설정되어 있으나 **`openai`** 가 이 Python에 설치되어 있지 않습니다. "
            f"터미널에서 실행: `{sys.executable} -m pip install openai` 후 앱을 다시 실행하세요."
        )

    render_top_nav()

    left, right = st.columns([7, 3], gap="medium")
    with left:
        with st.container(height=_split_h, border=False):
            if st.session_state.nav_tab == "factory":
                factory_dashboard(df)
            else:
                customer_dashboard(df)

    with right:
        with st.container(height=_split_h, border=False):
            chat_panel(df)


if __name__ == "__main__":
    main()
