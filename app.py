"""
손익분석 시스템 — Streamlit 대시보드
데이터: AI단기과제_더미데이터.xlsx (매출-원가자료 + 품목별제조원가) 또는 레거시 손익데이터 시트
"""

from __future__ import annotations

import html as html_module
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from data_loader import load_pl_dataframe, resolve_data_path

DATA_PATH = resolve_data_path()
EOK = 100_000_000  # 억 원 환산
M_WON = 1_000_000  # 백만 원(M) 표시
VARIABLE_COST_COLS = ["재료비", "노무비", "경비", "용기상각비"]
MFG_COST_COLS = ["재료비", "노무비", "경비"]

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
FACTORY_TAB_ORDER: tuple[str, ...] = ("M30", "M10", "M20")


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
        "filters_applied": False,
        "flt_year": None,
        "flt_factories": None,
        "flt_customers": None,
        "flt_items": None,
        "watch_detail_open": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_raw_data() -> pd.DataFrame:
    return load_pl_dataframe(DATA_PATH)


@st.cache_data(show_spinner=False)
def cached_data() -> pd.DataFrame:
    return load_raw_data()


def with_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """공헌이익 = 매출액 − 변동원가(재료·노무·경비·용기상각)."""
    out = df.copy()
    var_cost = out[VARIABLE_COST_COLS].sum(axis=1, numeric_only=True)
    out["공헌이익"] = out["매출액"] - var_cost
    out["공헌이익률"] = out["공헌이익"] / out["매출액"].replace(0, pd.NA)
    return out


def apply_global_filters(
    df: pd.DataFrame,
    year: int | None,
    factories: list[str] | None,
    customers: list[str] | None,
    items: list[str] | None,
) -> pd.DataFrame:
    """연도·공장·고객·품목 필터를 동시에 적용(AND)."""
    out = df
    if year is not None:
        out = out[out[YEAR_COL].astype(int) == year]
    if factories:
        out = out[out[FACTORY_COL].astype(str).isin(factories)]
    if customers:
        out = out[out[CUSTOMER_COL].astype(str).isin(customers)]
    if items:
        out = out[out[ITEM_COL].astype(str).isin(items)]
    return out


def apply_scope_filters(
    df: pd.DataFrame,
    factories: list[str] | None,
    customers: list[str] | None,
    items: list[str] | None,
) -> pd.DataFrame:
    """연도 제외 — 추이 차트용."""
    return apply_global_filters(df, None, factories, customers, items)


def _sanitize_multiselect(selected: list[str] | None, options: list[str]) -> list[str]:
    if not selected:
        return []
    opt_set = set(options)
    return [x for x in selected if x in opt_set]


def filter_factory_options(df: pd.DataFrame, year: int) -> list[str]:
    scoped = apply_global_filters(df, year, None, None, None)
    return factories_for_factory_tab(scoped[FACTORY_COL].dropna().unique())


def filter_customer_options(
    df: pd.DataFrame,
    year: int,
    factories: list[str] | None,
) -> list[str]:
    scoped = apply_global_filters(df, year, factories, None, None)
    return sorted(scoped[CUSTOMER_COL].dropna().astype(str).unique().tolist())


def filter_item_options(
    df: pd.DataFrame,
    year: int,
    factories: list[str] | None,
    customers: list[str] | None,
) -> list[str]:
    scoped = apply_global_filters(df, year, factories, customers, None)
    return sorted(scoped[ITEM_COL].dropna().astype(str).unique().tolist())


def inject_multiselect_autoclose_js() -> None:
    """멀티셀렉트·셀렉트: 마우스가 팝오버를 벗어나면 닫힘."""
    components.html(
        """
<script>
(function () {
  const root = window.parent.document;
  if (!root || root._plFilterAutoClose) return;
  root._plFilterAutoClose = true;

  const blurInputs = () => {
    root.querySelectorAll('[data-testid="stMultiSelect"] input, [data-testid="stSelectbox"] input')
      .forEach((el) => { try { el.blur(); } catch (e) {} });
  };

  const bindPopover = (pop) => {
    if (!pop || pop.dataset.plAutoClose) return;
    pop.dataset.plAutoClose = "1";
    pop.addEventListener("mouseleave", () => {
      setTimeout(() => {
        if (!pop.matches(":hover")) blurInputs();
      }, 120);
    });
  };

  const scan = () => root.querySelectorAll('[data-baseweb="popover"]').forEach(bindPopover);

  const obs = new MutationObserver(scan);
  obs.observe(root.body, { childList: true, subtree: true });
  scan();

  const wrap = root.querySelector(".filter-col-wrap");
  if (wrap) {
    wrap.addEventListener("mouseleave", (ev) => {
      if (!wrap.contains(ev.relatedTarget)) setTimeout(blurInputs, 80);
    });
  }
})();
</script>
        """,
        height=0,
        width=0,
    )


def fmt_eok(value: float) -> str:
    """억 원 단위 숫자 문자열 (기호 없음)."""
    return f"{value / EOK:,.2f}"


def trend_year_bounds(year: int, data_min_year: int = 2021) -> tuple[int, int]:
    """기준연도 포함 최근 5개년 (데이터 시작 연도 미만으로는 내려가지 않음)."""
    y_end = int(year)
    y_start = max(data_min_year, y_end - 4)
    return y_start, y_end


def trend_year_label(y_start: int, y_end: int) -> str:
    return f"{y_start}년~{y_end}년"


def yoy_pct(curr: float, prev: float) -> float | None:
    if prev == 0:
        return None
    return (curr - prev) / abs(prev) * 100.0


def kpi_delta_html(pct: float | None, *, unit: str = "%", invert: bool = False) -> str:
    if pct is None:
        return "<span class='kpi-delta neutral'>—</span>"
    good = pct >= 0 if not invert else pct <= 0
    cls = "up" if good else "down"
    arrow = "▲" if pct >= 0 else "▼"
    return f"<span class='kpi-delta {cls}'>{arrow} {abs(pct):.1f}{unit} YoY</span>"


def aggregate_year_metrics(df: pd.DataFrame, year: int) -> dict[str, float]:
    sub = df[df[YEAR_COL].astype(int) == year]
    sales = float(sub["매출액"].sum())
    contrib = float(sub["공헌이익"].sum())
    qty = float(sub["매출수량"].sum())
    mfg = float(sub[MFG_COST_COLS].sum(axis=1).sum())
    margin = (contrib / sales * 100.0) if sales else 0.0
    unit_mfg = (mfg / qty) if qty > 0 else 0.0
    op_profit = float(sub["영업이익"].sum())
    op_margin = (op_profit / sales * 100.0) if sales else 0.0
    return {
        "매출액": sales,
        "공헌이익": contrib,
        "공헌이익률": margin,
        "영업이익": op_profit,
        "영업이익률": op_margin,
        "단위제조원가": unit_mfg,
    }


def _watch_item_groups(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """품목·고객사 단위 집계 (요주의 품목 판별용)."""
    sub = df[df[YEAR_COL].astype(int) == year]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby([ITEM_COL, CUSTOMER_COL], as_index=False).agg(
        매출수량=("매출수량", "sum"),
        매출액=("매출액", "sum"),
        영업이익=("영업이익", "sum"),
    )
    g["영업이익율"] = (g["영업이익"] / g["매출액"].replace(0, pd.NA)).astype(float)
    return g


def count_watch_items(df: pd.DataFrame, year: int) -> int:
    g = _watch_item_groups(df, year)
    if g.empty:
        return 0
    bad = (g["영업이익"] < 0) | (g["영업이익율"] < 0.05)
    return int(bad.sum())


def watch_item_detail_rows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    g = _watch_item_groups(df, year)
    if g.empty:
        return g
    bad = (g["영업이익"] < 0) | (g["영업이익율"] < 0.05)
    out = g.loc[bad].copy()
    out["품목"] = out[ITEM_COL].astype(str)
    out["고객사"] = out[CUSTOMER_COL].astype(str)
    out["매출액(억)"] = (out["매출액"] / EOK).round(2)
    out["영업이익(억)"] = (out["영업이익"] / EOK).round(2)
    out["영업이익율(%)"] = (out["영업이익율"] * 100).round(2)
    return out.sort_values("매출액", ascending=False)[
        ["품목", "고객사", "매출수량", "매출액(억)", "영업이익(억)", "영업이익율(%)"]
    ]


def detect_anomalies(df: pd.DataFrame) -> list[str]:
    """공장별 재료비 전년 대비 급등(15%+) 감지."""
    alerts: list[str] = []
    years = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
    if len(years) < 2:
        return alerts
    y_curr, y_prev = years[-1], years[-2]
    for fac in sorted(df[FACTORY_COL].dropna().astype(str).unique()):
        c = df[(df[FACTORY_COL] == fac) & (df[YEAR_COL] == y_curr)]["재료비"].sum()
        p = df[(df[FACTORY_COL] == fac) & (df[YEAR_COL] == y_prev)]["재료비"].sum()
        pct = yoy_pct(float(c), float(p))
        if pct is not None and pct >= 15.0:
            alerts.append(f"{fac} 공장 원자재비 전년비 {pct:.0f}% 상승 감지")
    return alerts[:3]


def revenue_contribution_figure(by_year: pd.DataFrame, title: str) -> go.Figure:
    """매출(막대, 억 원) + 공헌이익률(꺾은선)."""
    by_year = by_year.sort_values(YEAR_COL).copy()
    x = by_year[YEAR_COL].astype(int).tolist()
    sales_eok = (by_year["매출액"].astype(float) / EOK).round(2)
    margin = (by_year["공헌이익"].astype(float) / by_year["매출액"].replace(0, pd.NA) * 100).round(1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="매출액",
            x=x,
            y=sales_eok,
            marker_color="#3b82f6",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="공헌이익률(%)",
            x=x,
            y=margin,
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(t=48, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        yaxis=dict(title="매출액 (억 원)", gridcolor="rgba(14,15,12,0.08)"),
        yaxis2=dict(title="공헌이익률 (%)", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="연도", tickmode="linear", dtick=1),
        **wise_plotly_layout(),
    )
    return fig


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
  .app-title-top-spacer {
    display: none;
  }
  section[data-testid="stMain"] {
    padding-top: 0.85rem !important;
  }
  section[data-testid="stMain"] > div {
    padding-top: 0 !important;
  }
  .block-container {
    padding-top: 3.5rem !important;
    max-width: 100% !important;
  }
  section.main > div {
    scroll-padding-top: 0.5rem;
  }
  h1, [data-testid="stHeader"] { color: var(--w-ink) !important; }
  h1 {
    font-weight: 900 !important;
    font-size: clamp(1.5rem, 2.2vw, 2rem) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.38 !important;
    padding-top: 0.65rem !important;
    padding-bottom: 0.2rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.35rem !important;
    overflow: visible !important;
  }
  section[data-testid="stMain"] [data-testid="element-container"] {
    overflow: visible !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    overflow: visible !important;
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
  .filter-panel {
    background: transparent;
    border: none;
    padding: 0;
  }
  .filter-col-wrap [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--w-canvas-soft) !important;
    border: none !important;
    box-shadow: none !important;
  }
  .filter-col-wrap [data-testid="stVerticalBlock"] {
    background: var(--w-canvas-soft) !important;
  }
  .filter-panel h4 {
    margin: 0 0 0.75rem 0;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--w-ink);
  }
  .filter-panel label, .filter-panel p {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    color: var(--w-mute) !important;
    text-transform: uppercase;
  }
  .anomaly-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: var(--w-radius-sm);
    padding: 0.65rem 0.75rem;
    font-size: 0.82rem;
    color: #92400e;
    margin-top: 0.75rem;
  }
  .kpi-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: #fff;
    border-radius: var(--w-radius-md);
    padding: 1rem 1.1rem;
    min-height: 108px;
  }
  .kpi-card.light {
    background: var(--w-canvas);
    color: var(--w-ink);
    border: 1px solid color-mix(in srgb, var(--w-ink) 10%, transparent);
  }
  .kpi-card.green {
    background: linear-gradient(135deg, #6d9f7c 0%, #9fd4a8 48%, #e4f7d6 100%);
    color: #0e0f0c;
    border: 1px solid color-mix(in srgb, #4a7c59 22%, transparent);
  }
  .kpi-card.green .kpi-label { opacity: 0.88; }
  .kpi-card.green .kpi-delta.up { color: #166534; }
  .kpi-card.green .kpi-delta.down { color: #b91c1c; }
  .kpi-card.warn {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
  }
  .kpi-card .kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    opacity: 0.9;
    margin-bottom: 0.35rem;
  }
  .kpi-card .kpi-value {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.02em;
  }
  .kpi-card .kpi-sub {
    font-size: 0.75rem;
    margin-top: 0.35rem;
    opacity: 0.85;
  }
  .kpi-delta { font-size: 0.78rem; font-weight: 600; }
  .kpi-delta.up { color: #86efac; }
  .kpi-delta.down { color: #fca5a5; }
  .kpi-card.light .kpi-delta.up { color: #16a34a; }
  .kpi-card.light .kpi-delta.down { color: #dc2626; }
  .kpi-delta.neutral { opacity: 0.7; }
  div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-weight: 600 !important;
  }
  [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"][kind="primary"] {
    background-color: #2563eb !important;
    color: #fff !important;
  }
  [data-testid="stAppViewContainer"] button[data-testid="baseButton-primary"][kind="primary"]:hover {
    background-color: #1d4ed8 !important;
  }
  .year-range-hint {
    font-size: 0.8rem;
    color: var(--w-body);
    margin: -0.25rem 0 0.75rem 0;
    padding: 0.35rem 0.5rem;
    background: color-mix(in srgb, var(--w-canvas) 70%, transparent);
    border-radius: var(--w-radius-sm);
  }
  .kpi-grid-row {
    align-items: stretch !important;
  }
  .kpi-grid-row > div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
  }
  .kpi-grid-row > div[data-testid="column"] > div {
    flex: 1 1 auto !important;
    width: 100% !important;
  }
  .kpi-card.green.watch-display {
    margin-bottom: 0 !important;
    cursor: pointer;
  }
  .kpi-card.green.watch-display.selected {
    box-shadow: 0 0 0 3px rgba(159, 232, 112, 0.85);
  }
  .kpi-card.green .kpi-value.watch-alert {
    color: #b91c1c;
  }
  .kpi-card.green .kpi-delta.watch-hint {
    color: #7f1d1d;
    opacity: 0.9;
  }
  div[data-testid="stHorizontalBlock"]:has(.watch-display) {
    align-items: stretch !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.watch-display) > div[data-testid="column"] {
    align-self: stretch !important;
  }
  div[data-testid="column"]:has(.watch-display) [data-testid="stVerticalBlock"] {
    position: relative !important;
    gap: 0 !important;
    height: 100% !important;
  }
  div[data-testid="column"]:has(.watch-display) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100% !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    z-index: 5;
  }
  div[data-testid="column"]:has(.watch-display) [data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  div[data-testid="column"]:has(.watch-display) [data-testid="stButton"] > button {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 0 !important;
  }
  div[data-testid="column"]:has(.watch-display) [data-testid="stButton"] > button:focus {
    outline: none !important;
    box-shadow: none !important;
  }
  div[data-testid="column"]:has(.watch-display) [data-testid="stButton"] p,
  div[data-testid="column"]:has(.watch-display) [data-testid="stButton"] [data-testid="stMarkdownContainer"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
  }
  .watch-detail-table {
    background: var(--w-canvas);
    border-radius: var(--w-radius-md);
    border: 1px solid color-mix(in srgb, var(--w-ink) 10%, transparent);
    padding: 0.5rem 0;
    margin-top: 0.5rem;
  }
  .watch-detail-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  .watch-detail-table th,
  .watch-detail-table td {
    padding: 0.55rem 0.65rem;
    border-bottom: 1px solid rgba(14,15,12,0.08);
    text-align: right;
  }
  .watch-detail-table th:first-child,
  .watch-detail-table td:first-child,
  .watch-detail-table th:nth-child(2),
  .watch-detail-table td:nth-child(2) {
    text-align: left;
  }
  .watch-detail-table th {
    font-weight: 700;
    color: var(--w-ink);
    background: rgba(255,255,255,0.6);
  }
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
    factories = {str(x) for x in df[FACTORY_COL].dropna().unique()}
    for f in sorted(factories, key=len, reverse=True):
        if f and f in question:
            return f
    m = re.search(r"\b(M\d{2})\b", question, re.I)
    if m:
        cand = m.group(1).upper()
        if cand in factories:
            return cand
    m = re.search(r"공장\s*([ABC])", question, re.I)
    if m:
        cand = f"공장{m.group(1).upper()}"
        if cand in factories:
            return cand
    m = re.search(r"([ABC])\s*공장", question, re.I)
    if m:
        cand = f"공장{m.group(1).upper()}"
        if cand in factories:
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


def _init_default_filters(df: pd.DataFrame) -> None:
    years = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
    if st.session_state.flt_year is None:
        st.session_state.flt_year = years[-1] if years else 2025
    if st.session_state.flt_factories is None:
        st.session_state.flt_factories = []
    if st.session_state.flt_customers is None:
        st.session_state.flt_customers = []
    if st.session_state.flt_items is None:
        st.session_state.flt_items = []


def render_global_filters(df: pd.DataFrame) -> tuple[int, list[str] | None, list[str] | None, list[str] | None]:
    """글로벌 필터 패널 — 조회 적용 시 AND 조건 반영."""
    _init_default_filters(df)
    years = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())
    inject_multiselect_autoclose_js()

    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("#### 글로벌 필터")

    year = st.selectbox(
        "YEAR(기준연도)",
        options=years,
        index=years.index(st.session_state.flt_year) if st.session_state.flt_year in years else len(years) - 1,
        format_func=lambda y: f"{int(y)}년",
        key="widget_year",
    )
    y_int = int(year)
    fac_options = filter_factory_options(df, y_int)
    fac_sel = st.multiselect(
        "FACTORY (공장)",
        options=fac_options,
        default=_sanitize_multiselect(st.session_state.flt_factories, fac_options),
        placeholder="전체 (M10, M20, M30)" if fac_options else "해당 연도 공장 없음",
        key="widget_factory",
    )
    fac_active = fac_sel or None

    cust_options = filter_customer_options(df, y_int, fac_active)
    cust_sel = st.multiselect(
        "CUSTOMER (고객)",
        options=cust_options,
        default=_sanitize_multiselect(st.session_state.flt_customers, cust_options),
        placeholder=f"전체 ({len(cust_options)}개 사)" if cust_options else "선택 가능한 고객 없음",
        key="widget_customer",
    )
    cust_active = cust_sel or None

    item_options = filter_item_options(df, y_int, fac_active, cust_active)

    def _item_label(code: str) -> str:
        s = str(code)
        return f"…{s[-8:]}" if len(s) > 10 else s

    item_sel = st.multiselect(
        "ITEM (품목)",
        options=item_options,
        default=_sanitize_multiselect(st.session_state.flt_items, item_options),
        placeholder=f"전체 ({len(item_options)}개 품목)" if item_options else "선택 가능한 품목 없음",
        format_func=_item_label,
        key="widget_item",
    )

    if st.button("조회 적용", type="primary", use_container_width=True, key="btn_apply_filters"):
        st.session_state.flt_year = y_int
        st.session_state.flt_factories = list(fac_sel)
        st.session_state.flt_customers = list(cust_sel)
        st.session_state.flt_items = list(item_sel)
        st.session_state.filters_applied = True
        st.session_state.watch_detail_open = False
        st.rerun()

    applied_year = int(st.session_state.flt_year)
    applied_fac = st.session_state.flt_factories or None
    applied_cust = st.session_state.flt_customers or None
    applied_item = st.session_state.flt_items or None

    y_start, y_end = trend_year_bounds(applied_year)
    st.markdown(
        f"<p class='year-range-hint'>표시 구간: <strong>{html_module.escape(trend_year_label(y_start, y_end))}</strong></p>",
        unsafe_allow_html=True,
    )

    alerts = detect_anomalies(apply_scope_filters(df, applied_fac, applied_cust, applied_item))
    if alerts:
        st.markdown(
            f"<div class='anomaly-box'><strong>AI 알림 (ANOMALY)</strong><br/>"
            f"{html_module.escape(alerts[0])}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    return applied_year, applied_fac, applied_cust, applied_item


def _toggle_watch_detail() -> None:
    st.session_state.watch_detail_open = not st.session_state.get("watch_detail_open", False)


def render_watch_toggle_card(count: int) -> None:
    """요주의 품목 KPI — 2행 녹색 카드, 카드 전체 클릭 시 하단 상세/차트 토글."""
    selected = bool(st.session_state.get("watch_detail_open", False))
    sel_cls = " selected" if selected else ""
    st.markdown(
        f'<div class="kpi-card green watch-display{sel_cls}">'
        f'<div class="kpi-label">요주의 품목 수</div>'
        f'<div class="kpi-value watch-alert">{count} 개</div>'
        f'<span class="kpi-delta neutral watch-hint">적자 또는 이익률 5% 미만</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.button(
        "\u200b",
        key="btn_watch_toggle",
        on_click=_toggle_watch_detail,
        help="클릭하여 상세 목록 표시/숨기기",
        use_container_width=True,
    )


def render_watch_detail_table(rows: pd.DataFrame) -> None:
    if rows.empty:
        st.info("요주의 품목이 없습니다.")
        return
    head = "".join(f"<th>{html_module.escape(c)}</th>" for c in rows.columns)
    body_parts = []
    for _, r in rows.iterrows():
        cells = "".join(f"<td>{html_module.escape(str(r[c]))}</td>" for c in rows.columns)
        body_parts.append(f"<tr>{cells}</tr>")
    html = (
        "<div class='watch-detail-table'>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, delta_html: str, *, variant: str = "blue", sub: str = "") -> None:
    cls = "kpi-card" if variant == "blue" else f"kpi-card {variant}"
    sub_html = f"<div class='kpi-sub'>{html_module.escape(sub)}</div>" if sub else ""
    st.markdown(
        f"<div class='{cls}'>"
        f"<div class='kpi-label'>{html_module.escape(label)}</div>"
        f"<div class='kpi-value'>{html_module.escape(value)}</div>"
        f"{delta_html}{sub_html}</div>",
        unsafe_allow_html=True,
    )


def executive_dashboard(
    df: pd.DataFrame,
    year: int,
    factories: list[str] | None,
    customers: list[str] | None,
    items: list[str] | None,
) -> None:
    scoped = apply_scope_filters(df, factories, customers, items)
    curr = aggregate_year_metrics(scoped, year)
    prev = aggregate_year_metrics(scoped, year - 1)

    sales_yoy = yoy_pct(curr["매출액"], prev["매출액"])
    contrib_yoy = yoy_pct(curr["공헌이익"], prev["공헌이익"])
    margin_pp = curr["공헌이익률"] - prev["공헌이익률"] if prev["매출액"] else None
    op_yoy = yoy_pct(curr["영업이익"], prev["영업이익"])
    op_margin_pp = curr["영업이익률"] - prev["영업이익률"] if prev["매출액"] else None
    watch_n = count_watch_items(scoped, year)

    if op_margin_pp is not None:
        op_delta = (
            f"<span class='kpi-delta {'down' if op_margin_pp < 0 else 'up'}'>"
            f"{'▼' if op_margin_pp < 0 else '▲'} {abs(op_margin_pp):.1f}%p YoY</span>"
        )
    else:
        op_delta = kpi_delta_html(None)
    if margin_pp is not None:
        margin_delta = (
            f"<span class='kpi-delta {'down' if margin_pp < 0 else 'up'}'>"
            f"{'▼' if margin_pp < 0 else '▲'} {abs(margin_pp):.1f}%p YoY</span>"
        )
    else:
        margin_delta = kpi_delta_html(None)

    st.markdown('<div class="kpi-grid-row">', unsafe_allow_html=True)
    r1a, r1b, r1c = st.columns(3)
    with r1a:
        render_kpi_card("총 매출액 (억)", fmt_eok(curr["매출액"]), kpi_delta_html(sales_yoy))
    with r1b:
        render_kpi_card("총 영업이익 (억)", fmt_eok(curr["영업이익"]), kpi_delta_html(op_yoy))
    with r1c:
        render_kpi_card("영업이익률", f"{curr['영업이익률']:.1f}%", op_delta)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="kpi-grid-row">', unsafe_allow_html=True)
    r2a, r2b, r2c = st.columns(3)
    with r2a:
        render_watch_toggle_card(watch_n)
    with r2b:
        render_kpi_card("총 공헌이익 (억)", fmt_eok(curr["공헌이익"]), kpi_delta_html(contrib_yoy), variant="green")
    with r2c:
        render_kpi_card("공헌이익률", f"{curr['공헌이익률']:.1f}%", margin_delta, variant="green")
    st.markdown("</div>", unsafe_allow_html=True)

    y_start, y_end = trend_year_bounds(year)
    trend = (
        scoped.groupby(YEAR_COL, as_index=False)
        .agg(매출액=("매출액", "sum"), 공헌이익=("공헌이익", "sum"))
        .sort_values(YEAR_COL)
    )
    trend = trend[(trend[YEAR_COL] >= y_start) & (trend[YEAR_COL] <= y_end)]

    scope_txt = "전사"
    if factories:
        scope_txt = ", ".join(factories)
    elif customers:
        scope_txt = "선택 고객"
    elif items:
        scope_txt = "선택 품목"

    chart_title = f"연도별 매출 및 공헌이익률 추이 ({scope_txt}, {trend_year_label(y_start, y_end)})"

    if st.session_state.get("watch_detail_open", False):
        st.markdown(f"**요주의 품목 상세** · {year}년 · 적자 또는 영업이익율 5% 미만")
        render_watch_detail_table(watch_item_detail_rows(scoped, year))
    else:
        st.plotly_chart(
            revenue_contribution_figure(trend, chart_title),
            use_container_width=True,
            key="pl_exec_trend",
        )


def analyst_dashboard(
    df: pd.DataFrame,
    year: int,
    factories: list[str] | None,
    customers: list[str] | None,
    items: list[str] | None,
) -> None:
    scoped = apply_global_filters(df, year, factories, customers, items)
    st.caption(f"{year}년 · 필터 동시 적용 결과 {len(scoped):,}건")

    summary = (
        scoped.groupby([FACTORY_COL, CUSTOMER_COL, ITEM_COL], as_index=False)
        .agg(
            매출수량=("매출수량", "sum"),
            매출액=("매출액", "sum"),
            공헌이익=("공헌이익", "sum"),
            영업이익=("영업이익", "sum"),
        )
        .sort_values("매출액", ascending=False)
    )
    summary["공헌이익률(%)"] = (
        (summary["공헌이익"] / summary["매출액"].replace(0, pd.NA) * 100).round(2)
    )
    disp = summary.copy()
    for c in ("매출액", "공헌이익", "영업이익"):
        disp[c] = (disp[c] / EOK).round(2)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("**원가 세부 (억 원, 상위 50)**")
    cost_y = scoped.groupby([FACTORY_COL, ITEM_COL], as_index=False)[COST_DETAIL_COLS].sum(numeric_only=True)
    sales_g = scoped.groupby([FACTORY_COL, ITEM_COL], as_index=False)["매출액"].sum()
    cost_y = cost_y.merge(sales_g, on=[FACTORY_COL, ITEM_COL], how="left").sort_values("매출액", ascending=False)
    st.dataframe(df_to_eok_display(cost_y.head(50), COST_DETAIL_COLS), use_container_width=True, hide_index=True)


def render_top_nav() -> None:
    st.markdown('<div class="app-title-top-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)
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


def year_label_short(y: int) -> str:
    """연도 표기용 뒤 두 자리 (예: 2025 → '25)."""
    return f"'{y % 100:02d}"


def customer_sales_sparkline_svg(
    sales_eok: list[float],
    profits_won: list[float],
    five_years: list[int],
    y_last: int,
    w: int = 132,
    h: int = 36,
) -> str:
    """5개년 매출 막대(행 내 상대 높이). 흑자: 하늘색·최종연 짙은 푸른색 / 적자: 옅은 빨강·최종연 빨간색."""
    if len(sales_eok) != len(five_years) or not five_years:
        return ""
    mx = max(sales_eok) if max(sales_eok) > 0 else 1.0
    n = len(five_years)
    gap = 2.0
    pad = 3.0
    band = (w - 2 * pad) / n
    bw = max(2.0, band - gap)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="연도별 매출">'
    ]
    for i, y in enumerate(five_years):
        sv = float(sales_eok[i])
        raw_p = profits_won[i]
        try:
            pv = float(raw_p)
        except (TypeError, ValueError):
            pv = 0.0
        if pd.isna(raw_p):
            pv = 0.0
        profit = pv >= 0.0
        is_last = y == y_last
        if profit:
            fill = "#1e40af" if is_last else "#7dd3fc"
        else:
            fill = "#dc2626" if is_last else "#fca5a5"
        h_bar = max(3.0, (sv / mx) * (h - 2 * pad))
        x0 = pad + i * band + gap / 2
        y0 = h - pad - h_bar
        parts.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bw:.2f}" height="{h_bar:.2f}" '
            f'rx="3" ry="3" fill="{fill}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def customer_compact_table_html(rows: list[dict], y_start: int, y_last: int) -> str:
    """고객 목록 5열 HTML (스파크라인 SVG 포함)."""
    thead = (
        "<thead><tr>"
        "<th>고객사</th>"
        f"<th>5년 매출<br/><span class='sub'>({y_start % 100:02d}년→{y_last % 100:02d}년 매출, 억)</span></th>"
        f"<th>연도별 매출추이<br/><span class='sub'>({year_label_short(y_start)}-{year_label_short(y_last)})</span></th>"
        f"<th>{year_label_short(y_last)} 영업이익<br/><span class='sub'>(억)</span></th>"
        f"<th>{year_label_short(y_last)} OPM<br/><span class='sub'>(%)</span></th>"
        "</tr></thead>"
    )
    body_parts: list[str] = []
    for r in rows:
        body_parts.append(
            "<tr>"
            f"<td class='name'>{html_module.escape(str(r['고객사']))}</td>"
            f"<td class='num'>{html_module.escape(r['매출_텍스트'])}</td>"
            f"<td class='spark'>{r['svg']}</td>"
            f"<td class='num'>{html_module.escape(r['이익_텍스트'])}</td>"
            f"<td class='num'>{html_module.escape(r['opm_텍스트'])}</td>"
            "</tr>"
        )
    style = (
        "<style>"
        ".cust-t{width:100%;border-collapse:collapse;font-size:14px;margin:0.25rem 0 0.75rem;}"
        ".cust-t th,.cust-t td{border-bottom:1px solid rgba(14,15,12,0.1);padding:8px 10px;vertical-align:middle;}"
        ".cust-t th{text-align:right;font-weight:600;color:#0e0f0c;background:rgba(255,255,255,0.65);}"
        ".cust-t th:first-child,.cust-t td:first-child{text-align:left;}"
        ".cust-t .sub{font-weight:400;font-size:11px;color:#454745;}"
        ".cust-t .name{font-weight:600;}"
        ".cust-t .num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;}"
        ".cust-t .spark{text-align:center;width:150px;}"
        "</style>"
    )
    return f"{style}<table class='cust-t'>{thead}<tbody>{''.join(body_parts)}</tbody></table>"


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

    y_start = y_last - 4
    five_years = list(range(y_start, y_last + 1))

    display_rows: list[dict] = []
    for _, crow in list_df.iterrows():
        sales_eok: list[float] = []
        profits_won: list[float] = []
        for y in five_years:
            sc = f"{y}_매출액"
            pc = f"{y}_영업이익"
            raw_s = crow[sc] if sc in crow.index else 0
            raw_p = crow[pc] if pc in crow.index else 0
            sv = pd.to_numeric(raw_s, errors="coerce")
            pv = pd.to_numeric(raw_p, errors="coerce")
            sales_eok.append(float(sv) / EOK if pd.notna(sv) else 0.0)
            profits_won.append(float(pv) if pd.notna(pv) else 0.0)
        mcol = f"{y_last}_영업이익율"
        raw_m = crow[mcol] if mcol in crow.index else pd.NA
        if pd.isna(raw_m):
            opm_txt = "—"
        else:
            opm_txt = f"{float(raw_m) * 100:.2f}%"
        last_p = crow[f"{y_last}_영업이익"] if f"{y_last}_영업이익" in crow.index else 0
        last_p_eok = float(pd.to_numeric(last_p, errors="coerce") or 0) / EOK
        display_rows.append(
            {
                "고객사": crow[CUSTOMER_COL],
                "매출_텍스트": f"{sales_eok[0]:.2f} -> {sales_eok[-1]:.2f}",
                "svg": customer_sales_sparkline_svg(sales_eok, profits_won, five_years, y_last),
                "이익_텍스트": f"{last_p_eok:.2f}",
                "opm_텍스트": opm_txt,
            }
        )

    st.markdown(f"**고객사 목록 ({y_last}년 매출액 기준 내림차순, {years[0]}~{y_last}년)**")
    st.caption(
        f"「5년 매출」: {y_start % 100:02d}년 매출액(억) → {y_last % 100:02d}년 매출액(억). "
        "「연도별 매출추이」: 막대 높이는 해당 고객의 5개년 매출 비율입니다. "
        "영업이익 **흑자** 연도는 하늘색(맨 오른쪽·최신 연도만 짙은 푸른색), **적자** 연도는 옅은 빨강(최신 연도만 빨간색)입니다."
    )
    st.markdown(customer_compact_table_html(display_rows, y_start, y_last), unsafe_allow_html=True)

    cust_options = list_df[CUSTOMER_COL].astype(str).tolist()
    if not cust_options:
        st.warning("매출처 데이터가 없습니다.")
        return

    sel = st.selectbox(
        "연도별·품목별 상세를 표시할 매출처",
        options=cust_options,
        index=0,
        key="customer_detail_pick",
    )

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
    st.set_page_config(
        page_title="AI 기반 원가데이터 분석",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    _split_h = min(920, max(520, int(st.session_state.get("_split_px", 820))))
    st.markdown(wise_streamlit_css(), unsafe_allow_html=True)

    if not DATA_PATH.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        st.stop()

    df = with_contribution(cached_data())

    vb, vm, _ = get_vlm_config()
    if vb and vm and not openai_available():
        st.warning(
            f"VLM이 설정되어 있으나 **`openai`** 가 이 Python에 설치되어 있지 않습니다. "
            f"터미널에서 실행: `{sys.executable} -m pip install openai` 후 앱을 다시 실행하세요."
        )

    st.markdown('<div class="app-title-top-spacer" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.title("AI 기반 원가데이터 분석 모니터링")

    col_filter, col_main, col_chat = st.columns([2.2, 4.8, 3], gap="medium")

    with col_filter:
        st.markdown('<div class="filter-col-wrap">', unsafe_allow_html=True)
        with st.container(height=_split_h, border=False):
            year, factories, customers, items = render_global_filters(df)
        st.markdown("</div>", unsafe_allow_html=True)

    filtered_df = apply_global_filters(df, year, factories, customers, items)
    trend_df = apply_scope_filters(df, factories, customers, items)

    with col_main:
        with st.container(height=_split_h, border=False):
            tab_exec, tab_analyst = st.tabs(
                ["경영진 대시보드형 (Top-Down)", "데이터 분석가형 (Bottom-Up)"]
            )
            with tab_exec:
                executive_dashboard(trend_df, year, factories, customers, items)
            with tab_analyst:
                analyst_dashboard(df, year, factories, customers, items)

    with col_chat:
        with st.container(height=_split_h, border=False):
            chat_panel(filtered_df)


if __name__ == "__main__":
    main()
