#!/usr/bin/env python3
# =====================================================================
# QQQ Ballast — 일일 신호 생성 + 텔레그램 발송
#
# 전략(백테스트 확정안, C안)
#   - 코어: QQQ 트랙2 (D=180, N1=10%에 25% 매도 -> N2=15%에 잔량 매도, DI/ADX 14, cd7, Y=ADX<9)
#   - 위성: QQQ가 비운 현금분을 TLT / GLD / XLE에 "신호 ON인 자산에만" 자산당 현금의 1/3씩 배분
#           (OFF 자산 몫은 현금으로 남김 = 1/3 고정)
#   - 신호는 미국장 종가로 확정, 체결은 다음 거래일 시가 가정
#
# 실행 예)
#   python ballast_signal.py --source csv --csv-dir . --dry-run     # 로컬 CSV로 테스트(발송 안 함)
#   python ballast_signal.py --dry-run                              # 야후 데이터로 테스트(발송 안 함)
#   python ballast_signal.py                                        # 실제 발송 (환경변수 필요)
#   python ballast_signal.py --check --source csv --csv-dir .       # 백테스트 재현 검증
#
# 환경변수: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
#
# [2026-11-23 검증 메모] (상세: QQQ_Ballast_종합요약.md 6~8절)
#   - 전략 로직은 변경하지 않음. 이 파일은 주석만 보강했고 계산 로직은 동일하다.
#   - 데이터는 배당 포함(조정 종가) 기준이다. 야후 auto_adjust=True / chart API의 adjclose 비율 조정이 이에 해당.
#     배당 제외 가격으로 백테스트하면 TLT 기여가 과소평가된다(TLT 제외 시 Calmar가 더 좋아 보이는 착시).
#   - QQQ 재진입은 "+DI가 -DI보다 위에 있는 상태"(교차 아님) + 청산 후 7거래일 경과(최소 대기일).
#   - GLD 신호는 파라미터 이웃 대비 뾰족한 봉우리(144조합 중 상위 8%). 체결 지연·비용에 취약하므로 주의.
#
# [2026-11-25] 메시지 하단을 "근거(세부 지표)" 블록으로 교체(build_extra_info). 신호 로직은 그대로이고
#   DI/ADX 값 노출용 di_adx_full을 추가했다(di_adx는 이를 호출하는 래퍼, 반환값 동일).
# =====================================================================
import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------- 전략 파라미터 (백테스트 확정값) ----------------------
# QQQ: cooldown=7은 "청산 후 최소 7거래일 재진입 금지"(강제). 5~10일이 안정 구간이고 3일 이하/15일 이상은 MDD 악화.
#      "종가>10일선" 재진입 확인은 QQQ 단독에선 MDD -20.3%→-18.4%였으나 위성 포함 전체 포트폴리오에선
#      개선 없음(Calmar 0.84→0.83) → 채택하지 않음.
QQQ_PARAMS = dict(D=180, N1=10, N2=15, frac1=0.25, di_period=14, cooldown=7, y_adx_thr=9)
# TLT: 20/110 근처(빠른선 10~50, 느린선 60~200)는 Calmar가 평탄해 튜닝 실익 없음. 배당 포함 기준에선 위성으로서 기여가 있어 유지.
TLT_PARAMS = dict(fast=20, slow=110)
# GLD: 현행 유지 결정. 단, 144조합 중 상위 8%의 뾰족한 봉우리(DI 4/6/8일로 바꾸면 Calmar 0.63~0.68로 하락),
#      연 31회 전환이라 체결 지연·비용에 취약. 대안 H1(150일선 월판정 + 고점대비 -4% 손절 + DI 재진입 + 쿨다운2)은
#      성과가 약간 낮지만(Calmar 0.81 vs 0.85) 1일 지연 MDD가 -26.5%(현행 -30.9%)로 더 견고.
GLD_PARAMS = dict(N=0.015, di_period=5, cooldown=2, y_adx_thr=15)
# XLE: 쿨다운 15 이상(15~25)이 안정 구간(5~10일은 Calmar 하락). 현행 유지.
XLE_PARAMS = dict(N=15, di_period=14, cooldown=15, y_adx_thr=15, roll_max_window=250)
N_SAT = 3                      # C안: 위성 자산당 현금의 1/3 고정
VOL_WINDOW = 20                # 변동성 참고 표시용(전략에는 미반영)
VOL_ALERT = 0.45

# 신호 warm-up 기준 시작일 (백테스트와 동일한 이력으로 계산해야 상태가 일치)
HIST_START = {"QQQ": "1999-03-10", "TLT": "2005-02-25", "GLD": "2005-02-25", "XLE": "2005-02-25"}
ASSETS = ["QQQ", "TLT", "GLD", "XLE"]
SATS = ["TLT", "GLD", "XLE"]

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


# ---------------------- 데이터 로드 ----------------------
def load_from_csv(csv_dir: str):
    out = {}
    for a in ASSETS:
        df = pd.read_csv(os.path.join(csv_dir, f"{a.lower()}_us_d.csv"), parse_dates=["Date"])
        df.columns = [c.strip() for c in df.columns]
        out[a] = df[["Date", "Open", "High", "Low", "Close"]].sort_values("Date").reset_index(drop=True)
    dates = {a: out[a]["Date"].iloc[-1].date() for a in ASSETS}
    return out, dates


def _trim_incomplete_bar(df: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    """장중에 받아온 오늘자 미완성 봉 제거 (ET 기준 오늘 날짜 + 아직 장중이면 마지막 행 삭제)."""
    if len(df) and df["Date"].iloc[-1].date() == now_et.date() and not session_closed_today(now_et):
        return df.iloc[:-1].reset_index(drop=True)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df[["Date", "Open", "High", "Low", "Close"]].dropna().sort_values("Date").reset_index(drop=True)


def _y_download(ticker, end):
    import yfinance as yf
    return _clean(yf.download(ticker, start=HIST_START[ticker], end=end, auto_adjust=True, progress=False))


def _y_history(ticker, end):
    import yfinance as yf
    return _clean(yf.Ticker(ticker).history(start=HIST_START[ticker], end=end, auto_adjust=True))


def _y_chart_api(ticker, end):
    """야후 chart API 직접 호출 (라이브러리 우회). adjclose 비율로 OHLC 조정."""
    import requests
    p1 = int(pd.Timestamp(HIST_START[ticker]).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())
    last = None
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}",
                             params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"]["adjclose"][0]["adjclose"]
            off = res["meta"].get("gmtoffset", -14400)
            d = pd.DataFrame({"Date": pd.to_datetime(res["timestamp"], unit="s") + pd.to_timedelta(off, unit="s"),
                              "Open": q["open"], "High": q["high"], "Low": q["low"],
                              "Close": q["close"], "Adj": adj})
            f = d["Adj"] / d["Close"]
            for c in ("Open", "High", "Low", "Close"):
                d[c] = d[c] * f
            return _clean(d.drop(columns="Adj").set_index("Date"))
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"chart API 실패: {last}")


def expected_session(now_et: datetime):
    """달력 기준 '이미 마감·확정됐어야 할' 마지막 평일 세션 (공휴일은 모름 -> 경고에만 사용)."""
    d = now_et.date()
    if (now_et.hour, now_et.minute) < (16, 30):
        d = d - pd.Timedelta(days=1)
    d = pd.Timestamp(d)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d.date()


def fetch_best(ticker: str, now_et: datetime):
    """야후의 3가지 경로(download / Ticker.history / chart API)를 모두 시도해 가장 최신 날짜를 채택.
    end는 '내일'로 명시(야후 end는 배타적이라 오늘 봉이 빠지는 문제 방지)."""
    end = (pd.Timestamp(now_et.date()) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    cands = []
    for name, fn in (("chart", _y_chart_api), ("download", _y_download), ("history", _y_history)):
        try:
            df = _trim_incomplete_bar(fn(ticker, end), now_et)
            if df.empty:
                raise RuntimeError("빈 데이터")
            cands.append((name, df))
            print(f"[{ticker}] {name}: 마지막 {df['Date'].iloc[-1].date()} 종가 {df['Close'].iloc[-1]:.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"[{ticker}] {name} 실패: {e}", file=sys.stderr)
    if not cands:
        raise RuntimeError(f"{ticker} 데이터를 가져오지 못했습니다")
    # 최신 날짜 우선, 동률이면 download > history > chart 순서 대신 목록 앞쪽(chart) 우선하지 않고 download 우선
    pri = {"download": 0, "history": 1, "chart": 2}
    cands.sort(key=lambda c: (-c[1]["Date"].iloc[-1].value, pri[c[0]]))
    return cands[0][1], cands[0][0]


def load_from_yahoo(max_rounds: int = 4):
    """4개 자산 조회. 기대 세션(달력 기준)보다 뒤처졌거나 자산 간 날짜가 다르면 2분 간격 재조회.
    끝까지 안 맞으면 경고와 함께 진행(침묵 발송 금지). 휴장일이면 경고가 오탐일 수 있음."""
    out, src_of, dates = {}, {}, {}

    def fetch_round():
        now_et = datetime.now(ET)
        for a in ASSETS:
            out[a], src_of[a] = fetch_best(a, now_et)
            dates[a] = out[a]["Date"].iloc[-1].date()
        return expected_session(now_et)

    exp = fetch_round()
    for rnd in range(max_rounds - 1):
        if len(set(dates.values())) <= 1 and min(dates.values()) >= exp:
            break
        print(f"재조회 {rnd + 1}: 기대 {exp}, " + ", ".join(f"{a}={dates[a]}" for a in ASSETS))
        time.sleep(120)
        exp = fetch_round()
    print("자산별 최신 데이터: " + ", ".join(f"{a}={dates[a]}({src_of[a]})" for a in ASSETS) + f" / 기대 {exp}")
    global EXPECTED
    EXPECTED = exp
    return out, dates


EXPECTED = None


# ---------------------- 공통 지표: DI/ADX (EWM) ----------------------
def di_adx_full(close: pd.Series, high: pd.Series, low: pd.Series, period: int):
    """+DI, -DI, ADX 값 전체(메시지 근거 표시용). 신호 계산(di_adx)과 같은 식을 쓴다."""
    up = high.diff()
    down = -low.diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = (100 * pd.Series(pdm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr).values
    mdi = (100 * pd.Series(mdm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr).values
    with np.errstate(invalid="ignore"):
        dx = (np.abs(pdi - mdi) / (pdi + mdi)) * 100
    adx = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().values
    return pdi, mdi, adx


def di_adx(close: pd.Series, high: pd.Series, low: pd.Series, period: int):
    pdi, mdi, adx = di_adx_full(close, high, low, period)
    di_up = np.nan_to_num(pdi > mdi, nan=0.0).astype(bool)
    return di_up, adx


# ---------------------- QQQ 트랙2 (비중, 추적 고점, 상태) ----------------------
def qqq_position(df, D, N1, N2, frac1, di_period, cooldown, y_adx_thr):
    """QQQ 코어 비중 계산.
    - 청산: 추적 고점 대비 -N1%면 frac1만 매도, -N2%면 전량 매도, ADX<y_adx_thr면 전량 매도(오신호 청산).
    - 재진입: 청산 후 cooldown 거래일이 지났고(최소 대기일, 강제) 그날 +DI > -DI(교차가 아니라 '+DI 우위 상태')이면 전량 재진입.
      (교차 순간만 허용하면 CAGR 11.1%/MDD -24.1%로 악화되어 상태 기준을 유지한다.)
    - 재진입 시 추적 고점은 180일 최고가(roll_max_D)로 리셋된다. 재진입 시점이 보통 이 고점 대비 -15% 아래라
      다음 날 바로 청산되는 '1~2일 탐색 매수'가 87회 중 67회 나타나는데, 이를 막으면(고점을 재진입일 종가로 리셋 등)
      오히려 MDD가 악화되므로 의도된 동작으로 둔다.
    - 체결 가정: 신호는 당일 종가 확정, 체결은 다음 거래일 시가. 체결이 1일 더 늦으면 MDD가 -20%대 중반으로 악화되는 취약점이 있다.
    """
    close, high, low = df["Close"], df["High"], df["Low"]
    di_up, adx = di_adx(close, high, low, di_period)
    c = close.values
    n = len(c)
    roll_max_D = close.rolling(D, min_periods=1).max().values
    pos = np.zeros(n)
    peak_arr = np.full(n, np.nan)
    state_arr = np.zeros(n, dtype=int)
    pos[0] = 1.0
    peak = roll_max_D[0]
    state = 2  # 2=전량, 1=frac1 매도 후, 0=현금
    days_since_exit = 9999
    peak_arr[0], state_arr[0] = peak, state
    for t in range(1, n):
        if state > 0:
            peak = max(peak, c[t])
            dd = c[t] / peak - 1
            y_trig = (not np.isnan(adx[t])) and adx[t] < y_adx_thr
            if state == 2 and dd <= -N1 / 100.0:
                state = 1
            if dd <= -N2 / 100.0:
                state = 0
                days_since_exit = 0
            elif y_trig:
                state = 0
                days_since_exit = 0
        else:
            days_since_exit += 1
            # 재진입: +DI 우위 '상태' + 쿨다운(최소 대기 거래일) 경과 (교차 조건 아님)
            if di_up[t] and days_since_exit >= cooldown:
                state = 2
                peak = roll_max_D[t]
        pos[t] = {2: 1.0, 1: (1 - frac1), 0: 0.0}[state]
        peak_arr[t], state_arr[t] = peak, state
    return pos, peak_arr, state_arr


# ---------------------- TLT (SMA 골든/데드크로스) ----------------------
def tlt_signal(df, fast, slow):
    sma_f = df["Close"].rolling(fast).mean()
    sma_s = df["Close"].rolling(slow).mean()
    sig = (sma_f > sma_s).fillna(False).values.astype(float)
    return sig, sma_f.values, sma_s.values


# ---------------------- GLD (확장형 고점 N% + DI/ADX) ----------------------
def gld_signal(df, N, di_period, cooldown, y_adx_thr):
    di_up, adx = di_adx(df["Close"], df["High"], df["Low"], di_period)
    c = df["Close"].values
    n = len(c)
    sig = np.zeros(n)
    peak_arr = np.full(n, np.nan)
    state, peak, cd = 0, -np.inf, 0
    for t in range(n):
        if state == 1:
            peak = max(peak, c[t])
            dd = c[t] / peak - 1
            y_trig = (not np.isnan(adx[t])) and adx[t] < y_adx_thr
            if y_trig or dd <= -N:
                state, cd, peak = 0, cooldown, -np.inf
        else:
            if cd > 0:
                cd -= 1
            elif di_up[t]:
                state, peak = 1, c[t]
        sig[t] = state
        peak_arr[t] = peak if state == 1 else np.nan
    return sig, peak_arr


# ---------------------- XLE (250일 고점 N% + DI/ADX) ----------------------
def xle_signal(df, N, di_period, cooldown, y_adx_thr, roll_max_window):
    di_up, adx = di_adx(df["Close"], df["High"], df["Low"], di_period)
    c = df["Close"].values
    n = len(c)
    roll_max = df["Close"].rolling(roll_max_window, min_periods=1).max().values
    sig = np.zeros(n)
    peak_arr = np.full(n, np.nan)
    peak = c[0]
    state = 0
    days_since_exit = 9999
    for t in range(1, n):
        if state == 1:
            peak = max(peak, c[t])
            dd = c[t] / peak - 1
            y_trig = (not np.isnan(adx[t])) and adx[t] < y_adx_thr
            if dd <= -N / 100.0 or y_trig:
                state, days_since_exit = 0, 0
        else:
            days_since_exit += 1
            if di_up[t] and days_since_exit >= cooldown:
                state = 1
                peak = roll_max[t]
        sig[t] = state
        peak_arr[t] = peak if state == 1 else np.nan
    return sig, peak_arr


# ---------------------- 신호 계산 + 공통 거래일 병합 ----------------------
def build_frame(data: dict) -> pd.DataFrame:
    """각 자산 신호를 '자기 전체 이력'으로 계산한 뒤 공통 거래일로 병합."""
    q = data["QQQ"].copy()
    q["pos"], q["peak"], q["state"] = qqq_position(q, **QQQ_PARAMS)
    q["pdi"], q["mdi"], q["adx"] = di_adx_full(q["Close"], q["High"], q["Low"], QQQ_PARAMS["di_period"])
    t = data["TLT"].copy()
    t["sig"], t["smaf"], t["smas"] = tlt_signal(t, **TLT_PARAMS)
    g = data["GLD"].copy()
    g["sig"], g["peak"] = gld_signal(g, **GLD_PARAMS)
    g["pdi"], g["mdi"], g["adx"] = di_adx_full(g["Close"], g["High"], g["Low"], GLD_PARAMS["di_period"])
    x = data["XLE"].copy()
    x["sig"], x["peak"] = xle_signal(x, **XLE_PARAMS)
    x["pdi"], x["mdi"], x["adx"] = di_adx_full(x["Close"], x["High"], x["Low"], XLE_PARAMS["di_period"])

    last_dates = {a: data[a]["Date"].iloc[-1] for a in ASSETS}  # 자산별 마지막 종가 날짜(메시지 표시용)

    def pick(df, cols, tag):
        return df[["Date"] + cols].rename(columns={c: f"{c}_{tag}" for c in cols})

    m = pick(q, ["Open", "Close", "pos", "peak", "state", "pdi", "mdi", "adx"], "QQQ")
    m = m.merge(pick(t, ["Open", "Close", "sig", "smaf", "smas"], "TLT"), on="Date")
    m = m.merge(pick(g, ["Open", "Close", "sig", "peak", "pdi", "mdi", "adx"], "GLD"), on="Date")
    m = m.merge(pick(x, ["Open", "Close", "sig", "peak", "pdi", "mdi", "adx"], "XLE"), on="Date")
    m = m.sort_values("Date").reset_index(drop=True)
    m.attrs["last_dates"] = last_dates
    return m


def target_weights(m: pd.DataFrame) -> dict:
    """당일 종가 기준 목표 비중(계좌 전체 대비). C안: 위성은 현금의 1/3 고정, OFF 몫은 현금."""
    cash = 1 - m["pos_QQQ"].values
    w = {"QQQ": m["pos_QQQ"].values}
    for a in SATS:
        w[a] = cash * m[f"sig_{a}"].values / N_SAT
    return w


# ---------------------- 메시지 구성 ----------------------
def fmt_w(x: float) -> str:
    s = f"{x * 100:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return f"{s}%"


def fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def build_weight_block(m, w, mode="full"):
    """최종 비중 4줄. mode='hold'면 변동 라벨 없이 '유지'만 표시(월요일/휴장 안내용)."""
    lines = []
    i = len(m) - 1
    for a in ASSETS:
        cur, prv = w[a][i], w[a][i - 1]
        if a == "QQQ":
            if mode == "hold" or abs(cur - prv) < 1e-9:
                icon, tag = ("🟠", "(유지)") if cur > 0 else ("⚪", "(전량 매도 상태)")
            elif cur < prv:
                icon = "🔴"
                if cur == 0:
                    dd = m["Close_QQQ"].iloc[i] / m["peak_QQQ"].iloc[i] - 1
                    reason = "손절 -%d%%" % QQQ_PARAMS["N2"] if dd <= -QQQ_PARAMS["N2"] / 100 else "ADX 소멸"
                    tag = f"(전량 매도 · {reason})"
                else:
                    tag = "(분할 매도 · -%d%% 손절, %d%% 매도)" % (QQQ_PARAMS["N1"], QQQ_PARAMS["frac1"] * 100)
            else:
                icon, tag = "🟢", ("(재진입)" if prv == 0 else "(비중 복원)")
            lines.append(f"{icon} QQQ  {fmt_w(cur)}  {tag}")
        else:
            on, pon = int(m[f"sig_{a}"].iloc[i]), int(m[f"sig_{a}"].iloc[i - 1])
            if mode == "hold":
                icon, tag = (("🟠", "(유지)") if cur > 0 else ("⚪", "(OFF)" if on == 0 else "(ON · 현금 없음)"))
            elif on != pon:
                if on == 1:
                    icon, tag = "🟢", "(신규 진입)" if cur > 0 else "(신호 ON · 현금 없음)"
                else:
                    icon, tag = "🔴", "(청산)"
            elif on == 1 and abs(cur - prv) > 1e-9:
                icon, tag = "🟠", "(비중 조정)"
            elif on == 1:
                icon, tag = ("🟠", "(유지)") if cur > 0 else ("⚪", "(ON · 현금 없음)")
            else:
                icon, tag = "⚪", "(OFF)"
            lines.append(f"{icon} {a}  {fmt_w(cur)}  {tag}")
    return "\n".join(lines)


def build_trades(m, w):
    """오늘 실행할 매매 (계좌 전체 대비 %p). 매도 먼저, 매수 나중."""
    i = len(m) - 1
    sells, buys = [], []
    for a in ASSETS:
        cur, prv = w[a][i], w[a][i - 1]
        d = cur - prv
        if abs(d) < 1e-9:
            continue
        if d < 0:
            verb = "전량 매도" if cur < 1e-9 else "일부 매도"
            sells.append((a, f"{a} {verb} {d * 100:.1f}%p ({fmt_w(prv)} → {fmt_w(cur)})"))
        else:
            verb = "신규 매수" if prv < 1e-9 else "추가 매수"
            buys.append((a, f"{a} {verb} +{d * 100:.1f}%p ({fmt_w(prv)} → {fmt_w(cur)})"))
    items = sells + buys
    return [t for _, t in items], [(a, ("매도" if (a, t) in sells else "매수")) for a, t in items]


# 임박 표시 기준 (표시 전용, 전략 로직과 무관)
IMM_STOP_FRAC = 0.2   # 손절선까지 남은 거리가 손절폭의 20% 이내
IMM_ADX = 2.0         # ADX가 청산선(Y)에서 2.0 이내
IMM_TLT = 0.3         # TLT SMA 격차 절대값 0.3% 이내
IMM_DI = 3.0          # 재진입 DI 격차 3.0 이내
WARN = "⚠️ 임박"


def _zero_days(sig) -> int:
    """끝에서부터 연속 0(청산) 일수. 청산한 날을 1로 센다."""
    k = 0
    for v in sig[::-1]:
        if v == 0:
            k += 1
        else:
            break
    return k


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _d(a: float, b: float, nd: int = 1) -> str:
    """전일 값 표기: (전일 b)"""
    return f"(전일 {b:.{nd}f})"


def _reentry_lines(m, a, i, sig, need, cd_name):
    """청산 상태 자산의 재진입 조건 체크리스트. need = 재진입에 필요한 경과 거래일."""
    k = _zero_days(sig[: i + 1]) - 1          # 청산일 = 0일째
    cd_ok = k >= need
    p1, m1 = m[f"pdi_{a}"].iloc[i], m[f"mdi_{a}"].iloc[i]
    p0, m0 = m[f"pdi_{a}"].iloc[i - 1], m[f"mdi_{a}"].iloc[i - 1]
    di_ok = p1 > m1
    adx1, adx0 = m[f"adx_{a}"].iloc[i], m[f"adx_{a}"].iloc[i - 1]
    lines = [
        f" 재진입 ① 쿨다운: 경과 {k}일 / 필요 {need}일 {'✔' if cd_ok else '✘'}",
        f" 재진입 ② +DI>-DI: {p1:.1f} vs {m1:.1f} {'✔' if di_ok else '✘'} (전일 {p0:.1f} vs {m0:.1f})",
        f" ADX {adx1:.1f} {_d(adx1, adx0)}",
    ]
    near_cd = (not cd_ok) and (need - k) <= 1
    near_di = (not di_ok) and (m1 - p1) <= IMM_DI
    if (cd_ok and near_di) or (di_ok and near_cd) or (near_cd and near_di):
        lines[1] += f"  {WARN}"
    return lines


def _hold_lines(m, a, i, stop_pct, y_thr, state_label=None):
    """보유 상태 자산: 고점 대비, 손절선까지 거리, ADX 여유."""
    c, pk = m[f"Close_{a}"].iloc[i], m[f"peak_{a}"].iloc[i]
    cp, pkp = m[f"Close_{a}"].iloc[i - 1], m[f"peak_{a}"].iloc[i - 1]
    dd = c / pk - 1
    dd_prev = (cp / pkp - 1) if not np.isnan(pkp) else None
    gap = stop_pct + dd * 100
    lines = [f" 추적고점 대비 {_pct(dd)}" + (f" (전일 {_pct(dd_prev)})" if dd_prev is not None else "")]
    near = gap <= stop_pct * IMM_STOP_FRAC
    tag = f"  {WARN}" if near else ""
    lines.append(f" {state_label or '손절'}(-{stop_pct:g}%)까지 {gap:.1f}%p{tag}")
    adx1, adx0 = m[f"adx_{a}"].iloc[i], m[f"adx_{a}"].iloc[i - 1]
    room = adx1 - y_thr
    tag = f"  {WARN}" if room <= IMM_ADX else ""
    lines.append(f" ADX {adx1:.1f} {_d(adx1, adx0)} / 청산선({y_thr})까지 {room:.1f}{tag}")
    return lines


def build_extra_info(m, w):
    """근거(세부 지표) 블록. 신호 계산에 쓴 값을 그대로 꺼내 보여준다(별도 재계산 없음)."""
    i = len(m) - 1
    ld = m.attrs["last_dates"]
    out = []

    def head(a, status):
        c, pc = m[f"Close_{a}"].iloc[i], m[f"Close_{a}"].iloc[i - 1]
        d0 = m["Date"].iloc[i]
        warn = f"  ⚠️ 원자료 마지막 날짜 {ld[a]:%m-%d}" if ld[a] != d0 else ""
        return [f"【{a}】 {status}", f" {d0:%m-%d} 종가 {c:,.2f} ({fmt_pct(c / pc - 1)}){warn}"]

    # QQQ
    state = int(m["state_QQQ"].iloc[i])
    L = head("QQQ", f"ON · 비중 {fmt_w(w['QQQ'][i])}" if state > 0 else "OFF")
    if state > 0:
        n = QQQ_PARAMS["N1"] if state == 2 else QQQ_PARAMS["N2"]
        label = "1차 손절" if state == 2 else "최종 손절"
        L += _hold_lines(m, "QQQ", i, n, QQQ_PARAMS["y_adx_thr"], label)
    else:
        L += _reentry_lines(m, "QQQ", i, m["state_QQQ"].values, QQQ_PARAMS["cooldown"], "QQQ")
    out.append("\n".join(L))

    # TLT
    sig = int(m["sig_TLT"].iloc[i])
    L = head("TLT", f"ON · 비중 {fmt_w(w['TLT'][i])}" if sig else "OFF")
    f1, s1 = m["smaf_TLT"].iloc[i], m["smas_TLT"].iloc[i]
    f0, s0 = m["smaf_TLT"].iloc[i - 1], m["smas_TLT"].iloc[i - 1]
    g1, g0 = (f1 / s1 - 1) * 100, (f0 / s0 - 1) * 100
    L.append(f" SMA{TLT_PARAMS['fast']} {f1:.2f} / SMA{TLT_PARAMS['slow']} {s1:.2f}")
    tag = f"  {WARN}" if abs(g1) <= IMM_TLT else ""
    target = "데드크로스" if sig else "골든크로스"
    dg = g1 - g0
    dg_txt = "변화 없음" if abs(dg) < 0.005 else f"{dg:+.2f}%p"
    L.append(f" 격차 {g1:+.2f}% (전일 {g0:+.2f}%, {dg_txt}) → {target}까지 {abs(g1):.2f}%p{tag}")
    out.append("\n".join(L))

    # GLD, XLE
    for a, P, stop, need in (("GLD", GLD_PARAMS, GLD_PARAMS["N"] * 100, GLD_PARAMS["cooldown"] + 1),
                             ("XLE", XLE_PARAMS, XLE_PARAMS["N"], XLE_PARAMS["cooldown"])):
        sig = int(m[f"sig_{a}"].iloc[i])
        L = head(a, f"ON · 비중 {fmt_w(w[a][i])}" if sig else "OFF")
        if sig:
            L += _hold_lines(m, a, i, stop, P["y_adx_thr"])
        else:
            L += _reentry_lines(m, a, i, m[f"sig_{a}"].values, need, a)
        out.append("\n".join(L))
    return "\n\n".join(out)


def build_sat_change(m):
    """위성 비중 변경: QQQ와 무관하게 위성 자신의 신호 ON/OFF 전환만 표시 (ON=100%, OFF=0%)."""
    i = len(m) - 1
    lines = []
    for a in SATS:
        on, pon = int(m[f"sig_{a}"].iloc[i]), int(m[f"sig_{a}"].iloc[i - 1])
        if on != pon:
            lines.append(f"{a}  {pon * 100}% → {on * 100}%")
    return "\n".join(lines)


def vol_reference(m):
    c = m["Close_QQQ"].values
    r = np.diff(c) / c[:-1]
    if len(r) < VOL_WINDOW:
        return None
    return float(np.std(r[-VOL_WINDOW:], ddof=1) * np.sqrt(252))


def build_message(m, now_kst: datetime, date_mismatch: str = "") -> str:
    """now_kst는 표시용(토·월요일 안내 문구 분기)일 뿐, 데이터 최신성 판정에는 쓰지 않는다 —
    "지금 몇 시니까 며칠 데이터가 있어야 한다"는 시계 기준 추측이 과거에 반복된 오탐의 원인이었음.
    대신 load_from_yahoo()가 자산 간 날짜 불일치를 이미 해소하려 시도했고, 그래도 안 맞으면
    date_mismatch에 그 사실만 경고로 받아 표시한다."""
    w = target_weights(m)
    bar_date = m["Date"].iloc[-1].date()
    title = f"📊 QQQ Ballast 신호 — {bar_date} 종가 기준"

    trades, verbs = build_trades(m, w)
    if trades:
        names = [f"{a} {v}" for a, v in verbs]
        summary = f"{', '.join(names)} ({len(trades)}건 매매)"
    else:
        summary = "변동 없음, 전 포지션 유지"

    out = [title, f"한줄요약: {summary}"]
    if date_mismatch:
        out.append(f"⚠️ {date_mismatch}")
    out += ["", build_weight_block(m, w), "", "──────────────", "💰 오늘 실행할 매매"]
    if trades:
        out += [f"{k}. {t}" for k, t in enumerate(trades, 1)]
        if now_kst.weekday() in (5, 0):  # 토요일 / 월요일 발송분(금요일 종가 기준)
            out.append("(다음 미국 개장일 시가 기준으로 실행)")
    else:
        out.append("- 없음")
    out.append("──────────────")
    out.append("")

    out += ["━━━ 근거 (세부 지표) ━━━", build_extra_info(m, w)]

    sat = build_sat_change(m)
    if sat:
        out += ["", "━━━ 위성 비중 변경 ━━━", sat]
    return "\n".join(out)


def session_closed_today(now_et: datetime) -> bool:
    """미국 정규장 마감(16:00 ET) 후 1시간 30분이 지났으면 오늘 세션 종가가 확정된 것으로 본다.
    데이터 제공처(야후/stooq)가 마감 직후 바로 일봉을 올리지 않는 경우가 있어 여유를 둠."""
    return (now_et.hour, now_et.minute) >= (17, 30)


# ---------------------- 텔레그램 ----------------------
def send_telegram(text: str):
    import requests
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    last = None
    for attempt in range(3):
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": text[:4000],
                                         "disable_web_page_preview": True}, timeout=20)
            if r.status_code == 200:
                return
            last = f"{r.status_code} {r.text}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"텔레그램 발송 실패: {last}")


# ---------------------- 백테스트 재현 검증 ----------------------
def check_backtest(m):
    """저장된 목표비중을 다음날 시가에 체결한다고 보고 성과를 재계산 (문서 수치와 대조용).
    비용·현금이자 없음. 데이터가 배당 포함인지(조정 종가)에 따라 문서 수치와 ±0.3~0.6%p 차이날 수 있다
    (TLT 배당 포함 시 CAGR이 더 높게 나옴). 정확한 대조는 문서와 같은 CSV로 해야 한다."""
    w = target_weights(m)
    n = len(m)
    r = np.zeros(n - 1)
    for a in ASSETS:
        o = m[f"Open_{a}"].values
        ret = o[1:] / o[:-1] - 1
        held = np.empty(n - 1)
        held[0] = w[a][0]
        held[1:] = w[a][: n - 2]
        r += held * ret
    eq = np.cumprod(1 + r)
    yrs = (m["Date"].iloc[-1] - m["Date"].iloc[0]).days / 365.25
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    print(f"기간 {m['Date'].iloc[0].date()} ~ {m['Date'].iloc[-1].date()} ({yrs:.1f}년)")
    print(f"CAGR {cagr:.2%}  MDD {mdd:.2%}  Calmar {cagr / abs(mdd):.3f}   (문서 기준 2026-09-28: 15.60% / -17.34% / 0.899, 2026-09-14까지. 데이터 소스에 따라 오차 가능)")


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "csv"], default="yahoo")
    ap.add_argument("--csv-dir", default=".")
    ap.add_argument("--dry-run", action="store_true", help="발송하지 않고 화면에만 출력")
    ap.add_argument("--force", action="store_true", help="(호환용, 현재는 동작에 영향 없음)")
    ap.add_argument("--check", action="store_true", help="백테스트 성과 재현 검증만 실행")
    ap.add_argument("--now-kst", default=None, help="테스트용 가짜 현재시각 (예: 2026-09-30T06:30)")
    args = ap.parse_args()

    try:
        if args.source == "csv":
            data, dates = load_from_csv(args.csv_dir)
        else:
            data, dates = load_from_yahoo()
        m = build_frame(data)

        if args.check:
            check_backtest(m)
            return

        date_mismatch = ""
        if len(set(dates.values())) > 1:
            date_mismatch = "자산별 데이터 기준일 불일치: " + ", ".join(f"{a} {d}" for a, d in dates.items())
        elif EXPECTED and min(dates.values()) < EXPECTED:
            date_mismatch = (f"데이터 최신 아님 (기대 {EXPECTED}, 실제 {min(dates.values())}). "
                             "휴장일이면 정상, 아니면 오늘 매매 보류 후 재실행하세요")

        if args.now_kst:
            now_kst = datetime.fromisoformat(args.now_kst).replace(tzinfo=KST)
        else:
            now_kst = datetime.now(KST)
        msg = build_message(m, now_kst, date_mismatch=date_mismatch)
        if args.dry_run:
            print(msg)
        else:
            send_telegram(msg)
            print("발송 완료")
    except Exception as e:  # noqa: BLE001
        err = f"❌ QQQ Ballast 실행 오류\n{type(e).__name__}: {e}"
        print(traceback.format_exc(), file=sys.stderr)
        if not args.dry_run and not args.check:
            try:
                send_telegram(err)
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()