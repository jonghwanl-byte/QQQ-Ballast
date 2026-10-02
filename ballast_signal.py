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
# =====================================================================
import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------- 전략 파라미터 (백테스트 확정값) ----------------------
QQQ_PARAMS = dict(D=180, N1=10, N2=15, frac1=0.25, di_period=14, cooldown=7, y_adx_thr=9)
TLT_PARAMS = dict(fast=20, slow=110)
GLD_PARAMS = dict(N=0.015, di_period=5, cooldown=2, y_adx_thr=15)
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
def load_from_csv(csv_dir: str) -> dict:
    out = {}
    for a in ASSETS:
        df = pd.read_csv(os.path.join(csv_dir, f"{a.lower()}_us_d.csv"), parse_dates=["Date"])
        df.columns = [c.strip() for c in df.columns]
        out[a] = df[["Date", "Open", "High", "Low", "Close"]].sort_values("Date").reset_index(drop=True)
    return out


def _trim_incomplete_bar(df: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    """장중에 받아온 오늘자 미완성 봉 제거 (ET 기준 오늘 날짜 + 마감 전이면 마지막 행 삭제)."""
    if len(df) and df["Date"].iloc[-1].date() == now_et.date() and not session_closed_today(now_et):
        return df.iloc[:-1].reset_index(drop=True)
    return df


def fetch_one_yahoo(ticker: str, now_et: datetime) -> pd.DataFrame:
    import yfinance as yf
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=HIST_START[ticker], auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise RuntimeError("빈 데이터")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
            df = df[["Date", "Open", "High", "Low", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
            return _trim_incomplete_bar(df, now_et)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"yahoo 다운로드 실패: {last_err}")


def fetch_one_stooq(ticker: str) -> pd.DataFrame:
    """야후가 지연될 때 쓰는 보조 소스. 보통 야후보다 장마감 후 더 빨리 갱신됨."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    df = pd.read_csv(url, parse_dates=["Date"])
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError("빈 데이터 또는 형식 오류")
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    return df


def load_from_yahoo() -> dict:
    """자산별로 야후 -> (지연 시) stooq 순으로 시도하고, 어느 자산이 지연됐는지 로그에 남긴다."""
    now_et = datetime.now(ET)
    exp = expected_session(now_et)
    out = {}
    status = []
    for a in ASSETS:
        df, src = None, None
        try:
            df = fetch_one_yahoo(a, now_et)
            src = "yahoo"
        except Exception as e:  # noqa: BLE001
            print(f"[{a}] yahoo 실패: {e}", file=sys.stderr)

        last = df["Date"].iloc[-1].date() if df is not None and len(df) else None
        if df is None or last is None or last < exp:
            try:
                df2 = fetch_one_stooq(a)
                last2 = df2["Date"].iloc[-1].date() if len(df2) else None
                if df is None or (last2 is not None and (last is None or last2 > last)):
                    df, src, last = df2, "stooq", last2
            except Exception as e:  # noqa: BLE001
                print(f"[{a}] stooq 실패: {e}", file=sys.stderr)

        if df is None or not len(df):
            raise RuntimeError(f"{a} 데이터를 야후/stooq 양쪽에서 모두 가져오지 못했습니다")
        out[a] = df
        status.append(f"{a}={last}({src})")

    print(f"예상 세션 {exp} | 자산별 최신 데이터: " + ", ".join(status))
    stale = [s for s in status if str(exp) not in s]
    if stale:
        print(f"⚠️ 예상 세션({exp})보다 오래된 자산: {', '.join(stale)}", file=sys.stderr)
    return out


# ---------------------- 공통 지표: DI/ADX (EWM) ----------------------
def di_adx(close: pd.Series, high: pd.Series, low: pd.Series, period: int):
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
    di_up = np.nan_to_num(pdi > mdi, nan=0.0).astype(bool)
    return di_up, adx


# ---------------------- QQQ 트랙2 (비중, 추적 고점, 상태) ----------------------
def qqq_position(df, D, N1, N2, frac1, di_period, cooldown, y_adx_thr):
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
    t = data["TLT"].copy()
    t["sig"], t["smaf"], t["smas"] = tlt_signal(t, **TLT_PARAMS)
    g = data["GLD"].copy()
    g["sig"], g["peak"] = gld_signal(g, **GLD_PARAMS)
    x = data["XLE"].copy()
    x["sig"], x["peak"] = xle_signal(x, **XLE_PARAMS)

    ath = {a: data[a]["Close"].max() for a in ASSETS}  # 역대 최고 종가(수정종가, 데이터 시작일 이후)

    def pick(df, cols, tag):
        return df[["Date"] + cols].rename(columns={c: f"{c}_{tag}" for c in cols})

    m = pick(q, ["Open", "Close", "pos", "peak", "state"], "QQQ")
    m = m.merge(pick(t, ["Open", "Close", "sig", "smaf", "smas"], "TLT"), on="Date")
    m = m.merge(pick(g, ["Open", "Close", "sig", "peak"], "GLD"), on="Date")
    m = m.merge(pick(x, ["Open", "Close", "sig", "peak"], "XLE"), on="Date")
    m = m.sort_values("Date").reset_index(drop=True)
    m.attrs["ath"] = ath
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


def build_extra_info(m, w):
    i = len(m) - 1
    ath = m.attrs["ath"]
    lines = []
    for a in ASSETS:
        c, pc = m[f"Close_{a}"].iloc[i], m[f"Close_{a}"].iloc[i - 1]
        lines.append(a)
        lines.append(f" 전일대비 {fmt_pct(c / pc - 1)} | 고점대비 {fmt_pct(c / ath[a] - 1)}")
        held = w[a][i] > 1e-9
        if not held:
            continue
        if a == "QQQ":
            state = int(m["state_QQQ"].iloc[i])
            n_pct = QQQ_PARAMS["N1"] if state == 2 else QQQ_PARAMS["N2"]
            dd = (c / m["peak_QQQ"].iloc[i] - 1) * 100
            gap = n_pct + dd
            label = "1차 손절선" if state == 2 else "손절선"
            if gap <= 0:
                lines.append(f" {label}(-{n_pct}%) 이미 초과 ({-gap:.1f}%p) — 다음 종가 기준 매도 신호 가능")
            else:
                lines.append(f" {label}(-{n_pct}%)까지 {gap:.1f}% 남음")
        elif a == "TLT":
            spread = (m["smaf_TLT"].iloc[i] / m["smas_TLT"].iloc[i] - 1) * 100
            lines.append(f" 데드크로스까지 SMA 격차 {spread:+.1f}%")
        else:
            n_pct = GLD_PARAMS["N"] * 100 if a == "GLD" else XLE_PARAMS["N"]
            dd = (c / m[f"peak_{a}"].iloc[i] - 1) * 100
            gap = n_pct + dd
            n_txt = f"{n_pct:.1f}".rstrip("0").rstrip(".")
            lines.append(f" 손절선(-{n_txt}%)까지 {gap:.1f}% 남음")
    return "\n".join(lines)


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


def build_message(m, now_kst: datetime, force: bool = False, now_et: datetime = None) -> str:
    w = target_weights(m)
    bar_date = m["Date"].iloc[-1].date()
    title = f"📊 QQQ Ballast 신호 — {bar_date} 종가 기준"

    # 휴장/데이터 미갱신 감지
    if now_et is not None and not force:
        exp = expected_session(now_et)
        if bar_date < exp:
            return "\n".join([
                f"📊 QQQ Ballast — {now_kst.strftime('%Y-%m-%d')}",
                f"⚠️ 미국장 휴장이거나 데이터가 아직 갱신되지 않았습니다. (마지막 데이터 {bar_date}, 예상 {exp})",
                "신호 변동 없음으로 간주하고 직전 비중을 유지하세요.",
                "데이터 지연이 의심되면 GitHub Actions에서 수동 재실행(Run workflow) 하세요.",
                "",
                build_weight_block(m, w, mode="hold"),
            ])

    trades, verbs = build_trades(m, w)
    if trades:
        names = [f"{a} {v}" for a, v in verbs]
        summary = f"{', '.join(names)} ({len(trades)}건 매매)"
    else:
        summary = "변동 없음, 전 포지션 유지"

    out = [title, f"한줄요약: {summary}", "", build_weight_block(m, w), "", "──────────────",
           "💰 오늘 실행할 매매"]
    if trades:
        out += [f"{k}. {t}" for k, t in enumerate(trades, 1)]
        if now_kst.weekday() in (5, 0):  # 토요일 / 월요일 발송분(금요일 종가 기준)
            out.append("(다음 미국 개장일 시가 기준으로 실행)")
    else:
        out.append("- 없음")
    out.append("──────────────")
    out.append("")

    out += ["━━━ 부가 정보 ━━━", build_extra_info(m, w)]

    sat = build_sat_change(m)
    if sat:
        out += ["", "━━━ 위성 비중 변경 ━━━", sat]
    return "\n".join(out)


def session_closed_today(now_et: datetime) -> bool:
    """미국 정규장 마감(16:00 ET) 후 15분이 지났으면 오늘 세션 종가가 확정된 것으로 본다."""
    return (now_et.hour, now_et.minute) >= (16, 15)


def expected_session(now_et: datetime):
    """지금 시점에서 종가가 확정되어 있어야 하는 가장 최근 미국 평일 세션 날짜."""
    d = now_et.date()
    if not session_closed_today(now_et):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


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
    """저장된 목표비중을 다음날 시가에 체결한다고 보고 성과를 재계산 (문서 수치와 대조용)."""
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
    print(f"CAGR {cagr:.2%}  MDD {mdd:.2%}  Calmar {cagr / abs(mdd):.3f}   (문서 기준: 15.60% / -17.34% / 0.899, 2026-09-14까지)")


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "csv"], default="yahoo")
    ap.add_argument("--csv-dir", default=".")
    ap.add_argument("--dry-run", action="store_true", help="발송하지 않고 화면에만 출력")
    ap.add_argument("--force", action="store_true", help="월요일/휴장 안내 로직을 무시하고 전체 메시지 생성")
    ap.add_argument("--check", action="store_true", help="백테스트 성과 재현 검증만 실행")
    ap.add_argument("--now-kst", default=None, help="테스트용 가짜 현재시각 (예: 2026-09-30T06:30)")
    args = ap.parse_args()

    try:
        if args.source == "csv":
            data = load_from_csv(args.csv_dir)
            m = build_frame(data)
        else:
            data = load_from_yahoo()
            m = build_frame(data)
            # 마감 직후라 데이터가 늦게 반영될 수 있어, 최신 세션이 안 보이면 최대 4회(5분 간격) 재시도
            now0 = datetime.now(KST)
            if not args.force and not args.check:
                for i in range(4):
                    if m["Date"].iloc[-1].date() >= expected_session(now0.astimezone(ET)):
                        break
                    print(f"재시도 {i + 1}/4 ...")
                    time.sleep(300)
                    data = load_from_yahoo()
                    m = build_frame(data)
                    now0 = datetime.now(KST)
        if args.check:
            check_backtest(m)
            return
        if args.now_kst:
            now_kst = datetime.fromisoformat(args.now_kst).replace(tzinfo=KST)
        else:
            now_kst = datetime.now(KST)
        now_et = now_kst.astimezone(ET) if args.source == "yahoo" else None
        msg = build_message(m, now_kst, force=args.force, now_et=now_et)
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
