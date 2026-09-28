# =====================================================================
# QQQ Ballast — QQQ 단일 투자 포트폴리오 백테스트 (정정본, 2026-09-28)
#
# 구조
# ----
# 1) 코어: QQQ 트랙2 (D·N 분할매도 + A·X·Y 재진입)
#    - D=180(확장형 고점), N1=10%(25% 매도) -> N2=15%(잔량 75% 매도)
#    - 재진입: DI/ADX(EWM) 14일, +DI 상향돌파(cooldown 7일), 오신호 매도 Y=ADX<9
# 2) 위성자산: QQQ 미보유 비중(현금분)을 TLT·GLD·XLE 중 "신호 on인 자산"에 배분
#    - TLT: SMA 20/110 골든/데드크로스
#    - GLD: 확장형고점 N=1.5% + DI/ADX(EWM) 5일 + cooldown 2일 + Y=ADX<15 (2026-09-28 변경)
#    - XLE: 250일 고점 N=15% + DI/ADX(EWM) 14일 + cooldown 15일 + Y=ADX<15
#    - ALLOC_MODE="fix3": 자산당 최대 현금의 1/3 (off 자산 몫은 현금으로 남김)  <- 기본값
#      ALLOC_MODE="1n"  : 신호 on인 자산끼리만 1/n (수익 높으나 MDD 큼)
# 3) 고변동성 필터(옵션): VOL_FILTER_THRESHOLD 설정 시 QQQ 20일 연환산 변동성이
#    임계값 초과인 날 위성 비중을 0으로 (보유 중이어도 현금화). 정정 후 성과 개선 효과
#    없음이 확인되어 기본값은 사용 안 함(None).
#
# 체결 가정 (정정됨)
# -----------------
# - 신호는 당일 종가로 확정, 체결은 "다음 거래일 시가"
# - 수익률 구간 [시가 t -> 시가 t+1] 에는 전날 종가(t-1) 기준 신호 pos[t-1]를 적용
# - 한국시간 새벽 미국장 종료 후 종가 분석 -> 그날 밤 국내 연금계좌 ETF 시초가 매매와 일치
#
# 정정 이력
# --------
# 초기 통합 백테스트는 pos[t]를 [시가 t -> 시가 t+1]에 적용해 신호를 하루 일찍
# 반영하는 룩어헤드 오류가 있었음(성과 과대평가). 본 파일은 정정본이며 이전 결과
# (CAGR 20.36% / MDD -16.68% / Calmar 1.220)는 폐기.
#
# 성과 (2005-02-25 ~ 2026-09-14, 21.5년, 트랙2, 신 GLD 반영, 비용·이자 제외)
# ----------------------------------------------------------
#   QQQ 단독                          : CAGR 12.44%  MDD -18.58%  Calmar 0.670
#   위성 TLT+GLD+XLE 1/3 고정 (기본값) : CAGR 15.60%  MDD -17.34%  Calmar 0.899
#   위성 GLD 단독                      : CAGR 16.35%  MDD -17.99%  Calmar 0.909
#   위성 TLT+GLD+XLE 1/n               : CAGR 17.75%  MDD -21.70%  Calmar 0.818
#
# 사용 데이터: qqq_us_d.csv, tlt_us_d.csv, gld_us_d.csv, xle_us_d.csv
#             (컬럼: Date,Open,High,Low,Close,Volume)
# =====================================================================
import pandas as pd
import numpy as np

# ---------------------- 파라미터 (최종 확정값) ----------------------
QQQ_PARAMS = dict(D=180, N1=10, N2=15, frac1=0.25, di_period=14, cooldown=7, y_adx_thr=9)
TLT_PARAMS = dict(fast=20, slow=110)
GLD_PARAMS = dict(N=0.015, di_period=5, cooldown=2, y_adx_thr=15)   # 2026-09-28 변경(GLD_위성전략_트랙2_분석결과.md)
XLE_PARAMS = dict(N=15, di_period=14, cooldown=15, y_adx_thr=15, roll_max_window=250)
ALLOC_MODE = "fix3"         # "fix3": 자산당 현금의 1/3 고정 / "1n": 신호 on인 자산끼리 1/n
VOL_FILTER_WINDOW = 20      # QQQ 일간수익률 변동성 계산 기간(거래일)
VOL_FILTER_THRESHOLD = None  # 예: 0.45 -> 연환산 변동성 45% 초과일에 위성 비중 0 (기본: 미사용)


# ---------------------- 데이터 로드 ----------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    return df[["Date", "Open", "High", "Low", "Close"]]


# ---------------------- 공통 지표: DI/ADX (EWM 방식) ----------------------
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


# ---------------------- QQQ 포지션 (D·N + A·X·Y, 트랙2) ----------------------
def qqq_position(close: np.ndarray, high: pd.Series, low: pd.Series,
                  D, N1, N2, frac1, di_period, cooldown, y_adx_thr) -> np.ndarray:
    close_s = pd.Series(close)
    di_up, adx = di_adx(close_s, high, low, di_period)
    n = len(close)
    roll_max_D = close_s.rolling(D, min_periods=1).max().values

    pos = np.zeros(n)
    pos[0] = 1.0
    peak = roll_max_D[0]
    state = 2  # 2=전량 보유, 1=frac1 매도한 상태, 0=현금
    days_since_exit = 9999

    for t in range(1, n):
        if state > 0:
            peak = max(peak, close[t])
            dd = close[t] / peak - 1
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
    return pos


# ---------------------- TLT 신호 (SMA 골든/데드크로스) ----------------------
def tlt_signal(close: pd.Series, fast: int, slow: int) -> np.ndarray:
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean()
    return (sma_f > sma_s).fillna(False).values.astype(float)


# ---------------------- GLD 신호 (확장형 D·N + A·X·Y) ----------------------
def gld_signal(close: pd.Series, high: pd.Series, low: pd.Series,
               N, di_period, cooldown, y_adx_thr) -> np.ndarray:
    di_up, adx = di_adx(close, high, low, di_period)
    c = close.values
    n = len(c)
    pos = np.zeros(n)
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
        pos[t] = state
    return pos


# ---------------------- XLE 신호 (확장형(250일) N% + A·X·Y) ----------------------
def xle_signal(close: pd.Series, high: pd.Series, low: pd.Series,
               N, di_period, cooldown, y_adx_thr, roll_max_window) -> np.ndarray:
    di_up, adx = di_adx(close, high, low, di_period)
    c = close.values
    n = len(c)
    roll_max = close.rolling(roll_max_window, min_periods=1).max().values
    pos = np.zeros(n)
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
        pos[t] = state
    return pos


# ---------------------- 통합 포트폴리오 시뮬레이션 ----------------------
def build_merged(qqq_csv, tlt_csv, gld_csv, xle_csv) -> pd.DataFrame:
    """각 자산 신호를 '자기 전체 이력'으로 계산한 뒤 공통 거래일로 병합 (원본 문서/실서비스와 동일 방식)."""
    q, t, g, x = (load_csv(p) for p in (qqq_csv, tlt_csv, gld_csv, xle_csv))
    q["pos_qqq"] = qqq_position(q["Close"].values, q["High"], q["Low"], **QQQ_PARAMS)
    t["sig_tlt"] = tlt_signal(t["Close"], **TLT_PARAMS)
    g["sig_gld"] = gld_signal(g["Close"], g["High"], g["Low"], **GLD_PARAMS)
    x["sig_xle"] = xle_signal(x["Close"], x["High"], x["Low"], **XLE_PARAMS)
    m = q[["Date", "Open", "Close", "pos_qqq"]].rename(columns={"Open": "Open_qqq", "Close": "Close_qqq"})
    m = m.merge(t[["Date", "Open", "sig_tlt"]].rename(columns={"Open": "Open_tlt"}), on="Date")
    m = m.merge(g[["Date", "Open", "sig_gld"]].rename(columns={"Open": "Open_gld"}), on="Date")
    m = m.merge(x[["Date", "Open", "sig_xle"]].rename(columns={"Open": "Open_xle"}), on="Date")
    return m.sort_values("Date").reset_index(drop=True)


def target_weights(m: pd.DataFrame, alloc_mode: str = None, vol_thr=None):
    """일별 '목표' 비중(당일 종가 신호 기준)을 계산. 실제 체결은 다음날 시가."""
    alloc_mode = alloc_mode or ALLOC_MODE
    vol_thr = VOL_FILTER_THRESHOLD if vol_thr is None else vol_thr

    pos_qqq, sig_tlt, sig_gld, sig_xle = m["pos_qqq"].values, m["sig_tlt"].values, m["sig_gld"].values, m["sig_xle"].values

    qqq_close = m["Close_qqq"].values
    qqq_ret = np.zeros(len(qqq_close))
    qqq_ret[1:] = qqq_close[1:] / qqq_close[:-1] - 1
    vol20 = pd.Series(qqq_ret).rolling(VOL_FILTER_WINDOW).std().values * np.sqrt(252)
    high_vol = (vol20 > vol_thr) if vol_thr else np.zeros(len(m), dtype=bool)

    on_tlt, on_gld, on_xle = (s * (~high_vol) for s in (sig_tlt, sig_gld, sig_xle))
    cash = 1 - pos_qqq
    cnt = on_tlt + on_gld + on_xle
    den = np.where(cnt == 0, 1, cnt) if alloc_mode == "1n" else 3.0
    w = dict(qqq=pos_qqq, tlt=cash * on_tlt / den, gld=cash * on_gld / den, xle=cash * on_xle / den)
    return w, dict(sig_tlt=sig_tlt, sig_gld=sig_gld, sig_xle=sig_xle, vol20=vol20, high_vol=high_vol)


def run_portfolio(m: pd.DataFrame, alloc_mode: str = None, vol_thr=None):
    """목표 비중을 하루 지연(다음날 시가 체결)시켜 [시가 t -> 시가 t+1] 수익률에 적용."""
    w, extra = target_weights(m, alloc_mode, vol_thr)
    n = len(m)
    dates = m["Date"].values
    strat_r = np.zeros(n - 1)
    for name, col in (("qqq", "Open_qqq"), ("tlt", "Open_tlt"), ("gld", "Open_gld"), ("xle", "Open_xle")):
        o = m[col].values
        r = o[1:] / o[:-1] - 1
        held = np.empty(n - 1)
        held[0] = w[name][0]
        held[1:] = w[name][: n - 2]      # held[t] = 전날 종가 기준 목표비중
        strat_r += held * r
    equity = np.cumprod(1 + strat_r)
    years = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(float) / 365.25
    cagr = equity[-1] ** (1 / years) - 1
    mdd = (equity / np.maximum.accumulate(equity) - 1).min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    total = sum(w[k] for k in w)
    metrics = dict(CAGR=cagr, MDD=mdd, Calmar=calmar, Years=years, AvgExposure=float(total.mean()))
    return metrics, w, extra


def latest_signal(m: pd.DataFrame, w: dict) -> dict:
    """가장 최근 종가 기준 목표비중과 전일 대비 변화. 텔레그램 알림용."""
    out = {"date": str(m["Date"].iloc[-1].date())}
    for k in ("qqq", "tlt", "gld", "xle"):
        out[k] = dict(today=float(w[k][-1]), yesterday=float(w[k][-2]))
    return out


if __name__ == "__main__":
    m = build_merged("qqq_us_d.csv", "tlt_us_d.csv", "gld_us_d.csv", "xle_us_d.csv")
    print(f"기간: {m['Date'].iloc[0].date()} ~ {m['Date'].iloc[-1].date()}")
    for mode in ("fix3", "1n"):
        metrics, w, _ = run_portfolio(m, alloc_mode=mode, vol_thr=None)
        print(f"[위성 {mode}] CAGR {metrics['CAGR']:.2%}  MDD {metrics['MDD']:.2%}  "
              f"Calmar {metrics['Calmar']:.3f}  평균투자비중 {metrics['AvgExposure']:.1%}")
    print("최신 신호:", latest_signal(m, w))
