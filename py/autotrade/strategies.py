"""매매 전략 모음.

모든 전략은 캔들 표(DataFrame: open/high/low/close/volume)를 받아서
아래 3가지 열을 가진 표를 돌려준다.

  buy        : True면 '이 캔들이 끝난 시점'에 매수 신호 -> 다음 캔들 시가에 매수
  sell       : True면 '이 캔들이 끝난 시점'에 매도 신호 -> 다음 캔들 시가에 매도
  breakout   : 숫자가 있으면 '이 캔들 진행 중' 가격이 그 값에 닿는 순간 매수
               (이 값은 캔들이 시작할 때 이미 알 수 있는 정보로만 계산해야 한다)
  size       : (선택) 돌파 매수 때 현금의 몇 %를 쓸지(0~1). 없으면 100%

미래 정보를 몰래 쓰지 않도록(look-ahead 방지) 모든 신호는
그 시점까지 확정된 데이터로만 계산한다.

새 전략을 추가하려면 Strategy를 상속해서 signals()를 만들고
아래 STRATEGIES 딕셔너리에 등록하면 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


EPS = 1e-9  # 계산 오차 수준의 차이는 '같은 값'으로 본다(PC와 핸드폰 결과를 똑같이 맞추기 위함)


def gt(a, b):
    """a가 b보다 '확실히' 크다 (상대 오차 1e-9 이내는 같은 값). NaN이면 False."""
    return (a - b) > EPS * b.abs() if hasattr(b, "abs") else (a - b) > EPS * abs(b)


class Strategy:
    name = "base"
    description = ""
    # 매수 후 몇 개 캔들 뒤에 무조건 파는지(None이면 매도 신호가 나올 때까지 보유)
    hold_bars: int | None = None

    def __init__(self, **params):
        self.params = params

    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({p})"

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _empty(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"buy": False, "sell": False, "breakout": np.nan}, index=df.index
        )


class VolatilityBreakout(Strategy):
    """변동성 돌파 전략.

    직전 캔들의 (고가 - 저가) = '변동폭'을 구한다.
    이번 캔들 시가 + 변동폭 × k 를 '목표가'로 정하고,
    가격이 목표가를 뚫고 올라가면 '상승 힘이 강하다'고 보고 매수한다.
    산 다음에는 다음 캔들 시가에 판다(짧게 먹고 빠지는 단타).

    ma > 0 이면 '시가가 최근 ma개 캔들 평균보다 높을 때(상승 추세)'만 매수한다.
    """
    name = "breakout"
    description = "변동성 돌파: 시가+전봉변동폭×k를 뚫으면 매수, 다음 봉 시가에 매도"
    hold_bars = 1

    def __init__(self, k: float = 0.5, ma: int = 0):
        super().__init__(k=k, ma=ma)
        self.k, self.ma = k, ma

    def signals(self, df):
        out = self._empty(df)
        prev_range = (df["high"] - df["low"]).shift(1)
        target = df["open"] + prev_range * self.k
        ok = target.notna()
        if self.ma > 0:
            # 이전 캔들들의 종가 평균(이번 캔들 종가는 아직 모르므로 shift)
            trend = ind.sma(df["close"], self.ma).shift(1)
            ok &= df["open"] > trend
        out["breakout"] = target.where(ok)
        return out


class MACross(Strategy):
    """이동평균 교차 전략.

    짧은 기간 평균선이 긴 기간 평균선을 아래에서 위로 뚫으면(골든크로스) 매수,
    위에서 아래로 뚫으면(데드크로스) 매도한다. 추세를 따라가는 방식.
    """
    name = "ma_cross"
    description = "이동평균 교차: 단기선이 장기선 위로(골든크로스) 매수, 아래로 매도"

    def __init__(self, short: int = 5, long: int = 20):
        super().__init__(short=short, long=long)
        self.short, self.long = short, long

    def signals(self, df):
        out = self._empty(df)
        s = ind.sma(df["close"], self.short)
        l = ind.sma(df["close"], self.long)
        above = gt(s, l)
        prev = above.shift(1, fill_value=False)
        out["buy"] = above & ~prev & l.notna()
        out["sell"] = ~above & prev
        return out


class RSIReversal(Strategy):
    """RSI 역추세 전략.

    RSI가 low(예: 30) 아래로 떨어졌다가 다시 위로 올라오면 '많이 떨어졌으니 반등'을 노려 매수,
    RSI가 high(예: 70) 위로 올라가면 매도한다.
    """
    name = "rsi"
    description = "RSI 역추세: RSI가 과매도 구간에서 올라오면 매수, 과매수면 매도"

    def __init__(self, period: int = 14, low: float = 30, high: float = 70):
        super().__init__(period=period, low=low, high=high)
        self.period, self.low, self.high = period, low, high

    def signals(self, df):
        out = self._empty(df)
        r = ind.rsi(df["close"], self.period)
        prev = r.shift(1)
        out["buy"] = (prev < self.low) & (r >= self.low)
        out["sell"] = r > self.high
        return out


class BollingerReversion(Strategy):
    """볼린저 밴드 평균회귀 전략.

    가격이 아래 밴드 밑으로 떨어지면 '평소보다 과하게 내렸다'고 보고 매수,
    가격이 중심선(평균)까지 돌아오면 매도한다.
    """
    name = "bollinger"
    description = "볼린저 밴드: 아래 밴드 밑으로 내려가면 매수, 중심선 회복 시 매도"

    def __init__(self, period: int = 20, std: float = 2.0):
        super().__init__(period=period, std=std)
        self.period, self.std = period, std

    def signals(self, df):
        out = self._empty(df)
        mid, upper, lower = ind.bollinger(df["close"], self.period, self.std)
        out["buy"] = gt(lower, df["close"])
        out["sell"] = gt(df["close"], mid)
        return out


class Donchian(Strategy):
    """터틀 트레이딩(돈치안 채널 돌파).

    1980년대 리처드 데니스가 초보자들('터틀')에게 가르친 추세추종 규칙.
    종가가 최근 entry개 캔들의 최고가를 넘으면 '새 고점 = 추세 시작'으로 보고 매수,
    최근 exit개 캔들의 최저가 아래로 내려가면 매도한다.
    """
    name = "donchian"
    description = "터틀 트레이딩: 최근 N봉 최고가를 넘으면 매수, 최근 M봉 최저가 밑이면 매도"

    def __init__(self, entry: int = 20, exit: int = 10):
        super().__init__(entry=entry, exit=exit)
        self.entry, self.exit = entry, exit

    def signals(self, df):
        out = self._empty(df)
        hi = df["high"].rolling(self.entry).max().shift(1)  # 이번 봉을 뺀 직전 N봉
        lo = df["low"].rolling(self.exit).min().shift(1)
        out["buy"] = df["close"] > hi
        out["sell"] = df["close"] < lo
        return out


class ConnorsRSI2(Strategy):
    """코너스 RSI-2 전략.

    래리 코너스가 책 'Short Term Trading Strategies That Work'(2008)에서 소개한 단기 전략.
    큰 흐름은 상승(가격 > 장기 평균선)인데 2봉짜리 RSI가 아주 낮으면(짧게 급락) 매수,
    가격이 단기 평균선 위로 회복하면 매도한다. '상승장의 눌림목 사기'.
    """
    name = "connors"
    description = "코너스 RSI-2: 상승 추세 중 짧은 급락(RSI2가 매우 낮음)에 매수, 단기 평균 회복 시 매도"

    def __init__(self, trend: int = 200, low: float = 10, exit_ma: int = 5):
        super().__init__(trend=trend, low=low, exit_ma=exit_ma)
        self.trend, self.low, self.exit_ma = trend, low, exit_ma

    def signals(self, df):
        out = self._empty(df)
        c = df["close"]
        r2 = ind.rsi(c, 2)
        out["buy"] = gt(c, ind.sma(c, self.trend)) & (r2 < self.low)
        out["sell"] = gt(c, ind.sma(c, self.exit_ma))
        return out


class MACDCross(Strategy):
    """MACD 교차 전략.

    제럴드 아펠이 1970년대에 만든 지표. MACD선(12봉 지수평균 - 26봉 지수평균)이
    시그널선(MACD의 9봉 지수평균)을 위로 뚫으면 매수, 아래로 뚫으면 매도한다.
    """
    name = "macd"
    description = "MACD: MACD선이 시그널선을 위로 뚫으면 매수, 아래로 뚫으면 매도"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast, self.slow, self.signal = fast, slow, signal

    def signals(self, df):
        out = self._empty(df)
        c = df["close"]
        macd = ind.ema(c, self.fast) - ind.ema(c, self.slow)
        sig = ind.ema(macd, self.signal)
        above = macd > sig
        prev = above.shift(1, fill_value=False)
        warm = pd.Series(np.arange(len(df)) >= self.slow, index=df.index)
        out["buy"] = above & ~prev & warm
        out["sell"] = ~above & prev
        return out


class KangBreakout(Strategy):
    """강환국식 변동성 돌파 (래리 윌리엄스 방법의 개선형).

    1) k를 고정하지 않고, 최근 noise개 캔들의 평균 '노이즈 비율'로 정한다.
       노이즈 비율 = 1 - |종가-시가| / (고가-저가)  (출렁임이 심할수록 1에 가까움 -> 더 신중하게 매수)
    2) 이동평균 점수: 시가가 직전 3·5·10·20봉 평균보다 높은 개수 / 4 만큼만 산다(0이면 안 삼).
    3) 변동성 조절: 직전 봉 변동폭이 크면 적게 산다. 비중 = target_vol / (직전 변동폭/직전 종가), 최대 1.
    실제 매수 비중 = 이동평균 점수 × 변동성 비중. 산 다음 봉 시가에 판다.
    """
    name = "kang"
    description = "강환국식 변동성 돌파: 노이즈로 k 자동 결정, 이동평균 점수와 변동성으로 매수 비중 조절"
    hold_bars = 1
    MAS = (3, 5, 10, 20)

    def __init__(self, noise: int = 20, target_vol: float = 0.02):
        super().__init__(noise=noise, target_vol=target_vol)
        self.noise, self.target_vol = noise, target_vol

    def signals(self, df):
        out = self._empty(df)
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        rng = h - l
        noise_ratio = (1 - (c - o).abs() / rng).where(rng > 0)
        k = noise_ratio.rolling(self.noise).mean().shift(1)
        prev_range = rng.shift(1)
        target = o + prev_range * k
        score = sum((o > ind.sma(c, n).shift(1)).astype(float) for n in self.MAS) / len(self.MAS)
        vol = prev_range / c.shift(1)
        vol_w = (self.target_vol / vol.where(vol > 0)).clip(upper=1.0)
        size = score * vol_w
        ok = target.notna() & (size > 0)
        out["breakout"] = target.where(ok)
        out["size"] = size.where(ok)
        return out


class CrabelNR7(Strategy):
    """토비 크라벨의 NR7 돌파.

    직전 캔들의 변동폭(고가-저가)이 최근 nr개 캔들 중 가장 좁았다면(시장이 조용했다면)
    이번 캔들에서 '시가 + 스트레치'를 뚫을 때 매수하고 다음 봉 시가에 판다.
    스트레치 = 최근 stretch개 캔들에서 시가와 가까운 쪽 끝(고가 또는 저가)까지 거리의 평균.
    """
    name = "nr7"
    description = "크라벨 NR7: 최근 N봉 중 가장 좁았던 봉 다음에, 시가+스트레치를 뚫으면 매수"
    hold_bars = 1

    def __init__(self, nr: int = 7, stretch: int = 10):
        super().__init__(nr=nr, stretch=stretch)
        self.nr, self.stretch = nr, stretch

    def signals(self, df):
        out = self._empty(df)
        o, h, l = df["open"], df["high"], df["low"]
        rng = h - l
        narrow = (rng <= rng.rolling(self.nr).min()).shift(1, fill_value=False)
        st = np.minimum(h - o, o - l).rolling(self.stretch).mean().shift(1)
        target = o + st
        out["breakout"] = target.where(narrow & target.notna())
        return out


class OpeningRangeBreakout(Strategy):
    """오프닝 레인지 돌파 (ORB, 토비 크라벨).

    업비트 하루가 시작되는 오전 9시부터 or_bars개 캔들(1시간봉이면 처음 1시간)의
    최고가·최저가를 '오프닝 레인지'로 정한다.
    그날 처음으로 그 최고가를 뚫으면 매수(하루 한 번만),
    캔들 종가가 레인지 최저가 밑으로 내려가면 다음 봉 시가에 손절,
    아니면 그날 마지막 캔들 다음(다음날 오전 9시 시가)에 판다.
    """
    name = "orb"
    description = "오프닝 레인지 돌파: 오전 9시 첫 캔들 고가를 그날 처음 뚫으면 매수, 다음날 9시 정리"
    DAY_START_HOUR = 9  # 업비트 일봉이 바뀌는 시각(한국시간)

    def __init__(self, or_bars: int = 1):
        super().__init__(or_bars=or_bars)
        self.or_bars = or_bars

    def signals(self, df):
        out = self._empty(df)
        n = len(df)
        if n == 0:
            return out
        h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
        epoch = ((df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype="int64")
        sec = epoch - self.DAY_START_HOUR * 3600  # 오전 9시를 하루의 시작으로 (한국시간 기준 시각)
        day = sec // 86400
        sod = sec - day * 86400                                     # 그날 9시부터 지난 초
        step = int(np.min(np.diff(sec))) if n > 1 else 3600         # 캔들 하나의 길이(초)
        brk = np.full(n, np.nan)
        or_lo = np.full(n, np.nan)
        cur, pos, valid, done, hi, lo = None, 0, False, False, np.nan, np.nan
        for i in range(n):
            if day[i] != cur:
                cur, pos, done = day[i], 0, False
                valid = sod[i] == 0  # 9시 캔들이 없으면 그날은 쉰다
            if valid:
                if pos < self.or_bars:  # 오프닝 레인지 만드는 중
                    hi = h[i] if pos == 0 else max(hi, h[i])
                    lo = l[i] if pos == 0 else min(lo, l[i])
                else:
                    or_lo[i] = lo
                    if not done:
                        brk[i] = hi
                        if h[i] >= hi:
                            done = True  # 하루 한 번만 (이번 캔들에서 돌파)
            pos += 1
        last_of_day = (sod + step) >= 86400  # 이 캔들이 끝나면 다음날 9시
        out["breakout"] = brk
        out["sell"] = last_of_day | (c < or_lo)  # 그날 마지막 캔들이거나 레인지 최저가 밑으로 마감
        return out


class IBSReversion(Strategy):
    """IBS(Internal Bar Strength) 역추세.

    IBS = (종가-저가) / (고가-저가). 0이면 그날 최저가에 마감, 1이면 최고가에 마감.
    IBS가 low보다 작으면(바닥 근처 마감) '내일 반등'을 노려 다음 봉 시가에 사고, 하루 뒤 시가에 판다.
    """
    name = "ibs"
    description = "IBS 역추세: 캔들이 바닥 근처에서 마감하면 다음 봉에 매수, 하루 뒤 매도"
    hold_bars = 1

    def __init__(self, low: float = 0.2):
        super().__init__(low=low)
        self.low = low

    def signals(self, df):
        out = self._empty(df)
        rng = df["high"] - df["low"]
        ibs = ((df["close"] - df["low"]) / rng).where(rng > 0)
        out["buy"] = ibs < self.low
        return out


STRATEGIES = {
    VolatilityBreakout.name: VolatilityBreakout,
    KangBreakout.name: KangBreakout,
    CrabelNR7.name: CrabelNR7,
    OpeningRangeBreakout.name: OpeningRangeBreakout,
    IBSReversion.name: IBSReversion,
    MACross.name: MACross,
    RSIReversal.name: RSIReversal,
    BollingerReversion.name: BollingerReversion,
    Donchian.name: Donchian,
    ConnorsRSI2.name: ConnorsRSI2,
    MACDCross.name: MACDCross,
}

# 전략 비교 때 시험해 볼 기본 파라미터 조합
DEFAULT_GRID = {
    "breakout": [dict(k=0.3), dict(k=0.5), dict(k=0.7), dict(k=0.5, ma=5)],
    "kang": [dict(noise=20, target_vol=0.02), dict(noise=20, target_vol=0.05)],
    "nr7": [dict(nr=7, stretch=10), dict(nr=4, stretch=10)],
    "orb": [dict(or_bars=1), dict(or_bars=2)],
    "ibs": [dict(low=0.2), dict(low=0.1)],
    "ma_cross": [dict(short=5, long=20), dict(short=10, long=30), dict(short=20, long=60)],
    "rsi": [dict(period=14, low=30, high=70), dict(period=14, low=25, high=60)],
    "bollinger": [dict(period=20, std=2.0), dict(period=20, std=2.5)],
    "donchian": [dict(entry=20, exit=10), dict(entry=55, exit=20)],
    "connors": [dict(trend=200, low=10, exit_ma=5), dict(trend=100, low=5, exit_ma=5)],
    "macd": [dict(fast=12, slow=26, signal=9)],
}


# 메뉴 프로그램에서 보여 줄 전략 이름과 값 설명: (값 이름, 한글 설명, 기본값, 고를 수 있는 값들)
DISPLAY_NAMES = {
    "breakout": "변동성 돌파",
    "kang": "강환국식 변동성 돌파",
    "nr7": "크라벨 NR7",
    "orb": "오프닝 레인지 돌파",
    "ibs": "IBS 역추세",
    "ma_cross": "이동평균 교차",
    "rsi": "RSI 역추세",
    "bollinger": "볼린저 밴드",
    "donchian": "터틀(돈치안 채널)",
    "connors": "코너스 RSI-2",
    "macd": "MACD 교차",
}

PARAM_INFO = {
    "breakout": [
        ("k", "돌파 계수 k (클수록 신중하게 매수)", 0.5, [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("ma", "추세 필터: 최근 N봉 평균 위일 때만 매수 (0 = 끔)", 0, [0, 3, 5, 10, 20]),
    ],
    "kang": [
        ("noise", "k를 정할 때 볼 최근 캔들 수", 20, [10, 20, 30]),
        ("target_vol", "목표 변동성 (클수록 많이 삼)", 0.02, [0.01, 0.02, 0.05, 1.0]),
    ],
    "nr7": [
        ("nr", "최근 N봉 중 가장 좁았으면 준비", 7, [4, 7]),
        ("stretch", "스트레치 평균 기간(봉 개수)", 10, [5, 10, 20]),
    ],
    "orb": [
        ("or_bars", "오전 9시부터 레인지를 만들 캔들 수", 1, [1, 2, 3]),
    ],
    "ibs": [
        ("low", "IBS가 이 값보다 작으면 매수", 0.2, [0.1, 0.15, 0.2, 0.25]),
    ],
    "ma_cross": [
        ("short", "단기 평균선 기간(봉 개수)", 5, [3, 5, 10, 20]),
        ("long", "장기 평균선 기간(봉 개수)", 20, [10, 20, 30, 60, 120]),
    ],
    "rsi": [
        ("period", "RSI 계산 기간(봉 개수)", 14, [7, 9, 14, 21]),
        ("low", "이 값 아래에서 올라오면 매수 (과매도)", 30, [20, 25, 30, 35]),
        ("high", "이 값을 넘으면 매도 (과매수)", 70, [55, 60, 65, 70, 75, 80]),
    ],
    "bollinger": [
        ("period", "평균 기간(봉 개수)", 20, [10, 20, 30, 50]),
        ("std", "밴드 폭 (클수록 더 많이 떨어져야 매수)", 2.0, [1.5, 2.0, 2.5, 3.0]),
    ],
    "donchian": [
        ("entry", "최근 N봉 최고가를 넘으면 매수", 20, [10, 20, 55]),
        ("exit", "최근 M봉 최저가 밑이면 매도", 10, [5, 10, 20]),
    ],
    "connors": [
        ("trend", "상승 추세 판단용 장기 평균(봉 개수)", 200, [50, 100, 200]),
        ("low", "RSI(2)가 이 값보다 낮으면 매수", 10, [5, 10, 15]),
        ("exit_ma", "가격이 이 평균 위로 오면 매도", 5, [3, 5, 10]),
    ],
    "macd": [
        ("fast", "빠른 평균(봉 개수)", 12, [8, 12]),
        ("slow", "느린 평균(봉 개수)", 26, [21, 26]),
        ("signal", "시그널선(봉 개수)", 9, [5, 9]),
    ],
}


def make_strategy(name: str, **params) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략: {name}. 가능: {list(STRATEGIES)}")
    return STRATEGIES[name](**params)
