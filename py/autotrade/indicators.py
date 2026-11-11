"""기술적 지표 계산 함수 모음. (pandas Series를 받아 Series를 돌려준다)"""
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    """단순 이동평균: 최근 n개 가격의 평균."""
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """지수 이동평균: 최근 가격에 더 큰 가중치를 주는 평균."""
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI(상대강도지수, 0~100). 보통 30 아래면 '과매도', 70 위면 '과매수'로 본다."""
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down
    return 100 - 100 / (1 + rs)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """볼린저 밴드: (중심선, 위 밴드, 아래 밴드). 중심선은 이동평균, 밴드는 ±k 표준편차."""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid, mid + k * std, mid - k * std
