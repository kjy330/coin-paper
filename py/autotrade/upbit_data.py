"""업비트 공개 시세(캔들) API에서 데이터를 받아오는 모듈.

- 시세 조회는 API 키가 필요 없는 '공개 API'라서 계정 없이도 동작한다.
- 업비트는 한 번에 최대 200개 캔들만 주므로, 여러 번 나눠서 받아 이어 붙인다.
- 받은 데이터는 data/ 폴더에 CSV로 저장해 두고 다음에 재사용한다(캐시).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_URL = "https://api.upbit.com/v1"
KST = timezone(timedelta(hours=9))

# 사람이 읽기 쉬운 이름 -> 업비트 API 경로
INTERVALS = {
    "1m": "minutes/1",
    "3m": "minutes/3",
    "5m": "minutes/5",
    "10m": "minutes/10",
    "15m": "minutes/15",
    "30m": "minutes/30",
    "60m": "minutes/60",
    "240m": "minutes/240",
    "1d": "days",
}

INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "60m": 60, "240m": 240, "1d": 1440,
}

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})


def _get(path: str, params: dict, retries: int = 5):
    """GET 요청. 요청이 너무 많다는 응답(429)이면 잠시 쉬었다가 다시 시도한다."""
    url = f"{BASE_URL}/{path}"
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=10)
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 429:
            time.sleep(0.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"업비트 API 요청 실패: {url} {params}")


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.rename(columns={
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    # 시각은 한국시간(KST) 기준 캔들 시작 시각
    df["time"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
    return df.sort_index()


def fetch_candles(market: str, interval: str, count: int = 200, to: datetime | None = None) -> pd.DataFrame:
    """최신 캔들 최대 200개를 받는다. 맨 마지막 행은 '아직 진행 중인' 캔들일 수 있다."""
    if interval not in INTERVALS:
        raise ValueError(f"지원하지 않는 interval: {interval}. 가능: {list(INTERVALS)}")
    params = {"market": market, "count": min(count, 200)}
    if to is not None:
        params["to"] = to.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = _get(f"candles/{INTERVALS[interval]}", params)
    return _to_frame(rows)


def fetch_history(market: str, interval: str, days: int, cache_dir: str = "data", refresh: bool = False) -> pd.DataFrame:
    """최근 `days`일치 캔들을 받아온다(여러 번 나눠 요청). CSV 캐시를 사용한다."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{market}_{interval}_{days}d.csv")
    if not refresh and os.path.exists(cache):
        age_h = (time.time() - os.path.getmtime(cache)) / 3600
        if age_h < 6:  # 6시간 이내에 받은 데이터면 재사용
            return pd.read_csv(cache, index_col="time", parse_dates=True)

    need = int(days * 1440 / INTERVAL_MINUTES[interval])
    frames = []
    to = None
    got = 0
    while got < need:
        df = fetch_candles(market, interval, 200, to)
        if df.empty:
            break
        frames.append(df)
        got += len(df)
        # 다음 요청은 이번에 받은 가장 오래된 캔들보다 이전 데이터
        oldest_kst = df.index[0].to_pydatetime().replace(tzinfo=KST)
        to = oldest_kst
        time.sleep(0.12)  # 초당 요청 제한(약 10회)을 지키기 위해
        if len(df) < 200:
            break
    if not frames:
        raise RuntimeError(f"{market} {interval} 데이터를 받지 못했습니다.")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    cutoff = out.index[-1] - pd.Timedelta(days=days)
    out = out[out.index > cutoff]
    out.to_csv(cache, index_label="time")
    return out


def fetch_since(market: str, interval: str, since: datetime) -> pd.DataFrame:
    """since(한국시간) 이후의 모든 캔들 + 지금 진행 중인 캔들까지 받는다(캐시 없음)."""
    since = pd.Timestamp(since)
    frames, to = [], None
    while True:
        df = fetch_candles(market, interval, 200, to)
        if df.empty:
            break
        frames.append(df)
        if df.index[0] <= since or len(df) < 200:
            break
        to = df.index[0].to_pydatetime().replace(tzinfo=KST)
        time.sleep(0.12)
    if not frames:
        raise RuntimeError(f"{market} {interval} 데이터를 받지 못했습니다.")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[out.index >= since]


def fetch_price(market: str) -> float:
    """현재가(가장 최근 체결가)."""
    rows = _get("ticker", {"markets": market})
    return float(rows[0]["trade_price"])


def top_krw_markets(n: int = 10) -> list[str]:
    """원화(KRW) 마켓 중 24시간 거래대금 상위 n개 종목."""
    markets = _get("market/all", {"isDetails": "false"})
    krw = [m["market"] for m in markets if m["market"].startswith("KRW-")]
    tickers = []
    for i in range(0, len(krw), 100):
        tickers += _get("ticker", {"markets": ",".join(krw[i:i + 100])})
        time.sleep(0.12)
    tickers.sort(key=lambda t: t.get("acc_trade_price_24h", 0), reverse=True)
    return [t["market"] for t in tickers[:n]]
