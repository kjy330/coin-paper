"""'시작일부터 다시 계산' 방식의 모의매매.

프로그램을 계속 켜 둘 필요가 없다. 확인할 때마다 업비트에서
'시작 시각 이후의 실제 캔들'을 받아, 시작 시각부터 지금까지 규칙대로 매매했다면
어떻게 됐을지를 처음부터 다시 계산한다. (꺼져 있던 동안의 매매도 빠짐없이 반영)

규칙
- 시작 시각 '이후에 새로 시작한' 캔들부터 매매한다(이미 진행 중이던 캔들은 건너뜀).
- 지표 계산용으로 시작 전 캔들 WARMUP개를 함께 받는다(매매는 하지 않음).
- 지금 진행 중인 캔들은 지금까지의 고가·저가로 돌파·손절만 확인하고,
  캔들 끝 신호는 캔들이 끝난 뒤에 반영한다.
- 같은 계산을 핸드폰 웹페이지(phone_web/index.html)도 똑같이 한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .backtest import run_backtest
from .engine import Costs, Risk
from .presets import PRESET_BY_ID
from .strategies import make_strategy
from .upbit_data import INTERVAL_MINUTES, KST, fetch_since

WARMUP = 260  # 200봉 평균 같은 긴 지표를 위한 준비 캔들 수


def forward_test(preset_id: str, market: str, start: datetime, cash: float = 1_000_000,
                 costs: Costs | None = None, now: datetime | None = None, df: pd.DataFrame | None = None,
                 end: datetime | None = None) -> dict:
    """end가 있으면(중지된 모의매매) 그 시각까지만 계산한다."""
    p = PRESET_BY_ID[preset_id]
    iv = p["interval"]
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[iv])
    start = pd.Timestamp(start)
    now = pd.Timestamp(now or datetime.now(KST).replace(tzinfo=None))
    if df is None:
        df = fetch_since(market, iv, start - step * WARMUP)
    if end is not None:
        now = min(now, pd.Timestamp(end))
        df = df[df.index < now]

    out = {"id": preset_id, "title": p["title"], "market": market, "interval": iv,
           "start": start.isoformat(sep=" ", timespec="minutes"), "cash0": cash,
           "updated": now.isoformat(sep=" ", timespec="minutes")}
    idx = df.index
    after = (idx >= start).nonzero()[0]
    if len(after) == 0:
        out.update(status="waiting", equity=cash, ret=0.0, trades=[], position=None,
                   first_bar=None, message="첫 캔들이 시작되기를 기다리는 중입니다.")
        return out
    trade_from = int(after[0])
    forming = idx[-1] + step > now
    s = make_strategy(p["strategy"], **p["params"])
    risk = Risk(position_size=1.0, stop_loss=p["stop_loss"], take_profit=p["take_profit"])
    r = run_backtest(df, s, cash, costs or Costs(), risk, trade_from=trade_from,
                     last_bar_forming=forming, close_at_end=False)
    acct, trader = r.account, r.trader
    last_price = float(df["close"].iloc[-1])
    eq = acct.equity(last_price)
    first_open = float(df["open"].iloc[trade_from])
    peak = r.equity.cummax()
    pos = None
    if acct.in_position:
        t = acct.open_trade
        pos = {"entry_time": str(t.entry_time), "entry_price": t.entry_price, "qty": t.qty,
               "cost": t.cost, "value": t.qty * last_price,
               "ret": t.qty * last_price * (1 - acct.costs.fee - acct.costs.slippage) / t.cost - 1}
    brk = s.signals(df)["breakout"].iloc[-1] if forming else float("nan")
    # 핸드폰 그래프용 자산 변화(최대 200점으로 줄임)
    k = max(1, len(r.equity) // 200)
    curve = [[str(t), float(v)] for t, v in list(r.equity.items())[::k]]
    if curve and curve[-1][0] != str(r.equity.index[-1]):
        curve.append([str(r.equity.index[-1]), float(r.equity.iloc[-1])])
    out.update(
        curve=curve,
        status="running", first_bar=str(idx[trade_from]), bars=len(df) - trade_from,
        last_price=last_price, equity=eq, ret=eq / cash - 1,
        bh_ret=last_price / first_open - 1,
        mdd=float(((r.equity - peak) / peak).min()),
        fees=acct.fees_paid, position=pos, pending=trader.pending,
        breakout_target=None if brk != brk else float(brk),
        trades=[{"entry_time": str(t.entry_time), "entry_price": t.entry_price,
                 "exit_time": str(t.exit_time), "exit_price": t.exit_price,
                 "pnl": t.pnl, "ret": t.ret, "reason": t.exit_reason} for t in acct.trades],
    )
    return out


def compute_runs(data: dict, costs: Costs | None = None) -> list:
    """runs.json 내용(data)의 모든 모의매매를 계산한다. PC 메뉴와 GitHub 자동 계산이 함께 쓴다."""
    results = []
    for run in data.get("runs", []):
        try:
            r = forward_test(run["id"], run["market"], run["start"], data.get("cash", 1_000_000),
                             costs, end=run.get("end"))
        except Exception as e:  # 네트워크 오류 등
            r = {"id": run["id"], "market": run["market"], "start": run["start"],
                 "status": "error", "message": str(e)[:200]}
        r["end"] = run.get("end")
        results.append(r)
    return results
