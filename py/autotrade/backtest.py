"""백테스트: 과거 캔들 데이터로 전략을 돌려 보고 성과를 계산한다."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import Account, Costs, Risk, Trader


@dataclass
class Result:
    label: str
    equity: pd.Series     # 캔들마다의 총자산(현금 + 코인 평가액)
    trades: list
    stats: dict


def run_backtest(df: pd.DataFrame, strategy, initial_cash: float = 1_000_000,
                 costs: Costs | None = None, risk: Risk | None = None,
                 quiet: bool = True, trade_from: int = 0, last_bar_forming: bool = False,
                 close_at_end: bool = True) -> Result:
    """trade_from: 이 번호의 캔들부터 매매(앞쪽은 지표 계산용 준비 데이터).
    last_bar_forming: 마지막 캔들이 아직 진행 중이면 True(그 캔들 끝의 신호는 아직 모름).
    close_at_end: 끝에서 보유 코인을 종가로 정리해 계산할지(모의매매는 False: 보유 유지)."""
    logger = logging.getLogger("autotrade")
    old_level = logger.level
    if quiet:
        logger.setLevel(logging.WARNING)  # 백테스트 중에는 매매 로그를 찍지 않음
    try:
        sig = strategy.signals(df)
        acct = Account(cash=initial_cash, costs=costs or Costs())
        trader = Trader(strategy, acct, risk)
        eq = np.empty(len(df))
        in_pos = np.zeros(len(df), dtype=bool)

        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        buys, sells = sig["buy"].to_numpy(), sig["sell"].to_numpy()
        brk = sig["breakout"].to_numpy(dtype=float)
        times = df.index

        n = len(df)
        for i in range(trade_from, n):
            t = times[i]
            trader.open_bar(t, o[i])
            trader.tick(t, o[i], h[i], l[i], brk[i])
            if not (last_bar_forming and i == n - 1):
                trader.close_bar(t, bool(buys[i]), bool(sells[i]))
            eq[i] = acct.equity(c[i])
            in_pos[i] = acct.in_position

        # 마지막에 들고 있는 코인은 마지막 종가로 정리해서 계산
        if close_at_end and acct.in_position:
            acct.sell(times[-1], c[-1], "백테스트 종료")
            eq[-1] = acct.cash
    finally:
        logger.setLevel(old_level)

    sl = slice(trade_from, None)
    equity = pd.Series(eq[sl], index=df.index[sl])
    stats = compute_stats(equity, acct.trades, df.iloc[sl], initial_cash, in_pos[sl], acct.fees_paid)
    res = Result(strategy.label(), equity, acct.trades, stats)
    res.account, res.trader = acct, trader
    return res


def compute_stats(equity: pd.Series, trades: list, df: pd.DataFrame, initial: float,
                  in_pos: np.ndarray, fees: float) -> dict:
    total_ret = equity.iloc[-1] / initial - 1
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min()
    bh_ret = df["close"].iloc[-1] / df["open"].iloc[0] - 1
    bh_eq = df["close"] / df["open"].iloc[0]
    bh_mdd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()

    rets = np.array([t.ret for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl <= 0)

    return {
        "수익률": total_ret,
        "보유수익률": bh_ret,          # 그냥 처음에 사서 끝까지 들고 있었을 때
        "MDD": mdd,                    # 최대 낙폭: 고점 대비 가장 크게 떨어졌던 비율
        "보유MDD": bh_mdd,
        "거래수": len(trades),
        "승률": len(wins) / len(trades) if trades else 0.0,
        "평균수익": wins.mean() if len(wins) else 0.0,
        "평균손실": losses.mean() if len(losses) else 0.0,
        "손익비": gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0),
        "보유시간비율": in_pos.mean(),  # 전체 기간 중 코인을 들고 있던 시간 비율
        "수수료합계": fees,
        "최종자산": equity.iloc[-1],
    }


def format_stats(s: dict) -> str:
    pf = "∞" if s["손익비"] == np.inf else f"{s['손익비']:.2f}"
    return (
        f"  수익률       {s['수익률']:+.2%}   (그냥 보유했다면 {s['보유수익률']:+.2%})\n"
        f"  최대낙폭(MDD) {s['MDD']:.2%}   (그냥 보유했다면 {s['보유MDD']:.2%})\n"
        f"  거래 횟수     {s['거래수']}회, 승률 {s['승률']:.1%}\n"
        f"  평균 수익/손실 {s['평균수익']:+.2%} / {s['평균손실']:+.2%}, 손익비 {pf}\n"
        f"  보유시간 비율 {s['보유시간비율']:.1%}, 낸 수수료 {s['수수료합계']:,.0f}원\n"
        f"  최종 자산     {s['최종자산']:,.0f}원"
    )


def trades_frame(trades: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "매수시각": t.entry_time, "매수가": t.entry_price,
        "매도시각": t.exit_time, "매도가": t.exit_price,
        "손익(원)": round(t.pnl), "수익률(%)": round(t.ret * 100, 3), "매도사유": t.exit_reason,
    } for t in trades])
