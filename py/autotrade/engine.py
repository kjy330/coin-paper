"""매매 엔진: 가상 계좌와 '언제 사고 파는지'를 처리하는 공통 로직.

백테스트(과거 데이터)와 모의매매(실시간)가 똑같은 코드를 쓰도록 만들어서
두 결과가 서로 비교 가능하도록 했다.

한 캔들(봉)을 처리하는 순서:
  1) open_bar  : 캔들 시작. 직전 캔들 끝에 예약된 매수/매도를 '시가'로 체결
  2) tick      : 캔들 진행 중. 손절/익절 확인, 돌파 가격 도달 시 매수
  3) close_bar : 캔들 끝. 전략 신호를 보고 다음 캔들 시가에 할 주문을 예약
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("autotrade")

MIN_ORDER_KRW = 5000  # 업비트 원화마켓 최소 주문 금액


@dataclass
class Costs:
    fee: float = 0.0005       # 거래 수수료(업비트 원화마켓 0.05%)
    slippage: float = 0.0005  # 슬리피지: 원하는 가격보다 불리하게 체결되는 정도(가정)


@dataclass
class Risk:
    position_size: float = 1.0  # 매수할 때 현금의 몇 %를 쓸지(1.0 = 100%)
    stop_loss: float = 0.0      # 손절 비율(0.02 = 산 가격보다 2% 내리면 팔기, 0이면 사용 안 함)
    take_profit: float = 0.0    # 익절 비율(0.03 = 3% 오르면 팔기, 0이면 사용 안 함)


@dataclass
class Trade:
    entry_time: object
    entry_price: float
    exit_time: object = None
    exit_price: float = 0.0
    qty: float = 0.0
    cost: float = 0.0       # 매수에 쓴 돈(수수료 포함)
    proceeds: float = 0.0   # 매도로 받은 돈(수수료 차감)
    exit_reason: str = ""

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost

    @property
    def ret(self) -> float:
        return self.pnl / self.cost if self.cost else 0.0


@dataclass
class Account:
    cash: float
    costs: Costs = field(default_factory=Costs)
    qty: float = 0.0
    open_trade: Trade | None = None
    trades: list = field(default_factory=list)
    fees_paid: float = 0.0

    @property
    def in_position(self) -> bool:
        return self.open_trade is not None

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def buy(self, time, price: float, size: float, reason: str = "") -> bool:
        spend = self.cash * size
        if spend < MIN_ORDER_KRW or self.in_position:
            return False
        fill = price * (1 + self.costs.slippage)
        fee = spend * self.costs.fee
        qty = (spend - fee) / fill
        self.cash -= spend
        self.qty = qty
        self.fees_paid += fee
        self.open_trade = Trade(entry_time=time, entry_price=fill, qty=qty, cost=spend)
        log.info("매수 %s | 가격 %s | 금액 %s원 | 사유: %s", time, f"{fill:,.2f}", f"{spend:,.0f}", reason)
        return True

    def sell(self, time, price: float, reason: str = "") -> bool:
        if not self.in_position:
            return False
        fill = price * (1 - self.costs.slippage)
        gross = self.qty * fill
        fee = gross * self.costs.fee
        self.cash += gross - fee
        self.fees_paid += fee
        t = self.open_trade
        t.exit_time, t.exit_price, t.proceeds, t.exit_reason = time, fill, gross - fee, reason
        self.trades.append(t)
        self.qty = 0.0
        self.open_trade = None
        log.info("매도 %s | 가격 %s | 손익 %s원 (%+.2f%%) | 사유: %s",
                 time, f"{fill:,.2f}", f"{t.pnl:,.0f}", t.ret * 100, reason)
        return True


class Trader:
    """전략 + 계좌 + 위험관리를 묶어서 캔들 단위로 매매를 진행한다."""

    def __init__(self, strategy, account: Account, risk: Risk | None = None):
        self.strategy = strategy
        self.acct = account
        self.risk = risk or Risk()
        self.pending: str | None = None  # 다음 캔들 시가에 할 주문: "buy" / "sell" / None
        self.bars_held = 0
        self.entered_this_bar = False

    # 1) 캔들 시작
    def open_bar(self, time, open_price: float):
        self.entered_this_bar = False
        if self.pending == "buy":
            if self.acct.buy(time, open_price, self.risk.position_size, "매수 신호"):
                self.bars_held = 0
        elif self.pending == "sell":
            reason = "보유기간 종료" if self._hold_expired() else "매도 신호"
            self.acct.sell(time, open_price, reason)
        self.pending = None

    # 2) 캔들 진행 중 (백테스트에서는 해당 캔들의 고가/저가, 모의매매에서는 현재가를 넣는다)
    def tick(self, time, open_price: float, high: float, low: float, breakout: float | None = None):
        a, r = self.acct, self.risk
        if a.in_position and not self.entered_this_bar:
            entry = a.open_trade.entry_price
            if r.stop_loss > 0:
                stop = entry * (1 - r.stop_loss)
                if low <= stop:
                    # 시가부터 이미 손절가 아래로 시작했다면(갭 하락) 시가에 팔린 것으로 본다
                    a.sell(time, min(open_price, stop), f"손절 {r.stop_loss:.1%}")
                    return
            if r.take_profit > 0:
                tp = entry * (1 + r.take_profit)
                if high >= tp:
                    a.sell(time, max(open_price, tp), f"익절 {r.take_profit:.1%}")
                    return
        if (not a.in_position and breakout is not None and breakout == breakout  # NaN 제외
                and high >= breakout):
            price = max(open_price, breakout)
            if a.buy(time, price, r.position_size, f"돌파가 {breakout:,.2f} 도달"):
                self.bars_held = 0
                self.entered_this_bar = True

    # 3) 캔들 끝
    def close_bar(self, time, buy: bool, sell: bool):
        if self.acct.in_position:
            self.bars_held += 1
            if self._hold_expired() or sell:
                self.pending = "sell"
        elif buy:
            self.pending = "buy"

    def _hold_expired(self) -> bool:
        hb = self.strategy.hold_bars
        return hb is not None and self.bars_held >= hb
