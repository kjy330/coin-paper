"""유명 트레이더·책에서 공개된 전략을 '그대로 골라 쓸 수 있게' 미리 맞춰 둔 설정(프리셋).

전략 종류, 세부 값, 캔들 단위, 시험 기간, 손절/익절까지 모두 정해져 있어서
사용자는 고르기만 하면 된다.

주의: 대부분 원래 주식·선물용으로 만들어졌고, 코인(24시간 거래, 큰 변동성)에 옮겨 온 것이다.
유명하다고 지금 코인 시장에서 돈을 번다는 보장은 없다 → recommend.py로 직접 확인한다.
"""

PRESETS = [
    {
        "id": "williams_breakout",
        "title": "변동성 돌파 (래리 윌리엄스)",
        "origin": "래리 윌리엄스, 1987년 선물 트레이딩 대회 우승자. 국내에서는 코인 자동매매 입문 전략으로 가장 널리 알려짐",
        "strategy": "breakout", "params": {"k": 0.5, "ma": 5},
        "interval": "1d", "days": 365, "stop_loss": 0.0, "take_profit": 0.0,
        "style": "하루 단타 (매일 아침 9시에 정리)",
        "how": "어제 하루 움직인 폭의 절반만큼 오늘 오르면 '오늘은 오르는 날'로 보고 산다. "
               "최근 5일 평균보다 높을 때만 사고, 다음날 아침 9시에 무조건 판다.",
        "risk_note": "하루만 들고 있어서 별도 손절이 없다(다음날 아침 자동 정리).",
    },
    {
        "id": "turtle",
        "title": "터틀 트레이딩 (리처드 데니스)",
        "origin": "1983년 리처드 데니스가 초보자 20여 명에게 가르쳐 큰 수익을 낸 것으로 유명한 규칙",
        "strategy": "donchian", "params": {"entry": 20, "exit": 10},
        "interval": "240m", "days": 180, "stop_loss": 0.05, "take_profit": 0.0,
        "style": "추세 따라가기 (며칠 ~ 몇 주 보유)",
        "how": "최근 20봉(4시간봉 기준 약 3일) 중 가장 높은 가격을 넘으면 큰 상승의 시작으로 보고 산다. "
               "최근 10봉 중 가장 낮은 가격 밑으로 떨어지면 판다.",
        "risk_note": "원래 규칙의 손절(변동폭 2배)을 단순화해 5% 손절을 넣었다. 원래 일봉용을 4시간봉에 맞췄다.",
    },
    {
        "id": "connors_rsi2",
        "title": "RSI-2 눌림목 매수 (래리 코너스)",
        "origin": "래리 코너스의 책 『Short Term Trading Strategies That Work』(2008)에 소개된 단기 전략",
        "strategy": "connors", "params": {"trend": 200, "low": 10, "exit_ma": 5},
        "interval": "60m", "days": 120, "stop_loss": 0.0, "take_profit": 0.0,
        "style": "짧은 반등 노리기 (몇 시간 ~ 하루)",
        "how": "큰 흐름은 오르는 중(최근 200봉 평균보다 위)인데 잠깐 확 떨어졌을 때 사고, "
               "최근 5봉 평균 위로 돌아오면 판다. 상승장에서 '세일할 때 사기'.",
        "risk_note": "코너스는 손절이 오히려 성과를 떨어뜨렸다고 해서 손절을 넣지 않았다. 원래 일봉용을 1시간봉에 맞췄다.",
    },
    {
        "id": "macd",
        "title": "MACD 교차 (제럴드 아펠)",
        "origin": "1970년대 제럴드 아펠이 만든 지표. 증권사 앱에 기본으로 들어 있을 만큼 대중적",
        "strategy": "macd", "params": {"fast": 12, "slow": 26, "signal": 9},
        "interval": "240m", "days": 180, "stop_loss": 0.05, "take_profit": 0.0,
        "style": "추세 따라가기 (하루 ~ 며칠)",
        "how": "단기 흐름이 장기 흐름보다 강해지는 순간(MACD선이 시그널선을 위로 돌파) 사고, 약해지면 판다.",
        "risk_note": "갑작스런 급락에 대비해 5% 손절을 넣었다.",
    },
    {
        "id": "golden_cross",
        "title": "골든크로스 50/200",
        "origin": "가장 오래되고 유명한 이동평균 규칙. 뉴스에서 '골든크로스 발생'이라고 하는 바로 그것",
        "strategy": "ma_cross", "params": {"short": 50, "long": 200},
        "interval": "240m", "days": 365, "stop_loss": 0.0, "take_profit": 0.0,
        "style": "느린 추세 따라가기 (몇 주 보유, 거래 적음)",
        "how": "50봉 평균선이 200봉 평균선을 위로 뚫으면(골든크로스) 사고, 아래로 뚫으면(데드크로스) 판다.",
        "risk_note": "거래가 매우 드물다. 단타보다는 '큰 하락을 피하는' 용도에 가깝다.",
    },
    {
        "id": "bollinger",
        "title": "볼린저 밴드 평균회귀 (존 볼린저)",
        "origin": "1980년대 존 볼린저가 만든 밴드 지표를 이용한 대표적인 '많이 내리면 사기' 방식",
        "strategy": "bollinger", "params": {"period": 20, "std": 2.0},
        "interval": "60m", "days": 120, "stop_loss": 0.03, "take_profit": 0.0,
        "style": "짧은 반등 노리기 (몇 시간 ~ 하루)",
        "how": "가격이 평소 움직이는 범위(밴드) 아래로 벗어나면 사고, 평균으로 돌아오면 판다.",
        "risk_note": "계속 떨어지는 장에서 크게 물리지 않도록 3% 손절을 넣었다.",
    },
    {
        "id": "rsi_wilder",
        "title": "RSI 30/70 (웰레스 와일더)",
        "origin": "1978년 웰레스 와일더가 만든 RSI 지표의 교과서적인 사용법",
        "strategy": "rsi", "params": {"period": 14, "low": 30, "high": 70},
        "interval": "60m", "days": 120, "stop_loss": 0.03, "take_profit": 0.0,
        "style": "짧은 반등 노리기 (몇 시간 ~ 며칠)",
        "how": "많이 팔려서 RSI가 30 아래로 갔다가 다시 올라오면 사고, 70 위(많이 오름)면 판다.",
        "risk_note": "3% 손절을 넣었다.",
    },
]

PRESET_BY_ID = {p["id"]: p for p in PRESETS}
