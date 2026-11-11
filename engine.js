/* 모의매매 계산 엔진 (Python의 autotrade 패키지를 그대로 옮긴 것).
 * PC 메뉴와 같은 규칙으로 '시작일부터 지금까지' 다시 계산한다. 실제 주문 기능은 없다.
 * 브라우저에서는 window.Engine, Node.js(일치 검사용)에서는 require()로 쓴다. */
(function (root) {
  "use strict";
  var MIN_ORDER_KRW = 5000;
  var WARMUP = 260;
  var IV_MIN = { "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "60m": 60, "240m": 240, "1d": 1440 };
  var IV_PATH = { "1m": "minutes/1", "3m": "minutes/3", "5m": "minutes/5", "10m": "minutes/10", "15m": "minutes/15",
    "30m": "minutes/30", "60m": "minutes/60", "240m": "minutes/240", "1d": "days" };
  var KST_MS = 9 * 3600 * 1000;

  // ---------- 시간: 한국시간을 'UTC처럼' 숫자로 다룬다 (Python의 naive KST와 같은 방식)
  function parseKst(s) { return Date.parse(String(s).replace(" ", "T").slice(0, 19).padEnd(19, ":00") + "Z"); }
  function fmtKst(ms) { return new Date(ms).toISOString().slice(0, 16).replace("T", " "); }
  function nowKst() { return Date.now() + KST_MS; }

  // ---------- 지표 (pandas와 같은 정의)
  function shift(a, k) { var o = new Array(a.length).fill(NaN); for (var i = k; i < a.length; i++) o[i] = a[i - k]; return o; }
  function sma(a, n) {
    var o = new Array(a.length).fill(NaN), s = 0;
    for (var i = 0; i < a.length; i++) { s += a[i]; if (i >= n) s -= a[i - n]; if (i >= n - 1) o[i] = s / n; }
    return o;
  }
  function ema(a, n) {
    var al = 2 / (n + 1), o = new Array(a.length);
    for (var i = 0; i < a.length; i++) o[i] = i === 0 ? a[0] : (1 - al) * o[i - 1] + al * a[i];
    return o;
  }
  function ewmMinPeriods(x, alpha, minp) {
    var o = new Array(x.length).fill(NaN), avg = NaN, cnt = 0;
    for (var i = 0; i < x.length; i++) {
      if (cnt === 0) { if (isNaN(x[i])) continue; avg = x[i]; cnt = 1; }
      else { avg = (1 - alpha) * avg + alpha * x[i]; cnt++; }
      if (cnt >= minp) o[i] = avg;
    }
    return o;
  }
  function rsi(c, n) {
    var up = new Array(c.length), dn = new Array(c.length);
    for (var i = 0; i < c.length; i++) {
      var d = i === 0 ? NaN : c[i] - c[i - 1];
      up[i] = isNaN(d) ? NaN : Math.max(d, 0); dn[i] = isNaN(d) ? NaN : Math.max(-d, 0);
    }
    var au = ewmMinPeriods(up, 1 / n, n), ad = ewmMinPeriods(dn, 1 / n, n);
    return au.map(function (u, i) { var rs = u / ad[i]; return 100 - 100 / (1 + rs); });
  }
  function rollStd(a, n) {
    var o = new Array(a.length).fill(NaN);
    for (var i = n - 1; i < a.length; i++) {
      var m = 0, j; for (j = i - n + 1; j <= i; j++) m += a[j]; m /= n;
      var v = 0; for (j = i - n + 1; j <= i; j++) v += (a[j] - m) * (a[j] - m);
      o[i] = Math.sqrt(v / (n - 1));
    }
    return o;
  }
  function rollExt(a, n, fn) {
    var o = new Array(a.length).fill(NaN);
    for (var i = n - 1; i < a.length; i++) { var v = a[i - n + 1]; for (var j = i - n + 2; j <= i; j++) v = fn(v, a[j]); o[i] = v; }
    return o;
  }

  // pandas rolling(n).mean()과 같음: 창 안에 NaN이 있으면 NaN
  function rollMean(a, n) {
    var o = new Array(a.length).fill(NaN);
    for (var i = n - 1; i < a.length; i++) { var s = 0; for (var j = i - n + 1; j <= i; j++) s += a[j]; o[i] = s / n; }
    return o;
  }

  // 계산 오차 수준의 차이는 '같은 값'으로 본다 (strategies.py의 gt와 같음). NaN이면 false.
  var EPS = 1e-9;
  function gt(a, b) { return (a - b) > EPS * Math.abs(b); }

  // ---------- 전략 (strategies.py와 같은 규칙)
  function empty(n) { return { buy: new Array(n).fill(false), sell: new Array(n).fill(false), breakout: new Array(n).fill(NaN) }; }
  var STRATEGIES = {
    breakout: { hold: 1, fn: function (d, p) {
      var n = d.o.length, out = empty(n), k = p.k == null ? 0.5 : p.k, ma = p.ma || 0;
      var trend = ma > 0 ? shift(sma(d.c, ma), 1) : null;
      for (var i = 1; i < n; i++) {
        var tgt = d.o[i] + (d.h[i - 1] - d.l[i - 1]) * k;
        var ok = !isNaN(tgt) && (!trend || d.o[i] > trend[i]);
        out.breakout[i] = ok ? tgt : NaN;
      }
      return out;
    } },
    kang: { hold: 1, fn: function (d, p) {
      var n = d.o.length, out = empty(n), N = p.noise || 20, tv = p.target_vol == null ? 0.02 : p.target_vol, i;
      out.size = new Array(n).fill(NaN);
      var rng = [], noise = [];
      for (i = 0; i < n; i++) { rng[i] = d.h[i] - d.l[i]; noise[i] = rng[i] > 0 ? 1 - Math.abs(d.c[i] - d.o[i]) / rng[i] : NaN; }
      var kk = shift(rollMean(noise, N), 1);
      var mas = [3, 5, 10, 20].map(function (m) { return shift(sma(d.c, m), 1); });
      for (i = 1; i < n; i++) {
        var tgt = d.o[i] + rng[i - 1] * kk[i];
        var score = 0; mas.forEach(function (m) { if (d.o[i] > m[i]) score++; }); score /= mas.length;
        var vol = rng[i - 1] / d.c[i - 1], vw = vol > 0 ? Math.min(1, tv / vol) : NaN, size = score * vw;
        if (!isNaN(tgt) && size > 0) { out.breakout[i] = tgt; out.size[i] = size; }
      }
      return out;
    } },
    nr7: { hold: 1, fn: function (d, p) {
      var n = d.o.length, out = empty(n), nr = p.nr || 7, sn = p.stretch || 10, i, rng = [], st0 = [];
      for (i = 0; i < n; i++) { rng[i] = d.h[i] - d.l[i]; st0[i] = Math.min(d.h[i] - d.o[i], d.o[i] - d.l[i]); }
      var rmin = rollExt(rng, nr, Math.min), st = shift(rollMean(st0, sn), 1);
      for (i = 1; i < n; i++) {
        var narrow = rng[i - 1] <= rmin[i - 1], tgt = d.o[i] + st[i];
        if (narrow && !isNaN(tgt)) out.breakout[i] = tgt;
      }
      return out;
    } },
    orb: { hold: null, fn: function (d, p) {
      var n = d.o.length, out = empty(n), ob = p.or_bars || 1, i, sec = [], day = [], sod = [];
      if (!n) return out;
      for (i = 0; i < n; i++) { sec[i] = Math.round(d.t[i] / 1000) - 9 * 3600; day[i] = Math.floor(sec[i] / 86400); sod[i] = sec[i] - day[i] * 86400; }
      var step = 3600;
      if (n > 1) { step = Infinity; for (i = 1; i < n; i++) step = Math.min(step, sec[i] - sec[i - 1]); }
      var cur = null, pos = 0, valid = false, done = false, hi = NaN, lo = NaN, orLo = new Array(n).fill(NaN);
      for (i = 0; i < n; i++) {
        if (day[i] !== cur) { cur = day[i]; pos = 0; done = false; valid = sod[i] === 0; }
        if (valid) {
          if (pos < ob) { hi = pos === 0 ? d.h[i] : Math.max(hi, d.h[i]); lo = pos === 0 ? d.l[i] : Math.min(lo, d.l[i]); }
          else { orLo[i] = lo; if (!done) { out.breakout[i] = hi; if (d.h[i] >= hi) done = true; } }
        }
        pos++;
      }
      for (i = 0; i < n; i++) out.sell[i] = (sod[i] + step) >= 86400 || d.c[i] < orLo[i];
      return out;
    } },
    ibs: { hold: 1, fn: function (d, p) {
      var n = d.o.length, out = empty(n), lo = p.low == null ? 0.2 : p.low;
      for (var i = 0; i < n; i++) { var r = d.h[i] - d.l[i]; out.buy[i] = r > 0 && (d.c[i] - d.l[i]) / r < lo; }
      return out;
    } },
    ma_cross: { hold: null, fn: function (d, p) {
      var n = d.c.length, out = empty(n), s = sma(d.c, p.short || 5), l = sma(d.c, p.long || 20), prev = false;
      for (var i = 0; i < n; i++) {
        var above = gt(s[i], l[i]);
        out.buy[i] = above && !prev && !isNaN(l[i]); out.sell[i] = !above && prev; prev = above;
      }
      return out;
    } },
    rsi: { hold: null, fn: function (d, p) {
      var n = d.c.length, out = empty(n), r = rsi(d.c, p.period || 14), lo = p.low == null ? 30 : p.low, hi = p.high == null ? 70 : p.high;
      for (var i = 0; i < n; i++) { var pr = i > 0 ? r[i - 1] : NaN; out.buy[i] = pr < lo && r[i] >= lo; out.sell[i] = r[i] > hi; }
      return out;
    } },
    bollinger: { hold: null, fn: function (d, p) {
      var n = d.c.length, out = empty(n), per = p.period || 20, k = p.std == null ? 2 : p.std, mid = sma(d.c, per), sd = rollStd(d.c, per);
      for (var i = 0; i < n; i++) { out.buy[i] = gt(mid[i] - k * sd[i], d.c[i]); out.sell[i] = gt(d.c[i], mid[i]); }
      return out;
    } },
    donchian: { hold: null, fn: function (d, p) {
      var n = d.c.length, out = empty(n);
      var hi = shift(rollExt(d.h, p.entry || 20, Math.max), 1), lo = shift(rollExt(d.l, p.exit || 10, Math.min), 1);
      for (var i = 0; i < n; i++) { out.buy[i] = d.c[i] > hi[i]; out.sell[i] = d.c[i] < lo[i]; }
      return out;
    } },
    connors: { hold: null, fn: function (d, p) {
      var n = d.c.length, out = empty(n), r2 = rsi(d.c, 2), tr = sma(d.c, p.trend || 200), ex = sma(d.c, p.exit_ma || 5), lo = p.low == null ? 10 : p.low;
      for (var i = 0; i < n; i++) { out.buy[i] = gt(d.c[i], tr[i]) && r2[i] < lo; out.sell[i] = gt(d.c[i], ex[i]); }
      return out;
    } },
    macd: { hold: null, fn: function (d, p) {
      var n = d.c.length, out = empty(n), fast = ema(d.c, p.fast || 12), slow = p.slow || 26, sl = ema(d.c, slow);
      var m = fast.map(function (v, i) { return v - sl[i]; }), sg = ema(m, p.signal || 9), prev = false;
      for (var i = 0; i < n; i++) {
        var above = m[i] > sg[i];
        out.buy[i] = above && !prev && i >= slow; out.sell[i] = !above && prev; prev = above;
      }
      return out;
    } }
  };

  // ---------- 가상 계좌와 매매 진행 (engine.py와 같은 순서)
  function Account(cash, costs) { this.cash = cash; this.costs = costs; this.qty = 0; this.open = null; this.trades = []; this.fees = 0; }
  Account.prototype.equity = function (price) { return this.cash + this.qty * price; };
  Account.prototype.buy = function (t, price, size, reason) {
    var spend = this.cash * size;
    if (spend < MIN_ORDER_KRW || this.open) return false;
    var fill = price * (1 + this.costs.slippage), fee = spend * this.costs.fee, qty = (spend - fee) / fill;
    this.cash -= spend; this.qty = qty; this.fees += fee;
    this.open = { entry_time: t, entry_price: fill, qty: qty, cost: spend };
    return true;
  };
  Account.prototype.sell = function (t, price, reason) {
    if (!this.open) return false;
    var fill = price * (1 - this.costs.slippage), gross = this.qty * fill, fee = gross * this.costs.fee;
    this.cash += gross - fee; this.fees += fee;
    var tr = this.open;
    tr.exit_time = t; tr.exit_price = fill; tr.proceeds = gross - fee; tr.reason = reason;
    tr.pnl = tr.proceeds - tr.cost; tr.ret = tr.pnl / tr.cost;
    this.trades.push(tr); this.qty = 0; this.open = null;
    return true;
  };
  function pct1(x) { return (x * 100).toFixed(1) + "%"; }

  function Trader(hold, acct, risk) { this.hold = hold; this.a = acct; this.r = risk; this.pending = null; this.held = 0; this.entered = false; }
  Trader.prototype.expired = function () { return this.hold != null && this.held >= this.hold; };
  Trader.prototype.openBar = function (t, o) {
    this.entered = false;
    if (this.pending === "buy") { if (this.a.buy(t, o, this.r.position_size, "매수 신호")) this.held = 0; }
    else if (this.pending === "sell") { this.a.sell(t, o, this.expired() ? "보유기간 종료" : "매도 신호"); }
    this.pending = null;
  };
  Trader.prototype.tick = function (t, o, h, l, brk, size) {
    var a = this.a, r = this.r;
    if (a.open && !this.entered) {
      var e = a.open.entry_price;
      if (r.stop_loss > 0) { var st = e * (1 - r.stop_loss); if (l <= st) { a.sell(t, Math.min(o, st), "손절 " + pct1(r.stop_loss)); return; } }
      if (r.take_profit > 0) { var tp = e * (1 + r.take_profit); if (h >= tp) { a.sell(t, Math.max(o, tp), "익절 " + pct1(r.take_profit)); return; } }
    }
    if (!a.open && !isNaN(brk) && h >= brk) {
      var frac = (size == null || isNaN(size)) ? r.position_size : r.position_size * size;
      if (a.buy(t, Math.max(o, brk), frac, "돌파가 도달")) { this.held = 0; this.entered = true; }
    }
  };
  Trader.prototype.closeBar = function (buy, sell) {
    if (this.a.open) { this.held++; if (this.expired() || sell) this.pending = "sell"; }
    else if (buy) this.pending = "buy";
  };

  // ---------- 시작일부터 다시 계산 (forward.py와 같은 규칙)
  // candles: [{t: ms(한국시간 기준), o, h, l, c}] 오래된 순
  function forwardTest(preset, market, candles, startMs, nowMs, cash, costs, endMs) {
    var step = IV_MIN[preset.interval] * 60000;
    if (endMs != null) { nowMs = Math.min(nowMs, endMs); candles = candles.filter(function (x) { return x.t < nowMs; }); }
    var out = { id: preset.id, title: preset.title, market: market, interval: preset.interval,
      start: fmtKst(startMs), cash0: cash, updated: fmtKst(nowMs) };
    var from = -1;
    for (var i = 0; i < candles.length; i++) if (candles[i].t >= startMs) { from = i; break; }
    if (from < 0) { out.status = "waiting"; out.equity = cash; out.ret = 0; out.trades = []; out.position = null; return out; }
    var n = candles.length, forming = candles[n - 1].t + step > nowMs;
    var d = { t: [], o: [], h: [], l: [], c: [] };
    candles.forEach(function (x) { d.t.push(x.t); d.o.push(x.o); d.h.push(x.h); d.l.push(x.l); d.c.push(x.c); });
    var strat = STRATEGIES[preset.strategy], sig = strat.fn(d, preset.params || {});
    var acct = new Account(cash, costs);
    var tr = new Trader(strat.hold, acct, { position_size: 1, stop_loss: preset.stop_loss || 0, take_profit: preset.take_profit || 0 });
    var peak = -Infinity, mdd = 0, curve = [];
    for (i = from; i < n; i++) {
      var t = fmtKst(candles[i].t);
      tr.openBar(t, d.o[i]);
      tr.tick(t, d.o[i], d.h[i], d.l[i], sig.breakout[i], sig.size ? sig.size[i] : NaN);
      if (!(forming && i === n - 1)) tr.closeBar(sig.buy[i], sig.sell[i]);
      var eq = acct.equity(d.c[i]);
      curve.push([candles[i].t, eq]);
      peak = Math.max(peak, eq); mdd = Math.min(mdd, (eq - peak) / peak);
    }
    var last = d.c[n - 1], equity = acct.equity(last), pos = null;
    if (acct.open) {
      var o = acct.open;
      pos = { entry_time: o.entry_time, entry_price: o.entry_price, qty: o.qty, cost: o.cost, value: o.qty * last,
        ret: o.qty * last * (1 - costs.fee - costs.slippage) / o.cost - 1 };
    }
    var brk = forming ? sig.breakout[n - 1] : NaN;
    out.status = "running"; out.first_bar = fmtKst(candles[from].t); out.bars = n - from;
    out.last_price = last; out.equity = equity; out.ret = equity / cash - 1;
    out.bh_ret = last / d.o[from] - 1; out.mdd = mdd; out.fees = acct.fees; out.position = pos;
    out.pending = tr.pending; out.breakout_target = isNaN(brk) ? null : brk;
    out.trades = acct.trades; out.curve = curve;
    return out;
  }

  // ---------- 업비트 공개 시세 받기 (API 키 불필요)
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  // 업비트는 요청이 몰리면 잠깐 거절(429)하는데, 그 응답에는 CORS 허용 헤더가 없어
  // 브라우저에서는 '네트워크 오류'로 보인다. 그래서 오류가 나면 점점 길게 쉬었다가 다시 시도한다.
  var lastCall = 0;
  async function getJson(url) {
    var waits = [600, 1200, 2500, 4000, 6000, 9000];
    for (var k = 0; k <= waits.length; k++) {
      var gap = Date.now() - lastCall;
      if (gap < 1100) await sleep(1100 - gap);  // 브라우저 요청은 업비트가 더 엄격히 제한하므로 넉넉히 띄움
      lastCall = Date.now();
      try {
        var res = await fetch(url, { cache: "no-store" });
        if (res.status === 429 || res.status >= 500) throw new Error("HTTP " + res.status);
        if (!res.ok) throw new Error("HTTP " + res.status);
        return await res.json();
      } catch (e) { if (k === waits.length) throw e; await sleep(waits[k]); }
    }
  }
  async function fetchSince(market, interval, sinceMs) {
    var all = [], to = null;
    for (var guard = 0; guard < 60; guard++) {
      var url = "https://api.upbit.com/v1/candles/" + IV_PATH[interval] + "?market=" + market + "&count=200" +
        (to ? "&to=" + encodeURIComponent(new Date(to - KST_MS).toISOString().slice(0, 19) + "Z") : "");
      var rows = await getJson(url);
      if (!rows || !rows.length) break;
      var batch = rows.map(function (r) { return { t: parseKst(r.candle_date_time_kst), o: r.opening_price, h: r.high_price, l: r.low_price, c: r.trade_price }; });
      batch.sort(function (a, b) { return a.t - b.t; });
      all = batch.concat(all);
      if (batch[0].t <= sinceMs || rows.length < 200) break;
      to = batch[0].t;
    }
    var seen = {}, out = [];
    all.sort(function (a, b) { return a.t - b.t; }).forEach(function (x) { if (!seen[x.t] && x.t >= sinceMs) { seen[x.t] = 1; out.push(x); } });
    return out;
  }
  async function runLive(preset, market, startStr, cash, costs, endStr) {
    var startMs = parseKst(startStr), step = IV_MIN[preset.interval] * 60000;
    var candles = await fetchSince(market, preset.interval, startMs - step * WARMUP);
    return forwardTest(preset, market, candles, startMs, nowKst(), cash, costs, endStr ? parseKst(endStr) : null);
  }

  var api = { forwardTest: forwardTest, runLive: runLive, fetchSince: fetchSince, parseKst: parseKst, fmtKst: fmtKst,
    nowKst: nowKst, STRATEGIES: STRATEGIES, WARMUP: WARMUP, IV_MIN: IV_MIN };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Engine = api;
})(this);
