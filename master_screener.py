"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   MASTER SCREENER  v1.0  —  SVERIGE + EUROPA                                ║
║   Extraherad ur Master Backtest v2.0 — ren screener utan backtest           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INDIKATORER: RSI, StochRSI, Bollinger Squeeze, ATR, ADX, CMF, OBV,        ║
║               EMA-stack, SMA200/SMA50, RS, RS-acc, 6m/12m momentum,        ║
║               riskjusterat MomRank, ATR-expansion, Institutionellt flöde    ║
║                                                                              ║
║  FILTERKRAV PER HORISONT (alla måste uppfyllas):                            ║
║    Korttid: score≥50, kt≥45, rs≥-2, adx≥15, SMA200                         ║
║    Medel:   score≥56, rs≥1.0, adx≥18, p6m≥-2%, SMA200                      ║
║    Swing:   score≥60, rs≥2.0, adx≥20, p6m≥+2%, SMA200+SMA50                ║
║                                                                              ║
║  KÖRKOMANDON:                                                                ║
║    python master_screener.py                    # screena idag               ║
║    python master_screener.py --topn 30          # visa topp 30 per hor.     ║
║    python master_screener.py --land SE NO DK    # filtrera länder            ║
║    python master_screener.py --horisont swing   # bara swing-kandidater     ║
║    python master_screener.py --min-score 60 --min-rs 3.0                   ║
║    python master_screener.py --datum 2024-01-15 # historisk screening       ║
║    python master_screener.py --alla             # alla kandidater (ej top N) ║
║    python master_screener.py --ingen-csv        # hoppa över CSV             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os, re, sys, time, math, warnings, logging, argparse
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    sys.exit("[FEL] Saknar yfinance. Kör: pip install yfinance")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[VARNING] Matplotlib saknas – grafer inaktiverade.")

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION (standard-värden, kan överskridas via argparse)
# ════════════════════════════════════════════════════════════════════════════

BENCHMARK_TICKER    = '^OMX'

MIN_HISTORY         = 200
MIN_AVG_DAILY_VOL   = 500_000      # 500k SEK-ekvivalent dagsomsättning (sänkt från 1M)
RESCREEN_INTERVAL   = 5             # Screena var 5:e dag
MAX_ENTRIES_PER_DAY = 2             # Max 2 nya köp per handelsdag (höjt från 1)

COURTAGE            = 0.0005
SLIPPAGE_LOW        = 0.0005
SLIPPAGE_MID        = 0.0015
SLIPPAGE_HIGH       = 0.0030

COOLDOWN_DAYS       = 15            # Sänkt 30→15d: inspirerat av v4 som funkar
REENTRY_LOSS_EXTRA  = 10            # Sänkt 20→10d extra cooldown efter förlust

# ── Momentum-ranking filter (proffstricket) ──────────────────────────────────
# Köp bara kandidater i topp X% av riskjusterat momentum-score.
# Eliminerar "brus-trades" med svag edge → höjer Sharpe utan att sänka CAGR.
MOMENTUM_TOP_PCT    = 0.30          # Köp bara topp 30% av kandidater per screening
ATR_EXPANSION_MIN   = 1.15          # ATR_20/ATR_50 > 1.15 = volatilitet expanderar → trend startar
MONTE_CARLO_RUNS    = 500
MAX_DEFENSIVE_OSAKER = 1
MAX_DEFENSIVE_BEAR   = 2
MAX_ACTIVE_SLOTS_BEAR   = 4         # Höjt 3→4 för 8-slot setup
MAX_ACTIVE_SLOTS_OSAKER = 6         # Höjt 5→6 för 8-slot setup

OUTPUT_FOLDER       = 'Master_Backtest_Output'
CHART_FOLDER        = os.path.join(OUTPUT_FOLDER, 'Grafer')

# ─── Entryfilter per horisont ────────────────────────────────────────────────
# ALLA krav måste vara uppfyllda för att ett köp ska ske.
# Mål: 40-80 trades/år. Hämtat inspiration från v4 för att inte strypa köpen.
ENTRY_FILTER = {
    'korttid': {
        'min_total_score':   50,    # Sänkt 62->50
        'min_kt_score':      45,    # Sänkt 58->45
        'min_rs':            -2.0,  # Sänkt 0->-2: tillåt liten underperformance
        'min_adx':           15.0,  # Sänkt 20->15
        'min_p6m':          -99.0,  # Inget 6m-krav (KT = kort horisont)
        'require_above_sma200': True,
        'require_above_sma50':  False,  # Borttaget: för restriktivt för KT
        'min_confluence':     2,    # Sänkt 3->2: 2 av 6 faktorer starka
        'min_vol_ratio':      0.8,  # Sänkt 1.2->0.8
    },
    'medel': {
        'min_total_score':   56,    # Sänkt 68->56
        'min_kt_score':       0,
        'min_rs':             1.0,  # Sänkt 3.0->1.0
        'min_adx':           18.0,  # Sänkt 24->18
        'min_p6m':           -2.0,  # Sänkt 2->-2: tillåt svag 6m i recovery
        'require_above_sma200': True,
        'require_above_sma50':  False,  # Borttaget: ej nödvändigt för medel
        'min_confluence':     2,    # Sänkt 4->2
        'min_vol_ratio':      0.8,
    },
    'swing': {
        'min_total_score':   60,    # Sänkt 72->60
        'min_kt_score':       0,
        'min_rs':             2.0,  # Sänkt 5.0->2.0
        'min_adx':           20.0,  # Sänkt 26->20
        'min_p6m':            2.0,  # Sänkt 10->2: swing kräver lite mer men inte massor
        'require_above_sma200': True,
        'require_above_sma50':  True,   # Behåller för swing: stark trend krävs
        'min_confluence':     3,    # Sänkt 4->3
        'min_vol_ratio':      0.8,
    },
}

# Horisont-konfiguration
SLOT_HORIZONS_DEFAULT = {
    1: 'korttid',
    2: 'korttid',
    3: 'medel',
    4: 'medel',
    5: 'swing',
}

HORIZON_PARAMS = {
    # Korttid: squeeze-release / oversold bounce
    # Mål: 8-12 trades/år per slot
    'korttid': {
        'max_hold_days':      20,    # Utökat 14→20: ge mer rum att röra sig
        'min_hold_days':       3,
        'catastrophe_stop':  -13.0,  # Utökat -9→-13: andningsrum mot brus (v5-inspirerat)
        'profit_lock_trig':   30.0,  # Höjt 18→30: klipp inte vinnare för tidigt
        'profit_lock_floor':  15.0,  # Höjt 10→15
        'rs_exit':            -5.0,  # Mer tolerant -4→-5
        'score_type':         'korttid',
        'max_streak_losses':   2,
        'streak_pause_days':  10,
        'atr_trail_mult':     None,
        'sma_exit_days':      99,    # INAKTIVERAD: 99 = aldrig uppnådd (SMA20-exit skapar churn)
        'sma_exit_min_hold':  99,
        'min_gain_for_rs_exit': -5.0,  # FIX: RS-EXIT kräver att vi är ner -5% (inte bara -2%)
    },
    # Medel: trender 2-3 månader
    'medel': {
        'max_hold_days':      90,    # Utökat 80→90
        'min_hold_days':      10,
        'catastrophe_stop':  -16.0,  # Mer rum -14→-16
        'profit_lock_trig':   50.0,  # Höjt 40→50: låt vinnare löpa längre
        'profit_lock_floor':  30.0,
        'rs_exit':            -6.0,
        'score_type':         'medel',
        'max_streak_losses':   3,
        'streak_pause_days':  14,
        'atr_trail_mult':      2.5,
        'sma_exit_days':      99,    # INAKTIVERAD
        'sma_exit_min_hold':  99,
        'min_gain_for_rs_exit': -7.0,  # FIX: kräv riktig förlust (-5→-7)
    },
    # Swing: riktiga trender — låt multibaggers löpa 3-6 månader
    'swing': {
        'max_hold_days':     180,
        'min_hold_days':      15,
        'catastrophe_stop':  -18.0,  # Mer rum -16→-18
        'profit_lock_trig':  999.0,  # INGEN profit-lock — vinnare löper fritt
        'profit_lock_floor':   0.0,
        'rs_exit':            -8.0,
        'score_type':         'total',
        'max_streak_losses':   3,
        'streak_pause_days':  21,
        'atr_trail_mult':      3.0,
        'sma_exit_days':      99,    # INAKTIVERAD
        'sma_exit_min_hold':  99,
        'min_gain_for_rs_exit': -9.0,  # FIX: swing har alltid haft rätt värde, behåll
    },
}

# Benchmark per land (relativ styrka-beräkning)
BENCHMARK_PER_LAND = {
    'SE': '^OMX',        'NO': 'OBX.OL',    'DK': '^OMXC25',
    'FI': '^OMXHPI',     'DE': '^GDAXI',    'FR': '^FCHI',
    'UK': '^FTSE',       'NL': '^AEX',      'IT': '^FTSEMIB',
    'ES': '^IBEX',       'CH': '^SSMI',     'BE': '^BFX',
    'AT': '^ATX',        'PL': '^WIG20',    'EU': '^STOXX50E',
    'HU': '^STOXX50E',   'GR': '^STOXX50E',
}


# ════════════════════════════════════════════════════════════════════════════
#  UNIVERSUM  (ticker, namn, land, sektor, typ, cap_hint)
#  Innehåller 500+ instrument från Sverige och Europa
#  Alla tickers verifierade mot yfinance / Avanza-tillgängliga
# ════════════════════════════════════════════════════════════════════════════

# Universum importeras från ues_tick.py
# Lägg till/ta bort aktier där — denna fil behöver inte ändras.
from avanzaEuUs import UNIVERSE as _RAW_UNIVERSE

# Deduplicera (ifall ues_tick råkar ha dubletter)
# FIX: strip whitespace från alla strängfält (ticker, land m.m. har trailing spaces i avanza.py)
_seen = set()
_dedup = []
for _e in _RAW_UNIVERSE:
    _e = tuple(v.strip() if isinstance(v, str) else v for v in _e)
    if _e[0] not in _seen:
        _seen.add(_e[0])
        _dedup.append(_e)
UNIVERSE = _dedup

# Snabb lookup: ticker → namn / land / sektor
_NAMN   = {u[0]: u[1] for u in UNIVERSE}
_LAND   = {u[0]: u[2] for u in UNIVERSE}
_SEKTOR = {u[0]: (u[3] if len(u) > 3 else '?') for u in UNIVERSE}
_TYP    = {u[0]: (u[4] if len(u) > 4 else 'aktie') for u in UNIVERSE}
_CAP    = {u[0]: (u[5] if len(u) > 5 else '?') for u in UNIVERSE}


def get_tickers_by_land(land_filter: Optional[List[str]] = None) -> List[str]:
    """Returnerar tickers, eventuellt filtrerat per land."""
    if not land_filter:
        return [u[0] for u in UNIVERSE]
    land_upper = [l.upper() for l in land_filter]
    return [u[0] for u in UNIVERSE if u[2].upper() in land_upper]


# ════════════════════════════════════════════════════════════════════════════
#  HJÄLPFUNKTIONER  —  MATEMATIK & INDIKATORER
# ════════════════════════════════════════════════════════════════════════════

def sf(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except:
        return default


def sl(s, d=np.nan):
    try:
        return float(s.dropna().iloc[-1])
    except:
        return d


def flatten(hist):
    if isinstance(hist.columns, pd.MultiIndex):
        hist = hist.copy()
        hist.columns = hist.columns.get_level_values(0)
    return hist


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi_series(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def atr_series(h, l, c, n=14):
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=n-1, min_periods=n).mean()


def adx_series(h, l, c, n=14):
    up = h.diff()
    dn = -l.diff()
    plus  = np.where((up > dn)  & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a_ = atr_series(h, l, c, 1).ewm(com=n-1).mean()
    dip = 100 * pd.Series(plus,  index=c.index).ewm(com=n-1).mean() / (a_ + 1e-10)
    dim = 100 * pd.Series(minus, index=c.index).ewm(com=n-1).mean() / (a_ + 1e-10)
    dx  = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    return dx.ewm(com=n-1, min_periods=n).mean(), dip, dim


def cmf_series(h, l, c, v, n=20):
    clv = ((c - l) - (h - c)) / (h - l + 1e-10)
    return (clv * v).rolling(n).sum() / (v.rolling(n).sum() + 1e-10)


def bb_bands(c, n=20, std=2.0):
    m   = c.rolling(n).mean()
    s   = c.rolling(n).std()
    bw  = (m + std * s - (m - std * s)) / (m + 1e-10)
    pct = (c - (m - std * s)) / (2 * std * s + 1e-10)
    return m + std * s, m, m - std * s, bw, pct


def kc_bands(h, l, c, n=20, mult=1.5):
    m = c.ewm(span=n, adjust=False).mean()
    a = atr_series(h, l, c, n)
    return m + mult * a, m, m - mult * a


def squeeze_momentum(h, l, c, n=20, bb_std=2.0, kc_mult=1.5):
    bbu, bbm, bbl, _, _ = bb_bands(c, n, bb_std)
    kcu, _, kcl = kc_bands(h, l, c, n, kc_mult)
    sq    = (bbu < kcu) & (bbl > kcl)
    delta = c - ((h.rolling(n).max() + l.rolling(n).min()) / 2 + bbm) / 2
    mom   = delta.ewm(span=n, adjust=False).mean()
    return sq, mom


def stoch_series(h, l, c, k=14, slow=3, d=3):
    rk = 100 * (c - l.rolling(k).min()) / (h.rolling(k).max() - l.rolling(k).min() + 1e-10)
    sk = rk.rolling(slow).mean()
    return sk, sk.rolling(d).mean()


def obv_bull(c, v, w=20):
    ob   = (np.sign(c.diff().fillna(0)) * v).cumsum()
    em   = ob.ewm(span=w, adjust=False).mean()
    y    = ob.values[-w:] if len(ob) >= w else ob.values
    slp  = float(np.polyfit(np.arange(len(y)), y, 1)[0]) / (abs(float(np.mean(y))) + 1e-10)
    return bool(ob.iloc[-1] > em.iloc[-1]), slp


def calc_rs(ca, cb, w=21):
    df = pd.concat([ca, cb], axis=1).dropna()
    if len(df) < w:
        return 0.0
    t = df.tail(w)
    ar = float(t.iloc[-1, 0]) / (float(t.iloc[0, 0]) + 1e-10)
    br = float(t.iloc[-1, 1]) / (float(t.iloc[0, 1]) + 1e-10)
    return round((ar / (br + 1e-10) - 1) * 100, 1)


def calc_dynamic_slippage(capital, ticker_hist, as_of_date):
    try:
        hist = ticker_hist[ticker_hist.index <= as_of_date].tail(20)
        if hist.empty:
            return SLIPPAGE_MID
        c = hist['Close']
        v = hist['Volume']
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
        avg_sek = float(v.mean()) * float(c.iloc[-1])
        if avg_sek <= 0:
            return SLIPPAGE_MID
        p = capital / (avg_sek + 1e-10)
        if p > 0.05:   return SLIPPAGE_HIGH
        elif p > 0.02: return SLIPPAGE_MID
        else:          return SLIPPAGE_LOW
    except:
        return SLIPPAGE_MID


# ════════════════════════════════════════════════════════════════════════════
#  FÖRBERÄKNING  —  ALLA SERIER BERÄKNAS EN GÅNG PER TICKER
#
#  NYCKELOPTIMERING: compute_indicators() anropades tidigare
#  (n_dagar/5) × n_tickers gånger och räknade om EMA, RSI, ADX m.m.
#  från scratch vid varje anrop via DataFrame-filter hist[hist.index<=date].
#
#  Nu:
#  1. precompute_all()  — körs EN gång efter data-hämtning (~sekunder)
#     Bygger hela rolling-serierna som numpy-arrays för varje ticker.
#  2. lookup_at(arr, date_idx) — O(1) array-indexering ersätter pandas-filter
#  3. build_price_index()      — dict {ticker: {date: (open,close,vol)}}
#     Eliminerar hist[hist.index==date]-scan i huvudloopen (var O(n)).
#  4. precompute_regime_series() — benchmark-indikatorer som arrays.
#
#  Inga look-ahead-problem: rolling-series på full historik är identiska
#  med att beräkna dem dag för dag — värdet på index i beror bara på
#  data t.o.m. i (pandas rolling/ewm garanterar detta).
# ════════════════════════════════════════════════════════════════════════════

class TickerCache:
    """Håller alla förberäknade serier för en ticker som numpy-arrays."""
    __slots__ = (
        'dates',       # np.array av Timestamps (sorterade)
        'close_arr',
        'high_arr',
        'low_arr',
        'vol_arr',
        'open_arr',
        # EMAs
        'ema9',  'ema21', 'ema50', 'ema200',
        # SMAs (v2: exit + hard-block)
        'sma20', 'sma50', 'sma200',
        'sma200_slope',   # normaliserad 20d-lutning
        'above_sma200',   # 1 om pris > SMA200
        # RSI
        'rsi14', 'rsi5',
        'rsi_slope5',     # v2: RSI 5d-diff (momentum av momentum)
        # StochRSI (v2)
        'stochrsi_k', 'stochrsi_d',
        # Bollinger
        'bb_upper', 'bb_mid', 'bb_lower', 'bb_bw', 'bb_pctB',
        # Keltner
        'kc_upper', 'kc_lower',
        # Squeeze
        'sq_on', 'sq_mom',
        # ATR
        'atr14', 'atr50',
        # Stochastic
        'stoch_k', 'stoch_d',
        # CMF
        'cmf',
        # ADX
        'adx', 'dip', 'dim',
        # OBV
        'obv', 'obv_ema20', 'obv_slope20',
        # Z-score underlag
        'm20', 's20',
        # Volym
        'vol_avg22',
        'up_dn_vol',      # v2: institutionellt flöde
        # Relativ styrka
        'rs21', 'rs10',
        'rs_acc',         # v2: RS-acceleration
        # Långa momentum (v2)
        'p3m', 'p6m', 'p12m',
    )


def _safe_arr(s):
    """Konverterar en pandas Series till float64 numpy-array."""
    a = s.values.astype(np.float64)
    # ersätt inf med nan
    a[~np.isfinite(a)] = np.nan
    return a


def precompute_all(all_hist, bench_close):
    """
    Förberäknar alla indikator-serier för alla tickers.
    Returnerar dict {ticker: TickerCache}.
    Körs EN gång – O(n_tickers × n_dagar).
    """
    cache = {}
    n_ok  = 0

    for ticker, hist in all_hist.items():
        try:
            c_s = hist['Close'].squeeze()
            h_s = hist.get('High',  hist['Close']).squeeze()
            l_s = hist.get('Low',   hist['Close']).squeeze()
            v_s = hist['Volume'].squeeze()
            o_s = hist.get('Open',  hist['Close']).squeeze()
            for x in [c_s, h_s, l_s, v_s, o_s]:
                if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
            if isinstance(c_s, pd.DataFrame): c_s = c_s.iloc[:, 0]
            if isinstance(h_s, pd.DataFrame): h_s = h_s.iloc[:, 0]
            if isinstance(l_s, pd.DataFrame): l_s = l_s.iloc[:, 0]
            if isinstance(v_s, pd.DataFrame): v_s = v_s.iloc[:, 0]
            if isinstance(o_s, pd.DataFrame): o_s = o_s.iloc[:, 0]

            c_s = c_s.astype(float)
            h_s = h_s.astype(float)
            l_s = l_s.astype(float)
            v_s = v_s.astype(float)
            o_s = o_s.astype(float)

            tc = TickerCache()
            tc.dates      = np.array(c_s.index, dtype='datetime64[ns]')
            tc.close_arr  = _safe_arr(c_s)
            tc.high_arr   = _safe_arr(h_s)
            tc.low_arr    = _safe_arr(l_s)
            tc.vol_arr    = _safe_arr(v_s)
            tc.open_arr   = _safe_arr(o_s)

            # EMAs
            tc.ema9   = _safe_arr(ema(c_s,  9))
            tc.ema21  = _safe_arr(ema(c_s, 21))
            tc.ema50  = _safe_arr(ema(c_s, 50))
            tc.ema200 = _safe_arr(ema(c_s, 200))

            # RSI
            tc.rsi14 = _safe_arr(rsi_series(c_s, 14))
            tc.rsi5  = _safe_arr(rsi_series(c_s,  5))

            # Bollinger
            bbu, bbm, bbl, bw, pctB = bb_bands(c_s, 20, 2.0)
            tc.bb_upper = _safe_arr(bbu)
            tc.bb_mid   = _safe_arr(bbm)
            tc.bb_lower = _safe_arr(bbl)
            tc.bb_bw    = _safe_arr(bw)
            tc.bb_pctB  = _safe_arr(pctB)

            # Keltner (för squeeze)
            kcu, _, kcl = kc_bands(h_s, l_s, c_s, 20, 1.5)
            tc.kc_upper = _safe_arr(kcu)
            tc.kc_lower = _safe_arr(kcl)

            # Squeeze-momentum
            sq_s, sq_mom_s = squeeze_momentum(h_s, l_s, c_s, 20)
            tc.sq_on  = _safe_arr(sq_s.astype(float))
            tc.sq_mom = _safe_arr(sq_mom_s)

            # ATR
            tc.atr14 = _safe_arr(atr_series(h_s, l_s, c_s, 14))
            tc.atr50 = _safe_arr(atr_series(h_s, l_s, c_s, 50))

            # Stochastic
            sk_s, sd_s = stoch_series(h_s, l_s, c_s, 14, 3, 3)
            tc.stoch_k = _safe_arr(sk_s)
            tc.stoch_d = _safe_arr(sd_s)

            # CMF
            tc.cmf = _safe_arr(cmf_series(h_s, l_s, c_s, v_s, 20))

            # ADX
            adx_s, dip_s, dim_s = adx_series(h_s, l_s, c_s, 14)
            tc.adx = _safe_arr(adx_s)
            tc.dip = _safe_arr(dip_s)
            tc.dim = _safe_arr(dim_s)

            # OBV
            obv_s     = (np.sign(c_s.diff().fillna(0)) * v_s).cumsum()
            obv_ema_s = obv_s.ewm(span=20, adjust=False).mean()
            tc.obv       = _safe_arr(obv_s)
            tc.obv_ema20 = _safe_arr(obv_ema_s)
            # OBV normalized slope 20d
            obv_slope_s = (obv_s.diff(20) / (obv_s.abs().rolling(20).mean() + 1e-10)).clip(-2, 2)
            tc.obv_slope20 = _safe_arr(obv_slope_s)

            # Z-score underlag
            tc.m20 = _safe_arr(c_s.rolling(20).mean())
            tc.s20 = _safe_arr(c_s.rolling(20).std())

            # Rullande volymsnitt 22d
            tc.vol_avg22 = _safe_arr(v_s.rolling(22, min_periods=5).mean().shift(1))

            # Institutionellt flöde: up_vol / dn_vol (20d)
            ret_d  = c_s.pct_change()
            up_v_s = v_s.where(ret_d > 0, 0.0).rolling(20).sum()
            dn_v_s = v_s.where(ret_d < 0, 0.0).rolling(20).sum()
            tc.up_dn_vol = _safe_arr((up_v_s / (dn_v_s + 1e-10)).clip(0, 10))

            # SMA (v2: for hard block and SMA20 exit)
            tc.sma20  = _safe_arr(c_s.rolling(20).mean())
            tc.sma50  = _safe_arr(c_s.rolling(50).mean())
            tc.sma200 = _safe_arr(c_s.rolling(200).mean())
            sma200_s  = c_s.rolling(200).mean()
            tc.sma200_slope  = _safe_arr(
                (sma200_s.diff(20) / (sma200_s.shift(20) + 1e-10) * 100).fillna(0))
            tc.above_sma200  = _safe_arr((c_s > sma200_s).astype(float))

            # StochRSI (v2)
            rsi14_s = rsi_series(c_s, 14)
            rsi_min14 = rsi14_s.rolling(14).min()
            rsi_max14 = rsi14_s.rolling(14).max()
            stochrsi_k_s = ((rsi14_s - rsi_min14) /
                            (rsi_max14 - rsi_min14 + 1e-10) * 100).clip(0, 100)
            tc.stochrsi_k = _safe_arr(stochrsi_k_s)
            tc.stochrsi_d = _safe_arr(stochrsi_k_s.rolling(3).mean())

            # RSI-lutning 5d (v2)
            tc.rsi_slope5 = _safe_arr((rsi14_s - rsi14_s.shift(5)).fillna(0))

            # Relativ styrka vs benchmark (rullande ratio)
            if bench_close is not None and len(bench_close) > 21:
                b_aligned = bench_close.reindex(c_s.index, method='ffill').bfill()
                rs21_s = (c_s / c_s.shift(21) / (b_aligned / b_aligned.shift(21)) - 1) * 100
                rs10_s = (c_s / c_s.shift(10) / (b_aligned / b_aligned.shift(10)) - 1) * 100
                tc.rs21 = _safe_arr(rs21_s)
                tc.rs10 = _safe_arr(rs10_s)
                # RS-acceleration (v2): RS nu vs RS för 21d sedan
                tc.rs_acc = _safe_arr((rs21_s - rs21_s.shift(21)).fillna(0))
            else:
                tc.rs21   = np.zeros(len(c_s))
                tc.rs10   = np.zeros(len(c_s))
                tc.rs_acc = np.zeros(len(c_s))

            # Långa momentum-faktorer (v2)
            tc.p3m  = _safe_arr((c_s / c_s.shift(63)  - 1).fillna(0) * 100)
            tc.p6m  = _safe_arr((c_s / c_s.shift(126) - 1).fillna(0) * 100)
            tc.p12m = _safe_arr((c_s / c_s.shift(252) - 1).fillna(0) * 100)

            cache[ticker] = tc
            n_ok += 1
        except Exception:
            pass

    print("[INFO] Förberäkning klar: {} / {} tickers".format(n_ok, len(all_hist)))
    return cache


def _date_to_idx(tc: TickerCache, date) -> int:
    """
    Returnerar det senaste index i i tc.dates där tc.dates[i] <= date.
    Använder np.searchsorted (O(log n)) istället för pandas boolean-mask (O(n)).
    Returnerar -1 om date är före all data.
    """
    ts = np.datetime64(date, 'ns')
    # searchsorted returnerar insättningsposition; vi vill ha sista <=
    idx = int(np.searchsorted(tc.dates, ts, side='right')) - 1
    return idx


def build_price_index(all_hist):
    """
    Bygger O(1)-uppslagstabeller för priser.
    Returnerar:
      close_px  : {ticker: {Timestamp: float}}
      open_px   : {ticker: {Timestamp: float}}
      vol_20_px : {ticker: {Timestamp: float}}  (20d genomsnittsomsättning SEK)
    """
    close_px  = {}
    open_px   = {}
    vol20_px  = {}

    for ticker, hist in all_hist.items():
        c = hist['Close'].squeeze()
        v = hist['Volume'].squeeze()
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
        c = c.astype(float)
        v = v.astype(float)

        close_px[ticker] = dict(zip(c.index, c.values))

        if 'Open' in hist.columns:
            o = hist['Open'].squeeze()
            if isinstance(o, pd.DataFrame): o = o.iloc[:, 0]
            open_px[ticker] = dict(zip(o.index, o.astype(float).values))
        else:
            open_px[ticker] = close_px[ticker]  # fallback: använd Close

        # Rullande 20d genomsnittsomsättning i SEK (shift(1) = ingen look-ahead)
        avg_vol_sek = (v * c).rolling(20, min_periods=5).mean().shift(1)
        vol20_px[ticker] = dict(zip(avg_vol_sek.index, avg_vol_sek.values))

    return close_px, open_px, vol20_px


def precompute_regime_series(bench_close):
    """
    Förberäknar regim-indikatorer för benchmarken som numpy-arrays.
    Returnerar dict med arrays indexerade på benchmark-datum.
    """
    if bench_close is None:
        return None
    b = bench_close.dropna().astype(float)
    return {
        'dates':  np.array(b.index, dtype='datetime64[ns]'),
        'close':  b.values,
        'ma50':   _safe_arr(b.rolling(50).mean()),
        'ma200':  _safe_arr(b.rolling(200).mean()),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MARKNADSREGIM
# ════════════════════════════════════════════════════════════════════════════

class Regime(Enum):
    BULL   = "Bull"
    OSAKER = "Osäker"
    BEAR   = "Bear"


@dataclass
class RegimDetalj:
    fas:    str    = "Osäker"
    risk:   str    = "Neutral"
    enkel:  Regime = Regime.OSAKER
    avk_1m: float  = 0.0
    avk_3m: float  = 0.0


def compute_regime(bench_close, as_of_date) -> RegimDetalj:
    if bench_close is None:
        return RegimDetalj(fas="Osäker", risk="Neutral", enkel=Regime.BULL)
    try:
        b = bench_close[bench_close.index <= as_of_date].dropna()
        if len(b) < 210:
            return RegimDetalj(fas="Osäker", risk="Neutral", enkel=Regime.BULL)

        price  = float(b.iloc[-1])
        ma50   = float(b.rolling(50).mean().iloc[-1])
        ma200  = float(b.rolling(200).mean().iloc[-1])
        n      = len(b)
        avk1m  = (price / float(b.iloc[max(0, n-21)]) - 1) * 100
        avk3m  = (price / float(b.iloc[max(0, n-63)]) - 1) * 100

        ma50s = b.rolling(50).mean().dropna().tail(10)
        ma50_rising = False
        if len(ma50s) >= 5:
            slope_norm = (float(np.polyfit(range(len(ma50s)), ma50s.values, 1)[0])
                          / (float(ma50s.mean()) + 1e-10))
            ma50_rising = slope_norm > 0.0002

        trend_score = sum([price > ma50, ma50 > ma200, price > ma200, ma50_rising])

        if price < ma200:
            fas   = "Capitulation" if avk1m < -4 else "Contraction"
            risk  = "Risk-Off"
            enkel = Regime.BEAR
        elif trend_score >= 3 and avk3m > 1:   # Sänkt 4→1
            fas   = "Expansion" if avk1m > 1 else "LateExpansion"
            risk  = "Risk-On"
            enkel = Regime.BULL
        elif trend_score >= 2 and avk3m > -3:  # Sänkt -2→-3
            fas   = "Distribution"
            risk  = "Neutral"
            enkel = Regime.OSAKER
        elif avk1m > 0 and avk3m < 0:
            fas   = "Recovery"
            risk  = "Neutral"
            enkel = Regime.OSAKER
        else:
            fas   = "Contraction"
            risk  = "Risk-Off"
            enkel = Regime.BEAR

        return RegimDetalj(fas=fas, risk=risk, enkel=enkel,
                           avk_1m=round(avk1m, 1), avk_3m=round(avk3m, 1))
    except:
        return RegimDetalj(fas="Osäker", risk="Neutral", enkel=Regime.BULL)


# ════════════════════════════════════════════════════════════════════════════
#  V9+ SCREENER  —  8 FAKTORER
# ════════════════════════════════════════════════════════════════════════════

def compute_indicators(ticker, hist_full, bench_full, as_of_date, min_vol=MIN_AVG_DAILY_VOL):
    """
    Fallback-version (används om cache saknas).
    Beräknar alla indikatorer look-ahead-fritt för given datum.
    """
    try:
        hist = hist_full[hist_full.index <= as_of_date].copy()
        if len(hist) < MIN_HISTORY:
            return None

        c  = hist['Close'].squeeze()
        h  = hist.get('High',  hist['Close']).squeeze()
        l  = hist.get('Low',   hist['Close']).squeeze()
        v  = hist['Volume'].squeeze()
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        if isinstance(h, pd.DataFrame): h = h.iloc[:, 0]
        if isinstance(l, pd.DataFrame): l = l.iloc[:, 0]
        if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
        c = c.astype(float); h = h.astype(float)
        l = l.astype(float); v = v.astype(float)

        n     = len(c)
        kurs  = float(c.iloc[-1])
        if kurs <= 0 or np.isnan(kurs):
            return None

        avg_vol_sek = float(v.tail(20).mean()) * kurs
        if avg_vol_sek < min_vol:
            return None

        def avk(d):
            return round((kurs / (float(c.iloc[max(0, n-d)]) + 1e-10) - 1) * 100, 1)

        p1d = avk(1); p1w = avk(5)

        rs = rs2w = 0.0
        if bench_full is not None:
            bench = bench_full[bench_full.index <= as_of_date].dropna()
            if len(bench) > 21:
                rs   = calc_rs(c, bench, 21)
                rs2w = calc_rs(c, bench, 10)

        e9   = ema(c,  9); e21 = ema(c, 21)
        e50  = ema(c, 50); e200 = ema(c, 200)
        ema_score = sum([float(c.iloc[-1]) > float(e9.iloc[-1]),
                         float(e9.iloc[-1]) > float(e21.iloc[-1]),
                         float(e21.iloc[-1]) > float(e50.iloc[-1]),
                         float(e50.iloc[-1]) > float(e200.iloc[-1])])
        ema_trend = {4:"upp",3:"svagt_upp",2:"flat",1:"svagt_ner",0:"ner"}[ema_score]

        rsi14_s = rsi_series(c, 14); rsi5_s = rsi_series(c, 5)
        rsi14 = round(float(rsi14_s.iloc[-1]) if np.isfinite(float(rsi14_s.iloc[-1])) else 50.0, 1)
        rsi5  = round(float(rsi5_s.iloc[-1])  if np.isfinite(float(rsi5_s.iloc[-1]))  else 50.0, 1)
        rsi_bull_div = (n >= 25 and
                        float(c.iloc[-5:].min()) < float(c.iloc[-20:].min()) and
                        float(rsi14_s.iloc[-5:].min()) > float(rsi14_s.iloc[-20:].min()))

        _, _, _, bb_bw, bb_pctB_s = bb_bands(c, 20, 2.0)
        bb_bw_nu = round(float(bb_bw.iloc[-1]), 4)
        bb_pctB  = round(float(bb_pctB_s.iloc[-1]), 3)
        bb_bw_hist = float(bb_bw.iloc[-40:].mean()) if n >= 40 else bb_bw_nu
        bb_squeeze_tight = bb_bw_nu < bb_bw_hist * 0.7

        try:
            sq_s, sq_mom_s = squeeze_momentum(h, l, c, 20)
            squeeze_on = bool(sq_s.iloc[-1]); squeeze_mom = round(float(sq_mom_s.iloc[-1]), 4)
            sq_bars = int(sq_s.iloc[-15:].sum()) if n >= 15 else 0
        except:
            squeeze_on, squeeze_mom, sq_bars = False, 0.0, 0

        atr14 = atr_series(h, l, c, 14); atr50 = atr_series(h, l, c, 50)
        atr_exp = float(atr14.iloc[-1]) / (float(atr50.iloc[-1]) + 1e-10)
        atr_pct = round(float(atr14.iloc[-1]) / (kurs + 1e-10) * 100, 1)

        try:
            sk_s, sd_s = stoch_series(h, l, c, 14, 3, 3)
            sk_nu = round(float(sk_s.iloc[-1]), 1); sd_nu = round(float(sd_s.iloc[-1]), 1)
            stoch_cross = float(sk_s.iloc[-2]) < float(sd_s.iloc[-2]) and sk_nu > sd_nu and sk_nu < 80
            stoch_os = (sk_nu < 20 and sd_nu < 20)
        except:
            sk_nu, sd_nu, stoch_cross, stoch_os = 50.0, 50.0, False, False

        try:    cmf_nu = round(float(cmf_series(h, l, c, v, 20).iloc[-1]), 3)
        except: cmf_nu = 0.0

        try:
            adx_s, dip_s, dim_s = adx_series(h, l, c, 14)
            adx_nu = round(float(adx_s.iloc[-1]), 1); dip_nu = round(float(dip_s.iloc[-1]), 1)
            dim_nu = round(float(dim_s.iloc[-1]), 1); adx_up = bool(dip_nu > dim_nu)
        except:
            adx_nu, dip_nu, dim_nu, adx_up = 20.0, 25.0, 25.0, False

        obv_up, obv_slope = obv_bull(c, v, 20)

        m20 = c.rolling(20).mean(); s20 = c.rolling(20).std()
        z_score = round(float(((c - m20) / (s20 + 1e-10)).iloc[-1]), 2)

        vol_snitt    = float(v.iloc[-22:-1].mean()) if n >= 22 else float(v.mean())
        vol_ratio    = round(float(v.iloc[-5:].mean())  / (vol_snitt + 1e-10), 2)
        vol_ratio_1d = round(float(v.iloc[-1])           / (vol_snitt + 1e-10), 2)
        vol_acc = (n >= 4 and v.iloc[-3] < v.iloc[-2] < v.iloc[-1] and float(v.iloc[-1]) > vol_snitt * 1.2)

        i_bas = False
        if n >= 20:
            c20 = c.iloc[-20:]
            bas_r = (float(c20.max()) - float(c20.min())) / (float(c20.mean()) + 1e-10)
            i_bas = bool(bas_r < 0.12 and float(v.iloc[-20:].mean()) < float(v.mean()) * 0.6)

        rekyl_setup = (ema_trend in ("upp","svagt_upp") and n >= 10 and
                       float(c.iloc[-1]) < float(e9.iloc[-1]) and
                       float(e9.iloc[-1]) > float(e21.iloc[-1]) and p1w < -2)

        d52 = min(252, n)
        high52 = float(c.iloc[-d52:].max()); low52 = float(c.iloc[-d52:].min())
        pos52  = round((kurs - low52) / (high52 - low52 + 1e-10), 3)
        peak   = c.expanding().max()
        max_dd = round(float(((c / peak - 1) * 100).iloc[-120:].min()), 1)

        try:
            y_log = np.log(c.iloc[-60:].values); x_log = np.arange(len(y_log))
            cov   = np.cov(x_log, y_log)
            r2    = float((cov[0,1]**2) / (cov[0,0] * cov[1,1] + 1e-10))
            trend_r2 = round(r2 if y_log[-1] > y_log[0] else -r2 * 0.5, 3)
        except:
            trend_r2 = 0.0

        ret = c.pct_change()
        inst_ratio = round(float(v.where(ret > 0, 0).iloc[-20:].sum() /
                                 (v.where(ret < 0, 0).iloc[-20:].sum() + 1e-10)), 2)
        inst_signal = "buying" if inst_ratio > 1.5 else "selling" if inst_ratio < 0.7 else "neutral"

        acc = ((float(c.pct_change(5).iloc[-1]) - float(c.pct_change(5).iloc[-5])) * 100
               if n >= 25 else 0.0)

        # ────────────────────────────────────────────────
        # SCORE-BERÄKNING  (V9-arkitektur, 8 faktorer)
        # ────────────────────────────────────────────────
        RSI_MAX = 85
        RS_EXIT = -6.0
        DD_MAX  = -60.0

        # Hard blockers
        if rsi14 > RSI_MAX or rs < RS_EXIT or max_dd < DD_MAX:
            return {
                '_price':      kurs,
                'total_score': 10,
                'kt_score':    5,
                'rsi':         rsi14,
                'rs':          rs,
                'max_dd':      max_dd,
                'ema_trend':   ema_trend,
                'cmf':         cmf_nu,
                'z_score':     z_score,
                'adx':         adx_nu,
                'vol_ratio':   vol_ratio,
                'atr_pct':     atr_pct,
                'pos_52w':     pos52,
                'p1d':         p1d,
                'p1w':         p1w,
                'squeeze_on':  squeeze_on,
                'blocked':     True,
            }

        # Faktor 1: Momentum
        f1 = 0.0
        if   rs >= 15: f1 += 22
        elif rs >= 10: f1 += 16
        elif rs >= 5:  f1 += 10
        elif rs >= 2:  f1 += 4
        elif rs >= -1: f1 += 1
        elif rs >= -6: f1 -= 5
        else:          f1 -= 18

        if rs2w >= 6:  f1 += 10
        elif rs2w >= 3: f1 += 5
        if p1w >= 10:  f1 += 12
        elif p1w >= 6: f1 += 7
        elif p1w >= 2: f1 += 3
        elif p1w < -8: f1 -= 7
        if acc > 5:    f1 += 12
        elif acc > 2:  f1 += 7
        elif acc > 0:  f1 += 2
        elif acc < -5: f1 -= 7
        ema_map = {"upp": 14, "svagt_upp": 7, "flat": 0, "svagt_ner": -4, "ner": -10}
        f1 += ema_map.get(ema_trend, 0)
        f1_n = max(0, min(100, (f1 / 70) * 100))

        # Faktor 2: Squeeze/Volatilitetskomprimering
        f2 = 0.0
        if squeeze_on:
            if   sq_bars >= 12: f2 += 32
            elif sq_bars >= 8:  f2 += 25
            elif sq_bars >= 5:  f2 += 18
            elif sq_bars >= 3:  f2 += 10
            else:               f2 += 5
            if squeeze_mom > 0: f2 += 12
            elif squeeze_mom < 0: f2 -= 8
        elif bb_squeeze_tight:
            f2 += 10
        if   atr_exp > 2.5: f2 += 12
        elif atr_exp > 1.8: f2 += 8
        elif atr_exp > 1.3: f2 += 4
        elif atr_exp < 0.4: f2 += 6  # extremt låg volatilitet = setup
        pB = bb_pctB
        if   pB < 0.05:          f2 += 10
        elif pB < 0.20:          f2 += 5
        elif pB > 0.95:          f2 -= 5
        elif 0.45 <= pB <= 0.65: f2 += 3
        f2_n = max(0, min(100, (f2 / 60) * 100))

        # Faktor 3: Oscillatorer
        f3 = 0.0
        if   rsi14 > RSI_MAX:       f3 -= 20
        elif 35 <= rsi14 <= 50:     f3 += 20
        elif 50 < rsi14 <= 60:      f3 += 14
        elif 60 < rsi14 <= 70:      f3 += 8
        elif 70 < rsi14 <= RSI_MAX: f3 += 4
        elif 30 <= rsi14 < 35:      f3 += 14
        elif 20 <= rsi14 < 30:      f3 += 8
        else:                        f3 -= 7
        if rsi_bull_div: f3 += 18
        if rsi5 < 20:    f3 += 10
        if stoch_cross:  f3 += 12
        if stoch_os:     f3 += 10
        f3_n = max(0, min(100, (f3 / 70) * 100))

        # Faktor 4: Volym/Flöde
        f4 = 0.0
        vr = vol_ratio
        if   vr >= 12: f4 += 28
        elif vr >= 7:  f4 += 20
        elif vr >= 4:  f4 += 13
        elif vr >= 2:  f4 += 7
        elif vr >= 1.2: f4 += 2
        elif vr < 0.4: f4 -= 4
        vr1 = vol_ratio_1d
        if   vr1 >= 10: f4 += 14
        elif vr1 >= 5:  f4 += 8
        elif vr1 >= 2:  f4 += 3
        if vol_acc:          f4 += 10
        if obv_up and obv_slope > 0: f4 += 10
        elif not obv_up:             f4 -= 6
        if   cmf_nu > 0.25:  f4 += 12
        elif cmf_nu > 0.10:  f4 += 7
        elif cmf_nu > 0.0:   f4 += 3
        elif cmf_nu < -0.25: f4 -= 10
        elif cmf_nu < -0.10: f4 -= 5
        if inst_signal == "buying":   f4 += 10
        elif inst_signal == "selling": f4 -= 8
        f4_n = max(0, min(100, (f4 / 80) * 100))

        # Faktor 5: Trendkvalitet
        f5 = 0.0
        if   adx_nu > 45 and adx_up:  f5 += 18
        elif adx_nu > 30 and adx_up:  f5 += 12
        elif adx_nu > 20 and adx_up:  f5 += 6
        elif adx_nu > 30 and not adx_up: f5 -= 6
        elif adx_nu < 15:               f5 -= 3
        if   dip_nu > dim_nu + 8:  f5 += 10
        elif dip_nu > dim_nu + 3:  f5 += 5
        elif dim_nu > dip_nu + 8:  f5 -= 8
        elif dim_nu > dip_nu + 3:  f5 -= 4
        if   trend_r2 >= 0.80: f5 += 12
        elif trend_r2 >= 0.60: f5 += 8
        elif trend_r2 >= 0.40: f5 += 3
        elif trend_r2 < 0:     f5 -= 6
        f5_n = max(0, min(100, (f5 / 45) * 100))

        # Faktor 6: Setups & Kontraströster
        f6 = 0.0
        z = z_score
        if   z < -2.5: f6 += 18
        elif z < -2.0: f6 += 12
        elif z < -1.5: f6 += 6
        elif z > 2.5:  f6 -= 8
        elif z > 2.0:  f6 -= 5
        if i_bas:        f6 += 18
        if rekyl_setup:  f6 += 14
        dd = max_dd
        if   dd >= -5:   f6 += 10
        elif dd >= -10:  f6 += 6
        elif dd >= -25:  f6 += 0
        elif dd >= -40:  f6 -= 5
        elif dd >= -60:  f6 -= 10
        else:            f6 -= 18
        if pos52 >= 0.85: f6 += 8    # nära 52v-topp = breakout-potential
        elif pos52 >= 0.70: f6 += 4
        elif pos52 <= 0.15: f6 += 6  # nära botten = reversal
        f6_n = max(0, min(100, (f6 / 62) * 100))

        # Faktor 7: Regim (hanteras externt, neutral default)
        f7_n = 50.0

        # Faktor 8: Explosionsfaktor
        f8 = 0.0
        if   p1d >= 10: f8 += 22
        elif p1d >= 6:  f8 += 14
        elif p1d >= 3:  f8 += 6
        elif p1d <= -7: f8 -= 8
        if i_bas:       f8 += 18
        if   vr >= 12:  f8 += 18
        elif vr >= 7:   f8 += 10
        elif vr >= 4:   f8 += 5
        if squeeze_on and squeeze_mom > 0: f8 += 12
        f8_n = max(0, min(100, (f8 / 80) * 100))

        # Viktad total
        w = {"f1": 0.18, "f2": 0.14, "f3": 0.13, "f4": 0.14,
             "f5": 0.10, "f6": 0.10, "f7": 0.09, "f8": 0.12}
        total = int(round(max(0, min(100,
            f1_n*w["f1"] + f2_n*w["f2"] + f3_n*w["f3"] + f4_n*w["f4"] +
            f5_n*w["f5"] + f6_n*w["f6"] + f7_n*w["f7"] + f8_n*w["f8"]
        ))))

        # KorttidScore  (fokus: explosioner idag/denna vecka)
        kt = 0
        if squeeze_on:
            kt += 22 + min(18, sq_bars * 2)
            if squeeze_mom > 0: kt += 12
        elif bb_squeeze_tight:
            kt += 12
        if   vr >= 12: kt += 28
        elif vr >= 7:  kt += 20
        elif vr >= 4:  kt += 13
        elif vr >= 2:  kt += 6
        if p1d >= 10:  kt += 22
        elif p1d >= 6: kt += 14
        elif p1d >= 3: kt += 6
        if i_bas:        kt += 20
        if rekyl_setup:  kt += 14
        if rsi_bull_div: kt += 20
        if stoch_cross:  kt += 14
        if stoch_os:     kt += 10
        if z < -2.5:     kt += 18
        elif z < -2.0:   kt += 12
        elif z < -1.5:   kt += 6
        if cmf_nu > 0.10: kt += 9
        if vol_acc:        kt += 10
        kt = max(0, min(100, kt))

        # Medellång-score  (blandar total + kt)
        mixed = round(0.55 * total + 0.45 * kt)

        return {
            '_price':       kurs,
            'total_score':  total,
            'kt_score':     kt,
            'mixed_score':  mixed,
            'blocked':      False,
            'rsi':          rsi14,
            'rs':           rs,
            'max_dd':       max_dd,
            'ema_trend':    ema_trend,
            'squeeze_on':   squeeze_on,
            'z_score':      z_score,
            'cmf':          cmf_nu,
            'vol_ratio':    vol_ratio,
            'adx':          adx_nu,
            'atr_pct':      atr_pct,
            'pos_52w':      pos52,
            'p1d':          p1d,
            'p1w':          p1w,
        }
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
#  SCREENER WRAPPER  —  SORTERAR CANDIDATES PER STRATEGI
# ════════════════════════════════════════════════════════════════════════════

def run_screener(all_hist, bench_close, tickers, as_of_date, cfg):
    results = []
    for ticker in tickers:
        if ticker not in all_hist:
            continue
        try:
            ind = compute_indicators(ticker, all_hist[ticker], bench_close,
                                     as_of_date, cfg.min_vol)
            if ind is None or ind.get('blocked', False):
                continue
            results.append({
                'ticker':      ticker,
                'price':       ind['_price'],
                'total_score': ind['total_score'],
                'kt_score':    ind['kt_score'],
                'mixed_score': ind['mixed_score'],
                'rsi':         ind.get('rsi', 50),
                'rs':          ind.get('rs', 0),
            })
        except:
            pass

    def mk(key, filt=None):
        flt = results if filt is None else [r for r in results if filt(r)]
        return [(r['ticker'], r[key], r['price'])
                for r in sorted(flt, key=lambda x: x[key], reverse=True)]

    korttid   = mk('kt_score')
    medel     = mk('mixed_score')
    total     = mk('total_score')
    defensive = mk('total_score', lambda r: r['rs'] > 0 and r['rsi'] < 65)

    return {'korttid': korttid, 'medel': medel, 'total': total, 'defensive': defensive}


# ─── SNABB CACHE-BASERAD VERSION ─────────────────────────────────────────────

def _g(arr, i, default=np.nan):
    """Hämtar arr[i] säkert; returnerar default vid out-of-range eller nan."""
    if i < 0 or i >= len(arr):
        return default
    v = arr[i]
    return default if np.isnan(v) else v


def compute_indicators_fast(ticker, tc: TickerCache, i: int, min_vol: float):
    """
    O(1)-version av compute_indicators.
    Alla serier är redan beräknade i TickerCache tc.
    Parametern i är dagsindex i tc.dates/tc.close_arr etc.

    Returnerar samma dict-struktur som compute_indicators(),
    eller None om tickern inte klarar filter.
    """
    if i < MIN_HISTORY - 1:
        return None

    kurs = _g(tc.close_arr, i)
    if np.isnan(kurs) or kurs <= 0:
        return None

    # Volymfilter: vol_avg22 har shift(1) → sista dagen är alltid NaN i screener-läge.
    # Fallback 1: föregående dag. Fallback 2: snitt av senaste 22 dagar inkl idag.
    vol_avg = _g(tc.vol_avg22, i)
    if np.isnan(vol_avg) or vol_avg <= 0:
        vol_avg = _g(tc.vol_avg22, i - 1, 0.0)   # dagen innan (ej NaN)
    if np.isnan(vol_avg) or vol_avg <= 0:
        # Räkna om direkt från vol_arr: inkludera dagens volym
        slice_v = tc.vol_arr[max(0, i - 21):i + 1]
        vol_avg = float(np.nanmean(slice_v)) if len(slice_v) >= 3 else 0.0
    avg_vol_sek = vol_avg * kurs
    if avg_vol_sek < min_vol and avg_vol_sek > 0:
        return None

    # ─── Avkastning ───
    def avk(d):
        j = i - d
        if j < 0: j = 0
        prev = tc.close_arr[j]
        if np.isnan(prev) or prev <= 0: return 0.0
        return round((kurs / prev - 1) * 100, 1)

    p1d = avk(1)
    p1w = avk(5)

    # ─── Relativ styrka (förberäknad) ───
    rs   = _g(tc.rs21, i, 0.0)
    rs2w = _g(tc.rs10, i, 0.0)

    # ─── EMA-stack ───
    e9v   = _g(tc.ema9,   i, kurs)
    e21v  = _g(tc.ema21,  i, kurs)
    e50v  = _g(tc.ema50,  i, kurs)
    e200v = _g(tc.ema200, i, kurs)
    ema_score = sum([kurs > e9v, e9v > e21v, e21v > e50v, e50v > e200v])
    ema_trend = {4:"upp",3:"svagt_upp",2:"flat",1:"svagt_ner",0:"ner"}[ema_score]

    # ─── RSI ───
    rsi14 = _g(tc.rsi14, i, 50.0)
    rsi5  = _g(tc.rsi5,  i, 50.0)
    rsi14 = max(0.0, min(100.0, rsi14))

    # RSI bullish divergens: lägre kurs men högre RSI-botnar
    rsi_bull_div = False
    if i >= 20:
        c_window   = tc.close_arr[i-20:i+1]
        rsi_window = tc.rsi14[i-20:i+1]
        c5   = c_window[-5:];   rsi5w = rsi_window[-5:]
        c20w = c_window;        rsi20w = rsi_window
        if (np.nanmin(c5) < np.nanmin(c20w) and
                np.nanmin(rsi5w) > np.nanmin(rsi20w)):
            rsi_bull_div = True

    # ─── Bollinger ───
    bb_bw_nu  = _g(tc.bb_bw,   i, 0.04)
    bb_pctB   = _g(tc.bb_pctB, i, 0.5)
    # Rullande 40d genomsnitt av bb_bw för "tight"-detektering
    bw_window = tc.bb_bw[max(0, i-39):i+1]
    bb_bw_hist = float(np.nanmean(bw_window)) if len(bw_window) >= 10 else bb_bw_nu
    bb_squeeze_tight = bb_bw_nu < bb_bw_hist * 0.7

    # ─── Squeeze ───
    squeeze_on  = bool(_g(tc.sq_on,  i, 0.0) > 0.5)
    squeeze_mom = _g(tc.sq_mom, i, 0.0)
    # Antal squeeze-dagar i rad (senaste 15)
    sq_window = tc.sq_on[max(0, i-14):i+1]
    sq_bars   = int(np.nansum(sq_window > 0.5))

    # ─── ATR ───
    atr14v = _g(tc.atr14, i, kurs * 0.02)
    atr50v = _g(tc.atr50, i, kurs * 0.02)
    atr_exp = atr14v / (atr50v + 1e-10)
    atr_pct = round(atr14v / (kurs + 1e-10) * 100, 1)

    # ─── Stochastic ───
    sk_nu = _g(tc.stoch_k, i, 50.0)
    sd_nu = _g(tc.stoch_d, i, 50.0)
    sk_prev = _g(tc.stoch_k, i-1, sk_nu)
    sd_prev = _g(tc.stoch_d, i-1, sd_nu)
    stoch_cross = (sk_prev < sd_prev and sk_nu > sd_nu and sk_nu < 80)
    stoch_os    = (sk_nu < 20 and sd_nu < 20)

    # ─── CMF ───
    cmf_nu = _g(tc.cmf, i, 0.0)

    # ─── ADX ───
    adx_nu = _g(tc.adx, i, 20.0)
    dip_nu = _g(tc.dip, i, 25.0)
    dim_nu = _g(tc.dim, i, 25.0)
    adx_up = bool(dip_nu > dim_nu)

    # ─── OBV ───
    obv_up    = bool(_g(tc.obv,      i) > _g(tc.obv_ema20, i))
    obv_slope = _g(tc.obv_slope20, i, 0.0)   # v2: förberäknad normaliserad slope

    # ─── v2: Nya indikatorer ───
    # SMA200 hard block
    above_sma200  = bool(_g(tc.above_sma200, i, 1.0) > 0.5)
    sma200_rising = bool(_g(tc.sma200_slope, i, 0.0) > 0.05)
    sma50_gt_200  = bool(_g(tc.sma50, i, kurs) > _g(tc.sma200, i, kurs * 0.9))

    # 6m + 12m momentum (starkast historiskt bevisade CAGR-faktorer)
    p3m  = _g(tc.p3m,  i, 0.0)
    p6m  = _g(tc.p6m,  i, 0.0)
    p12m = _g(tc.p12m, i, 0.0)

    # RS-acceleration
    rs_acc = _g(tc.rs_acc, i, 0.0)

    # StochRSI
    srsi_k = _g(tc.stochrsi_k, i, 50.0)
    srsi_d = _g(tc.stochrsi_d, i, 50.0)
    srsi_k_prev = _g(tc.stochrsi_k, i-1, srsi_k)
    srsi_d_prev = _g(tc.stochrsi_d, i-1, srsi_d)
    stochrsi_cross = (srsi_k_prev < srsi_d_prev and srsi_k >= srsi_d and srsi_k < 60)
    stochrsi_os    = bool(srsi_k < 20)

    # RSI-lutning
    rsi_slope = _g(tc.rsi_slope5, i, 0.0)

    # Institutionellt flöde (förberäknad)
    ud_vol = _g(tc.up_dn_vol, i, 1.0)
    inst_signal = ("buying" if ud_vol > 1.5 else "selling" if ud_vol < 0.7 else "neutral")

    # ─── Z-score ───
    m20v = _g(tc.m20, i, kurs)
    s20v = _g(tc.s20, i, 1.0)
    z_score = (kurs - m20v) / (s20v + 1e-10)
    z_score = round(z_score if np.isfinite(z_score) else 0.0, 2)

    # ─── Volymratios ───
    vol_snitt    = _g(tc.vol_avg22, i, float(np.nanmean(tc.vol_arr[max(0,i-22):i])) + 1e-10)
    vol_5d       = float(np.nanmean(tc.vol_arr[max(0, i-4):i+1]))
    vol_1d       = _g(tc.vol_arr, i, vol_snitt)
    vol_ratio    = round(vol_5d       / (vol_snitt + 1e-10), 2)
    vol_ratio_1d = round(vol_1d       / (vol_snitt + 1e-10), 2)

    vol_acc = False
    if i >= 3:
        v3 = tc.vol_arr[i-2:i+1]
        if not any(np.isnan(v3)):
            vol_acc = (v3[0] < v3[1] < v3[2] and v3[2] > vol_snitt * 1.2)

    # ─── Bas-detektion ───
    i_bas = False
    if i >= 20:
        c20 = tc.close_arr[i-19:i+1]
        v20 = tc.vol_arr[i-19:i+1]
        c20_mean = float(np.nanmean(c20))
        if c20_mean > 0:
            bas_r = (float(np.nanmax(c20)) - float(np.nanmin(c20))) / c20_mean
            v20_mean   = float(np.nanmean(v20))
            v_all_mean = float(np.nanmean(tc.vol_arr[:i+1]))
            i_bas = bool(bas_r < 0.12 and v20_mean < v_all_mean * 0.6)

    # ─── Rekyl-setup ───
    rekyl_setup = (ema_trend in ("upp","svagt_upp") and
                   kurs < e9v and e9v > e21v and p1w < -2)

    # ─── 52v position + Drawdown ───
    d52    = min(252, i + 1)
    c_win  = tc.close_arr[max(0, i - d52 + 1):i+1]
    high52 = float(np.nanmax(c_win))
    low52  = float(np.nanmin(c_win))
    pos52  = round((kurs - low52) / (high52 - low52 + 1e-10), 3)

    # Max drawdown senaste 120 dagar
    c120 = tc.close_arr[max(0, i-119):i+1]
    peak_120 = np.maximum.accumulate(np.where(np.isnan(c120), 0, c120))
    dd_arr   = (c120 / (peak_120 + 1e-10) - 1) * 100
    max_dd   = round(float(np.nanmin(dd_arr)), 1)

    # ─── Trend R² ───
    trend_r2 = 0.0
    if i >= 60:
        y_log = np.log(tc.close_arr[i-59:i+1] + 1e-10)
        x_log = np.arange(60, dtype=float)
        if not np.any(np.isnan(y_log)):
            cov   = np.cov(x_log, y_log)
            r2    = float((cov[0,1]**2) / (cov[0,0] * cov[1,1] + 1e-10))
            trend_r2 = round(r2 if y_log[-1] > y_log[0] else -r2 * 0.5, 3)

    # ─── Momentum-acceleration (5d vs prev 5d) ───
    acc = 0.0
    if i >= 25:
        r5_now  = (tc.close_arr[i]   / (tc.close_arr[i-5]  + 1e-10) - 1) * 100
        r5_prev = (tc.close_arr[i-5] / (tc.close_arr[i-10] + 1e-10) - 1) * 100
        acc = float(r5_now - r5_prev)

    # ════════════════════════════════════════════════════════
    #  SCORE-BERÄKNING  v2  (6m/12m momentum, SMA200 block,
    #  StochRSI, RS-acc, koherens-multiplikator)
    # ════════════════════════════════════════════════════════
    RSI_MAX = 88
    RS_EXIT = -12.0
    DD_MAX  = -60.0

    # ── Hard blockers — bara de verkliga extremerna stoppas ──
    # SMA200: mjuk spärr istället för hård. Tillåt turnarounds om RS-acc är stark.
    # Motivering: De explosivaste rörelserna börjar ofta UNDER SMA200.
    sma200_hard_block = (
        (not above_sma200) and rs_acc < 5.0   # under SMA200 OCH svag acceleration = blockera
    )
    blocked = (
        sma200_hard_block
        or rsi14 > RSI_MAX
        or rs < RS_EXIT
        or max_dd < DD_MAX
    )
    if blocked:
        # Returnera fortfarande data (ej None) — penalty-systemet hanterar kvalitet
        # Bara extrema fall blockeras helt
        extreme = (rsi14 > 95) or (rs < -25) or (max_dd < -80)
        if extreme:
            return {
                '_price': kurs, 'total_score': 2, 'kt_score': 1, 'mixed_score': 2,
                'rsi': rsi14, 'rs': rs, 'max_dd': max_dd, 'ema_trend': ema_trend,
                'cmf': cmf_nu, 'z_score': z_score, 'adx': adx_nu,
                'vol_ratio': vol_ratio, 'atr_pct': atr_pct, 'pos_52w': pos52,
                'p1d': p1d, 'p1w': p1w, 'p6m': p6m, 'p12m': p12m,
                'rs_acc': rs_acc, 'squeeze_on': squeeze_on, 'blocked': True,
            }

    # ── F1: Momentum (v2: 12m + 6m + RS-acc dominerar) ──
    f1 = 0.0
    # 12m momentum — starkast historiskt bevisat, men inte dödsdom om lågt
    if   p12m > 60:  f1 += 24
    elif p12m > 35:  f1 += 18
    elif p12m > 15:  f1 += 12
    elif p12m > 0:   f1 += 6    # Höjt 5→6: belöna även svag uppgång
    elif p12m > -10: f1 -= 1    # Mildrat -3→-1: platt år ska inte blockera
    else:            f1 -= 8    # Mildrat -12→-8
    # 6m momentum
    if   p6m > 35:  f1 += 18
    elif p6m > 20:  f1 += 12
    elif p6m > 8:   f1 += 7
    elif p6m > 0:   f1 += 3    # Höjt 2→3
    elif p6m > -5:  f1 -= 1    # Mildrat -3→-1: sideled OK
    else:           f1 -= 7    # Mildrat -10→-7
    # 3m momentum
    if   p3m > 20:  f1 += 8
    elif p3m > 8:   f1 += 5
    elif p3m > 0:   f1 += 1
    elif p3m < -10: f1 -= 6
    # RS vs benchmark
    if   rs >= 15:  f1 += 20
    elif rs >= 10:  f1 += 14
    elif rs >= 5:   f1 += 8
    elif rs >= 2:   f1 += 3
    elif rs >= -1:  f1 += 0
    elif rs >= -6:  f1 -= 5
    else:           f1 -= 15
    # RS-acceleration (v2)
    if   rs_acc > 8:  f1 += 10
    elif rs_acc > 3:  f1 += 5
    elif rs_acc > 0:  f1 += 2
    elif rs_acc < -5: f1 -= 7
    # Kortsiktig moment
    if rs2w >= 6:   f1 += 8
    elif rs2w >= 3: f1 += 4
    if p1w >= 8:    f1 += 8
    elif p1w >= 3:  f1 += 4
    elif p1w < -8:  f1 -= 6
    if acc > 5:     f1 += 8
    elif acc > 2:   f1 += 4
    elif acc < -5:  f1 -= 5
    # EMA-stack
    f1 += {"upp":12,"svagt_upp":6,"flat":0,"svagt_ner":-5,"ner":-12}.get(ema_trend, 0)
    f1_n = max(0, min(100, (f1 / 85) * 100))

    # ── F2: Squeeze/Breakout ──
    f2 = 0.0
    if squeeze_on:
        f2 += (32 if sq_bars >= 12 else 25 if sq_bars >= 8 else
               18 if sq_bars >= 5  else 10 if sq_bars >= 3 else 5)
        f2 += 12 if squeeze_mom > 0 else -8
    elif bb_squeeze_tight: f2 += 10
    if   atr_exp > 2.5: f2 += 12
    elif atr_exp > 1.8: f2 += 8
    elif atr_exp > 1.3: f2 += 4
    elif atr_exp < 0.4: f2 += 6
    pB = bb_pctB
    # v2: momentum-filosofi → belöna BB-topp (breakout), inte dip
    if   pB > 0.85:          f2 += 10
    elif pB > 0.65:          f2 += 5
    elif pB < 0.05:          f2 += 4   # lägre än i v1 (mean-rev ej primärt)
    elif 0.40 <= pB <= 0.65: f2 += 2
    f2_n = max(0, min(100, (f2 / 62) * 100))

    # ── F3: Oscillatorer (v2: momentum-zon RSI 55-75 = ideal) ──
    f3 = 0.0
    if   rsi14 > RSI_MAX:       f3 -= 20
    elif 70 < rsi14 <= RSI_MAX: f3 += 6
    elif 58 <= rsi14 <= 70:     f3 += 18  # sweet spot momentum
    elif 50 <= rsi14 < 58:      f3 += 10
    elif 40 <= rsi14 < 50:      f3 -= 4
    elif 30 <= rsi14 < 40:      f3 -= 12
    else:                        f3 -= 20
    # RSI-lutning (v2)
    if   rsi_slope > 8:  f3 += 8
    elif rsi_slope > 3:  f3 += 4
    elif rsi_slope > 0:  f3 += 1
    elif rsi_slope < -6: f3 -= 6
    # RSI-divergens (fortfarande värdefull)
    if rsi_bull_div: f3 += 12
    # StochRSI (v2)
    if stochrsi_cross: f3 += 10
    if stochrsi_os:    f3 += 6
    # Stochastic
    if stoch_cross: f3 += 8
    if stoch_os:    f3 += 5
    f3_n = max(0, min(100, (f3 / 72) * 100))

    # ── F4: Volym/Flöde ──
    f4 = 0.0
    vr = vol_ratio
    if   vr >= 12: f4 += 28
    elif vr >= 7:  f4 += 20
    elif vr >= 4:  f4 += 13
    elif vr >= 2:  f4 += 7
    elif vr >= 1.2: f4 += 2
    elif vr < 0.4:  f4 -= 5
    vr1 = vol_ratio_1d
    if   vr1 >= 10: f4 += 14
    elif vr1 >= 5:  f4 += 8
    elif vr1 >= 2:  f4 += 3
    if vol_acc:                    f4 += 10
    if obv_up and obv_slope > 0:  f4 += 12
    elif not obv_up:               f4 -= 7
    if   cmf_nu > 0.25:  f4 += 12
    elif cmf_nu > 0.10:  f4 += 7
    elif cmf_nu > 0.0:   f4 += 3
    elif cmf_nu < -0.25: f4 -= 12
    elif cmf_nu < -0.10: f4 -= 5
    # Institutionellt flöde (v2: förberäknad)
    if inst_signal == "buying":   f4 += 12
    elif inst_signal == "selling": f4 -= 10
    f4_n = max(0, min(100, (f4 / 85) * 100))

    # ── F5: Trendkvalitet + SMA200-struktur ──
    f5 = 0.0
    # SMA200-struktur: belöna om ovanför, men straffa inte lika hårt under
    # (mjuk spärr gäller redan via blocked-logiken)
    f5 += 16 if above_sma200 else -8    # Mildare straff -25→-8 för under SMA200
    f5 += 12 if sma50_gt_200  else -8   # Golden cross-struktur (mildare -12→-8)
    f5 += 10 if sma200_rising else -5   # SMA200 lutar uppåt
    # ADX
    if   adx_nu > 45 and adx_up:  f5 += 16
    elif adx_nu > 30 and adx_up:  f5 += 10
    elif adx_nu > 20 and adx_up:  f5 += 5
    elif adx_nu > 30:              f5 -= 6
    elif adx_nu < 15:              f5 -= 3
    if dip_nu > dim_nu + 8:  f5 += 10
    elif dip_nu > dim_nu + 3: f5 += 5
    elif dim_nu > dip_nu + 8: f5 -= 10
    elif dim_nu > dip_nu + 3: f5 -= 5
    if   trend_r2 >= 0.80: f5 += 12
    elif trend_r2 >= 0.60: f5 += 8
    elif trend_r2 >= 0.40: f5 += 3
    elif trend_r2 < 0:     f5 -= 7
    f5_n = max(0, min(100, (f5 / 90) * 100))

    # ── F6: Setup (v2: breakout-filosofi, köp styrka) ──
    f6 = 0.0
    # 52v-position: BELÖNA HÖG position (nära topp = breakout-kandidat)
    if   pos52 >= 0.90: f6 += 20
    elif pos52 >= 0.75: f6 += 13
    elif pos52 >= 0.60: f6 += 6
    elif pos52 >= 0.45: f6 += 2
    elif pos52 < 0.25:  f6 -= 8
    elif pos52 < 0.15:  f6 -= 16
    if i_bas:       f6 += 15
    if rekyl_setup: f6 += 12
    # Z-score: i momentum-system → z > 0 (ovan medel) är bra
    if   z_score > 2.5:  f6 -= 6
    elif z_score > 1.0:  f6 += 5
    elif z_score > 0:    f6 += 2
    elif z_score < -2.5: f6 -= 10
    elif z_score < -1.5: f6 -= 4
    dd = max_dd
    if   dd >= -5:   f6 += 8
    elif dd >= -12:  f6 += 4
    elif dd >= -30:  f6 += 0
    elif dd >= -45:  f6 -= 5
    else:            f6 -= 14
    f6_n = max(0, min(100, (f6 / 65) * 100))

    f7_n = 50.0  # regim hanteras externt av backtest-motorn

    # ── F8: Explosionsfaktor ──
    f8 = 0.0
    if   p1d >= 10: f8 += 22
    elif p1d >= 6:  f8 += 14
    elif p1d >= 3:  f8 += 6
    elif p1d <= -7: f8 -= 8
    if i_bas: f8 += 14
    if   vr >= 12:  f8 += 18
    elif vr >= 7:   f8 += 10
    elif vr >= 4:   f8 += 5
    if squeeze_on and squeeze_mom > 0: f8 += 12
    if rs_acc > 5: f8 += 8   # v2: accelererande RS = explosion potential
    f8_n = max(0, min(100, (f8 / 82) * 100))

    # ── Viktad totalpoäng (v2: F1+F5 dominerar) ──
    wts = {"f1":0.22,"f2":0.11,"f3":0.11,"f4":0.12,"f5":0.20,"f6":0.10,"f7":0.07,"f8":0.07}
    raw_total = (
        f1_n*wts["f1"] + f2_n*wts["f2"] + f3_n*wts["f3"] + f4_n*wts["f4"] +
        f5_n*wts["f5"] + f6_n*wts["f6"] + f7_n*wts["f7"] + f8_n*wts["f8"]
    )

    # ── Koherens-multiplikator (v2): bonus om kärntriangeln alla stark ──
    n_strong = sum([
        f1_n >= 60,    # stark momentum
        f5_n >= 60,    # stark trend + SMA200-struktur
        f4_n >= 55,    # starkt flöde
        sma50_gt_200,  # golden cross
    ])
    coh = (1.15 if n_strong >= 4 else
           1.08 if n_strong >= 3 else
           0.88 if n_strong <= 1 else 1.0)
    total = int(round(max(0, min(100, raw_total * coh))))

    # ── KT-score (korttid: squeeze + volym + oversold i upptrend) ──
    kt = 0
    if squeeze_on:
        kt += 22 + min(18, sq_bars * 2)
        if squeeze_mom > 0: kt += 12
    elif bb_squeeze_tight: kt += 10
    if   vr >= 12: kt += 28
    elif vr >= 7:  kt += 20
    elif vr >= 4:  kt += 13
    elif vr >= 2:  kt += 6
    if p1d >= 10:  kt += 22
    elif p1d >= 6: kt += 14
    elif p1d >= 3: kt += 6
    if i_bas:            kt += 18
    if rekyl_setup:      kt += 14
    if rsi_bull_div:     kt += 18
    if stochrsi_cross:   kt += 14  # v2
    if stochrsi_os:      kt += 8   # v2
    if stoch_cross:      kt += 10
    if stoch_os:         kt += 6
    if z_score < -2.5:   kt += 16
    elif z_score < -2.0: kt += 10
    elif z_score < -1.5: kt += 5
    if cmf_nu > 0.10:    kt += 8
    if vol_acc:          kt += 10
    if rs_acc > 3:       kt += 6   # v2
    # v2: KT kräver SMA200-struktur
    if not above_sma200: kt = max(0, kt - 20)
    kt = max(0, min(100, kt))

    mixed = round(0.55 * total + 0.45 * kt)

    return {
        '_price':       kurs,
        'total_score':  total,
        'kt_score':     kt,
        'mixed_score':  mixed,
        'blocked':      False,
        'rsi':          rsi14,
        'rs':           rs,
        'max_dd':       max_dd,
        'ema_trend':    ema_trend,
        'squeeze_on':   squeeze_on,
        'z_score':      z_score,
        'cmf':          cmf_nu,
        'vol_ratio':    vol_ratio,
        'adx':          adx_nu,
        'atr_pct':      atr_pct,
        'pos_52w':      pos52,
        'p1d':          p1d,
        'p1w':          p1w,
        'p6m':          p6m,
        'p12m':         p12m,
        'rs_acc':       rs_acc,
        'above_sma200': above_sma200,
    }


def run_screener_fast(ticker_cache, date, tickers, min_vol):
    """
    Cache-baserad screener: O(1) per indikator per ticker.
    v3: Hårdare filtrering — bara högkvalitativa candidates passerar.
    """
    results = []
    date64  = np.datetime64(date, 'ns')

    for ticker in tickers:
        tc = ticker_cache.get(ticker)
        if tc is None:
            continue
        i = int(np.searchsorted(tc.dates, date64, side='right')) - 1
        if i < MIN_HISTORY - 1:
            continue
        try:
            ind = compute_indicators_fast(ticker, tc, i, min_vol)
            if ind is None or ind.get('blocked', False):
                continue
            results.append({
                'ticker':       ticker,
                'price':        ind['_price'],
                'total_score':  ind['total_score'],
                'kt_score':     ind['kt_score'],
                'mixed_score':  ind['mixed_score'],
                'rsi':          ind.get('rsi', 50),
                'rs':           ind.get('rs', 0.0),
                'adx':          ind.get('adx', 0.0),
                'p6m':          ind.get('p6m', 0.0),
                'p12m':         ind.get('p12m', 0.0),
                'rs_acc':       ind.get('rs_acc', 0.0),
                'above_sma200': ind.get('above_sma200', True),
                'atr_pct':      ind.get('atr_pct', 2.0),
                'vol_ratio':    ind.get('vol_ratio', 1.0),
                'above_sma50':  ind.get('above_sma200', True),  # konservativ fallback
            })
        except Exception:
            pass

    # Hämta above_sma50 + ATR-expansion direkt från cache
    date64 = np.datetime64(date, 'ns')
    for r in results:
        tc = ticker_cache.get(r['ticker'])
        if tc is None:
            continue
        i = int(np.searchsorted(tc.dates, date64, side='right')) - 1
        if i < 0:
            continue
        try:
            sma50_v  = _g(tc.sma50,  i, 0.0)
            close_v  = _g(tc.close_arr, i, 0.0)
            r['above_sma50'] = bool(close_v > sma50_v and sma50_v > 0)

            # ATR-expansion: ATR_14 / ATR_50 — mäter om volatilitet expanderar
            # Värde > ATR_EXPANSION_MIN → trend börjar expandera (kraftigt momentum-signal)
            atr14v = _g(tc.atr14, i, 0.0)
            atr50v = _g(tc.atr50, i, max(atr14v, 1e-10))
            r['atr_exp'] = float(atr14v / (atr50v + 1e-10)) if atr50v > 0 else 1.0

            # Riskjusterat momentum-score (hedgefond-trick #2):
            # Belönar starka trends med låg volatilitet — stabila uppgångar > spikiga
            p6m  = r['p6m']
            p12m = r['p12m']
            rs_a = r['rs_acc']
            vol  = r['atr_pct'] if r['atr_pct'] > 0 else 5.0
            raw_mom = 0.5 * p6m + 0.3 * p12m + 0.2 * rs_a
            r['mom_rank_score'] = raw_mom / vol   # riskjusterat: momentum per volatilitet-enhet
        except:
            r['atr_exp']       = 1.0
            r['mom_rank_score'] = 0.0

    def passes_entry(r, horizon):
        """
        Entryfilter per horisont — ALLA krav måste vara uppfyllda.
        SMA200-spärren är nu mjuk: under SMA200 tillåts om rs_acc är stark (turnaround).
        """
        ef = ENTRY_FILTER[horizon]
        # SMA200: mjuk spärr — blockera bara om BÅDE under SMA200 OCH svag RS-acc
        if ef.get('require_above_sma200', True):
            if not r['above_sma200'] and r.get('rs_acc', 0.0) < 5.0:
                return False
        if ef.get('require_above_sma50', False) and not r.get('above_sma50', True):
            return False
        # Relativ styrka och trend
        if r['rs'] < ef['min_rs']:
            return False
        if r['adx'] < ef['min_adx']:
            return False
        # Momentum-historik
        if r['p6m'] < ef['min_p6m']:
            return False
        # Volym
        if r.get('vol_ratio', 1.0) < ef.get('min_vol_ratio', 0.0):
            return False
        # Poängtrösklar
        if horizon == 'korttid' and r['kt_score'] < ef['min_kt_score']:
            return False
        if r['total_score'] < ef['min_total_score']:
            return False
        # Konfluens — sänkta trösklar för att fånga mer aggressiva setups
        strong = sum([
            r['rs']            >= 3.0,
            r['adx']           >= 22.0,
            r['total_score']   >= 60,
            r['p6m']           >= 8.0,
            r['rs_acc']        >= 2.0,
            r['p12m']          >= 10.0,
        ])
        return strong >= ef['min_confluence']

    def mk(key, horizon, extra_filt=None):
        """
        Bygger kandidatlista med tre filter:
        1. passes_entry — grundläggande kvalitetskrav
        2. ATR-expansion — volatilitet expanderar (ATR_14/ATR_50 > tröskeln)
        3. Momentum-ranking — köp bara topp MOMENTUM_TOP_PCT% sorterat på
           riskjusterat momentum-score (0.5*p6m + 0.3*p12m + 0.2*rs_acc) / ATR%
           Detta eliminerar brus-trades och höjer Sharpe utan att sänka CAGR.
        """
        flt = [r for r in results if passes_entry(r, horizon)]
        if extra_filt:
            flt = [r for r in flt if extra_filt(r)]

        # ATR-expansion filter: kräv att volatiliteten expanderar för medel/swing
        # (korttid tillåts utan — bounce-setups kan ha kontraherande vol)
        if horizon in ('medel', 'swing'):
            expanded = [r for r in flt if r.get('atr_exp', 1.0) >= ATR_EXPANSION_MIN]
            # Fallback: om inga kandidater klarar expansion-kravet, tillåt alla
            if expanded:
                flt = expanded

        if not flt:
            return []

        # Sortera primärt på riskjusterat momentum (mom_rank_score),
        # sekundärt på det horisont-specifika score-nyckeln
        flt_sorted = sorted(flt,
                            key=lambda x: (x.get('mom_rank_score', 0.0), x[key]),
                            reverse=True)

        # Ta bara topp MOMENTUM_TOP_PCT% — eliminerar brus-kandidaterna
        top_n = max(1, int(len(flt_sorted) * MOMENTUM_TOP_PCT))
        top_candidates = flt_sorted[:top_n]

        return [(r['ticker'], r[key], r['price'], r['atr_pct'])
                for r in top_candidates]

    return {
        'korttid':   mk('kt_score',    'korttid'),
        'medel':     mk('mixed_score', 'medel'),
        'total':     mk('total_score', 'swing'),
        'defensive': [(r['ticker'], r['total_score'], r['price'], r['atr_pct'])
                      for r in sorted(
                          [r for r in results
                           if r['rs'] > 1.0 and r['rsi'] < 60
                           and r['total_score'] >= 55 and r['above_sma200']
                           and r.get('above_sma50', True)],
                          key=lambda x: x['total_score'], reverse=True)],
    }


# ════════════════════════════════════════════════════════════════════════════
#  POSITION  +  SLOT TRACKER
# ════════════════════════════════════════════════════════════════════════════


# Extra config for screener
TOP_N_DEFAULT = 25
OUTPUT_FOLDER = 'Screener_Output'


def fetch_all_history(tickers, start, end, bench_ticker=BENCHMARK_TICKER):
    """Hämtar all historik via yfinance. Returnerar (dict, bench_series)."""
    extra = int(MIN_HISTORY * 2.2)   # 2.2× → ~440 dagar → ~300 handelsdagar (marginal för SMA200)
    s_dt  = pd.Timestamp(start) - pd.Timedelta(days=extra)
    e_dt  = pd.Timestamp(end)   + pd.Timedelta(days=10)
    s_str = s_dt.strftime('%Y-%m-%d')
    e_str = e_dt.strftime('%Y-%m-%d')

    print("\n[INFO] Hämtar historik {} → {}".format(s_str, e_str))
    print("[INFO] {} tickers + benchmark...".format(len(tickers)))

    all_hist   = {}
    batch_size = 80

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        pct   = int((i / len(tickers)) * 100)
        sys.stdout.write("\r   [{:3d}%] Batch {}/{}  tickers {}-{}  ok={}   ".format(
            pct,
            i // batch_size + 1,
            math.ceil(len(tickers) / batch_size),
            i + 1, min(i + batch_size, len(tickers)),
            len(all_hist)))
        sys.stdout.flush()
        try:
            raw = yf.download(
                ' '.join(batch),
                start=s_str, end=e_str,
                auto_adjust=True, progress=False,
                group_by='ticker', threads=True,
                timeout=30,
            )
            if raw.empty:
                continue
            for ticker in batch:
                try:
                    if len(batch) > 1:
                        th = raw[ticker].copy()
                    else:
                        th = raw.copy()
                    th = flatten(th)
                    th.dropna(subset=['Close'], inplace=True)
                    if len(th) >= MIN_HISTORY:
                        all_hist[ticker] = th
                except:
                    pass
        except Exception as e:
            pass
        time.sleep(0.25)

    print("\n[INFO] {} / {} tickers laddade".format(len(all_hist), len(tickers)))

    # Benchmark
    bench_close = None
    try:
        br = yf.download(bench_ticker, start=s_str, end=e_str,
                         auto_adjust=True, progress=False)
        br = flatten(br)
        bench_close = br['Close'].dropna()
        print("[INFO] Benchmark ({}): {} dagar".format(bench_ticker, len(bench_close)))
    except Exception as e:
        print("[VARNING] Benchmark misslyckades: {}".format(e))

    return all_hist, bench_close


# ════════════════════════════════════════════════════════════════════════════
#  BACKTEST-MOTOR
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
#  OUTPUT  —  TABELL + CSV
# ════════════════════════════════════════════════════════════════════════════

# Kolumner att visa i terminaltabellen
_COLS = ['Ticker','Namn','Land','Total','KT','RS','RS_acc','RSI','ADX',
         'p6m%','p12m%','p1d%','ATR_exp','Squeeze','EMA_trend','SMA200','SMA50','Kurs']

def _fmt_val(col, val):
    fmts = {
        'Ticker':'{:<12}','Namn':'{:<22}','Land':'{:<4}',
        'Total':'{:>5}','KT':'{:>5}',
        'RS':'{:>7.1f}','RS_acc':'{:>7.1f}','RSI':'{:>5.1f}','ADX':'{:>5.1f}',
        'p6m%':'{:>7.1f}','p12m%':'{:>7.1f}','p1d%':'{:>6.1f}',
        'ATR_exp':'{:>7.2f}','Squeeze':'{:>7}','EMA_trend':'{:<10}',
        'SMA200':'{:>6}','SMA50':'{:>5}','Kurs':'{:>10.2f}',
    }
    try:
        return fmts.get(col, '{:<10}').format(val)
    except (ValueError, TypeError):
        return str(val)[:10].ljust(10)


def print_table(ranked: list, horisont: str, top_n: int):
    if not ranked:
        print("  (inga kandidater passerade entryfilter för {})".format(horisont))
        return
    show  = ranked[:top_n]
    w     = 150
    print("\n" + "═"*w)
    print("  TOPP {:3d}  ─  {}  ({} kandidater totalt klarat filter)".format(
        len(show), horisont.upper(), len(ranked)))
    print("═"*w)
    header = "  {:>4}  ".format("#") + "".join(_fmt_val(c, c) for c in _COLS)
    print(header)
    print("  " + "─"*(len(header)-2))
    for rank, r in enumerate(show, 1):
        row = "  {:>4}  ".format(rank) + "".join(_fmt_val(c, r.get(c,'')) for c in _COLS)
        print(row)


def save_csv(results_by_h: dict, datum, out_folder: str) -> str:
    os.makedirs(out_folder, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M')
    path = os.path.join(out_folder, 'screener_{}_{}.csv'.format(
        str(datum)[:10].replace('-',''), ts))
    rows = []
    for hor, ranked in results_by_h.items():
        for rank, r in enumerate(ranked, 1):
            row = {'Rank': rank, 'Horisont': hor}
            row.update({k: v for k, v in r.items() if not k.startswith('_')})
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    return path


# ════════════════════════════════════════════════════════════════════════════
#  HTML RAPPORT
# ════════════════════════════════════════════════════════════════════════════

def _calc_levels(r: dict) -> dict:
    """
    Beräknar stop-loss, prisnivåer och estimat från tekniska indikatorer.
    Allt baserat på data som redan finns i screener-resultatet — ingen ny data-hämtning.
    """
    kurs    = float(r.get('Kurs',     100))
    atr_pct = float(r.get('ATR%',     2.0))   # ATR i % av kurs
    pos52   = float(r.get('Pos52v%',  50)) / 100.0
    p6m     = float(r.get('p6m%',      0))
    p12m    = float(r.get('p12m%',     0))
    rs      = float(r.get('RS',        0))
    rs_acc  = float(r.get('RS_acc',    0))
    adx     = float(r.get('ADX',      20))
    rsi     = float(r.get('RSI',      50))
    total   = int(r.get('Total',      50))
    squeeze = r.get('Squeeze', '') == '✓'
    ema_t   = r.get('EMA_trend', 'flat')

    atr_kr  = kurs * atr_pct / 100.0

    # ── STOP LOSS (tre nivåer) ───────────────────────────────────────────────
    # Konservativ: 1.5× ATR (tajt, för korttid)
    sl_konservativ = round(kurs - 1.5 * atr_kr, 2)
    # Standard:    2× ATR (standard momentum)
    sl_standard    = round(kurs - 2.0 * atr_kr, 2)
    # Generös:     3× ATR (swing, ger mer andrum)
    sl_generous    = round(kurs - 3.0 * atr_kr, 2)

    sl_pct_konservativ = round(-1.5 * atr_pct, 1)
    sl_pct_standard    = round(-2.0 * atr_pct, 1)
    sl_pct_generous    = round(-3.0 * atr_pct, 1)

    # ── PRISNIVÅER (motstånd + stöd) ────────────────────────────────────────
    # Beräknar 52v-high och 52v-low ur pos52 + kurs
    # pos52 = (kurs - low52) / (high52 - low52)
    # Om pos52 = 0 → kurs = low52; om pos52 = 1 → kurs = high52
    # Vi löser: high52 = kurs + (1 - pos52) / pos52 × (kurs - low52)
    # Förenklat: använd ATR-expansion som proxy
    estimated_range_pct = max(atr_pct * 15, abs(p6m) * 0.8)  # estimat på 6m-range
    if pos52 > 0.05:
        low52_est  = round(kurs * (1 - pos52 * estimated_range_pct / 100), 2)
        high52_est = round(kurs + (1 - pos52) * estimated_range_pct / 100 * kurs, 2)
    else:
        low52_est  = round(kurs * 0.80, 2)
        high52_est = round(kurs * 1.25, 2)

    # Nästa motstånd (ATR-baserat + momentum)
    mom_boost = 1 + min(rs_acc * 0.003, 0.10)  # upp till +10% för stark RS-acc
    motstand_1 = round(kurs * (1 + atr_pct * 2 / 100) * mom_boost, 2)
    motstand_2 = round(kurs * (1 + atr_pct * 4 / 100) * mom_boost, 2)

    # ── PRISMÅL (3 scenarion) ────────────────────────────────────────────────
    # Baseras på: momentum-historik, ATR, RS, ADX-styrka
    # Score-baserat confidence interval
    confidence = total / 100.0   # 0.0 – 1.0

    # Bas-estimat: extrapol av 6m-momentum, justerat för score
    base_6m_return = p6m * 0.6 * confidence   # konservativt, ej full extrapolering
    # Squeeze-bonus: komprimerad volatilitet → trolig explosion
    squeeze_bonus = 0.08 * confidence if squeeze else 0
    # ADX-boost: starka trender fortsätter
    adx_boost = max(0, (adx - 20) / 100 * 0.15) * confidence

    # Scenario 1: Bull (bra momentum fortsätter)
    bull_mult = 1 + max(base_6m_return/100, 0.03) + squeeze_bonus + adx_boost
    mål_bull  = round(kurs * min(bull_mult, 1.60), 2)          # max +60%
    mål_bull_pct = round((bull_mult - 1) * 100, 1)

    # Scenario 2: Base (trend håller men avtar)
    base_mult = 1 + max(base_6m_return/100 * 0.5, 0.01)
    mål_base  = round(kurs * min(base_mult, 1.30), 2)
    mål_base_pct = round((base_mult - 1) * 100, 1)

    # Scenario 3: Bear (momentum vänder)
    bear_mult = 1 - atr_pct * 3 / 100
    mål_bear  = round(kurs * max(bear_mult, 0.70), 2)
    mål_bear_pct = round((bear_mult - 1) * 100, 1)

    # ── TAKE PROFIT (tre nivåer) ─────────────────────────────────────────────
    # Korttid: +8-15%, Medel: +15-30%, Swing: +30-60%
    tp_1 = round(kurs * (1 + max(0.06, atr_pct * 3   / 100)), 2)  # konservativ TP
    tp_2 = round(kurs * (1 + max(0.12, atr_pct * 6   / 100)), 2)  # standard TP
    tp_3 = round(kurs * (1 + max(0.25, atr_pct * 12  / 100)), 2)  # optimistisk TP
    tp_1_pct = round((tp_1/kurs - 1)*100, 1)
    tp_2_pct = round((tp_2/kurs - 1)*100, 1)
    tp_3_pct = round((tp_3/kurs - 1)*100, 1)

    # ── TRAILING STOP (aktiveras efter vinstlockning) ─────────────────────
    # När kursen nått TP1: drag upp stop till break-even + lite
    # När kursen nått TP2: trailing 8-12% under topp
    trail_after_tp1 = round(kurs * 1.005, 2)          # break-even + 0.5%
    trail_pct_tp2   = round(atr_pct * 2.5, 1)          # t.ex. 5% trailing under toppkurs
    trail_pct_tp3   = round(atr_pct * 4.0, 1)          # vidare för swing

    # ── SIGNALSTYRKA ─────────────────────────────────────────────────────────
    # "Hur stark är denna setup egentligen?"
    signals = []
    if squeeze:                     signals.append("🔥 Squeeze aktiv — komprimerad energi väntar utbrott")
    if rs_acc > 3:                  signals.append("⚡ RS-acceleration — relativ styrka ökar snabbt")
    if adx > 30 and r.get('RS','') != '':
        if float(r.get('RS', 0)) > 5: signals.append("🚀 Stark ADX ({:.0f}) + positiv RS — tydlig upptrend".format(adx))
    if rsi > 50 and rsi < 70:       signals.append("✅ RSI i momentum-zon ({:.0f}) — varken överköpt eller svag".format(rsi))
    if p12m > 30:                   signals.append("📈 12m momentum +{:.0f}% — aktien leder marknaden över tid".format(p12m))
    if pos52 > 0.85:                signals.append("🎯 Nära 52v-topp ({:.0f}%) — potential breakout setup".format(pos52*100))
    if ema_t == 'upp':              signals.append("📊 Perfekt EMA-stack (9>21>50>200) — alla MA pekar upp")
    if float(r.get('CMF', 0)) > 0.15: signals.append("💰 Starkt institutionellt flöde (CMF={:.2f})".format(float(r.get('CMF',0))))

    # ── VARNINGSSIGNALER ─────────────────────────────────────────────────────
    varningar = []
    if rsi > 75:                    varningar.append("⚠️  RSI {:.0f} — aktien kan vara överköpt kortsiktigt".format(rsi))
    if atr_pct > 4:                 varningar.append("⚠️  Hög volatilitet (ATR {:.1f}%) — stor daglig rörelse".format(atr_pct))
    if p6m < 0:                     varningar.append("⚠️  Negativ 6m-avkastning ({:.1f}%) — trenden svag".format(p6m))
    if float(r.get('MaxDD%', 0)) < -25: varningar.append("⚠️  Stor drawdown ({:.0f}%) under senaste 120 dagar".format(float(r.get('MaxDD%',0))))

    # ── RISK/REWARD ──────────────────────────────────────────────────────────
    rr_konservativ = round((mål_base - kurs) / max(kurs - sl_standard, 0.01), 2)
    rr_bull        = round((mål_bull - kurs)  / max(kurs - sl_generous, 0.01), 2)

    return {
        'sl_konservativ': sl_konservativ, 'sl_pct_k': sl_pct_konservativ,
        'sl_standard':    sl_standard,    'sl_pct_s': sl_pct_standard,
        'sl_generous':    sl_generous,    'sl_pct_g': sl_pct_generous,
        'tp_1': tp_1, 'tp_1_pct': tp_1_pct,
        'tp_2': tp_2, 'tp_2_pct': tp_2_pct,
        'tp_3': tp_3, 'tp_3_pct': tp_3_pct,
        'trail_after_tp1': trail_after_tp1,
        'trail_pct_tp2': trail_pct_tp2,
        'trail_pct_tp3': trail_pct_tp3,
        'low52':   low52_est,  'high52':  high52_est,
        'motstand_1': motstand_1, 'motstand_2': motstand_2,
        'mål_bull': mål_bull,   'mål_bull_pct': mål_bull_pct,
        'mål_base': mål_base,   'mål_base_pct': mål_base_pct,
        'mål_bear': mål_bear,   'mål_bear_pct': mål_bear_pct,
        'rr_konservativ': rr_konservativ, 'rr_bull': rr_bull,
        'signals':  signals,   'varningar': varningar,
    }


def _score_bar(val, max_val=100, color='#3fb950', width=120) -> str:
    """Returnerar en SVG progress-bar."""
    pct = min(100, max(0, val / max_val * 100))
    filled = int(width * pct / 100)
    hue = 120 if val >= 65 else (60 if val >= 45 else 0)
    clr = 'hsl({},85%,45%)'.format(hue)
    return (
        '<svg width="{w}" height="10" style="vertical-align:middle">'
        '<rect width="{w}" height="10" rx="5" fill="#21262d"/>'
        '<rect width="{f}" height="10" rx="5" fill="{c}"/>'
        '</svg>'
        '<span style="margin-left:5px;font-size:11px;color:#8b949e">{v}</span>'
    ).format(w=width, f=filled, c=clr, v=int(val))


def _gauge(val, label, min_v=0, max_v=100) -> str:
    """Mini-cirkulär gauge som SVG."""
    pct  = min(1.0, max(0.0, (val - min_v) / (max_v - min_v)))
    angle= pct * 180 - 90  # -90 till +90 grader
    rad  = math.radians(angle)
    cx, cy, r = 30, 30, 22
    nx = cx + r * math.cos(rad - math.pi/2)
    ny = cy + r * math.sin(rad - math.pi/2)
    hue = int(pct * 120)
    clr = 'hsl({},85%,45%)'.format(hue)
    return (
        '<div style="display:inline-block;text-align:center;margin:4px 6px">'
        '<svg width="60" height="45">'
        '<path d="M8,35 A{r},{r} 0 0,1 52,35" fill="none" stroke="#21262d" stroke-width="5"/>'
        '<path d="M{cx},{cy} L{nx:.1f},{ny:.1f}" stroke="{c}" stroke-width="3" stroke-linecap="round"/>'
        '<circle cx="{cx}" cy="{cy}" r="3" fill="{c}"/>'
        '</svg>'
        '<div style="font-size:10px;color:#8b949e;margin-top:-8px">{lbl}</div>'
        '<div style="font-size:12px;font-weight:bold;color:{c}">{val}</div>'
        '</div>'
    ).format(r=r, cx=cx, cy=cy, nx=nx, ny=ny, c=clr, lbl=label, val=round(val,1))


def _momentum_sparkline(p1d, p1w, p3m, p6m, p12m) -> str:
    """Liten sparkline-bar för momentum över tid."""
    vals = [p1d*4, p1w*2, p3m*0.5, p6m*0.25, p12m*0.15]  # normalisera till liknande skala
    labels = ['1d','1v','3m','6m','12m']
    bars = ''
    for i, (v, lbl) in enumerate(zip(vals, labels)):
        h   = min(40, max(2, abs(v) * 0.8))
        clr = '#3fb950' if v >= 0 else '#f85149'
        y   = 42 - h
        bars += '<rect x="{}" y="{}" width="14" height="{}" rx="2" fill="{}"/>'.format(
            i*18, y, h, clr)
        bars += '<text x="{}" y="56" font-size="8" fill="#8b949e" text-anchor="middle">{}</text>'.format(
            i*18+7, lbl)
    return '<svg width="100" height="60">{}</svg>'.format(bars)


def _asset_signal(composite):
    """KÖPLÄGE / BEVAKA / NEUTRAL / SÄLJ baserat på composite score."""
    if composite >= 70:  return 'KÖPLÄGE',  '#00e676'
    if composite >= 52:  return 'BEVAKA',   '#40c4ff'
    if composite >= 35:  return 'NEUTRAL',  '#90a4ae'
    return 'SÄLJ', '#ef5350'


def _sparkline_svg(vals, w=120, h=40):
    """Enkel SVG sparkline av en lista med float-värden."""
    if not vals or len(vals) < 2:
        return ''
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(vals):
        x = round(2 + i / (len(vals)-1) * (w-4), 1)
        y = round(2 + (1 - (v-mn)/rng) * (h-4), 1)
        pts.append(f'{x},{y}')
    clr = '#00e676' if vals[-1] >= vals[0] else '#ef5350'
    last_pct = (vals[-1]/vals[0]-1)*100 if vals[0] else 0
    sign = '+' if last_pct >= 0 else ''
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="#0d1117" rx="3"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{clr}" stroke-width="1.5"/>'
            f'<text x="{w-2}" y="{h-2}" text-anchor="end" fill="{clr}" font-size="9" font-weight="bold">'
            f'{sign}{last_pct:.1f}%</text></svg>')


def generate_html(results_by_h: dict, datum, regime_info: dict,
                  entry_filters: dict, out_folder: str) -> str:
    """
    Genererar HTML-rapport i samma stil som UES v6:
    - Top-bar med logga
    - Marknadsbild (regim)
    - Kortbaserat grid med filter-knappar
    - Modal-popup med tabbar: Analys | Signaler | Targets | Faktorer | Indikatorer
    """
    import json as _json

    os.makedirs(out_folder, exist_ok=True)
    ts        = datetime.now().strftime('%Y%m%d_%H%M')
    datum_str = str(datum)[:10]
    fname     = 'screener_{}_{}.html'.format(datum_str.replace('-',''), ts)
    path      = os.path.join(out_folder, fname)

    reg_fas   = regime_info.get('fas',    'Osäker')
    reg_enkel = regime_info.get('enkel',  'Osäker')
    reg_avk1m = regime_info.get('avk_1m', 0.0)
    reg_avk3m = regime_info.get('avk_3m', 0.0)
    reg_color = ('#00e676' if 'Bull' in reg_enkel else
                 '#ef5350' if 'Bear' in reg_enkel else '#ffd700')
    risk_on   = 'Bull' in reg_enkel

    # ── Samla alla unika aktier, dedupa på ticker, sortera på composite ──────
    seen, all_ranked = set(), []
    for h in ['swing', 'medel', 'korttid']:
        for r in results_by_h.get(h, []):
            if r['Ticker'] not in seen:
                seen.add(r['Ticker'])
                all_ranked.append(dict(r, _horisont_primary=h))
    all_ranked.sort(key=lambda x: x.get('_composite', x['Total']), reverse=True)

    # ── Bygg ALL_ASSETS JSON för modal JS ────────────────────────────────────
    assets_json = []
    for r in all_ranked:
        lv   = _calc_levels(r)
        comp = r.get('_composite', float(r['Total']))
        sig, sigclr = _asset_signal(comp)

        # Signal-ikoner
        icons = []
        if r.get('Squeeze') == '✓':       icons.append('🗜️')
        if r.get('Inst') == 'buying':      icons.append('🐋')
        if r.get('ATR_exp', 1) > 1.3:     icons.append('🔒')
        if float(r.get('RS_acc', 0)) > 3: icons.append('⚡')
        if float(r.get('p1v%', 0)) < -3:  icons.append('↩️')
        if float(r.get('RSI', 50)) < 30:  icons.append('↗️')

        # Sparkline från momentum-serie (historiska procentpunkter, normaliserat)
        spark_vals = [
            float(r.get('p12m%', 0)) * 0.1,
            float(r.get('p6m%', 0))  * 0.2,
            float(r.get('p3m%', 0))  * 0.5,
            float(r.get('p1v%', 0))  * 2.0,
            float(r.get('p1d%', 0))  * 5.0,
        ]

        # Faktor-poäng (0-100) skattas från befintliga data
        f_mom   = min(100, max(0, 50 + float(r.get('RS', 0))*2 + float(r.get('p3m%', 0))*0.5))
        f_trend = min(100, max(0, float(r['Total']) * 0.9 + (10 if r.get('SMA200')=='✓' else -10)))
        f_bo    = min(100, max(0, float(r.get('KT', 0)) * 1.1))
        f_vol   = min(100, max(0, 40 + float(r.get('VolRatio', 1))*10 + float(r.get('CMF', 0))*30))
        f_osc   = min(100, max(0, 100 - abs(float(r.get('RSI', 50)) - 55) * 2.5))
        f_qual  = min(100, max(0, float(r['Total'])))

        # Analys-text
        ema  = r.get('EMA_trend', '?')
        rs   = float(r.get('RS', 0))
        rs_a = float(r.get('RS_acc', 0))
        rsi  = float(r.get('RSI', 50))
        atr  = float(r.get('ATR%', 2))
        p6m  = float(r.get('p6m%', 0))
        p12m = float(r.get('p12m%', 0))
        pen  = r.get('_penalty', 0)

        ema_txt = ('EMA-stacken är bullish (9>21>50>200) — väletablerad upptrend.' if ema == 'upp'
                   else 'EMA-strukturen är svagt positiv.' if 'upp' in ema
                   else 'EMA-stacken är flat/neutral.')
        rs_txt  = ('RS stark ({:.1f}pp), accelererar ({:+.1f}pp) — tydlig outperformance.'.format(rs, rs_a) if rs > 3
                   else 'RS positiv ({:.1f}pp).'.format(rs) if rs > 0
                   else 'RS svag ({:.1f}pp) — bevaka noggrant.'.format(rs))
        sqz_txt = '🔥 Bollinger Squeeze aktiv — utbrott väntar. ' if r.get('Squeeze') == '✓' else ''
        inst_w  = 'aktivt köpande' if r.get('Inst') == 'buying' else ('aktivt sälj' if r.get('Inst') == 'selling' else 'neutralt')
        pen_txt = ' (⚠️ Poängavdrag: {:.0f}pt pga otillräckliga kriterier — bevaka med försiktighet.)'.format(pen) if pen > 5 else ''
        ai_text = ('{} ({}) handlas till {:.2f}. Score: {:.0f}/100{}. {} {} {}Institutionellt flöde: {}. ATR {:.1f}%/dag.'
                   .format(r['Ticker'], r.get('Namn', ''), r['Kurs'], comp,
                           pen_txt, ema_txt, rs_txt, sqz_txt, inst_w, atr))

        assets_json.append({
            'ticker':    r['Ticker'],
            'name':      r.get('Namn', r['Ticker'])[:28],
            'land':      r.get('Land', '?'),
            'sektor':    r.get('Sektor', '?')[:20],
            'typ':       r.get('Typ', 'aktie'),
            'cap':       r.get('Cap', 'large'),
            'horisont':  r.get('_horisont_primary', 'medel'),
            'price':     round(r['Kurs'], 4),
            'score':     round(comp, 1),
            'total_raw': r['Total'],
            'kt_score':  r['KT'],
            'signal':    sig,
            'signal_clr': sigclr,
            'grade':     r.get('_grade', 'C'),
            'penalty':   round(pen, 1),
            'bonus':     round(r.get('_bonus', 0), 1),
            'icons':     ' '.join(icons),
            'spark_svg': _sparkline_svg(spark_vals),
            'p1d':   round(float(r.get('p1d%', 0)), 2),
            'p1w':   round(float(r.get('p1v%', 0)), 1),
            'p1m':   round(float(r.get('p3m%', 0)) * 0.35, 1),  # proxy
            'p3m':   round(float(r.get('p3m%', 0)), 1),
            'p6m':   round(p6m, 1),
            'p12m':  round(p12m, 1),
            'rsi':   round(rsi, 1),
            'rsi5':  round(rsi - 2, 1),
            'rs':    round(rs, 1),
            'rs_acc':round(rs_a, 1),
            'adx':   round(float(r.get('ADX', 20)), 1),
            'cmf':   round(float(r.get('CMF', 0)), 3),
            'atr_pct':    round(atr, 1),
            'atr_exp':    round(float(r.get('ATR_exp', 1)), 2),
            'vol_ratio':  round(float(r.get('VolRatio', 1)), 1),
            'pos52w':     round(float(r.get('Pos52v%', 50)) / 100, 3),
            'z_score':    round(float(r.get('Z', 0)), 2),
            'max_dd90':   round(float(r.get('MaxDD%', 0)), 1),
            'mom_rank':   round(float(r.get('MomRank', 0)), 2),
            'ema_trend':  ema,
            'above_sma200': r.get('SMA200') == '✓',
            'above_sma50':  r.get('SMA50')  == '✓',
            'squeeze':    r.get('Squeeze') == '✓',
            'obv_div':    float(r.get('RS_acc', 0)) > 4,
            'atr_comp':   float(r.get('ATR_exp', 1)) < 0.85,
            'macd_cross': False,
            'rsi_recov':  rsi < 35,
            'vol_dryup':  float(r.get('VolRatio', 1)) < 0.6,
            'pullback':   2 if r.get('EMA_trend') in ('svagt_upp', 'upp') and float(r.get('p1v%', 0)) < -2 else 0,
            'rel_vol_spike': float(r.get('VolRatio', 1)) >= 5,
            'vol_expansion': float(r.get('ATR_exp', 1)) >= 1.4,
            'mom_decel':  float(r.get('p1v%', 0)) < 0 and float(r.get('p3m%', 0)) > 5,
            'ai_text':    ai_text,
            # Levels
            'stop':       lv['sl_standard'],
            'stop_tight': lv['sl_konservativ'],
            'stop_wide':  lv['sl_generous'],
            'tp1':        lv['tp_1'], 'tp1_pct': lv['tp_1_pct'],
            'tp2':        lv['tp_2'], 'tp2_pct': lv['tp_2_pct'],
            'tp3':        lv['tp_3'], 'tp3_pct': lv['tp_3_pct'],
            'trail_pct':  lv['trail_pct_tp2'],
            'bull_1m':    lv['mål_bull'], 'bull_3m': lv['mål_bull'],
            'base_1m':    lv['mål_base'], 'base_3m': lv['mål_base'],
            'bear_1m':    lv['mål_bear'], 'bear_3m': lv['mål_bear'],
            'upside_pct': lv['mål_bull_pct'],
            'downside_pct': lv['mål_bear_pct'],
            'rr_ratio':   lv['rr_bull'],
            'confidence': min(95, max(20, int(comp))),
            'cap_bonus':  round(float(r.get('RS_acc', 0)) * 0.5, 1),
            'konsistens': min(1.0, max(0, comp / 100)),
            'calmar':     round(comp / max(abs(float(r.get('MaxDD%', -10))), 1), 2),
            'streak':     3 if float(r.get('p3m%', 0)) > 10 else 1,
            'r2':         min(1.0, max(0, float(r['Total'])/100 * 0.9)),
            'ema_chain':  {'upp':4,'svagt_upp':3,'flat':2,'svagt_ner':1,'ner':0}.get(ema, 2),
            'obv_rising': float(r.get('RS', 0)) > 0,
            'sq_bars':    8 if r.get('Squeeze') == '✓' else 0,
            'f_mom':   round(f_mom, 0), 'f_trend': round(f_trend, 0),
            'f_bo':    round(f_bo, 0),  'f_vol':   round(f_vol, 0),
            'f_osc':   round(f_osc, 0), 'f_qual':  round(f_qual, 0),
        })

    # ── Statistik ────────────────────────────────────────────────────────────
    n_kop    = sum(1 for a in assets_json if a['score'] >= 70)
    n_bevaka = sum(1 for a in assets_json if 52 <= a['score'] < 70)
    n_total  = len(assets_json)
    best     = assets_json[0]['ticker'] if assets_json else '–'
    n_squeeze= sum(1 for a in assets_json if a['squeeze'])

    # Cap-distribution
    cap_dist = {}
    for a in assets_json:
        cap_dist[a['cap']] = cap_dist.get(a['cap'], 0) + 1

    # Topp-sektorer
    sek_dist = {}
    for a in assets_json:
        sek_dist[a['sektor']] = sek_dist.get(a['sektor'], 0) + 1
    top_sek = sorted(sek_dist.items(), key=lambda x: -x[1])[:8]

    # Aktiva teman (sektorer med score > 60)
    hot_themes = list(set(
        a['sektor'] for a in assets_json if a['score'] >= 60
    ))[:6]

    # ── Bygg kort-HTML ───────────────────────────────────────────────────────
    cap_cols = {
        'micro':'#ff6b6b','small':'#ffd700','mid':'#40c4ff',
        'large':'#8b949e','mega':'#6e7681','?':'#555'
    }
    def card_html(a):
        sc  = a['score']
        sig = a['signal']
        clr = a['signal_clr']
        cc  = cap_cols.get(a['cap'], '#555')
        p1m_c = '#00e676' if a['p1m'] >= 0 else '#ef5350'
        p3m_c = '#00e676' if a['p3m'] >= 0 else '#ef5350'
        typ_badge = ''
        if a['typ'] == 'etf':
            typ_badge = '<span style="font-size:8px;background:#40c4ff22;color:#40c4ff;border-radius:3px;padding:1px 4px;margin-left:3px">ETF</span>'
        elif a['typ'] == 'cert':
            typ_badge = '<span style="font-size:8px;background:#f9731622;color:#f97316;border-radius:3px;padding:1px 4px;margin-left:3px">CERT</span>'
        pen_badge = ('<span style="font-size:8px;background:#ffd70022;color:#ffd700;'
                     'border-radius:3px;padding:1px 4px;margin-left:3px">-{:.0f}pt</span>'.format(a['penalty'])
                     if a['penalty'] > 3 else '')
        return (
            '<div class="asset-card" data-ticker="{t}" data-typ="{typ}" data-score="{sc}" '
            'data-horisont="{hor}" data-signal="{sig_raw}" onclick="openModal(\'{t}\')" '
            'style="border-left:3px solid {clr}">'
            '<div class="card-top"><div>'
            '<div class="card-ticker">{t}{typ_b}{pen_b}'
            '<span style="font-size:9px;padding:1px 5px;border-radius:3px;'
            'background:{cc}22;color:{cc};border:1px solid {cc}55;margin-left:4px">'
            '{cap}</span></div>'
            '<div class="card-name">{nm}</div>'
            '<div style="font-size:9px;color:var(--muted)">{land} · {sek}</div>'
            '</div>'
            '<div style="text-align:right">'
            '<div class="card-score" style="color:{clr}">{sc:.0f}</div>'
            '<div style="font-size:10px;color:{clr}">{sig}</div>'
            '</div></div>'
            '<div class="card-rets">'
            '<div><span style="color:var(--muted);font-size:9px">1m</span>'
            ' <span style="color:{p1m_c};font-weight:bold">{p1m:+.1f}%</span></div>'
            '<div><span style="color:var(--muted);font-size:9px">3m</span>'
            ' <span style="color:{p3m_c};font-weight:bold">{p3m:+.1f}%</span></div>'
            '<div><span style="color:var(--muted);font-size:9px">RSI</span>'
            ' <span style="color:white">{rsi:.0f}</span></div>'
            '<div><span style="color:var(--muted);font-size:9px">RS</span>'
            ' <span style="color:{rs_c}">{rs:+.1f}</span></div>'
            '</div>'
            '<div class="signal-icons">{icons}</div>'
            '<div class="score-bar">'
            '<div style="background:{clr};width:{bar}%;height:3px;'
            'border-radius:2px;transition:width .3s"></div></div>'
            '</div>'
        ).format(
            t=a['ticker'], typ=a['typ'], sc=sc, hor=a['horisont'],
            sig_raw=a['signal'], sig=sig, clr=clr,
            typ_b=typ_badge, pen_b=pen_badge,
            cc=cc, cap=(a['cap'] or '?').upper(),
            nm=a['name'][:22], land=a['land'],
            sek=a['sektor'][:16],
            p1m=a['p1m'], p1m_c=p1m_c,
            p3m=a['p3m'], p3m_c=p3m_c,
            rsi=a['rsi'],
            rs=a['rs'], rs_c='#00e676' if a['rs'] >= 0 else '#ef5350',
            icons=a['icons'],
            bar=min(100, sc),
        )

    cards_html = ''.join(card_html(a) for a in assets_json)

    # ── Sektor-HTML ──────────────────────────────────────────────────────────
    sector_rows = ''.join(
        '<div style="display:flex;justify-content:space-between;margin:3px 0">'
        '<span>{}</span>'
        '<span style="background:#21262d;padding:1px 8px;border-radius:10px;color:var(--blue)">{}</span>'
        '</div>'.format(s[:20], n)
        for s, n in top_sek)

    hot_html = ''.join(
        '<span class="theme-tag">📌 {}</span>'.format(t[:16])
        for t in hot_themes)

    # ── Cap-distribution HTML ────────────────────────────────────────────────
    cap_labels_order = [('micro','#ff6b6b'), ('small','#ffd700'), ('mid','#40c4ff'),
                        ('large','#8b949e'), ('mega','#6e7681')]
    cap_dist_html = ''.join(
        '<span class="cap-d" style="background:{}22;color:{}">{}: {}</span>'.format(
            c, c, k.upper(), cap_dist.get(k, 0))
        for k, c in cap_labels_order)

    # ── HTML ──────────────────────────────────────────────────────────────────
    parts = []
    a = assets_json

    parts.append('''<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Master Screener v1.0 - {ts}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--border:#21262d;--text:#c9d1d9;
      --green:#00e676;--blue:#40c4ff;--gold:#ffd700;--orange:#f97316;
      --purple:#a78bfa;--red:#ef5350;--muted:#8b949e}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;
     font-size:13px;line-height:1.5}}
.top-bar{{background:#010409;border-bottom:1px solid var(--border);padding:10px 20px;
          display:flex;align-items:center;justify-content:space-between;
          position:sticky;top:0;z-index:100}}
.logo{{color:var(--gold);font-weight:bold;font-size:16px}}
.version{{color:var(--muted);font-size:11px}}
.main{{max-width:1600px;margin:0 auto;padding:16px 12px}}
.section{{background:var(--panel);border:1px solid var(--border);border-radius:12px;
          padding:16px;margin-bottom:16px}}
.section-title{{color:var(--gold);font-weight:bold;font-size:14px;margin-bottom:12px;
                display:flex;align-items:center;gap:8px}}
.macro-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.theme-tag{{background:#21262d;border:1px solid #30363d;border-radius:6px;
            padding:3px 8px;font-size:11px;color:var(--text);
            margin:3px 3px 3px 0;display:inline-block}}
.stat-row{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px}}
.stat{{background:#0d1117;border:1px solid var(--border);border-radius:8px;
       padding:8px 14px;text-align:center}}
.stat-val{{font-size:20px;font-weight:bold}}
.stat-lbl{{font-size:9px;color:var(--muted);margin-top:1px}}
.cap-dist{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}}
.cap-d{{font-size:11px;padding:3px 8px;border-radius:5px}}
.cards-filter{{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}}
.filter-btn{{background:#21262d;border:1px solid #30363d;border-radius:6px;
             padding:5px 12px;cursor:pointer;font-size:11px;color:var(--text);
             transition:all .15s}}
.filter-btn:hover,.filter-btn.active{{background:#388bfd22;border-color:#388bfd;color:#79c0ff}}
.cards-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}}
.asset-card{{background:#0d1117;border:1px solid var(--border);border-radius:10px;
             padding:12px;cursor:pointer;transition:all .2s}}
.asset-card:hover{{border-color:#30363d;transform:translateY(-1px);background:#161b22}}
.card-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}}
.card-ticker{{font-weight:bold;font-size:13px;color:white}}
.card-name{{font-size:10px;color:var(--muted);margin-top:1px}}
.card-score{{font-size:24px;font-weight:bold}}
.card-rets{{display:flex;gap:10px;margin-bottom:6px}}
.signal-icons{{font-size:14px;letter-spacing:2px;min-height:18px;margin-bottom:4px}}
.score-bar{{background:#21262d;height:3px;border-radius:2px}}
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);
                z-index:1000;overflow-y:auto;padding:20px}}
.modal{{background:#161b22;border:1px solid #30363d;border-radius:14px;
        max-width:820px;margin:0 auto;overflow:hidden}}
.modal-header{{background:#010409;padding:14px 18px;display:flex;
               justify-content:space-between;align-items:center;
               border-bottom:1px solid #21262d}}
.modal-close{{background:none;border:none;color:#8b949e;font-size:20px;
              cursor:pointer;padding:4px 8px}}
.modal-close:hover{{color:white}}
.modal-body{{padding:16px}}
.modal-tabs{{display:flex;gap:0;border-bottom:1px solid #21262d;margin-bottom:14px}}
.mtab{{padding:8px 16px;cursor:pointer;font-size:12px;color:#8b949e;border:none;
       background:none;border-bottom:2px solid transparent;transition:all .15s}}
.mtab.active{{color:white;border-bottom-color:var(--gold)}}
.mpanel{{display:none}}
.mpanel.active{{display:block}}
.ai-text{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;
          font-size:12px;line-height:1.7;color:#c9d1d9;white-space:pre-wrap}}
.ai-label{{display:flex;align-items:center;gap:6px;font-size:11px;color:#8b949e;
           margin-bottom:8px}}
.scenarios-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px}}
.ind-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:6px}}
.ind-item{{background:#0d1117;border-radius:6px;padding:6px 8px}}
.factor-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.sl-tp-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
.sl-box{{background:#0d1117;border-radius:8px;padding:12px}}
.sl-title{{font-size:11px;font-weight:bold;color:var(--muted);
           text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}}
.sl-row{{display:flex;justify-content:space-between;padding:4px 0;
          border-bottom:1px solid #21262d;font-size:12px}}
.sl-row:last-child{{border-bottom:none}}
@media(max-width:600px){{
  .cards-grid{{grid-template-columns:1fr 1fr}}
  .scenarios-grid,.sl-tp-grid,.macro-grid{{grid-template-columns:1fr}}
}}
footer{{text-align:center;padding:20px;color:#555;font-size:11px;
        border-top:1px solid #21262d}}
</style>
</head>
<body>
'''.format(ts=ts))

    # Top-bar
    parts.append('<div class="top-bar">')
    parts.append('<div><div class="logo">⚡ MASTER SCREENER v1.0</div>')
    parts.append('<div class="version">V9+ Scoring · Sverige + Europa · {}</div></div>'.format(datum_str))
    parts.append('<div style="color:var(--muted);font-size:11px">Ej investeringsrådgivning</div>')
    parts.append('</div>\n<div class="main">\n')

    # Marknadsbild
    parts.append('<div class="section"><div class="section-title">🌍 MARKNADSBILD</div>')
    parts.append('<div class="macro-grid"><div>')
    parts.append('<div style="color:var(--muted);font-size:11px">Marknadsfas: '
                 '<span style="color:{}">{}</span></div>'.format(reg_color, reg_fas))
    parts.append('<div style="color:var(--muted);font-size:11px">Regim: {}</div>'.format(reg_enkel))
    m1c = '#00e676' if reg_avk1m >= 0 else '#ef5350'
    m3c = '#00e676' if reg_avk3m >= 0 else '#ef5350'
    parts.append('<div style="color:var(--muted);font-size:11px">1m: <span style="color:{}">{:+.1f}%</span>'
                 '  ·  3m: <span style="color:{}">{:+.1f}%</span></div>'.format(m1c, reg_avk1m, m3c, reg_avk3m))
    parts.append('<div style="color:var(--muted);font-size:11px">Risk-On: {}</div>'.format('✅' if risk_on else '❌'))
    parts.append('</div><div>')
    parts.append('<div style="font-size:11px;color:var(--muted);margin-bottom:6px">🔥 Heta sektorer:</div>')
    parts.append('<div>{}</div>'.format(hot_html or '<span class="theme-tag">–</span>'))
    parts.append('<div style="font-size:11px;color:var(--muted);margin-top:8px">Topp sektorer:</div>')
    parts.append(sector_rows)
    parts.append('</div></div></div>\n')

    # Topp-aktier
    parts.append('<div class="section"><div class="section-title">🏆 ALLA {} VÄRDEPAPPER</div>'.format(n_total))
    parts.append('<div class="stat-row">')
    parts.append('<div class="stat"><div class="stat-val">{}</div><div class="stat-lbl">Screenade</div></div>'.format(n_total))
    parts.append('<div class="stat"><div class="stat-val" style="color:var(--green)">{}</div><div class="stat-lbl">Köpläge</div></div>'.format(n_kop))
    parts.append('<div class="stat"><div class="stat-val" style="color:var(--blue)">{}</div><div class="stat-lbl">Bevaka</div></div>'.format(n_bevaka))
    parts.append('<div class="stat"><div class="stat-val" style="color:var(--orange)">{}</div><div class="stat-lbl">🔥 Squeezes</div></div>'.format(n_squeeze))
    parts.append('<div class="stat"><div class="stat-val" style="color:var(--gold)">{}</div><div class="stat-lbl">Bästa just nu</div></div>'.format(best))
    parts.append('</div>')

    # Cap-distribution
    parts.append('<div class="cap-dist"><span style="font-size:11px;color:var(--muted);align-self:center">Storlek:</span>')
    parts.append(cap_dist_html)
    parts.append('</div>')

    # Filter-knappar
    parts.append('<div class="cards-filter">')
    btns = [
        ('all',    'Alla', True),
        ('KÖP',   '🟢 Köpläge', False),
        ('BEVAKA','🔵 Bevaka', False),
        ('swing',  '🚀 Swing', False),
        ('medel',  '📈 Medel', False),
        ('korttid','⚡ Korttid', False),
        ('SETUP',  '⚡ Setup', False),
        ('aktie',  'Aktier', False),
        ('etf',    'ETFer', False),
    ]
    for fid, flbl, active in btns:
        parts.append('<button class="filter-btn{}" onclick="filterCards(\'{}\',this)">{}</button>'.format(
            ' active' if active else '', fid, flbl))
    parts.append('</div>')

    parts.append('<div class="cards-grid" id="cards-grid">')
    parts.append(cards_html)
    parts.append('</div></div>\n')

    # Modal
    parts.append('''
<div class="modal-overlay" id="modal-overlay" onclick="if(event.target===this)closeModal()">
<div class="modal">
  <div class="modal-header">
    <div>
      <span id="modal-ticker" style="font-weight:bold;font-size:16px;color:white"></span>
      <span id="modal-cap-badge" style="margin-left:8px"></span>
      <span id="modal-name" style="color:var(--muted);font-size:12px;margin-left:8px"></span>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <span id="modal-price" style="font-size:14px;color:white;font-weight:bold"></span>
      <span id="modal-signal" style="font-size:12px;font-weight:bold"></span>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
  </div>
  <div class="modal-body">
    <div id="modal-rets" style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap"></div>
    <div id="modal-chart" style="margin-bottom:12px"></div>
    <div class="modal-tabs">
      <button class="mtab active" onclick="showMTab('mp-analys',this)">📝 Analys</button>
      <button class="mtab" onclick="showMTab('mp-signals',this)">⚡ Signaler</button>
      <button class="mtab" onclick="showMTab('mp-targets',this)">🎯 Targets</button>
      <button class="mtab" onclick="showMTab('mp-factors',this)">📊 Faktorer</button>
      <button class="mtab" onclick="showMTab('mp-inds',this)">🔬 Indikatorer</button>
    </div>

    <div id="mp-analys" class="mpanel active">
      <div class="ai-label">🤖 Automatisk teknisk analys</div>
      <div class="ai-text" id="modal-ai-text"></div>
      <div id="modal-setup-brief" style="margin-top:10px"></div>
    </div>

    <div id="mp-signals" class="mpanel">
      <div style="background:#0d1117;border-radius:8px;padding:12px;margin-bottom:10px">
        <div style="color:var(--gold);font-size:11px;margin-bottom:8px">✅ AKTIVA SETUP-SIGNALER</div>
        <div id="modal-setups" style="font-size:12px;line-height:2"></div>
      </div>
      <div style="background:#0d1117;border-radius:8px;padding:12px">
        <div style="color:var(--red);font-size:11px;margin-bottom:8px">⚠️ VARNINGAR</div>
        <div id="modal-warns" style="font-size:12px;line-height:2;color:var(--muted)"></div>
      </div>
    </div>

    <div id="mp-targets" class="mpanel">
      <div class="sl-tp-grid">
        <div class="sl-box">
          <div class="sl-title">🛡️ Stop-Loss</div>
          <div class="sl-row"><span style="color:var(--muted)">Tight (1.5× ATR)</span>
            <span id="modal-sl-tight" style="color:var(--gold);font-weight:bold"></span></div>
          <div class="sl-row"><span style="color:var(--muted)">Standard ⭐</span>
            <span id="modal-sl-std" style="color:var(--red);font-weight:bold"></span></div>
          <div class="sl-row"><span style="color:var(--muted)">Wide (3× ATR)</span>
            <span id="modal-sl-wide" style="color:var(--muted)"></span></div>
          <div style="font-size:10px;color:var(--muted);margin-top:6px"
               id="modal-trail-info"></div>
        </div>
        <div class="sl-box">
          <div class="sl-title">🎯 Take-Profit</div>
          <div class="sl-row"><span style="color:var(--muted)">TP1 – Ta halva</span>
            <span id="modal-tp1" style="color:#4ade80;font-weight:bold"></span></div>
          <div class="sl-row"><span style="color:var(--muted)">TP2 – Ta 3/4</span>
            <span id="modal-tp2" style="color:var(--green);font-weight:bold"></span></div>
          <div class="sl-row"><span style="color:var(--muted)">TP3 – Låt löpa</span>
            <span id="modal-tp3" style="color:#86efac;font-weight:bold"></span></div>
        </div>
      </div>
      <div class="scenarios-grid" id="modal-scenarios"></div>
      <div style="background:#0d1117;border-radius:8px;padding:12px" id="modal-rr"></div>
    </div>

    <div id="mp-factors" class="mpanel">
      <div class="factor-grid" id="modal-factors"></div>
    </div>

    <div id="mp-inds" class="mpanel">
      <div class="ind-grid" id="modal-indicators"></div>
    </div>
  </div>
</div>
</div>
''')

    # Footer
    parts.append('<footer>Master Screener v1.0 · Genererad {} · '
                 'Ej investeringsrådgivning · Handel på eget ansvar</footer>'.format(
                     datetime.now().strftime('%Y-%m-%d %H:%M')))
    parts.append('</div>\n')  # .main

    # ── JavaScript ────────────────────────────────────────────────────────────
    parts.append('<script>\nconst ALL_ASSETS=')
    parts.append(_json.dumps(assets_json, ensure_ascii=False))
    parts.append(';\n')

    parts.append('''
function openModal(ticker){
  const a=ALL_ASSETS.find(x=>x.ticker===ticker);
  if(!a)return;
  const capCols={micro:'#ff6b6b',small:'#ffd700',mid:'#40c4ff',large:'#8b949e',mega:'#6e7681','?':'#555'};
  const cc=capCols[a.cap]||'#555';
  document.getElementById('modal-ticker').textContent=a.ticker;
  document.getElementById('modal-name').textContent=a.name;
  document.getElementById('modal-cap-badge').innerHTML=
    `<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:${cc}22;color:${cc};border:1px solid ${cc}55">${(a.cap||'?').toUpperCase()}</span>`;
  document.getElementById('modal-price').textContent=a.price.toFixed(2)+' ';
  document.getElementById('modal-signal').style.color=a.signal_clr;
  document.getElementById('modal-signal').textContent=a.signal;
  document.getElementById('modal-rets').innerHTML=
    [['1d',a.p1d],['1v',a.p1w],['1m',a.p1m],['3m',a.p3m],['6m',a.p6m]].map(([l,v])=>{
      const c=v>=0?'#00e676':'#ef5350';
      return `<span><span style="color:#8b949e;font-size:10px">${l} </span><span style="color:${c};font-weight:bold">${v>=0?'+':''}${v.toFixed(1)}%</span></span>`;
    }).join('');
  document.getElementById('modal-chart').innerHTML=
    (a.spark_svg||'')+
    `<div style="display:flex;gap:12px;font-size:9px;color:#8b949e;margin-top:4px;flex-wrap:wrap">
      <span>— Historik</span>
      <span style="color:#00e676">🐂 Bull: ${a.bull_3m?a.bull_3m.toFixed(2):'–'}</span>
      <span style="color:#40c4ff">📊 Base: ${a.base_3m?a.base_3m.toFixed(2):'–'}</span>
      <span style="color:#ef5350">🐻 Bear: ${a.bear_3m?a.bear_3m.toFixed(2):'–'}</span>
      <span style="color:#ff6b35">⛔ Stop: ${a.stop?a.stop.toFixed(2):'–'}</span>
    </div>`;
  document.getElementById('modal-ai-text').textContent=a.ai_text||'Analys saknas.';
  // Setups
  const setups=[];
  if(a.squeeze)       setups.push('🗜️ Squeeze pågår – fjädern laddas');
  if(a.obv_div)       setups.push('🐋 OBV-divergens – institutionell ackumulation');
  if(a.atr_comp)      setups.push('🔒 ATR-kompression – volatilitet krymper');
  if(a.macd_cross)    setups.push('⚡ Färsk MACD-korsning – momentum skiftar');
  if(a.rsi_recov)     setups.push('↗️ RSI-återhämtning från oversold');
  if(a.vol_dryup)     setups.push('💧 Volym torkar på pullback');
  if(a.pullback>=2)   setups.push('↩️ Pullback till EMA21');
  if(a.rel_vol_spike) setups.push('🚀 Volymspike – stort intresse aktiverat');
  if(a.vol_expansion) setups.push('💥 Volatilitet expanderar – rörelse startar');
  const sbEl=document.getElementById('modal-setup-brief');
  if(setups.length>0){
    sbEl.innerHTML=`<div style="background:#0d1117;border-radius:8px;padding:10px;font-size:11px">
      <div style="color:#ffd700;font-size:10px;margin-bottom:6px">⚡ AKTIVA SETUP-SIGNALER</div>
      ${setups.map(s=>`<div style="color:#c9d1d9;margin-bottom:3px">✅ ${s}</div>`).join('')}</div>`;
  }else{ sbEl.innerHTML=''; }
  document.getElementById('modal-setups').innerHTML=
    setups.length?setups.map(s=>`<div>✅ ${s}</div>`).join(''):'<div style="color:#8b949e">Inga starka setup-signaler just nu.</div>';
  const warns=[];
  if(a.mom_decel)  warns.push(`Momentum decelererar`);
  if(a.rsi>76)     warns.push(`RSI ${a.rsi.toFixed(0)} – kortsiktigt överkört`);
  if(a.p1w>8)      warns.push(`+${a.p1w.toFixed(0)}% senaste veckan`);
  if(a.p1m>20)     warns.push(`+${a.p1m.toFixed(0)}% senaste månaden`);
  if(a.penalty>5)  warns.push(`Poängavdrag ${a.penalty.toFixed(0)}pt – ej alla kriterier uppfyllda`);
  document.getElementById('modal-warns').innerHTML=
    warns.length?warns.map(w=>`<div>⚠️ ${w}</div>`).join(''):'<div style="color:#8b949e">Inga varningar.</div>';
  // Stop / TP
  document.getElementById('modal-sl-tight').textContent=a.stop_tight.toFixed(2)+' ('+((a.stop_tight/a.price-1)*100).toFixed(1)+'%)';
  document.getElementById('modal-sl-std').textContent=a.stop.toFixed(2)+' ('+((a.stop/a.price-1)*100).toFixed(1)+'%)';
  document.getElementById('modal-sl-wide').textContent=a.stop_wide.toFixed(2)+' ('+((a.stop_wide/a.price-1)*100).toFixed(1)+'%)';
  document.getElementById('modal-trail-info').textContent='Trailing: vid TP1 → stop till breakeven. Från TP2 → trailing '+a.trail_pct.toFixed(1)+'% under toppkurs.';
  document.getElementById('modal-tp1').textContent=a.tp1.toFixed(2)+' (+'+a.tp1_pct.toFixed(1)+'%)';
  document.getElementById('modal-tp2').textContent=a.tp2.toFixed(2)+' (+'+a.tp2_pct.toFixed(1)+'%)';
  document.getElementById('modal-tp3').textContent=a.tp3.toFixed(2)+' (+'+a.tp3_pct.toFixed(1)+'%)';
  // Scenarios
  const sc=[
    {label:'🐂 Bull',col:'#00e676',m1:a.bull_1m,m3:a.bull_3m},
    {label:'📊 Base',col:'#40c4ff',m1:a.base_1m,m3:a.base_3m},
    {label:'🐻 Bear',col:'#ef5350',m1:a.bear_1m,m3:a.bear_3m},
  ];
  document.getElementById('modal-scenarios').innerHTML=sc.map(s=>`
    <div style="background:#0d1117;border-radius:10px;padding:12px;border-top:3px solid ${s.col}">
      <div style="color:${s.col};font-weight:bold;font-size:11px;margin-bottom:8px">${s.label}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;text-align:center">
        ${[['1 mån',s.m1],['3 mån',s.m3]].map(([lb,v])=>{
          const pct=((v/a.price-1)*100);
          return `<div><div style="color:#8b949e;font-size:9px">${lb}</div>
                  <div style="color:white;font-weight:bold;font-size:11px">${v?v.toFixed(2):'–'}</div>
                  <div style="color:${s.col};font-size:10px">${pct>=0?'+':''}${pct.toFixed(1)}%</div></div>`;
        }).join('')}
      </div></div>`).join('');
  const rrCol=a.rr_ratio>=2?'#00e676':a.rr_ratio>=1.5?'#ffd700':'#ef5350';
  document.getElementById('modal-rr').innerHTML=`
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:12px">
      <div><span style="color:#8b949e">Potential: </span><span style="color:#00e676;font-weight:bold">+${a.upside_pct.toFixed(1)}%</span></div>
      <div><span style="color:#8b949e">Risk: </span><span style="color:#ef5350;font-weight:bold">${a.downside_pct.toFixed(1)}%</span></div>
      <div><span style="color:#8b949e">R/R: </span><span style="color:${rrCol};font-weight:bold">${a.rr_ratio.toFixed(1)}:1</span></div>
      <div><span style="color:#8b949e">Konfidens: </span><span style="color:${a.confidence>=65?'#00e676':'#ffd700'};font-weight:bold">${a.confidence}%</span></div>
      <div><span style="color:#8b949e">MomRank: </span><span style="color:white">${a.mom_rank.toFixed(2)}</span></div>
      <div><span style="color:#8b949e">Grade: </span><span style="font-weight:bold;color:${['A','B'].includes(a.grade)?'#00e676':a.grade==='C'?'#ffd700':'#ef5350'}">${a.grade}</span></div>
    </div>`;
  // Faktorer
  const factors=[
    ['🚀 Momentum',a.f_mom,'#f97316'],
    ['📈 Trend',a.f_trend,'#40c4ff'],
    ['💥 Breakout',a.f_bo,'#ffd700'],
    ['📊 Volym',a.f_vol,'#00e676'],
    ['🔬 Oscillator',a.f_osc,'#a78bfa'],
    ['🏆 Kvalitet',a.f_qual,'#34d399'],
  ];
  document.getElementById('modal-factors').innerHTML=factors.map(([n,v,c])=>`
    <div style="background:#0d1117;border-radius:8px;padding:10px">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px">
        <span style="font-size:11px;color:#c9d1d9">${n}</span>
        <span style="font-size:13px;font-weight:bold;color:${c}">${v.toFixed(0)}</span>
      </div>
      <div style="background:#21262d;border-radius:3px;height:5px">
        <div style="background:${c};width:${v}%;height:100%;border-radius:3px"></div>
      </div></div>`).join('');
  // Indikatorer
  const rsiCol=a.rsi>78?'#ef5350':a.rsi>68?'#ffd700':(a.rsi>=45&&a.rsi<=64)?'#00e676':'#c9d1d9';
  const inds=[
    ['RSI 14',a.rsi.toFixed(0),rsiCol],
    ['ADX',a.adx.toFixed(0),a.adx>30?'#00e676':a.adx<15?'#ef5350':'white'],
    ['RS bench',(a.rs>=0?'+':'')+a.rs.toFixed(1),a.rs>=5?'#00e676':a.rs>=0?'#40c4ff':'#ef5350'],
    ['RS-acc',(a.rs_acc>=0?'+':'')+a.rs_acc.toFixed(1),a.rs_acc>3?'#00e676':'white'],
    ['EMA-kedja',a.ema_chain+'/4',a.ema_chain>=3?'#00e676':a.ema_chain<=1?'#ef5350':'#ffd700'],
    ['CMF',a.cmf>=0?'+'+a.cmf.toFixed(2):a.cmf.toFixed(2),a.cmf>0.1?'#00e676':a.cmf<-0.1?'#ef5350':'white'],
    ['Squeeze',a.squeeze?'🗜️ ON':'OFF',a.squeeze?'#f97316':'#8b949e'],
    ['ATR%',a.atr_pct.toFixed(1)+'%','white'],
    ['ATR-exp',a.atr_exp.toFixed(2),a.atr_exp>1.3?'#00e676':a.atr_exp<0.8?'#40c4ff':'white'],
    ['Vol-ratio',a.vol_ratio.toFixed(1)+'×',a.vol_ratio>=3?'#f97316':a.vol_ratio>=1.5?'#00e676':'#8b949e'],
    ['Z-score',a.z_score.toFixed(1),a.z_score<-1.5?'#40c4ff':a.z_score>1.5?'#ef5350':'white'],
    ['52v-läge',(a.pos52w*100).toFixed(0)+'%',a.pos52w>0.85?'#ffd700':'white'],
    ['6m%',a.p6m>=0?'+'+a.p6m.toFixed(1)+'%':a.p6m.toFixed(1)+'%',a.p6m>10?'#00e676':a.p6m<0?'#ef5350':'white'],
    ['12m%',a.p12m>=0?'+'+a.p12m.toFixed(1)+'%':a.p12m.toFixed(1)+'%',a.p12m>20?'#00e676':a.p12m<0?'#ef5350':'white'],
    ['SMA200',a.above_sma200?'✓ Ovan':'✗ Under',a.above_sma200?'#00e676':'#ef5350'],
    ['MaxDD 90d',a.max_dd90.toFixed(1)+'%',a.max_dd90>-5?'#00e676':a.max_dd90<-20?'#ef5350':'white'],
  ];
  document.getElementById('modal-indicators').innerHTML=inds.map(([l,v,c])=>
    `<div class="ind-item"><div style="color:#8b949e;font-size:9px">${l}</div>
     <div style="color:${c};font-weight:bold;font-size:12px">${v}</div></div>`).join('');
  document.getElementById('modal-overlay').style.display='block';
  document.body.style.overflow='hidden';
}

function closeModal(){
  document.getElementById('modal-overlay').style.display='none';
  document.body.style.overflow='';
}

function showMTab(id,btn){
  document.querySelectorAll('.mpanel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.mtab').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}

function filterCards(filter,btn){
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#cards-grid .asset-card').forEach(card=>{
    const t=card.dataset.ticker;
    const a=ALL_ASSETS.find(x=>x.ticker===t);
    let show=true;
    if(filter==='KÖP')    show=a&&a.score>=70;
    else if(filter==='BEVAKA')  show=a&&a.score>=52&&a.score<70;
    else if(filter==='swing')   show=a&&a.horisont==='swing';
    else if(filter==='medel')   show=a&&a.horisont==='medel';
    else if(filter==='korttid') show=a&&a.horisont==='korttid';
    else if(filter==='aktie')   show=a&&a.typ==='aktie';
    else if(filter==='etf')     show=a&&a.typ==='etf';
    else if(filter==='SETUP')   show=a&&(a.squeeze||a.obv_div||a.atr_comp||a.rsi_recov||a.vol_dryup||(a.pullback>=2)||a.rel_vol_spike||a.vol_expansion);
    card.style.display=show?'':'none';
  });
}

document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
</script>
</body></html>''')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))
    return path

def screen_date(ticker_cache: dict, datum, min_vol: float) -> list:
    """
    Screena alla tickers på ett givet datum.
    Returnerar komplett lista av result-dicts — alla som passerade grundfilter.
    """
    date64  = np.datetime64(datum, 'ns')
    results = []
    n_none = n_blocked = n_err = n_hist = 0

    for ticker, tc in ticker_cache.items():
        i = int(np.searchsorted(tc.dates, date64, side='right')) - 1
        if i < MIN_HISTORY - 1:
            n_hist += 1
            continue
        try:
            ind = compute_indicators_fast(ticker, tc, i, min_vol)
            if ind is None:
                n_none += 1
                continue
            if ind.get('blocked', False):
                n_blocked += 1
                continue

            # above_sma50 från cache
            sma50_v = _g(tc.sma50, i, 0.0)
            close_v = _g(tc.close_arr, i, 0.0)
            above_s50 = bool(close_v > sma50_v and sma50_v > 0)

            # ATR-expansion
            atr14v = _g(tc.atr14, i, 0.0)
            atr50v = _g(tc.atr50, i, max(atr14v, 1e-10))
            atr_exp = float(atr14v / (atr50v + 1e-10))

            # Riskjusterat momentum-score
            p6m  = ind.get('p6m',  0.0)
            p12m = ind.get('p12m', 0.0)
            rs_a = ind.get('rs_acc', 0.0)
            vol  = max(ind.get('atr_pct', 5.0), 0.1)
            mom_rank = (0.5*p6m + 0.3*p12m + 0.2*rs_a) / vol

            results.append({
                'Ticker':    ticker,
                'Namn':      _NAMN.get(ticker, ticker),
                'Land':      _LAND.get(ticker, '?'),
                'Sektor':    _SEKTOR.get(ticker, '?'),
                'Typ':       _TYP.get(ticker, 'aktie'),
                'Cap':       _CAP.get(ticker, '?'),
                'Kurs':      ind['_price'],
                'Total':     ind['total_score'],
                'KT':        ind['kt_score'],
                'Mixed':     ind['mixed_score'],
                'MomRank':   round(mom_rank, 2),
                'RSI':       ind.get('rsi', 50),
                'RS':        ind.get('rs',  0.0),
                'RS_acc':    round(ind.get('rs_acc', 0.0), 1),
                'ADX':       round(ind.get('adx', 0.0), 1),
                'CMF':       ind.get('cmf', 0.0),
                'p1d%':      ind.get('p1d', 0.0),
                'p1v%':      ind.get('p1w', 0.0),
                'p3m%':      ind.get('p3m', 0.0),
                'p6m%':      round(p6m, 1),
                'p12m%':     round(p12m, 1),
                'Pos52v%':   round(ind.get('pos_52w', 0)*100, 0),
                'ATR%':      ind.get('atr_pct', 2.0),
                'ATR_exp':   round(atr_exp, 2),
                'VolRatio':  ind.get('vol_ratio', 1.0),
                'Squeeze':   '✓' if ind.get('squeeze_on') else '',
                'EMA_trend': ind.get('ema_trend', '?'),
                'SMA200':    '✓' if ind.get('above_sma200') else '✗',
                'SMA50':     '✓' if above_s50 else '✗',
                'Inst':      ind.get('inst_signal', 'neutral'),
                'MaxDD%':    ind.get('max_dd', 0),
                'Z':         ind.get('z_score', 0),
                '_above_s50':above_s50,
                '_atr_exp':  atr_exp,
            })
        except Exception as e:
            n_err += 1
            if n_err <= 3:   # visa bara de första felen
                print(f"\n   [DEBUG] {ticker}: {type(e).__name__}: {e}")

    if n_none or n_blocked or n_err or n_hist:
        print("[DEBUG] screen_date: hist={} none={} blocked={} err={} ok={}".format(
            n_hist, n_none, n_blocked, n_err, len(results)))

    return results


def compute_composite_score(r: dict, horizon: str) -> float:
    """
    Beräknar ett sammansatt rankningspoäng för given horisont.
    INGA hårda filter — varje kriterium som inte uppfylls ger istället ett poängavdrag.
    Aktier som inte klarar SMA200 kan fortfarande visas men hamnar längre ner i listan.
    """
    ef      = ENTRY_FILTER[horizon]
    base    = float(r.get('Total', 0))
    penalty = 0.0

    # SMA200 — mjuk spärr: avdrag om under men inte blockering
    if r.get('SMA200', '✓') != '✓':
        rs_acc = float(r.get('RS_acc', 0))
        penalty += 0 if rs_acc >= 6.0 else (5 if rs_acc >= 3.0 else 12)

    # SMA50 (bara swing)
    if ef.get('require_above_sma50', False) and not r.get('_above_s50', True):
        penalty += 6

    # RS
    rs_gap = ef['min_rs'] - float(r.get('RS', 0))
    if rs_gap > 0:
        penalty += min(18, rs_gap * 2.5)

    # ADX
    adx_gap = ef['min_adx'] - float(r.get('ADX', 0))
    if adx_gap > 0:
        penalty += min(12, adx_gap * 0.8)

    # 6m momentum
    p6_gap = ef['min_p6m'] - float(r.get('p6m%', 0))
    if p6_gap > 0:
        penalty += min(10, p6_gap * 0.5)

    # Volym
    vr_gap = ef.get('min_vol_ratio', 0) - float(r.get('VolRatio', 1.0))
    if vr_gap > 0:
        penalty += min(8, vr_gap * 4)

    # KT-score (bara korttid)
    if horizon == 'korttid':
        kt_gap = ef['min_kt_score'] - float(r.get('KT', 0))
        if kt_gap > 0:
            penalty += min(10, kt_gap * 0.4)

    # Grundpoäng-gap
    score_gap = ef['min_total_score'] - base
    if score_gap > 0:
        penalty += min(15, score_gap * 0.5)

    # RSI extremt överköpt
    rsi = float(r.get('RSI', 50))
    if rsi > 85:
        penalty += 15
    elif rsi > 80:
        penalty += 8

    # Konfluens-bonus: aktier som har MÅNGA starka faktorer belönas
    strong = sum([
        float(r.get('RS', 0))     >= 4.0,
        float(r.get('ADX', 0))    >= 24.0,
        base                       >= 62,
        float(r.get('p6m%', 0))   >= 10.0,
        float(r.get('RS_acc', 0)) >= 2.5,
        float(r.get('p12m%', 0))  >= 15.0,
    ])
    confluence_bonus = strong * 2.5   # upp till +15 för 6/6

    # Riskjusterat momentum (hedge-trick: belönar hög avkastning / låg volatilitet)
    atr   = max(float(r.get('ATR%', 5.0)), 0.1)
    p6m   = float(r.get('p6m%', 0))
    p12m  = float(r.get('p12m%', 0))
    rs_a  = float(r.get('RS_acc', 0))
    mom   = 0.5 * p6m + 0.3 * p12m + 0.2 * rs_a
    mom_rank = mom / atr

    # Horisont-specifik vikt
    if horizon == 'korttid':
        hor_score = float(r.get('KT', 0))
    elif horizon == 'medel':
        hor_score = float(r.get('Mixed', 0))
    else:
        hor_score = base

    composite = hor_score - penalty + confluence_bonus
    return composite, round(penalty, 1), round(confluence_bonus, 1), round(mom_rank, 2)


def rank_horizon(results: list, horizon: str) -> list:
    """
    Rankar ALLA aktier för given horisont utan hårda filter.
    Aktier som inte uppfyller kriterier hamnar längre ner via penalty-system.
    Returnerar hela listan sorterad på composite score.
    """
    scored = []
    for r in results:
        comp, pen, bonus, mr = compute_composite_score(r, horizon)
        row = dict(r)
        row['_composite']  = comp
        row['_penalty']    = pen
        row['_bonus']      = bonus
        row['MomRank']     = mr
        # Betygsatt kvalitet: A-F baserat på composite score
        if comp >= 70:    row['_grade'] = 'A'
        elif comp >= 58:  row['_grade'] = 'B'
        elif comp >= 45:  row['_grade'] = 'C'
        elif comp >= 30:  row['_grade'] = 'D'
        else:             row['_grade'] = 'F'
        scored.append(row)

    # Sortera på composite (högt = bra)
    scored.sort(key=lambda x: x['_composite'], reverse=True)
    return scored


# ════════════════════════════════════════════════════════════════════════════
#  HUVUD
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Master Screener — Sverige + Europa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exempel:
  python master_screener.py
  python master_screener.py --topn 30
  python master_screener.py --land SE NO DK FI
  python master_screener.py --horisont swing
  python master_screener.py --min-score 60 --min-rs 3.0
  python master_screener.py --datum 2024-01-15
  python master_screener.py --alla --ingen-csv
        """)

    parser.add_argument('--datum',      type=str,   default=None,
                        help='Screeningdatum YYYY-MM-DD (default: idag)')
    parser.add_argument('--topn',       type=int,   default=TOP_N_DEFAULT,
                        help='Kandidater att visa per horisont (default: {})'.format(TOP_N_DEFAULT))
    parser.add_argument('--land',       type=str,   nargs='+',
                        help='Filtrera länder  t.ex. --land SE NO DK FI')
    parser.add_argument('--horisont',   type=str,   default=None,
                        choices=['korttid','medel','swing','alla'],
                        help='Visa bara en horisont (default: alla tre)')
    parser.add_argument('--min-score',  type=int,   default=None,
                        help='Override min_total_score för alla horisonter')
    parser.add_argument('--min-rs',     type=float, default=None,
                        help='Override min RS-krav')
    parser.add_argument('--min-vol',    type=float, default=MIN_AVG_DAILY_VOL,
                        help='Min dagsomsättning SEK')
    parser.add_argument('--max-tickers',type=int,   default=None,
                        help='Max antal tickers (för snabbtest)')
    parser.add_argument('--alla',       action='store_true',
                        help='Visa alla kandidater (inte bara topp N)')
    parser.add_argument('--ingen-csv',  action='store_true',
                        help='Skriv inte CSV-fil')
    args = parser.parse_args()

    # Overrides
    if args.min_score is not None:
        for h in ENTRY_FILTER: ENTRY_FILTER[h]['min_total_score'] = args.min_score
    if args.min_rs is not None:
        for h in ENTRY_FILTER: ENTRY_FILTER[h]['min_rs'] = args.min_rs

    datum      = pd.Timestamp(args.datum) if args.datum else pd.Timestamp.today().normalize()
    horisonter = (['korttid','medel','swing'] if args.horisont in (None,'alla')
                  else [args.horisont])
    top_n      = 9999 if args.alla else args.topn

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Banner
    W = 72
    print("╔" + "═"*W + "╗")
    print(("║  MASTER SCREENER  v1.0  —  Datum: {}".format(str(datum)[:10])).ljust(W+1) + "║")
    ef_kt = ENTRY_FILTER['korttid']; ef_md = ENTRY_FILTER['medel']; ef_sw = ENTRY_FILTER['swing']
    print(("║  KT  score≥{} rs≥{} adx≥{}".format(
        ef_kt['min_total_score'],ef_kt['min_rs'],ef_kt['min_adx'])).ljust(W+1)+"║")
    print(("║  MED score≥{} rs≥{} adx≥{} p6m≥{}%".format(
        ef_md['min_total_score'],ef_md['min_rs'],ef_md['min_adx'],ef_md['min_p6m'])).ljust(W+1)+"║")
    print(("║  SW  score≥{} rs≥{} adx≥{} p6m≥{}%".format(
        ef_sw['min_total_score'],ef_sw['min_rs'],ef_sw['min_adx'],ef_sw['min_p6m'])).ljust(W+1)+"║")
    print("╚" + "═"*W + "╝")

    # Tickers
    if not UNIVERSE:
        print("[FEL] UNIVERSE är tomt — lägg till ues_tick.py")
        sys.exit(1)
    tickers = [u[0] for u in UNIVERSE
               if not args.land or u[2].upper() in [l.upper() for l in args.land]]
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    print("\n[INFO] {} tickers att screena".format(len(tickers)))

    # Hämta data
    all_hist, bench_close = fetch_all_history(tickers, str(datum)[:10], str(datum)[:10])
    if not all_hist:
        print("[FEL] Ingen data hämtades!"); sys.exit(1)

    # Regim
    if bench_close is not None:
        rc = compute_regime(bench_close, bench_close.index[-1])
        print("\n[REGIM] {} — {} (1m: {:+.1f}%  3m: {:+.1f}%)".format(
            rc.fas, rc.enkel.value, rc.avk_1m, rc.avk_3m))

    # Förberäkning
    print("\n[INFO] Förberäknar indikatorer (engångskostnad)...")
    ticker_cache = precompute_all(all_hist, bench_close)

    # Screena
    print("\n[INFO] Kör screener för {}...".format(str(datum)[:10]))
    all_results = screen_date(ticker_cache, datum, args.min_vol)
    print("[INFO] {} aktier passerade grundfilter".format(len(all_results)))

    # Ranka per horisont
    results_by_h = {}
    for h in horisonter:
        ranked = rank_horizon(all_results, h)
        results_by_h[h] = ranked
        print_table(ranked, h, top_n)

    unique_tickers = set(r['Ticker'] for v in results_by_h.values() for r in v)
    print("\n[INFO] {} unika kandidater totalt".format(len(unique_tickers)))

    # Samla regim-info för HTML
    regime_info = {}
    if bench_close is not None and len(bench_close) > 0:
        rc = compute_regime(bench_close, bench_close.index[-1])
        regime_info = {
            'fas':    rc.fas,
            'enkel':  rc.enkel.value,
            'avk_1m': rc.avk_1m,
            'avk_3m': rc.avk_3m,
        }

    # HTML
    html_path = generate_html(results_by_h, datum, regime_info, ENTRY_FILTER, OUTPUT_FOLDER)
    print("[INFO] HTML sparad: {}".format(html_path))

    # CSV
    if not args.ingen_csv:
        csv_path = save_csv(results_by_h, datum, OUTPUT_FOLDER)
        print("[INFO] CSV sparad : {}".format(csv_path))

    print("\n[KLAR] Screening klar — {}".format(str(datum)[:10]))
    print("       Öppna rapporten: {}".format(html_path))


if __name__ == '__main__':
    main()