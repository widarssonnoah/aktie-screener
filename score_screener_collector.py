"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   SCORE SCREENER — DATAINSAMLARE  v1.0                                      ║
║   Tekniska + fundamentala nyckeltal → interaktiv HTML-poängsättning         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  VAD DEN GÖR:                                                                ║
║  Detta skript gör INGEN filtrering och INGEN poängsättning. Det enda        ║
║  jobbet är att samla in tekniska (från prishistorik) och fundamentala       ║
║  (från yfinance) nyckeltal för alla aktier i Avanza-universumet och         ║
║  paketera dem i EN fristående HTML-fil.                                     ║
║                                                                              ║
║  All poängsättning sker sedan i HTML-filen, i din webbläsare, i realtid:    ║
║  Du sätter ett "idealintervall" (min–max) för valfria nyckeltal. Värden     ║
║  INOM intervallet = 100 poäng. Värden UTANFÖR avtar poängen — ju längre     ║
║  bort, desto lägre — enligt en jämn avklingningskurva där ett steg lika     ║
║  stort som intervallets bredd ungefär halverar poängen.                    ║
║                                                                              ║
║  Exempel: P/E-tal, idealintervall 5–15.                                     ║
║    P/E 10  → 100 poäng (inom intervallet)                                   ║
║    P/E 16  → ~93 poäng (precis utanför — nästan 100)                        ║
║    P/E 25  → ~50 poäng (en hel intervallbredd bortom gränsen)               ║
║    P/E 45  → ~6  poäng (tre intervallbredder bortom)                        ║
║                                                                              ║
║  Ett TOMT fält = nyckeltalet räknas inte med i totalpoängen alls (varken    ║
║  positivt eller negativt). En etta i min ELLER max = 0 är ett GILTIGT       ║
║  målvärde (t.ex. "vill ha kursen 0-10% över SMA50") — skiljer sig från      ║
║  tomt fält, som betyder "bry dig inte om det här nyckeltalet".              ║
║                                                                              ║
║  Om ett nyckeltal saknas för en specifik aktie (t.ex. P/E för ett bolag     ║
║  utan vinst) och du HAR satt ett intervall för det: den aktien får 0 poäng  ║
║  på just det nyckeltalet (visas som "–" i tabellen) — men bara om du valt   ║
║  att inkludera nyckeltalet. Håll koll på "N/A"-antalet per nyckeltal i UI:t.║
║                                                                              ║
║  DATAKÄLLOR (gratis):                                                       ║
║  - Teknisk data: prishistorik via yfinance (samma motor som               ║
║    master_screener.py / turnaround_screener_v5.py)                          ║
║  - Fundamental data: yfinance .info (best effort — täckningen varierar,    ║
║    särskilt för mindre europeiska bolag)                                    ║
║                                                                              ║
║  KÖRKOMANDON:                                                                ║
║    python score_screener_collector.py                                      ║
║    python score_screener_collector.py --land SE NO DK                       ║
║    python score_screener_collector.py --max-tickers 200      # snabbtest    ║
║    python score_screener_collector.py --ingen-fundamenta     # snabbare     ║
║    python score_screener_collector.py --workers 12                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os, sys, time, json, warnings, logging, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from master_screener import (
        fetch_all_history, precompute_all,
        UNIVERSE, _NAMN, _LAND, _SEKTOR, _TYP, _CAP,
        BENCHMARK_TICKER, MIN_HISTORY, MIN_AVG_DAILY_VOL,
        _g, TickerCache,
    )
except ImportError as e:
    sys.exit("[FEL] Kan inte importera master_screener.py: {}\n"
             "Lägg score_screener_collector.py i samma mapp som master_screener.py "
             "och avanzaEuUs.py.".format(e))

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_FOLDER      = 'ScoreScreener_Output'
FUND_CACHE_FILENAME = 'fundamenta_cache_score.json'
MIN_VOL_SEK         = MIN_AVG_DAILY_VOL
DEFAULT_WORKERS     = 4


# ════════════════════════════════════════════════════════════════════════════
#  NYCKELTALS-REGISTER
#  Detta är den enda "sanningskällan" för vilka nyckeltal som finns — HTML-UI:t
#  byggs dynamiskt utifrån denna lista, så nya nyckeltal kan läggas till här
#  utan att röra JavaScript-koden.
# ════════════════════════════════════════════════════════════════════════════

METRIC_REGISTRY = [
    # ── TEKNISKA ──────────────────────────────────────────────────────────
    {'key': 'RSI14',        'label': 'RSI (14)',                    'grupp': 'Teknisk', 'enhet': '',  'dec': 1},
    {'key': 'RSI_slope5',   'label': 'RSI momentum (5d förändring)', 'grupp': 'Teknisk', 'enhet': '',  'dec': 1},
    {'key': 'ADX',          'label': 'ADX (trendstyrka)',            'grupp': 'Teknisk', 'enhet': '',  'dec': 1},
    {'key': 'DI_diff',      'label': 'DI+ minus DI- (trendriktning)', 'grupp': 'Teknisk', 'enhet': '', 'dec': 1},
    {'key': 'CMF',          'label': 'Chaikin Money Flow',           'grupp': 'Teknisk', 'enhet': '',  'dec': 3},
    {'key': 'UpDnVol',      'label': 'Upp/ned-volymkvot',            'grupp': 'Teknisk', 'enhet': '×', 'dec': 2},
    {'key': 'OBV_diff_pct', 'label': 'OBV vs OBV-snitt',             'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_EMA9',    'label': 'Avstånd till EMA9',            'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_EMA21',   'label': 'Avstånd till EMA21',           'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_EMA50',   'label': 'Avstånd till EMA50',           'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_EMA200',  'label': 'Avstånd till EMA200',          'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_SMA50',   'label': 'Avstånd till SMA50',           'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_SMA200',  'label': 'Avstånd till SMA200',          'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'RS21',         'label': 'Relativ styrka vs index (21d)', 'grupp': 'Teknisk', 'enhet': '', 'dec': 1},
    {'key': 'RS_acc',       'label': 'RS-acceleration',              'grupp': 'Teknisk', 'enhet': '',  'dec': 1},
    {'key': 'ATR_pct',      'label': 'ATR (volatilitet)',            'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'BB_pctB',      'label': 'Bollinger %B',                 'grupp': 'Teknisk', 'enhet': '',  'dec': 2},
    {'key': 'BB_bw',        'label': 'Bollinger bandbredd',          'grupp': 'Teknisk', 'enhet': '',  'dec': 3},
    {'key': 'VolRatio',     'label': 'Volymkvot (5d/22d)',           'grupp': 'Teknisk', 'enhet': '×', 'dec': 2},
    {'key': 'StochK',       'label': 'Stochastic %K',                'grupp': 'Teknisk', 'enhet': '',  'dec': 1},
    {'key': 'StochD',       'label': 'Stochastic %D',                'grupp': 'Teknisk', 'enhet': '',  'dec': 1},
    {'key': 'Squeeze',      'label': 'Squeeze aktiv (1=ja, 0=nej)',   'grupp': 'Teknisk', 'enhet': '',  'dec': 0},
    {'key': 'p1w',          'label': 'Avkastning 1 vecka',           'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'p1m',          'label': 'Avkastning 1 månad',           'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'p3m',          'label': 'Avkastning 3 månader',         'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'p6m',          'label': 'Avkastning 6 månader',         'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'p12m',         'label': 'Avkastning 12 månader',        'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_52wHigh', 'label': 'Avstånd till 52v-högsta',      'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},
    {'key': 'Dist_52wLow',  'label': 'Avstånd till 52v-lägsta',      'grupp': 'Teknisk', 'enhet': '%', 'dec': 1},

    # ── FUNDAMENTALA ──────────────────────────────────────────────────────
    {'key': 'PE',            'label': 'P/E-tal',                     'grupp': 'Fundamental', 'enhet': '',  'dec': 1},
    {'key': 'ForwardPE',     'label': 'Framåtblickande P/E',         'grupp': 'Fundamental', 'enhet': '',  'dec': 1},
    {'key': 'PB',            'label': 'P/B-tal',                     'grupp': 'Fundamental', 'enhet': '',  'dec': 2},
    {'key': 'PS',            'label': 'P/S-tal',                     'grupp': 'Fundamental', 'enhet': '',  'dec': 2},
    {'key': 'PEG',           'label': 'PEG-tal',                     'grupp': 'Fundamental', 'enhet': '',  'dec': 2},
    {'key': 'EV_EBITDA',     'label': 'EV/EBITDA',                   'grupp': 'Fundamental', 'enhet': '',  'dec': 1},
    {'key': 'DivYield',      'label': 'Direktavkastning',            'grupp': 'Fundamental', 'enhet': '%', 'dec': 2},
    {'key': 'PayoutRatio',   'label': 'Utdelningsandel',             'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'ROE',           'label': 'Avkastning på eget kapital',  'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'ROA',           'label': 'Avkastning på totalt kapital', 'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'DebtToEquity',  'label': 'Skuldsättningsgrad (D/E)',    'grupp': 'Fundamental', 'enhet': '',  'dec': 1},
    {'key': 'CurrentRatio',  'label': 'Kassalikviditet',             'grupp': 'Fundamental', 'enhet': '',  'dec': 2},
    {'key': 'GrossMargin',   'label': 'Bruttomarginal',              'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'OpMargin',      'label': 'Rörelsemarginal',             'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'ProfitMargin',  'label': 'Vinstmarginal',               'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'EarnGrowth',    'label': 'Vinsttillväxt (kvartal)',     'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'RevGrowth',     'label': 'Omsättningstillväxt',         'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'Beta',          'label': 'Beta',                        'grupp': 'Fundamental', 'enhet': '',  'dec': 2},
    {'key': 'AnalystUpside', 'label': 'Analytikerriktkurs, uppsida', 'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
    {'key': 'RecMean',       'label': 'Analytikerbetyg (1=köp,5=sälj)', 'grupp': 'Fundamental', 'enhet': '', 'dec': 2},
    {'key': 'ShortPctFloat', 'label': 'Blankad andel av floaten',    'grupp': 'Fundamental', 'enhet': '%', 'dec': 1},
]
METRIC_KEYS = [m['key'] for m in METRIC_REGISTRY]


# ════════════════════════════════════════════════════════════════════════════
#  TEKNISK EXTRAKTION  (allt redan förberäknat i TickerCache — ingen extra
#  nätverkstrafik, bara utläsning + enkla kvoter/avstånd)
# ════════════════════════════════════════════════════════════════════════════

def extract_technical_metrics(tc: TickerCache, i: int, min_vol: float) -> Optional[dict]:
    if i < MIN_HISTORY - 1:
        return None
    kurs = _g(tc.close_arr, i)
    if np.isnan(kurs) or kurs <= 0:
        return None

    vol_avg = _g(tc.vol_avg22, i)
    if np.isnan(vol_avg) or vol_avg <= 0:
        sl_v = tc.vol_arr[max(0, i - 21):i + 1]
        vol_avg = float(np.nanmean(sl_v)) if len(sl_v) >= 3 else 0.0
    avg_vol_sek = vol_avg * kurs
    if min_vol > 0 and 0 < avg_vol_sek < min_vol:
        return None

    def pct_ago(d):
        j = max(0, i - d)
        prev = tc.close_arr[j]
        if np.isnan(prev) or prev <= 0:
            return np.nan
        return (kurs / prev - 1) * 100

    def dist(arr_val):
        if np.isnan(arr_val) or arr_val <= 0:
            return np.nan
        return (kurs / arr_val - 1) * 100

    obv_now = _g(tc.obv, i)
    obv_ema = _g(tc.obv_ema20, i)
    obv_diff = np.nan
    if not np.isnan(obv_now) and not np.isnan(obv_ema) and abs(obv_ema) > 1e-9:
        obv_diff = (obv_now - obv_ema) / abs(obv_ema) * 100

    vol_snitt = _g(tc.vol_avg22, i)
    if np.isnan(vol_snitt) or vol_snitt <= 0:
        vol_snitt = float(np.nanmean(tc.vol_arr[max(0, i - 22):i])) if i >= 3 else np.nan
    vol_5d = float(np.nanmean(tc.vol_arr[max(0, i - 4):i + 1]))
    vol_ratio = (vol_5d / vol_snitt) if (vol_snitt and vol_snitt > 0) else np.nan

    dip_nu, dim_nu = _g(tc.dip, i), _g(tc.dim, i)
    di_diff = (dip_nu - dim_nu) if not (np.isnan(dip_nu) or np.isnan(dim_nu)) else np.nan

    atr14 = _g(tc.atr14, i)
    atr_pct = (atr14 / kurs * 100) if (not np.isnan(atr14) and kurs > 0) else np.nan

    lookback = min(252, i + 1)
    window = tc.close_arr[i - lookback + 1:i + 1]
    window = window[~np.isnan(window)]
    hi52 = float(np.max(window)) if len(window) else np.nan
    lo52 = float(np.min(window)) if len(window) else np.nan
    dist_hi = ((kurs / hi52 - 1) * 100) if (not np.isnan(hi52) and hi52 > 0) else np.nan
    dist_lo = ((kurs / lo52 - 1) * 100) if (not np.isnan(lo52) and lo52 > 0) else np.nan

    vals = {
        'RSI14':        _g(tc.rsi14, i),
        'RSI_slope5':   _g(tc.rsi_slope5, i),
        'ADX':          _g(tc.adx, i),
        'DI_diff':      di_diff,
        'CMF':          _g(tc.cmf, i),
        'UpDnVol':      _g(tc.up_dn_vol, i),
        'OBV_diff_pct': obv_diff,
        'Dist_EMA9':    dist(_g(tc.ema9, i)),
        'Dist_EMA21':   dist(_g(tc.ema21, i)),
        'Dist_EMA50':   dist(_g(tc.ema50, i)),
        'Dist_EMA200':  dist(_g(tc.ema200, i)),
        'Dist_SMA50':   dist(_g(tc.sma50, i)),
        'Dist_SMA200':  dist(_g(tc.sma200, i)),
        'RS21':         _g(tc.rs21, i),
        'RS_acc':       _g(tc.rs_acc, i),
        'ATR_pct':      atr_pct,
        'BB_pctB':      _g(tc.bb_pctB, i),
        'BB_bw':        _g(tc.bb_bw, i),
        'VolRatio':     vol_ratio,
        'StochK':       _g(tc.stoch_k, i),
        'StochD':       _g(tc.stoch_d, i),
        'Squeeze':      _g(tc.sq_on, i),
        'p1w':          pct_ago(5),
        'p1m':          pct_ago(21),
        'p3m':          _g(tc.p3m, i),
        'p6m':          _g(tc.p6m, i),
        'p12m':         _g(tc.p12m, i),
        'Dist_52wHigh': dist_hi,
        'Dist_52wLow':  dist_lo,
    }
    clean = {k: round(float(v), 4) for k, v in vals.items() if v is not None and not np.isnan(v)}
    clean['_Kurs'] = round(float(kurs), 4)
    return clean
    def extract_chart_series(tc: TickerCache, i: int, lookback: int = 200) -> dict:
    """
    Komprimerad priskurve-historik för graf-funktionen i webbläsaren.
    Endast stängningskurs + volym sparas (inte OHLC) för att hålla
    filstorleken hanterbar över ~1100+ aktier. Alla tekniska overlays
    (EMA/SMA/Bollinger/Fibonacci/zigzag) beräknas sedan direkt i
    JavaScript utifrån denna serie, så ingen ytterligare data behöver
    skickas med.
    """
    start = max(0, i - lookback + 1)
    close_slice = tc.close_arr[start:i + 1]
    vol_slice = tc.vol_arr[start:i + 1]
    dates_slice = tc.dates[start:i + 1]

    if len(close_slice) < 20:
        return None

    close_list = [round(float(c), 4) if not np.isnan(c) else None for c in close_slice]
    vol_list = [int(v) if not np.isnan(v) else None for v in vol_slice]

    d0 = str(pd.Timestamp(dates_slice[0]).date())
    d1 = str(pd.Timestamp(dates_slice[-1]).date())

    return {'c': close_list, 'v': vol_list, 'd0': d0, 'd1': d1}


# ════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL EXTRAKTION  (yfinance .info, best effort, cachad + parallell)
# ════════════════════════════════════════════════════════════════════════════

def _cache_path() -> str:
    return os.path.join(OUTPUT_FOLDER, FUND_CACHE_FILENAME)


def _load_fund_cache() -> dict:
    p = _cache_path()
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_fund_cache(cache: dict) -> None:
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        with open(_cache_path(), 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except Exception:
        pass


def _pct(x):
    """yfinance ger tillväxt/marginal/avkastning som andelar (0.12 = 12%) — gör om till procent."""
    return None if x is None else x * 100.0


def _finite(x) -> bool:
    try:
        xf = float(x)
        return xf == xf and xf not in (float('inf'), float('-inf'))
    except Exception:
        return False


def fetch_one_fundamental(ticker: str, retries: int = 1) -> dict:
    """
    Best effort — returnerar tomt dict om något går fel eller yfinance saknas.

    Yahoo Finance är känt för att strypa (rate-limita) .info-anrop vid tät
    parallell trafik, vilket ofta visar sig som ett nästan tomt svar snarare
    än ett tydligt fel. Vi gör därför EN extra retry med kort paus om svaret
    verkar misstänkt tomt (< 5 fält), innan vi ger upp för den tickern.
    """
    if yf is None:
        return {}

    info = {}
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        if len(info) >= 5:
            break
        time.sleep(0.4 + attempt * 0.6)   # backoff innan ev. nytt försök

    if len(info) < 5:
        return {}   # gav upp — troligen strypt av Yahoo eller ogiltig ticker

    kurs = info.get('currentPrice') or info.get('regularMarketPrice')
    target = info.get('targetMeanPrice')
    upside = None
    if kurs and target and kurs > 0:
        upside = (target / kurs - 1) * 100.0

    out = {
        'PE':            info.get('trailingPE'),
        'ForwardPE':     info.get('forwardPE'),
        'PB':            info.get('priceToBook'),
        'PS':            info.get('priceToSalesTrailing12Months'),
        'PEG':           info.get('pegRatio') or info.get('trailingPegRatio'),
        'EV_EBITDA':     info.get('enterpriseToEbitda'),
        'DivYield':      _pct(info.get('dividendYield')),
        'PayoutRatio':   _pct(info.get('payoutRatio')),
        'ROE':           _pct(info.get('returnOnEquity')),
        'ROA':           _pct(info.get('returnOnAssets')),
        'DebtToEquity':  info.get('debtToEquity'),
        'CurrentRatio':  info.get('currentRatio'),
        'GrossMargin':   _pct(info.get('grossMargins')),
        'OpMargin':      _pct(info.get('operatingMargins')),
        'ProfitMargin':  _pct(info.get('profitMargins')),
        'EarnGrowth':    _pct(info.get('earningsQuarterlyGrowth')),
        'RevGrowth':     _pct(info.get('revenueGrowth')),
        'Beta':          info.get('beta'),
        'AnalystUpside': upside,
        'RecMean':       info.get('recommendationMean'),
        'ShortPctFloat': _pct(info.get('shortPercentOfFloat')),
    }
    return {k: round(float(v), 4) for k, v in out.items() if v is not None and _finite(v)}


def fetch_fundamentals_batch(tickers: list, workers: int) -> dict:
    """Hämtar fundamenta för alla tickers parallellt (I/O-bundet), med daglig cache."""
    cache = _load_fund_cache()
    today_str = str(pd.Timestamp.today().date())
    result = {}
    to_fetch = []

    for tk in tickers:
        entry = cache.get(tk)
        if entry and entry.get('_date') == today_str:
            result[tk] = {k: v for k, v in entry.items() if k != '_date'}
        else:
            to_fetch.append(tk)

    print("[INFO] Fundamenta: {} från cache, {} att hämta ({} parallella anrop)...".format(
        len(result), len(to_fetch), workers))

    changed = False
    done = 0
    t0 = time.time()
    if to_fetch:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_one_fundamental, tk): tk for tk in to_fetch}
            for fut in as_completed(futures):
                tk = futures[fut]
                try:
                    data = fut.result()
                except Exception:
                    data = {}
                result[tk] = data
                entry = dict(data)
                entry['_date'] = today_str
                cache[tk] = entry
                changed = True
                done += 1
                if done % 50 == 0 or done == len(to_fetch):
                    elapsed = time.time() - t0
                    print("   ... {}/{} klara ({:.0f}s)".format(done, len(to_fetch), elapsed))

    if changed:
        _save_fund_cache(cache)
    return result


# ════════════════════════════════════════════════════════════════════════════
#  KOMBINERA TILL SLUTGILTIGA RECORDS
# ════════════════════════════════════════════════════════════════════════════

def build_records(ticker_cache: dict, datum, min_vol: float, fund_data: dict) -> list:
    date64 = np.datetime64(datum, 'ns')
    records = []
    for ticker, tc in ticker_cache.items():
        i = int(np.searchsorted(tc.dates, date64, side='right')) - 1
        try:
            tech = extract_technical_metrics(tc, i, min_vol)
        except Exception:
            tech = None
        if tech is None:
            continue
               kurs = tech.pop('_Kurs')
        metrics = dict(tech)
        metrics.update(fund_data.get(ticker, {}))
        hist = extract_chart_series(tc, i)
        rec = {
            'Ticker': ticker.strip(), 'Namn': _NAMN.get(ticker, ticker).strip(),
            'Land': _LAND.get(ticker, '?').strip(), 'Sektor': _SEKTOR.get(ticker, '?').strip(),
            'Cap': _CAP.get(ticker, '?'), 'Kurs': kurs,
            'metrics': metrics,
        }
        if hist is not None:
            rec['hist'] = hist
        records.append(rec)
    records.sort(key=lambda r: r['Ticker'])
    return records


def print_coverage_report(records: list) -> None:
    """
    v1.1: Skriver ut hur många aktier som faktiskt har data för VARJE
    nyckeltal. Används för att skilja på "nyckeltalet fungerar men få
    aktier har det" (normalt, särskilt för fundamenta på mindre bolag)
    från "något är trasigt" (nästan 0% täckning på ett nyckeltal som
    borde finnas för de flesta aktier, t.ex. RSI eller P/E).
    """
    n = len(records)
    if n == 0:
        return
    W = 70
    print("\n" + "═"*W)
    print("  DATATÄCKNING PER NYCKELTAL  ({} aktier totalt)".format(n))
    print("═"*W)
    for grp in ('Teknisk', 'Fundamental'):
        print("\n  {}:".format(grp))
        for m in METRIC_REGISTRY:
            if m['grupp'] != grp:
                continue
            cnt = sum(1 for r in records if m['key'] in r['metrics'])
            pct = cnt / n * 100
            flag = '  ⚠️ LÅG TÄCKNING' if pct < 15 else ''
            print("    {:<16} {:>5}/{:<5} ({:>5.1f}%){}".format(m['key'], cnt, n, pct, flag))
    print()
    print("  Tomma/låga rader ovan betyder att just det nyckeltalet saknas för de flesta")
    print("  aktier i den här körningen — vanligast för fundamenta (Yahoo Finance har")
    print("  ojämn täckning, särskilt för mindre nordiska/europeiska bolag, och kan även")
    print("  strypa (rate-limita) för många parallella anrop). Prova --workers 2-3 eller")
    print("  kör igen om många fundamentala rader har låg täckning.\n")


# ════════════════════════════════════════════════════════════════════════════
#  HTML-GENERERING  — självständig fil, all poängsättning sker i webbläsaren
# ════════════════════════════════════════════════════════════════════════════

_CSS = """
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;
      --accent:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:-apple-system,Segoe UI,Roboto,sans-serif;
     margin:0;padding:20px 24px 60px;}
h1{font-size:21px;margin:0 0 2px 0;color:#e6edf3;}
.sub{color:var(--muted);font-size:12.5px;margin-bottom:18px;line-height:1.5;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:16px;}
.panel h2{font-size:14px;margin:0 0 10px 0;color:#e6edf3;}
.metricgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px;}
.mrow{display:flex;align-items:center;gap:6px;background:#0d1117;border:1px solid var(--border);
      border-radius:7px;padding:6px 8px;font-size:12px;}
.mrow .mlabel{flex:1;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.mrow input{width:52px;background:#010409;border:1px solid var(--border);color:var(--text);
            border-radius:5px;padding:3px 4px;font-size:12px;text-align:center;}
.mrow input.w{width:38px;}
.mrow .unit{color:var(--muted);font-size:10px;width:14px;}
.mrow .na{font-size:10px;color:var(--muted);width:44px;text-align:right;}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px;}
.toolbar input[type=text]{background:#010409;border:1px solid var(--border);color:var(--text);
            border-radius:7px;padding:7px 10px;font-size:13px;}
.toolbar select{background:#010409;border:1px solid var(--border);color:var(--text);
            border-radius:7px;padding:7px 10px;font-size:13px;}
button{background:#21262d;border:1px solid var(--border);color:var(--text);border-radius:7px;
       padding:7px 12px;font-size:12.5px;cursor:pointer;}
button:hover{background:#30363d;}
button.primary{background:#238636;border-color:#2ea043;}
button.primary:hover{background:#2ea043;}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
th{text-align:left;color:var(--muted);font-weight:600;padding:8px 8px;border-bottom:1px solid var(--border);
   cursor:pointer;white-space:nowrap;position:sticky;top:0;background:var(--bg);}
th:hover{color:var(--text);}
td{padding:7px 8px;border-bottom:1px solid #21262d;white-space:nowrap;}
tr.datarow:hover{background:#161b22;cursor:pointer;}
.scorebar{display:inline-block;width:60px;height:7px;border-radius:4px;background:#21262d;overflow:hidden;
          vertical-align:middle;margin-right:6px;}
.scorebar .fill{height:100%;}
.detailrow td{background:#0d1117;padding:10px 14px;}
.detailgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px;font-size:11.5px;}
.detailgrid .dm{background:#161b22;border:1px solid var(--border);border-radius:6px;padding:5px 8px;}
.detailgrid .dm .k{color:var(--muted);}
footer{margin-top:24px;color:#6e7681;font-size:11px;text-align:center;}
.smallnote{color:var(--muted);font-size:11px;margin-top:4px;}
.count{color:var(--muted);font-size:12px;}
textarea{width:100%;background:#010409;border:1px solid var(--border);color:var(--text);
         border-radius:7px;padding:8px;font-size:11px;font-family:monospace;min-height:70px;}
         .chartwrap{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:10px;}
.chartcontrols{display:flex;gap:6px;align-items:center;margin-bottom:8px;flex-wrap:wrap;}
.chartcontrols button{padding:4px 10px;font-size:11px;}
.chartcontrols button.active{background:#238636;border-color:#2ea043;color:#fff;}
.chartcanvas{width:100%;height:260px;display:block;}
.chartlegend{display:flex;gap:12px;flex-wrap:wrap;font-size:10.5px;color:var(--muted);margin-top:6px;}
.chartlegend span{display:inline-flex;align-items:center;gap:4px;}
.chartlegend .sw{width:10px;height:3px;display:inline-block;border-radius:2px;}
.chartnote{font-size:10.5px;color:#6e7681;margin-top:6px;line-height:1.4;}
"""

_JS = r"""
// Poängsättningsformel — S-formad (logistisk) avklingning.
// Inom [lo,hi] = 100. Utanför avtar poängen enligt en sigmoid: långsamt
// nära kanten, snabbast runt DECAY_MIDPOINT (mätt i antal intervallbredder
// från kanten), och planar sedan ut mot 0 långt bort.
const DECAY_STEEPNESS = 7;    // högre = skarpare "knä"
const DECAY_MIDPOINT  = 0.65; // var (i intervallbredder) nedgången är som snabbast

function metricScore(value, lo, hi) {
    if (value === null || value === undefined || isNaN(value)) return null;
    if (value >= lo && value <= hi) return 100;
    let width = hi - lo;
    if (width < 1e-9) width = Math.max(Math.abs(hi) * 0.1, 0.5);
    const dist = value < lo ? (lo - value) : (value - hi);
    const norm = dist / width;
    let score = 100 / (1 + Math.exp(DECAY_STEEPNESS * (norm - DECAY_MIDPOINT)));
    if (score < 0.5) score = 0;
    return Math.min(100, Math.max(0, score));
}

let CONFIG = {};   // { metricKey: {lo, hi, w} }  — saknad nyckel = inaktiv

function getActiveMetrics() {
    return Object.keys(CONFIG).filter(k => CONFIG[k] && CONFIG[k].lo !== null && CONFIG[k].hi !== null);
}

function computeRecordScore(rec) {
    const active = getActiveMetrics();
    if (active.length === 0) return {total: null, breakdown: {}};
    let wsum = 0, ssum = 0;
    const breakdown = {};
    for (const key of active) {
        const cfg = CONFIG[key];
        const raw = rec.metrics.hasOwnProperty(key) ? rec.metrics[key] : null;
        let sc;
        if (raw === null || raw === undefined) {
            sc = 0;   // aktivt nyckeltal men data saknas för denna aktie -> 0 poäng
        } else {
            sc = metricScore(raw, cfg.lo, cfg.hi);
        }
        const w = (cfg.w === null || cfg.w === undefined || isNaN(cfg.w)) ? 1 : cfg.w;
        breakdown[key] = {score: sc, raw: raw, weight: w};
        wsum += w;
        ssum += sc * w;
    }
    const total = wsum > 0 ? ssum / wsum : null;
    return {total, breakdown};
}

function scoreColor(s) {
    if (s === null) return '#6e7681';
    if (s >= 75) return '#3fb950';
    if (s >= 50) return '#d29922';
    if (s >= 25) return '#e3a008';
    return '#f85149';
}

function fmtVal(v, dec, unit) {
    if (v === null || v === undefined || isNaN(v)) return '–';
    return v.toFixed(dec) + (unit || '');
}

let SORT_KEY = '_total', SORT_DIR = -1;
let EXPANDED = new Set();

function buildSettingsPanel() {
    const groups = {};
    for (const m of METRICS) {
        (groups[m.grupp] = groups[m.grupp] || []).push(m);
    }
    const root = document.getElementById('settingsRoot');
    root.innerHTML = '';
    for (const grp of Object.keys(groups)) {
        const panel = document.createElement('div');
        panel.className = 'panel';
        const h = document.createElement('h2');
        h.textContent = grp + ' — idealintervall (tomt = ej med i poängen)';
        panel.appendChild(h);
        const grid = document.createElement('div');
        grid.className = 'metricgrid';
        for (const m of groups[grp]) {
            const row = document.createElement('div');
            row.className = 'mrow';
            row.innerHTML =
                '<span class="mlabel" title="' + m.label + '">' + m.label + '</span>' +
                '<input type="number" step="any" class="lo" data-key="' + m.key + '" placeholder="min">' +
                '<span>–</span>' +
                '<input type="number" step="any" class="hi" data-key="' + m.key + '" placeholder="max">' +
                '<span class="unit">' + (m.enhet || '') + '</span>' +
                '<input type="number" step="any" class="w" data-key="' + m.key + '" placeholder="1" title="Vikt (standard 1)">' +
                '<span class="na" id="na_' + m.key + '"></span>';
            grid.appendChild(row);
        }
        panel.appendChild(grid);
        root.appendChild(panel);
    }
    root.querySelectorAll('input').forEach(inp => {
        inp.addEventListener('input', onSettingsChange);
    });
}

let debounceTimer = null;
function onSettingsChange() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { readConfigFromUI(); renderTable(); }, 150);
}

function readConfigFromUI() {
    CONFIG = {};
    document.querySelectorAll('#settingsRoot .lo').forEach(loInp => {
        const key = loInp.dataset.key;
        const hiInp = document.querySelector('#settingsRoot .hi[data-key="' + key + '"]');
        const wInp = document.querySelector('#settingsRoot .w[data-key="' + key + '"]');
        const loVal = loInp.value.trim(), hiVal = hiInp.value.trim();
        if (loVal === '' || hiVal === '') return;   // tomt = inaktivt
        const lo = parseFloat(loVal), hi = parseFloat(hiVal);
        if (isNaN(lo) || isNaN(hi)) return;
        const wVal = wInp.value.trim();
        const w = wVal === '' ? 1 : parseFloat(wVal);
        CONFIG[key] = {lo: Math.min(lo, hi), hi: Math.max(lo, hi), w: isNaN(w) ? 1 : w};
    });
}

function applyConfigToUI() {
    document.querySelectorAll('#settingsRoot .lo').forEach(loInp => {
        const key = loInp.dataset.key;
        const hiInp = document.querySelector('#settingsRoot .hi[data-key="' + key + '"]');
        const wInp = document.querySelector('#settingsRoot .w[data-key="' + key + '"]');
        const cfg = CONFIG[key];
        loInp.value = cfg ? cfg.lo : '';
        hiInp.value = cfg ? cfg.hi : '';
        wInp.value = (cfg && cfg.w !== 1) ? cfg.w : '';
    });
}

function updateNACounts(scored) {
    const active = getActiveMetrics();
    for (const m of METRICS) {
        const el = document.getElementById('na_' + m.key);
        if (!el) continue;
        if (!active.includes(m.key)) { el.textContent = ''; continue; }
        let na = 0;
        for (const r of scored) if (!r.rec.metrics.hasOwnProperty(m.key)) na++;
        el.textContent = na > 0 ? (na + ' saknar') : '';
    }
}

function getFiltered() {
    const q = document.getElementById('searchBox').value.trim().toLowerCase();
    const land = document.getElementById('landFilter').value;
    const minScore = parseFloat(document.getElementById('minScore').value) || 0;
    let scored = RECORDS.map(rec => {
        const r = computeRecordScore(rec);
        return {rec, total: r.total, breakdown: r.breakdown};
    });
    updateNACounts(scored);
    scored = scored.filter(s => {
        if (q && !(s.rec.Ticker.toLowerCase().includes(q) || s.rec.Namn.toLowerCase().includes(q))) return false;
        if (land && s.rec.Land !== land) return false;
        if (s.total !== null && s.total < minScore) return false;
        return true;
    });
    scored.sort((a, b) => {
        let av, bv;
        if (SORT_KEY === '_total') { av = a.total; bv = b.total; }
        else if (SORT_KEY === '_kurs') { av = a.rec.Kurs; bv = b.rec.Kurs; }
        else { av = a.rec[SORT_KEY]; bv = b.rec[SORT_KEY]; }
        if (av === null || av === undefined) av = -Infinity;
        if (bv === null || bv === undefined) bv = -Infinity;
        if (typeof av === 'string') return SORT_DIR * av.localeCompare(bv);
        return SORT_DIR * (av - bv);
    });
    return scored;
}

function populateLandFilter() {
    const sel = document.getElementById('landFilter');
    const lands = Array.from(new Set(RECORDS.map(r => r.Land))).sort();
    for (const l of lands) {
        const opt = document.createElement('option');
        opt.value = l; opt.textContent = l;
        sel.appendChild(opt);
    }
}

function renderTable() {
    const scored = getFiltered();
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = '';
    document.getElementById('rowCount').textContent = scored.length + ' av ' + RECORDS.length + ' aktier';

    scored.forEach((s, idx) => {
        const tr = document.createElement('tr');
        tr.className = 'datarow';
        const col = scoreColor(s.total);
        const totalTxt = s.total === null ? '–' : s.total.toFixed(1);
        const barW = s.total === null ? 0 : s.total;
        tr.innerHTML =
            '<td>' + (idx + 1) + '</td>' +
            '<td><b>' + s.rec.Ticker + '</b></td>' +
            '<td>' + s.rec.Namn + '</td>' +
            '<td>' + s.rec.Land + '</td>' +
            '<td>' + s.rec.Sektor + '</td>' +
            '<td>' + s.rec.Kurs.toFixed(2) + '</td>' +
            '<td><span class="scorebar"><span class="fill" style="width:' + barW + '%;background:' + col + '"></span></span>' +
            '<b style="color:' + col + '">' + totalTxt + '</b></td>';
        tr.addEventListener('click', () => toggleDetail(s.rec.Ticker));
        tbody.appendChild(tr);

        if (EXPANDED.has(s.rec.Ticker)) {
            const dtr = document.createElement('tr');
            dtr.className = 'detailrow';
            const td = document.createElement('td');
            td.colSpan = 7;
            const grid = document.createElement('div');
            grid.className = 'detailgrid';
            const active = getActiveMetrics();
            if (active.length === 0) {
                grid.innerHTML = '<div class="dm">Inga nyckeltal aktiva — sätt minst ett intervall ovan.</div>';
            }
            for (const key of active) {
                const m = METRICS_BY_KEY[key];
                const b = s.breakdown[key];
                const dc = scoreColor(b.score);
                grid.innerHTML +=
                    '<div class="dm"><span class="k">' + m.label + ':</span> ' +
                    fmtVal(b.raw, m.dec, m.enhet) +
                    ' <b style="color:' + dc + '">(' + (b.score === null ? '–' : b.score.toFixed(0)) + 'p)</b></div>';
            }
            td.appendChild(grid);
            dtr.appendChild(td);
            tbody.appendChild(dtr);
        }
    });
}

function toggleDetail(ticker) {
    if (EXPANDED.has(ticker)) EXPANDED.delete(ticker); else EXPANDED.add(ticker);
    renderTable();
}

function exportCSV() {
    const scored = getFiltered();
    const active = getActiveMetrics();
    const header = ['Ticker', 'Namn', 'Land', 'Sektor', 'Kurs', 'TotalPoang'].concat(active);
    const lines = [header.join(';')];
    for (const s of scored) {
        const row = [s.rec.Ticker, s.rec.Namn, s.rec.Land, s.rec.Sektor, s.rec.Kurs,
                     s.total === null ? '' : s.total.toFixed(1)];
        for (const key of active) {
            const b = s.breakdown[key];
            row.push(b.score === null ? '' : b.score.toFixed(1));
        }
        lines.push(row.join(';'));
    }
    const blob = new Blob(['\ufeff' + lines.join('\n')], {type: 'text/csv;charset=utf-8;'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'score_screener_export.csv';
    a.click();
}

function exportSettings() {
    document.getElementById('settingsJson').value = JSON.stringify(CONFIG, null, 1);
}

function importSettings() {
    try {
        const parsed = JSON.parse(document.getElementById('settingsJson').value);
        CONFIG = parsed || {};
        applyConfigToUI();
        renderTable();
    } catch (e) {
        alert('Kunde inte tolka JSON: ' + e.message);
    }
}

function clearAll() {
    CONFIG = {};
    applyConfigToUI();
    renderTable();
}

const PRESETS = {
  value: {
    'PE':           {lo: 5,  hi: 15, w: 1},
    'PB':           {lo: 0.5, hi: 2, w: 1},
    'ROE':          {lo: 12, hi: 40, w: 1},
    'DivYield':     {lo: 2,  hi: 6,  w: 1},
    'DebtToEquity': {lo: 0,  hi: 80, w: 1},
    'RSI14':        {lo: 45, hi: 65, w: 1},
    'Dist_SMA200':  {lo: 2,  hi: 20, w: 1},
  },
  turnaround: {
    'RSI14': {lo: 35, hi: 52, w: 1.5},
    'RSI_slope5': {lo: 5, hi: 30, w: 1.5},
    'StochK': {lo: 20, hi: 55, w: 1},
    'StochD': {lo: 20, hi: 55, w: 1},
    'CMF': {lo: 0.03, hi: 0.35, w: 1.5},
    'OBV_diff_pct': {lo: 0, hi: 20, w: 1},
    'UpDnVol': {lo: 1.1, hi: 3, w: 1},
    'VolRatio': {lo: 1.1, hi: 2.8, w: 1},
    'Dist_52wLow': {lo: 0, hi: 25, w: 1},
    'Dist_52wHigh': {lo: -65, hi: -15, w: 0.5},
    'p1w': {lo: 0, hi: 10, w: 1},
    'p1m': {lo: -8, hi: 15, w: 1},
    'p3m': {lo: -40, hi: 5, w: 0.5},
    'ADX': {lo: 10, hi: 28, w: 0.5},
    'DI_diff': {lo: -5, hi: 15, w: 1},
    'Dist_SMA50': {lo: -15, hi: 5, w: 1},
    'Dist_EMA21': {lo: -10, hi: 5, w: 1},
    'BB_pctB': {lo: 0.2, hi: 0.65, w: 1},
    'DebtToEquity': {lo: 0, hi: 150, w: 0.5},
    'CurrentRatio': {lo: 0.8, hi: 4, w: 0.5},
  },
  breakout: {
    'BB_bw': {lo: 0.01, hi: 0.08, w: 1.5},
    'Squeeze': {lo: 1, hi: 1, w: 1.5},
    'ATR_pct': {lo: 1, hi: 4, w: 1},
    'Dist_52wHigh': {lo: -15, hi: 0, w: 1.5},
    'VolRatio': {lo: 1.2, hi: 3, w: 1.5},
    'RSI14': {lo: 55, hi: 70, w: 1},
    'ADX': {lo: 15, hi: 30, w: 1},
    'DI_diff': {lo: 0, hi: 20, w: 1},
    'Dist_EMA9': {lo: 0, hi: 5, w: 1},
    'Dist_EMA21': {lo: -2, hi: 8, w: 1},
    'BB_pctB': {lo: 0.7, hi: 1, w: 1},
    'StochK': {lo: 60, hi: 90, w: 1},
    'StochD': {lo: 55, hi: 85, w: 0.5},
    'CMF': {lo: 0.05, hi: 0.3, w: 1},
    'p1w': {lo: 0, hi: 8, w: 1},
    'p1m': {lo: -2, hi: 15, w: 1},
    'UpDnVol': {lo: 1.2, hi: 3.5, w: 1},
  },
  trend: {
    'ADX': {lo: 25, hi: 60, w: 1.5},
    'DI_diff': {lo: 10, hi: 40, w: 1.5},
    'RS21': {lo: 5, hi: 50, w: 1.5},
    'RS_acc': {lo: 0, hi: 20, w: 1},
    'Dist_EMA21': {lo: 0, hi: 10, w: 1},
    'Dist_EMA50': {lo: 0, hi: 15, w: 1},
    'Dist_SMA200': {lo: 5, hi: 40, w: 1},
    'p1m': {lo: 2, hi: 20, w: 1},
    'p3m': {lo: 5, hi: 40, w: 1},
    'p6m': {lo: 10, hi: 60, w: 1},
    'p12m': {lo: 10, hi: 100, w: 0.5},
    'RSI14': {lo: 50, hi: 75, w: 1},
    'Dist_52wHigh': {lo: -10, hi: 0, w: 1},
    'OBV_diff_pct': {lo: 5, hi: 40, w: 1},
    'CMF': {lo: 0.05, hi: 0.3, w: 1},
    'UpDnVol': {lo: 1.2, hi: 4, w: 1},
  },
  stable: {
    'PE': {lo: 10, hi: 30, w: 1},
    'ForwardPE': {lo: 10, hi: 28, w: 1},
    'PB': {lo: 1, hi: 8, w: 0.5},
    'PEG': {lo: 0.5, hi: 2.5, w: 1},
    'ROE': {lo: 15, hi: 40, w: 1.5},
    'ROA': {lo: 5, hi: 20, w: 1},
    'ProfitMargin': {lo: 10, hi: 40, w: 1},
    'OpMargin': {lo: 10, hi: 35, w: 1},
    'GrossMargin': {lo: 30, hi: 80, w: 0.5},
    'DebtToEquity': {lo: 0, hi: 100, w: 1.5},
    'CurrentRatio': {lo: 1, hi: 3, w: 1},
    'DivYield': {lo: 0, hi: 5, w: 0.5},
    'PayoutRatio': {lo: 0, hi: 70, w: 0.5},
    'EarnGrowth': {lo: 0, hi: 25, w: 1},
    'RevGrowth': {lo: 0, hi: 20, w: 1},
    'Beta': {lo: 0.5, hi: 1.3, w: 1.5},
    'AnalystUpside': {lo: 0, hi: 30, w: 0.5},
    'RecMean': {lo: 1, hi: 2.5, w: 0.5},
    'ATR_pct': {lo: 0, hi: 4, w: 0.5},
    'Dist_SMA200': {lo: -5, hi: 20, w: 0.5},
    'RSI14': {lo: 30, hi: 70, w: 0.3},
  },
  dividend: {
    'DivYield': {lo: 3, hi: 8, w: 2},
    'PayoutRatio': {lo: 20, hi: 65, w: 1.5},
    'DebtToEquity': {lo: 0, hi: 90, w: 1.5},
    'CurrentRatio': {lo: 1, hi: 3, w: 1},
    'ROE': {lo: 10, hi: 35, w: 1},
    'ProfitMargin': {lo: 5, hi: 35, w: 1},
    'Beta': {lo: 0.3, hi: 1.1, w: 1.5},
    'PE': {lo: 8, hi: 22, w: 1},
    'EarnGrowth': {lo: -5, hi: 15, w: 0.5},
    'RecMean': {lo: 1, hi: 2.8, w: 0.5},
  },
  deepvalue: {
    'PE': {lo: 3, hi: 10, w: 1.5},
    'PB': {lo: 0.2, hi: 1, w: 1.5},
    'PS': {lo: 0.2, hi: 1.2, w: 1},
    'EV_EBITDA': {lo: 2, hi: 7, w: 1},
    'PEG': {lo: 0, hi: 1.2, w: 1},
    'DebtToEquity': {lo: 0, hi: 100, w: 1},
    'CurrentRatio': {lo: 1, hi: 4, w: 1},
    'ROE': {lo: 5, hi: 30, w: 0.5},
    'DivYield': {lo: 0, hi: 10, w: 0.5},
    'Dist_52wLow': {lo: 0, hi: 30, w: 1},
  },
  momentum: {
    'p1w': {lo: 3, hi: 20, w: 1.5},
    'p1m': {lo: 8, hi: 40, w: 1.5},
    'p3m': {lo: 15, hi: 80, w: 1.5},
    'p6m': {lo: 25, hi: 120, w: 1},
    'p12m': {lo: 30, hi: 200, w: 1},
    'RS21': {lo: 15, hi: 60, w: 1.5},
    'RS_acc': {lo: 5, hi: 30, w: 1},
    'ADX': {lo: 25, hi: 65, w: 1},
    'DI_diff': {lo: 15, hi: 45, w: 1},
    'RSI14': {lo: 60, hi: 85, w: 1},
    'VolRatio': {lo: 1.3, hi: 4, w: 1},
    'OBV_diff_pct': {lo: 10, hi: 50, w: 1},
    'Dist_52wHigh': {lo: -8, hi: 0, w: 1},
  },
};

function applyPreset(name) {
    if (!name || !PRESETS[name]) return;
    CONFIG = JSON.parse(JSON.stringify(PRESETS[name]));  // djup kopia
    applyConfigToUI();
    renderTable();
}

document.addEventListener('DOMContentLoaded', () => {
    window.METRICS_BY_KEY = {};
    for (const m of METRICS) METRICS_BY_KEY[m.key] = m;
    buildSettingsPanel();
    populateLandFilter();
    document.getElementById('searchBox').addEventListener('input', renderTable);
    document.getElementById('landFilter').addEventListener('change', renderTable);
    document.getElementById('minScore').addEventListener('input', renderTable);
    document.getElementById('exportBtn').addEventListener('click', exportCSV);
    document.getElementById('exportSettingsBtn').addEventListener('click', exportSettings);
    document.getElementById('importSettingsBtn').addEventListener('click', importSettings);
    document.getElementById('clearBtn').addEventListener('click', clearAll);
    document.getElementById('presetSelect').addEventListener('change', (e) => applyPreset(e.target.value));
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sort;
            if (SORT_KEY === key) SORT_DIR *= -1; else { SORT_KEY = key; SORT_DIR = -1; }
            renderTable();
        });
    });
    renderTable();
});
"""


def generate_html(records: list, datum_str: str) -> str:
    parts = ['<!DOCTYPE html><html lang="sv"><head><meta charset="utf-8">',
             '<title>Score Screener</title><style>{}</style></head><body>'.format(_CSS)]
    parts.append('<h1>🎯 Score Screener — {} aktier</h1>'.format(len(records)))
    parts.append(
        '<div class="sub">Data hämtad {}. Sätt idealintervall (min–max) för valfria nyckeltal nedan — '
        'värden inom intervallet ger 100 poäng, värden utanför avtar poängen ju längre bort de är. '
        'Tomt fält = nyckeltalet räknas inte med. Allt beräknas direkt i webbläsaren, ingen data skickas '
        'någonstans. Ej investeringsrådgivning.</div>'.format(datum_str))

    parts.append('<div class="panel">'
                 '<div class="toolbar">'
                 '<select id="presetSelect"><option value="">— Välj ett preset —</option>'
                 '<option value="value">Klassisk value</option>'
                 '<option value="turnaround">Turnaround-kandidater</option>'
                 '<option value="breakout">Breakout — snart utbrott</option>'
                 '<option value="trend">Ren trendföljande</option>'
                 '<option value="stable">Stora stabila bolag — köp & håll</option>'
                 '<option value="dividend">Utdelningsportfölj</option>'
                 '<option value="deepvalue">Deep value / contrarian</option>'
                 '<option value="momentum">Momentum / aggressiv tillväxt</option>'
                 '</select>'
                 '<button id="clearBtn">Rensa alla intervall</button>'
                 '</div>'
                 '<div id="settingsRoot"></div>'
                 '<div class="smallnote">Vikt (högra, smala fältet) är valfri — standard är 1. Högre vikt '
                 'väger tyngre i totalpoängen.</div>'
                 '</div>')

    parts.append('<div class="panel">'
                 '<h2>Spara/dela inställningar</h2>'
                 '<textarea id="settingsJson" placeholder="Klistra in tidigare exporterade inställningar här..."></textarea>'
                 '<div class="toolbar" style="margin-top:8px">'
                 '<button id="exportSettingsBtn">Visa nuvarande inställningar som JSON</button>'
                 '<button id="importSettingsBtn" class="primary">Använd inklistrad JSON</button>'
                 '</div>'
                 '<div class="smallnote">Inställningarna sparas INTE automatiskt (ingen data lagras i webbläsaren) '
                 '— kopiera JSON-texten och spara den själv om du vill återanvända dina intervall nästa gång.</div>'
                 '</div>')

    parts.append('<div class="toolbar">'
                 '<input type="text" id="searchBox" placeholder="Sök ticker eller namn...">'
                 '<select id="landFilter"><option value="">Alla länder</option></select>'
                 '<span>Min. totalpoäng: <input type="number" id="minScore" value="0" style="width:55px"></span>'
                 '<button id="exportBtn">Exportera till CSV</button>'
                 '<span id="rowCount" class="count"></span>'
                 '</div>')

    parts.append('<table><thead><tr>'
                 '<th>#</th>'
                 '<th data-sort="Ticker">Ticker</th>'
                 '<th data-sort="Namn">Namn</th>'
                 '<th data-sort="Land">Land</th>'
                 '<th data-sort="Sektor">Sektor</th>'
                 '<th data-sort="_kurs">Kurs</th>'
                 '<th data-sort="_total">Totalpoäng</th>'
                 '</tr></thead><tbody id="tbody"></tbody></table>')

    parts.append('<footer>Score Screener v1.0 · Data hämtad {} · '
                 'Interaktiv poängsättning sker lokalt i din webbläsare · Ej investeringsrådgivning</footer>'.format(datum_str))

    parts.append('<script>')
    parts.append('const RECORDS = ' + json.dumps(records, ensure_ascii=False) + ';')
    parts.append('const METRICS = ' + json.dumps(METRIC_REGISTRY, ensure_ascii=False) + ';')
    parts.append(_JS)
    parts.append('</script></body></html>')
    return ''.join(parts)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Score Screener — samlar teknisk + fundamental data till en interaktiv HTML-fil',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exempel:
  python score_screener_collector.py
  python score_screener_collector.py --land SE NO DK
  python score_screener_collector.py --max-tickers 200
  python score_screener_collector.py --ingen-fundamenta
  python score_screener_collector.py --workers 12
        """)
    parser.add_argument('--land', type=str, nargs='+', default=None)
    parser.add_argument('--min-vol', type=float, default=MIN_VOL_SEK)
    parser.add_argument('--max-tickers', type=int, default=None)
    parser.add_argument('--ingen-fundamenta', action='store_true',
                        help='Hoppa över yfinance .info-hämtning (mycket snabbare, endast teknisk data)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Antal parallella yfinance-anrop för fundamenta (default: {})'.format(DEFAULT_WORKERS))
    parser.add_argument('--out', type=str, default=None, help='Filnamn för HTML-utfilen')
    args = parser.parse_args()

    datum = pd.Timestamp.today().normalize()
    datum_str = str(datum)[:10]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    W = 76
    print("╔" + "═"*W + "╗")
    print(("║  SCORE SCREENER — DATAINSAMLARE  v1.0  —  {}".format(datum_str)).ljust(W+1) + "║")
    print(("║  Samlar {} tekniska + {} fundamentala nyckeltal".format(
        sum(1 for m in METRIC_REGISTRY if m['grupp'] == 'Teknisk'),
        sum(1 for m in METRIC_REGISTRY if m['grupp'] == 'Fundamental'))).ljust(W+1) + "║")
    print("╚" + "═"*W + "╝")

    tickers = [u[0] for u in UNIVERSE
               if not args.land or u[2].upper() in [l.upper() for l in args.land]]
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    print("\n[INFO] {} tickers i universumet".format(len(tickers)))

    all_hist, bench_close = fetch_all_history(tickers, datum_str, datum_str)
    if not all_hist:
        print("[FEL] Ingen data hämtades!")
        sys.exit(1)

    print("[INFO] Förberäknar tekniska indikatorer...")
    ticker_cache = precompute_all(all_hist, bench_close)

    fund_data = {}
    if not args.ingen_fundamenta:
        fund_data = fetch_fundamentals_batch(list(ticker_cache.keys()), args.workers)
        n_empty = sum(1 for v in fund_data.values() if not v)
        if len(fund_data) > 0:
            print("[INFO] Fundamenta: {}/{} tickers gav NOLL fält tillbaka ({:.0f}%) — "
                  "dessa ströks troligen av Yahoo eller saknar täckning helt".format(
                      n_empty, len(fund_data), n_empty / len(fund_data) * 100))
    else:
        print("[INFO] Hoppar över fundamenta (--ingen-fundamenta)")

    print("[INFO] Bygger records...")
    records = build_records(ticker_cache, datum, args.min_vol, fund_data)
    print("[INFO] {} aktier med tillräcklig data".format(len(records)))

    print_coverage_report(records)

    print("[INFO] Genererar HTML...")
    html = generate_html(records, datum_str)
    out_name = args.out or 'score_screener_{}.html'.format(datum_str)
    html_path = os.path.join(OUTPUT_FOLDER, out_name)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print("[SPARAT] {}".format(html_path))
    print("\n[KLAR] Öppna filen i valfri webbläsare för att sätta idealintervall och se poängsättningen live.")


if __name__ == '__main__':
    main()
