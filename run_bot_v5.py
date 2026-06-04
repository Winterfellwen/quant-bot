"""
DOGE QuantBot v5 — Web Service wrapper for Render free plan
- Trading loop in background thread
- HTTP server on $PORT for health checks
"""
import json, time, sys, logging, warnings, os, hashlib, secrets
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
from threading import Thread, Event
from flask import Flask, request, session, redirect, url_for, render_template, jsonify
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger("quantbot_v5")

SYMBOL = "DOGE/USDT:USDT"
TIMEFRAME = "60min"  # Huobi API uses '60min', not '1h'
CANDLES_LIMIT = 2000
INITIAL_TRAIN_BARS = 500
RETRAIN_EVERY = 48
LOOP_INTERVAL = 180
ORDER_SIZE = 5  # Default fallback; overridden by dynamic sizing
TARGET_CONTRACTS = 5  # 500 DOGE at 5x leverage (~ $9.4 margin)
ABSOLUTE_MAX_CONTRACTS = 10
MIN_USDT_TO_TRADE = 1.0
CONTRACT_SIZE = 100  # Each contract = 100 DOGE
LEVERAGE_SAFETY_FACTOR = 0.85  # Use 85% of available balance

FEATURE_NAMES = [
    'r1','r3','r5','r12','r24','hl_pct','co_pct','pos20','pos50',
    'rsi7','rsi14','macd','bb_pos','bb_width','atr14','volatility',
    'vr5','vr20','adx14','ma20_dev','ma50_dev','ma100_dev',
    'ma20_slope','ma50_slope','ma100_slope','body_ratio','dir3',
    'obv_roc','btc_r1','btc_r5','funding','oi_chg','oi_diverge','fg','fg_ma'
]

CONFIG_FILE = "bot_config.json"
HISTORY_FILE = "bot_history.json"
SESSION_SECRET = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(16)
DEFAULT_CONFIG = {
    'username': 'admin',
    'password_hash': hashlib.sha256('admin'.encode()).hexdigest(),
    'api_key': os.environ.get('HTX_API_KEY', ''),
    'api_secret': os.environ.get('HTX_API_SECRET', ''),
    'bot_enabled': True,
}

STRATEGY_SUMMARY = (
    "使用 LightGBM 预测下一根小时线的涨跌方向。\n"
    "始终保持多头或空头仓位，依据概率翻仓；\n"
    "根据回撤动态调整杠杆：>20% 时 5x，>40% 时 3x，否则 10x；\n"
    "每 48 根新数据重新训练一次模型；\n"
    "界面可查看策略、持仓、历史收益、交易记录，并在线启停与修改 Huobi API Key。"
)

# Trading state (for HTTP health endpoint)
trading_state = {
    "status": "starting",
    "last_check": None,
    "current_position": None,
    "last_prob": None,
    "cumulative_return": 1.0,
    "equity": 1.0,
    "drawdown_pct": 0.0,
    "leverage": 10,
}

# Recent log buffer for /logs endpoint
recent_logs = []
class LogCollector(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        recent_logs.append(msg)
        if len(recent_logs) > 100:
            recent_logs.pop(0)

_log_collector = LogCollector()
_log_collector.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S'))
logging.getLogger().addHandler(_log_collector)

# Manual control flags (thread-safe)
resize_requested = Event()

bot_stop_event = Event()
bot_thread = None

app = Flask(__name__)
app.secret_key = SESSION_SECRET

config = None
trade_history = []
performance_history = []


def load_config():
    global config
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        else:
            cfg = DEFAULT_CONFIG.copy()
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to load config, using defaults: {e}")
        cfg = DEFAULT_CONFIG.copy()
    config = cfg
    return config


def save_config():
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")


def load_history():
    global trade_history, performance_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                h = json.load(f)
                trade_history = h.get('trades', [])
                performance_history = h.get('performance', [])
        else:
            trade_history = []
            performance_history = []
            save_history()
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")
        trade_history = []
        performance_history = []
    return {'trades': trade_history, 'performance': performance_history}


def save_history():
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'trades': trade_history, 'performance': performance_history}, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")
def get_api_keys():
    key = (config.get('api_key') or os.environ.get('HTX_API_KEY', '')) if config else os.environ.get('HTX_API_KEY', '')
    secret = (config.get('api_secret') or os.environ.get('HTX_API_SECRET', '')) if config else os.environ.get('HTX_API_SECRET', '')
    return key, secret


def record_trade_event(event, details=None):
    # Backwards-compatible: allow calling with a single dict argument containing full entry
    if isinstance(event, dict) and details is None:
        src = event.copy()
        t = src.pop('time', None) or datetime.now().isoformat(sep=' ', timespec='seconds')
        ev = src.pop('event', None)
        # If caller passed an explicit 'event', use it; otherwise try 'type' or fallback to 'unknown'
        if ev is None:
            ev = src.pop('type', 'unknown')
        entry = {
            'time': t,
            'event': ev,
            'details': src if src else None,
        }
    else:
        entry = {
            'time': datetime.now().isoformat(sep=' ', timespec='seconds'),
            'event': event,
            'details': details,
        }
    trade_history.insert(0, entry)
    if len(trade_history) > 200:
        trade_history.pop()
    save_history()


def record_performance(cumulative, equity, drawdown):
    entry = {
        'time': datetime.now().isoformat(sep=' ', timespec='seconds'),
        'cumulative_return': cumulative,
        'equity': equity,
        'drawdown_pct': drawdown,
    }
    performance_history.insert(0, entry)
    if len(performance_history) > 200:
        performance_history.pop()
    save_history()


def verify_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest() == config.get('password_hash')


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapper


def sign_huobi(method, host, path, params, secret):
    """Sign Huobi API request using HMAC SHA256 (private endpoint auth)."""
    import hmac, hashlib, base64
    from urllib.parse import urlencode
    sorted_params = sorted(params.items())
    query_string = urlencode(sorted_params)
    payload = f"{method}\n{host}\n{path}\n{query_string}"
    digest = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode('utf-8')

def huobi_signed_post(path, body, timeout=15):
    """Make a signed POST request to Huobi API (api.hbdm.com)."""
    import requests
    from datetime import datetime, timezone
    from urllib.parse import urlencode
    KEY, SEC = get_api_keys()
    host = "api.hbdm.com"
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    params = {
        'AccessKeyId': KEY,
        'SignatureMethod': 'HmacSHA256',
        'SignatureVersion': '2',
        'Timestamp': timestamp,
    }
    signature = sign_huobi('POST', host, path, params, SEC)
    params['Signature'] = signature
    url = f"https://{host}{path}?{urlencode(params)}"
    r = requests.post(url, json=body, timeout=timeout)
    return r.json()

def fetch_ohlcv_http(symbol_code='DOGE-USDT', period=TIMEFRAME, limit=CANDLES_LIMIT):
    """Fetch OHLCV kline directly from Huobi (public, no auth)."""
    import requests
    url = f"https://api.hbdm.com/linear-swap-ex/market/history/kline?contract_code={symbol_code}&period={period}&size={limit}"
    r = requests.get(url, timeout=30)
    data = r.json()
    if data.get('status') == 'ok' and data.get('data'):
        candles = []
        for d in data['data']:
            candles.append([int(d['id']), float(d['open']), float(d['high']),
                           float(d['low']), float(d['close']), float(d['vol'])])
        candles.sort(key=lambda x: x[0])
        return candles
    return []

def fetch_funding_rate_http(symbol_code='DOGE-USDT'):
    """Fetch funding rate directly from Huobi."""
    import requests
    url = f"https://api.hbdm.com/linear-swap-api/v1/swap_funding_rate?contract_code={symbol_code}"
    r = requests.get(url, timeout=15)
    data = r.json()
    if data.get('status') == 'ok' and data.get('data'):
        return float(data['data'].get('funding_rate', 0))
    return 0

def fetch_open_interest_http(symbol_code='DOGE-USDT'):
    """Fetch open interest history directly."""
    import requests
    body = {'contract_code': symbol_code, 'period': '60min', 'amount_type': 1, 'size': 5}
    r = requests.post("https://api.hbdm.com/linear-swap-api/v1/swap_his_open_interest", json=body, timeout=15)
    data = r.json()
    ticks = []
    if data.get('status') == 'ok' and data.get('data'):
        d = data['data']
        if 'tick' in d: ticks = d['tick']
        elif isinstance(d, list) and len(d) > 0 and 'tick' in d[0]: ticks = d[0]['tick']
    return ticks

def fetch_position_http():
    """Fetch current position via signed HTTP (no ccxt). Raises on API error."""
    # Dry-run: do not call external API
    dry = os.environ.get('DRY_RUN') == '1' or (config and config.get('dry_run'))
    if dry:
        return None, 0, 0
    try:
        data = huobi_signed_post('/linear-swap-api/v1/swap_position_info', {'contract_code': 'DOGE-USDT'})
    except Exception as e:
        raise Exception(f"Position fetch failed: {e}")
    if not isinstance(data, dict):
        raise Exception(f"Position API unexpected response: {data}")
    if data.get('status') != 'ok':
        raise Exception(f"Position API error: {data.get('err_msg', data)}")
    for p in (data.get('data') or []):
        try:
            contracts = float(p.get('volume', 0) or 0)
        except Exception:
            contracts = 0
        if contracts > 0:
            direction = p.get('direction', '')
            side = 'long' if direction == 'buy' else 'short'
            entry = 0
            try:
                entry = float(p.get('cost_hold', 0)) if float(p.get('cost_hold', 0)) > 0 else float(p.get('cost_open', 0))
            except Exception:
                entry = 0
            return side, entry, int(contracts)
    return None, 0, 0

def set_leverage_http(leverage):
    """Set leverage via signed HTTP."""
    try:
        data = huobi_signed_post('/linear-swap-api/v1/swap_switch_lever_rate',
                                 {'contract_code': 'DOGE-USDT', 'lever_rate': leverage})
        if data.get('status') != 'ok':
            logger.warning(f"set_leverage_http: {data.get('err_msg', data)}")
    except Exception as e:
        logger.error(f"set_leverage_http: {e}")

def create_order_http(direction, offset, contracts, leverage):
    """Create order via signed HTTP (market order). Raises on failure."""
    set_leverage_http(leverage)
    body = {
        'contract_code': 'DOGE-USDT',
        'direction': direction,
        'offset': offset,
        'volume': contracts,
        'order_price_type': 'market',
        'lever_rate': leverage,
    }
    # Support dry-run mode via env var or config flag
    dry = os.environ.get('DRY_RUN') == '1' or (config and config.get('dry_run'))
    if dry:
        # simulate an execution at last market price
        try:
            candles = fetch_ohlcv_http('DOGE-USDT', TIMEFRAME, 1)
            last_price = candles[-1][4] if candles else 0
        except Exception:
            last_price = 0
        fake = {'status': 'ok', 'data': {'order_id': 'dryrun-'+str(int(time.time())), 'filled_avg_price': last_price, 'filled_volume': contracts, 'fee': 0}}
        logger.info(f"DRY RUN Order simulated: {direction} {offset} x{contracts} @ {leverage}x price={last_price}")
        return fake

    data = huobi_signed_post('/linear-swap-api/v1/swap_order', body)
    if data.get('status') != 'ok':
        raise Exception(f"Huobi order failed: {data.get('err_msg', data)}")
    logger.info(f"Order OK: {direction} {offset} x{contracts} @ {leverage}x")
    return data

def get_unified_balance():
    """
    Query USDT balance from Huobi unified account via direct API.
    Returns dict: {available, total, frozen, unrealized_pnl, margin_static, error?}
    Works for both merged and unified accounts (err_code 4002 workaround).
    """
    try:
        import requests
        from datetime import datetime, timezone
        from urllib.parse import urlencode
        KEY, SEC = get_api_keys()
        if not KEY or not SEC:
            return {'available': 0, 'total': 0, 'frozen': 0, 'unrealized_pnl': 0, 'error': 'no_keys'}
        host = "api.hbdm.com"
        path = "/linear-swap-api/v3/unified_account_info"
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        params = {
            'AccessKeyId': KEY,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': timestamp,
        }
        signature = sign_huobi('POST', host, path, params, SEC)
        params['Signature'] = signature
        url = f"https://{host}{path}?{urlencode(params)}"
        r = requests.post(url, timeout=15)
        data = r.json()
        if data.get('status') == 'ok' or data.get('code') == 200:
            for asset in data.get('data', []):
                if asset.get('margin_asset') == 'USDT':
                    return {
                        'available': float(asset.get('withdraw_available', 0) or 0),
                        'total': float(asset.get('margin_balance', 0) or 0),
                        'frozen': float(asset.get('margin_frozen', 0) or 0),
                        'unrealized_pnl': float(asset.get('cross_profit_unreal', 0) or 0),
                        'margin_static': float(asset.get('margin_static', 0) or 0),
                    }
            return {'available': 0, 'total': 0, 'frozen': 0, 'unrealized_pnl': 0, 'error': 'no_usdt'}
        else:
            logger.warning(f"Unified account error: {data.get('err_msg') or data.get('msg')}")
            return {'available': 0, 'total': 0, 'frozen': 0, 'unrealized_pnl': 0, 'error': data.get('err_msg') or data.get('msg')}
    except Exception as e:
        logger.error(f"Unified account fetch failed: {e}")
        return {'available': 0, 'total': 0, 'frozen': 0, 'unrealized_pnl': 0, 'error': str(e)}

def calculate_dynamic_size(available_usdt, price, leverage, existing_contracts=0):
    """
    Calculate optimal order size in contracts based on available balance and target leverage.
    Target: 5 contracts (500 DOGE) at 5x leverage (~$9.4 margin).
    Caps scale up with available balance.
    Returns: int (number of contracts) or 0 if insufficient balance.
    """
    if available_usdt is None or available_usdt < MIN_USDT_TO_TRADE:
        return 0
    notional_per_contract = CONTRACT_SIZE * price
    margin_per_contract = notional_per_contract / max(leverage, 1)
    # Consider margin already occupied by existing contracts so we don't double-count.
    existing_margin = (existing_contracts * CONTRACT_SIZE * price) / max(leverage, 1)
    usable = max(0.0, available_usdt * LEVERAGE_SAFETY_FACTOR + existing_margin)
    max_contracts = int(usable / margin_per_contract)
    if max_contracts <= 0:
        return 0
    # Scale cap based on available funds
    if available_usdt < 20:
        cap = 5
    elif available_usdt < 50:
        cap = 7
    else:
        cap = ABSOLUTE_MAX_CONTRACTS
    # Keep a safety buffer of one contract margin to avoid touching edge cases
    try:
        buffer_contracts = 1
        available_contracts = max(0, max_contracts - buffer_contracts)
    except Exception:
        available_contracts = max_contracts
    return min(available_contracts, cap)

def compute_features(df, btc_ret1=0, btc_ret5=0, funding=0, oi_chg=0, oi_diverge=0, fg=50):
    c,h,l,v,o=df['close'],df['high'],df['low'],df['volume'],df['open']
    out={}
    for p in [1,3,5,12,24]: out[f'r{p}']=c.pct_change(p)
    out['hl_pct']=(h-l)/(c+1e-9); out['co_pct']=(c-o)/(o+1e-9)
    for pd_ in [20,50]:
        rh=h.rolling(pd_).max(); rl=l.rolling(pd_).min()
        out[f'pos{pd_}']=(c-rl)/(rh-rl+1e-9)
    for pd_ in [7,14]:
        delta=c.diff(); gain=delta.clip(lower=0).rolling(pd_).mean(); loss=(-delta.clip(upper=0)).rolling(pd_).mean()
        out[f'rsi{pd_}']=100-100/(1+gain/(loss+1e-9))
    ef=c.ewm(span=12,adjust=False).mean(); es=c.ewm(span=26,adjust=False).mean()
    line=ef-es; sig=line.ewm(span=9,adjust=False).mean(); out['macd']=line-sig
    mid=c.rolling(20).mean(); std=c.rolling(20).std()
    out['bb_pos']=(c-mid)/(2*std+1e-9); out['bb_width']=(4*std)/(mid+1e-9)
    tr=np.maximum(h-l,np.maximum((h-c.shift(1)).abs(),(l-c.shift(1)).abs()))
    out['atr14']=tr.rolling(14).mean()/c
    out['volatility']=c.pct_change().rolling(14).std()
    out['vr5']=v/(v.rolling(5).mean()+1e-9)-1; out['vr20']=v/(v.rolling(20).mean()+1e-9)-1
    dmp=h.diff().clip(lower=0); dmn=(-l.diff()).clip(lower=0)
    dmp_s=dmp.rolling(14).mean(); dmn_s=dmn.rolling(14).mean()
    dx=(abs(dmp_s-dmn_s)/(dmp_s+dmn_s+1e-9))*100; out['adx14']=dx.rolling(14).mean().fillna(0)
    for pd_ in [20,50,100]:
        ma=c.rolling(pd_).mean()
        out[f'ma{pd_}_dev']=c/(ma+1e-9)-1; out[f'ma{pd_}_slope']=(ma-ma.shift(pd_))/(ma.shift(pd_)+1e-9)
    body=abs(c-o); tr2=h-l+1e-9
    out['body_ratio']=body/tr2; out['dir3']=(c>o).astype(int).rolling(3).sum()
    obv_=(np.sign(c.diff())*v).fillna(0).cumsum(); out['obv_roc']=obv_.pct_change(10)
    n=len(c)
    out['btc_r1']=np.full(n,btc_ret1); out['btc_r5']=np.full(n,btc_ret5)
    out['funding']=np.full(n,funding); out['oi_chg']=np.full(n,oi_chg)
    out['oi_diverge']=np.full(n,oi_diverge); out['fg']=np.full(n,fg); out['fg_ma']=np.full(n,fg)
    arrs=[]
    for name in FEATURE_NAMES:
        v=out[name]
        arrs.append(v.values if hasattr(v,'values') else v)
    return np.nan_to_num(np.column_stack(arrs),nan=0.0)

def fetch_all():
    """Fetch all market data via direct HTTP (no ccxt)."""
    import requests
    candles = fetch_ohlcv_http('DOGE-USDT', TIMEFRAME, CANDLES_LIMIT)
    if not candles:
        raise Exception("fetch_ohlcv_http returned empty candles")
    df = pd.DataFrame(candles, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='s'); df.set_index('ts', inplace=True)
    btc_r1=0; btc_r5=0
    try:
        btc = fetch_ohlcv_http('BTC-USDT', TIMEFRAME, 6)
        bc = [b[4] for b in btc]
        if len(bc)>1: btc_r1=(bc[-1]-bc[-2])/bc[-2]
        if len(bc)>5: btc_r5=(bc[-1]-bc[-6])/bc[-6]
    except: pass
    funding = 0
    try:
        funding = fetch_funding_rate_http('DOGE-USDT')
    except: pass
    oi_chg=0; oi_diverge=0
    try:
        ticks = fetch_open_interest_http('DOGE-USDT')
        if ticks:
            ov=[float(t['value']) for t in ticks[-5:]]
            if len(ov)>4: oi_chg=(ov[-1]-ov[0])/ov[0]
            if len(ov)>1 and len(df)>1:
                last_d=df['close'].iloc[-1]/df['close'].iloc[-2]-1
                oi_d=ov[-1]/ov[-2]-1
                oi_diverge=1 if np.sign(oi_d)!=np.sign(last_d) else 0
    except: pass
    fg=50
    try:
        r=requests.get('https://api.alternative.me/fng/?limit=1',headers={'User-Agent':'Mozilla/5.0'},timeout=10)
        fg=int(r.json()['data'][0]['value'])
    except: pass
    return df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg

def get_dynamic_leverage(drawdown_pct):
    if drawdown_pct > 40: return 3
    elif drawdown_pct > 20: return 5
    return 10

def get_position(ex=None):
    """Return (side, entry_price, contracts) using direct HTTP (ex arg ignored)."""
    return fetch_position_http()

def get_unrealized_pnl(cur_side, entry_price, current_price, leverage):
    if cur_side is None or entry_price == 0: return 1.0
    if cur_side == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * leverage
    else:
        pnl_pct = (entry_price - current_price) / entry_price * leverage
    return 1.0 + pnl_pct

def close_position(ex, side, leverage, contracts):
    """Close position via direct HTTP, then verify it's actually closed."""
    if contracts <= 0: return
    direction = 'sell' if side == 'long' else 'buy'
    resp = create_order_http(direction, 'close', contracts, leverage)
    details = {'action': 'close', 'side': side, 'contracts': contracts, 'leverage': leverage, 'api_response': resp}
    # Try to extract executed price/volume/fee from response if present
    if isinstance(resp, dict) and 'data' in resp:
        d = resp.get('data')
        if isinstance(d, dict):
            details.update({
                'executed_price': d.get('filled_avg_price') or d.get('price') or None,
                'executed_volume': d.get('filled_volume') or d.get('filled_amount') or None,
                'fee': d.get('fee') or None,
            })
    record_trade_event('close', details)
    logger.info(f"Close order sent: {side} x{contracts}")
    # If dry-run, skip external verification
    dry = os.environ.get('DRY_RUN') == '1' or (config and config.get('dry_run'))
    if dry:
        logger.info("Dry-run mode: skipping close verification")
        return

    # Wait and verify position is actually closed (handle transient errors)
    for attempt in range(5):
        time.sleep(2)
        try:
            _, _, remaining = fetch_position_http()
            if remaining <= 0:
                logger.info(f"Position confirmed closed (attempt {attempt+1}, remaining={remaining})")
                return
        except Exception as e:
            logger.warning(f"Close verify attempt {attempt+1} failed (transient): {e}")
            continue  # Retry on transient error
    # Final check - raise if still open
    try:
        _, _, remaining = fetch_position_http()
        if remaining > 0:
            raise Exception(f"Position NOT closed after order! Remaining: {remaining} contracts")
        logger.info(f"Position confirmed closed (final check, remaining={remaining})")
    except Exception as e:
        if "NOT closed" in str(e):
            raise
        raise Exception(f"Cannot verify close status: {e}. Assuming position still open.")

def add_to_position(ex, side, contracts, leverage):
    """Add contracts via direct HTTP."""
    try:
        if contracts <= 0: return
        direction = 'buy' if side == 'long' else 'sell'
        resp = create_order_http(direction, 'open', contracts, leverage)
        details = {'action': 'add', 'side': side, 'contracts': contracts, 'leverage': leverage, 'api_response': resp}
        if isinstance(resp, dict) and 'data' in resp:
            d = resp.get('data')
            if isinstance(d, dict):
                details.update({
                    'executed_price': d.get('filled_avg_price') or d.get('price') or None,
                    'executed_volume': d.get('filled_volume') or d.get('filled_amount') or None,
                    'fee': d.get('fee') or None,
                })
        record_trade_event('increase', details)
        logger.info(f"Added {contracts} to {side} position")
    except Exception as e:
        logger.error(f"Add failed: {e}")

def reduce_position(ex, side, contracts, leverage):
    """Reduce contracts via direct HTTP."""
    try:
        if contracts <= 0: return
        direction = 'sell' if side == 'long' else 'buy'
        resp = create_order_http(direction, 'close', contracts, leverage)
        details = {'action': 'reduce', 'side': side, 'contracts': contracts, 'leverage': leverage, 'api_response': resp}
        if isinstance(resp, dict) and 'data' in resp:
            d = resp.get('data')
            if isinstance(d, dict):
                details.update({
                    'executed_price': d.get('filled_avg_price') or d.get('price') or None,
                    'executed_volume': d.get('filled_volume') or d.get('filled_amount') or None,
                    'fee': d.get('fee') or None,
                })
        record_trade_event('reduce', details)
        logger.info(f"Reduced {contracts} from {side} position")
    except Exception as e:
        logger.error(f"Reduce failed: {e}")

def open_position(ex, side, leverage, contracts):
    """Open new position with direct HTTP."""
    try:
        if contracts <= 0: return
        direction = 'buy' if side == 'long' else 'sell'
        resp = create_order_http(direction, 'open', contracts, leverage)
        details = {'action': 'open', 'side': side, 'contracts': contracts, 'leverage': leverage, 'api_response': resp}
        if isinstance(resp, dict) and 'data' in resp:
            d = resp.get('data')
            if isinstance(d, dict):
                details.update({
                    'executed_price': d.get('filled_avg_price') or d.get('price') or None,
                    'executed_volume': d.get('filled_volume') or d.get('filled_amount') or None,
                    'fee': d.get('fee') or None,
                })
        record_trade_event('open', details)
        logger.info(f"Opened {side} x{contracts} @ {leverage}x")
    except Exception as e:
        logger.error(f"Open failed: {e}")

def trading_loop():
    logger.info("=== QuantBot v5 Trading Loop STARTING ===")  # Debug line
    import lightgbm as lgb
    logger.info("=== QuantBot v5 Trading Loop Started ===")

    # Initial fetch with retry (Huobi ccxt may fail on Render)
    ex = None
    for attempt in range(6):
        try:
            df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg = fetch_all()
            logger.info(f"Init fetch OK: {len(df)} candles, price={df['close'].iloc[-1]:.5f}")
            break
        except Exception as e:
            logger.error(f"Init fetch {attempt+1}/6 failed: {e}")
            time.sleep(30)
    else:
        logger.error("Init fetch failed after 6 attempts")
        trading_state['status'] = 'error'
        return
    X = compute_features(df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg)
    c = df['close'].values
    y_binary = np.zeros(len(c))
    y_binary[:-1] = (c[1:] > c[:-1]).astype(int)
    y_binary[-1] = np.nan
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_binary)
    Xt = X[valid]; yt = y_binary[valid]
    if len(Xt) < 300:
        logger.error(f"Insufficient data: {len(Xt)}")
        trading_state['status'] = 'error'
        return

    train_size = min(INITIAL_TRAIN_BARS, len(Xt) - 2)
    train_valid = ~np.isnan(yt[:train_size])
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.03,
        num_leaves=20, reg_lambda=3, min_child_samples=30,
        random_state=42, verbose=-1, force_row_wise=True,
        class_weight='balanced'
    )

    # Initial training
    model.fit(Xt[:train_size][train_valid], yt[:train_size][train_valid])
    logger.info(f"Initial training: {train_valid.sum()} samples, {len(df)} candles")

    cur_side, entry_price, cur_contracts = get_position(None)
    logger.info(f"Current position: {cur_side or 'none'}, entry: {entry_price}, contracts: {cur_contracts}")

    first_bar = True
    last_retrain_size = len(df)
    cumulative_return = 1.0
    peak_return = 1.0
    trading_state['status'] = 'running'

    while not bot_stop_event.is_set():
        try:
            df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg = fetch_all()
            X = compute_features(df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg)

            if len(df) - last_retrain_size >= RETRAIN_EVERY:
                c = df['close'].values
                y_binary = np.zeros(len(c))
                y_binary[:-1] = (c[1:] > c[:-1]).astype(int)
                y_binary[-1] = np.nan
                valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_binary)
                Xt_rt = X[valid]; yt_rt = y_binary[valid]
                rt_valid = ~np.isnan(yt_rt)
                model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.03,
                    num_leaves=20, reg_lambda=3, min_child_samples=30,
                    random_state=42, verbose=-1, force_row_wise=True,
                    class_weight='balanced'
                )
                model.fit(Xt_rt[rt_valid], yt_rt[rt_valid])
                last_retrain_size = len(df)
                logger.info(f"Retrained: {rt_valid.sum()} samples")

            prob_up = float(model.predict_proba(X[-1:])[0][1])
            cur_side, entry_price, cur_contracts = get_position(None)
            price = float(df['close'].iloc[-1])

            # Use last known leverage to estimate unrealized PnL for equity calculation,
            # then compute drawdown and choose dynamic leverage.
            last_leverage = trading_state.get('leverage', 10)
            unrealized = get_unrealized_pnl(cur_side, entry_price, price, last_leverage)
            estimated_equity = cumulative_return * unrealized
            dd_pct = (1 - estimated_equity / max(peak_return, cumulative_return)) * 100
            leverage = get_dynamic_leverage(max(dd_pct, 0))

            # Get balance and calculate dynamic order size
            balance = get_unified_balance()
            available_usdt = balance.get('available', 0)
            order_size = calculate_dynamic_size(available_usdt, price, leverage, existing_contracts=cur_contracts)

            # Update state for HTTP
            trading_state.update({
                'last_check': datetime.now().isoformat(),
                'current_position': cur_side,
                'current_contracts': cur_contracts,
                'last_prob': prob_up,
                'cumulative_return': cumulative_return,
                'equity': estimated_equity,
                'drawdown_pct': dd_pct,
                'leverage': leverage,
                'available_usdt': available_usdt,
                'total_usdt': balance.get('total', 0),
                'unrealized_pnl': balance.get('unrealized_pnl', 0),
                'next_order_size': order_size,
            })
            record_performance(cumulative_return, estimated_equity, dd_pct)

            if first_bar:
                # On first loop after start/restart: prefer to preserve any existing exchange position.
                first_bar = False
                if cur_side is None:
                    if order_size <= 0:
                        logger.warning(f"Insufficient balance ({available_usdt:.4f} USDT) to open initial position, will retry next loop")
                    else:
                        init_side = "long" if prob_up > 0.5 else "short"
                        open_position(ex, init_side, leverage, order_size)
                        logger.info(f"Initial entry: {init_side} x{order_size} prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} dd={dd_pct:.1f}% bal={available_usdt:.2f}")
                else:
                    # There is an existing position on the exchange. If its direction matches current model
                    # prediction, do not close+reopen — only rebalance (add/reduce) towards the target size.
                    should_long = prob_up > 0.5
                    should_short = prob_up < 0.5
                    if (should_long and cur_side == 'long') or (should_short and cur_side == 'short'):
                        # same direction: adjust size to order_size
                        if order_size > cur_contracts:
                            diff = order_size - cur_contracts
                            add_to_position(ex, cur_side, diff, leverage)
                            logger.info(f"Startup REBALANCE +{diff}: {cur_side} {cur_contracts}→{order_size} prob={prob_up:.4f} bal={available_usdt:.2f}")
                            record_trade_event({
                                'event': 'startup_rebalance',
                                'action': 'add',
                                'side': cur_side,
                                'from_contracts': cur_contracts,
                                'to_contracts': order_size,
                                'diff': diff,
                                'leverage': leverage,
                                'price': price,
                                'note': 'startup_rebalance_add'
                            })
                        elif order_size < cur_contracts:
                            diff = cur_contracts - order_size
                            reduce_position(ex, cur_side, diff, leverage)
                            logger.info(f"Startup REBALANCE -{diff}: {cur_side} {cur_contracts}→{order_size} prob={prob_up:.4f} bal={available_usdt:.2f}")
                            record_trade_event({
                                'event': 'startup_rebalance',
                                'action': 'reduce',
                                'side': cur_side,
                                'from_contracts': cur_contracts,
                                'to_contracts': order_size,
                                'diff': diff,
                                'leverage': leverage,
                                'price': price,
                                'note': 'startup_rebalance_reduce'
                            })
                        else:
                            logger.info(f"Startup: holding existing {cur_side} x{cur_contracts} (matches target size)")
                            record_trade_event({
                                'event': 'startup_rebalance',
                                'action': 'hold',
                                'side': cur_side,
                                'contracts': cur_contracts,
                                'leverage': leverage,
                                'price': price,
                                'note': 'startup_rebalance_hold'
                            })
                    else:
                        # Opposite direction: perform normal flip (close then open)
                        if cur_side == 'short' and should_long:
                            realized = get_unrealized_pnl('short', entry_price, price, leverage)
                            close_position(ex, 'short', leverage, cur_contracts)
                            cumulative_return *= realized
                            peak_return = max(peak_return, cumulative_return)
                            time.sleep(1)
                            if order_size > 0:
                                open_position(ex, 'long', leverage, order_size)
                                logger.info(f"Startup Flip LONG x{order_size}: prob={prob_up:.4f} lev={leverage}x price={price:.4f} cum={cumulative_return:.3f}x")
                                record_trade_event({
                                    'event': 'startup_flip',
                                    'from_side': 'short',
                                    'to_side': 'long',
                                    'from_contracts': cur_contracts,
                                    'to_contracts': order_size,
                                    'leverage': leverage,
                                    'price': price,
                                    'note': 'startup_flip_long'
                                })
                        elif cur_side == 'long' and should_short:
                            realized = get_unrealized_pnl('long', entry_price, price, leverage)
                            close_position(ex, 'long', leverage, cur_contracts)
                            cumulative_return *= realized
                            peak_return = max(peak_return, cumulative_return)
                            time.sleep(1)
                            if order_size > 0:
                                open_position(ex, 'short', leverage, order_size)
                                logger.info(f"Startup Flip SHORT x{order_size}: prob={prob_up:.4f} lev={leverage}x price={price:.4f} cum={cumulative_return:.3f}x")
                                record_trade_event({
                                    'event': 'startup_flip',
                                    'from_side': 'long',
                                    'to_side': 'short',
                                    'from_contracts': cur_contracts,
                                    'to_contracts': order_size,
                                    'leverage': leverage,
                                    'price': price,
                                    'note': 'startup_flip_short'
                                })
                # skip the rest of this iteration to avoid double-handling
                continue

            # Handle manual resize request
            if resize_requested.is_set():
                logger.info(f"MANUAL RESIZE triggered (was {cur_side} x{cur_contracts})")
                if cur_side is not None and cur_contracts > 0:
                    close_position(ex, cur_side, leverage, cur_contracts)
                    time.sleep(2)
                    cur_side, entry_price, cur_contracts = get_position(None)
                if cur_side is None or cur_contracts == 0:
                    new_size = calculate_dynamic_size(available_usdt, price, leverage, existing_contracts=cur_contracts)
                    if new_size > 0:
                        side = "long" if prob_up > 0.5 else "short"
                        open_position(ex, side, leverage, new_size)
                        logger.info(f"MANUAL RESIZE done: {side} x{new_size} prob={prob_up:.4f} bal={available_usdt:.2f}")
                    else:
                        logger.warning(f"MANUAL RESIZE failed: insufficient balance {available_usdt:.4f}")
                resize_requested.clear()
                continue  # Skip remaining logic this iteration

            # Re-enter if no position (e.g. closed externally)
            if cur_side is None:
                if order_size > 0:
                    side = "long" if prob_up > 0.5 else "short"
                    open_position(ex, side, leverage, order_size)
                    logger.info(f"Re-entry (was None): {side} x{order_size} prob={prob_up:.4f} bal={available_usdt:.2f}")
                else:
                    logger.warning(f"No position, insufficient balance {available_usdt:.4f}, waiting")
            elif should_long and cur_side == "short":
                realized = get_unrealized_pnl("short", entry_price, price, leverage)
                close_position(ex, "short", leverage, cur_contracts)  # Raises on failure
                cumulative_return *= realized  # Only update AFTER close verified
                peak_return = max(peak_return, cumulative_return)
                time.sleep(1)
                new_size = calculate_dynamic_size(available_usdt, price, leverage)
                if new_size > 0:
                    open_position(ex, "long", leverage, new_size)
                    logger.info(f"Flip LONG x{new_size}: prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} cum={cumulative_return:.3f}x dd={dd_pct:.1f}% bal={available_usdt:.2f}")
                else:
                    logger.warning(f"Skipped LONG open: insufficient balance {available_usdt:.4f}")
            elif should_short and cur_side == "long":
                realized = get_unrealized_pnl("long", entry_price, price, leverage)
                close_position(ex, "long", leverage, cur_contracts)  # Raises on failure
                cumulative_return *= realized  # Only update AFTER close verified
                peak_return = max(peak_return, cumulative_return)
                time.sleep(1)
                new_size = calculate_dynamic_size(available_usdt, price, leverage)
                if new_size > 0:
                    open_position(ex, "short", leverage, new_size)
                    logger.info(f"Flip SHORT x{new_size}: prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} cum={cumulative_return:.3f}x dd={dd_pct:.1f}% bal={available_usdt:.2f}")
                else:
                    logger.warning(f"Skipped SHORT open: insufficient balance {available_usdt:.4f}")
            # Same direction - rebalance incrementally (add without close+reopen)
            elif cur_side is not None:
                if cur_contracts < order_size:
                    diff = order_size - cur_contracts
                    add_to_position(ex, cur_side, diff, leverage)
                    logger.info(f"REBALANCE +{diff}: {cur_side} {cur_contracts}→{order_size} prob={prob_up:.4f} bal={available_usdt:.2f}")
                elif cur_contracts > order_size:
                    diff = cur_contracts - order_size
                    reduce_position(ex, cur_side, diff, leverage)
                    logger.info(f"REBALANCE -{diff}: {cur_side} {cur_contracts}→{order_size} prob={prob_up:.4f} bal={available_usdt:.2f}")
                else:
                    logger.info(f"Holding {cur_side} x{cur_contracts}: prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} cum={cumulative_return:.3f}x dd={dd_pct:.1f}% bal={available_usdt:.2f} (size matches)")

        except Exception as e:
            logger.error(f"Loop error: {e}")
            import traceback; traceback.print_exc()

        if bot_stop_event.wait(LOOP_INTERVAL):
            break

    trading_state['status'] = 'stopped'

def is_bot_running():
    return bot_thread is not None and bot_thread.is_alive()


def start_trading():
    global bot_thread, bot_stop_event
    if is_bot_running():
        return False
    bot_stop_event.clear()
    bot_thread = Thread(target=trading_loop, daemon=True)
    bot_thread.start()
    trading_state['status'] = 'starting'
    return True


def stop_trading():
    if is_bot_running():
        bot_stop_event.set()
        trading_state['status'] = 'stopping'
        return True
    trading_state['status'] = 'stopped'
    return False


@app.route('/health')
def health():
    return jsonify(trading_state)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == config.get('username') and verify_password(password):
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='用户名或密码错误')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def dashboard():
    return render_template(
        'dashboard.html',
        username=config.get('username'),
        strategy=STRATEGY_SUMMARY,
        state=trading_state,
        trades=trade_history[:20],
        performance=performance_history[:20],
    )


@app.route('/start', methods=['POST'])
@login_required
def start():
    config['bot_enabled'] = True
    save_config()
    start_trading()
    return redirect(url_for('dashboard'))


@app.route('/stop', methods=['POST'])
@login_required
def stop():
    config['bot_enabled'] = False
    save_config()
    stop_trading()
    return redirect(url_for('dashboard'))


@app.route('/resize', methods=['POST'])
@login_required
def resize():
    if resize_requested.is_set():
        logger.info('Resize request already pending')
    else:
        resize_requested.set()
        logger.info('Manual resize queued')
    return redirect(url_for('dashboard'))


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    message = None
    if request.method == 'POST':
        config['api_key'] = request.form.get('api_key', '').strip()
        config['api_secret'] = request.form.get('api_secret', '').strip()
        new_username = request.form.get('username', '').strip()
        if new_username:
            config['username'] = new_username
        new_password = request.form.get('password', '')
        if new_password:
            config['password_hash'] = hashlib.sha256(new_password.encode('utf-8')).hexdigest()
            message = '设置已保存，密码已更新。'
        else:
            message = '设置已保存。'
        save_config()
    return render_template('settings.html', config=config, message=message)


@app.route('/api/status')
@login_required
def api_status():
    return jsonify(trading_state)


@app.route('/api/history')
@login_required
def api_history():
    return jsonify({'trades': trade_history, 'performance': performance_history})


@app.route('/api/position')
@login_required
def api_position():
    try:
        side, entry, contracts = fetch_position_http()
        result = {'positions': []}
        if side is not None and contracts > 0:
            result['positions'].append({
                'symbol': 'DOGE/USDT:USDT',
                'side': side,
                'contracts': contracts,
                'entry_price': entry,
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def keep_alive():
    """Self-ping every 50 seconds to prevent Render free plan from sleeping."""
    import requests
    port = int(os.environ.get('PORT', 8080))
    url = f"http://127.0.0.1:{port}/health"
    while True:
        time.sleep(50)
        try:
            r = requests.get(url, timeout=10)
            logger.info(f"[KeepAlive] Self-ping OK ({r.status_code})")
        except Exception as e:
            logger.warning(f"[KeepAlive] Self-ping failed: {e}")


if __name__ == "__main__":
    config = load_config()
    history = load_history()
    trade_history = history['trades']
    performance_history = history['performance']

    if config.get('bot_enabled'):
        start_trading()

    alive_thread = Thread(target=keep_alive, daemon=True)
    alive_thread.start()

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
