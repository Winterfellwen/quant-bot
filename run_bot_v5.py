"""
DOGE QuantBot v5 — Web Service wrapper for Render free plan
- Trading loop in background thread
- HTTP server on $PORT for health checks
"""
import json, time, sys, logging, warnings, os
import numpy as np
import pandas as pd
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event
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

def get_api_keys():
    key = os.environ.get('HTX_API_KEY', '')
    secret = os.environ.get('HTX_API_SECRET', '')
    return key, secret

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
    data = huobi_signed_post('/linear-swap-api/v1/swap_position_info', {'contract_code': 'DOGE-USDT'})
    if data.get('status') != 'ok':
        raise Exception(f"Position API error: {data.get('err_msg', data)}")
    for p in (data.get('data') or []):
        contracts = float(p.get('volume', 0) or 0)
        if contracts > 0:
            direction = p.get('direction', '')
            side = 'long' if direction == 'buy' else 'short'
            entry = float(p.get('cost_hold', 0)) if float(p.get('cost_hold', 0)) > 0 else float(p.get('cost_open', 0))
            return side, entry, int(contracts)  # Return as int for volume field
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
    data = huobi_signed_post('/linear-swap-api/v1/swap_order', body)
    if data.get('status') != 'ok':
        raise Exception(f"Huobi order failed: {data.get('err_msg', data)}")
    logger.info(f"Order OK: {direction} {offset} x{contracts} @ {leverage}x")

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

def calculate_dynamic_size(available_usdt, price, leverage):
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
    usable = available_usdt * LEVERAGE_SAFETY_FACTOR
    max_contracts = int(usable / margin_per_contract)
    # Scale cap based on available funds
    if available_usdt < 20:
        cap = 5
    elif available_usdt < 50:
        cap = 7
    else:
        cap = ABSOLUTE_MAX_CONTRACTS
    return max(1, min(max_contracts, cap))

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
    create_order_http(direction, 'close', contracts, leverage)
    logger.info(f"Close order sent: {side} x{contracts}")
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
        create_order_http(direction, 'open', contracts, leverage)
        logger.info(f"Added {contracts} to {side} position")
    except Exception as e:
        logger.error(f"Add failed: {e}")

def reduce_position(ex, side, contracts, leverage):
    """Reduce contracts via direct HTTP."""
    try:
        if contracts <= 0: return
        direction = 'sell' if side == 'long' else 'buy'
        create_order_http(direction, 'close', contracts, leverage)
        logger.info(f"Reduced {contracts} from {side} position")
    except Exception as e:
        logger.error(f"Reduce failed: {e}")

def open_position(ex, side, leverage, contracts):
    """Open new position with direct HTTP."""
    try:
        if contracts <= 0: return
        direction = 'buy' if side == 'long' else 'sell'
        create_order_http(direction, 'open', contracts, leverage)
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
    model.fit(Xt[:train_size][train_valid], yt[:train_size][train_valid])
    logger.info(f"Initial training: {train_valid.sum()} samples, {len(df)} candles")

    cur_side, entry_price, cur_contracts = get_position(None)
    logger.info(f"Current position: {cur_side or 'none'}, entry: {entry_price}, contracts: {cur_contracts}")

    first_bar = True
    last_retrain_size = len(df)
    cumulative_return = 1.0
    peak_return = 1.0
    trading_state['status'] = 'running'

    while True:
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

            unrealized = get_unrealized_pnl(cur_side, entry_price, price, 10)
            estimated_equity = cumulative_return * unrealized
            dd_pct = (1 - estimated_equity / max(peak_return, cumulative_return)) * 100
            leverage = get_dynamic_leverage(max(dd_pct, 0))

            # Get balance and calculate dynamic order size
            balance = get_unified_balance()
            available_usdt = balance.get('available', 0)
            order_size = calculate_dynamic_size(available_usdt, price, leverage)

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

            if first_bar and cur_side is None:
                if order_size <= 0:
                    logger.warning(f"Insufficient balance ({available_usdt:.4f} USDT) to open initial position, will retry next loop")
                    first_bar = False
                else:
                    init_side = "long" if prob_up > 0.5 else "short"
                    open_position(ex, init_side, leverage, order_size)
                    logger.info(f"Initial entry: {init_side} x{order_size} prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} dd={dd_pct:.1f}% bal={available_usdt:.2f}")
                    first_bar = False
            else:
                first_bar = False
                should_long = prob_up > 0.5
                should_short = prob_up < 0.5

                # Handle manual resize request
                if resize_requested.is_set():
                    logger.info(f"MANUAL RESIZE triggered (was {cur_side} x{cur_contracts})")
                    if cur_side is not None and cur_contracts > 0:
                        close_position(ex, cur_side, leverage, cur_contracts)
                        time.sleep(2)
                        cur_side, entry_price, cur_contracts = get_position(None)
                    if cur_side is None or cur_contracts == 0:
                        new_size = calculate_dynamic_size(available_usdt, price, leverage)
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
                    realized = get_unrealized_pnl("short", entry_price, price, 10)
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
                    realized = get_unrealized_pnl("long", entry_price, price, 10)
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

        time.sleep(LOOP_INTERVAL)

# HTTP server for Render health checks
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps(trading_state, default=str)
            self.wfile.write(response.encode())
        elif self.path == '/balance':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            try:
                balance = get_unified_balance()
                self.wfile.write(json.dumps(balance, default=str).encode())
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e), 'available': 0, 'total': 0}).encode())
        elif self.path == '/position':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
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
                self.wfile.write(json.dumps(result, default=str).encode())
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        elif self.path == '/diag':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            import requests
            results = {}
            # Test api.hbdm.com connectivity
            try:
                t0 = time.time()
                r = requests.get("https://api.hbdm.com/linear-swap-ex/market/detail/merged?contract_code=DOGE-USDT", timeout=10)
                results['api_hbdm_com'] = {'status': r.status_code, 'ms': int((time.time()-t0)*1000), 'ok': r.status_code == 200}
            except Exception as e:
                results['api_hbdm_com'] = {'error': str(e), 'ok': False}
            # Test api.hbdm.vn connectivity
            try:
                t0 = time.time()
                r = requests.get("https://api.hbdm.vn/linear-swap-ex/market/detail/merged?contract_code=DOGE-USDT", timeout=10)
                results['api_hbdm_vn'] = {'status': r.status_code, 'ms': int((time.time()-t0)*1000), 'ok': r.status_code == 200}
            except Exception as e:
                results['api_hbdm_vn'] = {'error': str(e), 'ok': False}
            self.wfile.write(json.dumps(results, default=str).encode())
        elif self.path == '/logs':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'recent': recent_logs}, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress logs

    def do_POST(self):
        if self.path == '/resize':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            if resize_requested.is_set():
                response = json.dumps({'status': 'already_pending', 'message': 'Resize already requested, waiting for next loop iteration'})
            else:
                resize_requested.set()
                response = json.dumps({'status': 'queued', 'message': 'Resize will execute on next loop iteration (within 180s). Bot will close current position and reopen with dynamic size based on current signal.'})
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()

def keep_alive():
    """Self-ping every 50 seconds to prevent Render free plan from sleeping."""
    import requests
    port = int(os.environ.get('PORT', 8080))
    url = f"http://127.0.0.1:{port}/health"
    while True:
        time.sleep(50)  # 50 seconds
        try:
            r = requests.get(url, timeout=10)
            logger.info(f"[KeepAlive] Self-ping OK ({r.status_code})")
        except Exception as e:
            logger.warning(f"[KeepAlive] Self-ping failed: {e}")

def start_http_server():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f"HTTP server listening on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    # Start HTTP server in main thread (required by Render web service)
    # Start trading loop in background thread
    trading_thread = Thread(target=trading_loop, daemon=True)
    trading_thread.start()

    # Start keep-alive thread (prevents Render free plan from sleeping)
    alive_thread = Thread(target=keep_alive, daemon=True)
    alive_thread.start()

    # Run HTTP server in main thread
    start_http_server()
