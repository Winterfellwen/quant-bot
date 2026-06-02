"""
DOGE QuantBot v5 — 分类器预测 · 动态杠杆 · 永远持仓
1h K线, LGBMClassifier, 概率阈值翻转, 动态杠杆(10x->5x->3x)
"""
import json, time, sys, logging, warnings, os
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger("quantbot_v5")

# ═══ CONFIG ═══
SYMBOL = "DOGE/USDT:USDT"
TIMEFRAME = "1h"
CANDLES_LIMIT = 2000
INITIAL_TRAIN_BARS = 500  # increased from 200 for better initial model
RETRAIN_EVERY = 48
LOOP_INTERVAL = 180
ORDER_SIZE = 10

# ═══ FEATURES ═══
FEATURE_NAMES = [
    'r1','r3','r5','r12','r24','hl_pct','co_pct','pos20','pos50',
    'rsi7','rsi14','macd','bb_pos','bb_width','atr14','volatility',
    'vr5','vr20','adx14','ma20_dev','ma50_dev','ma100_dev',
    'ma20_slope','ma50_slope','ma100_slope','body_ratio','dir3',
    'obv_roc','btc_r1','btc_r5','funding','oi_chg','oi_diverge','fg','fg_ma'
]

def get_api_keys():
    """Read API keys from environment variables (Render-compatible)"""
    key = os.environ.get('HTX_API_KEY', '')
    secret = os.environ.get('HTX_API_SECRET', '')
    if not key or not secret:
        # Fallback: try Windows env vars (local dev)
        import subprocess
        try:
            key = subprocess.run(['powershell','-Command',"[System.Environment]::GetEnvironmentVariable('HTX_API_KEY','User')"],
                               capture_output=True,text=True).stdout.strip()
            secret = subprocess.run(['powershell','-Command',"[System.Environment]::GetEnvironmentVariable('HTX_API_SECRET','User')"],
                                  capture_output=True,text=True).stdout.strip()
        except:
            pass
    return key, secret

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
    feats=np.column_stack(arrs)
    return np.nan_to_num(feats,nan=0.0)

def fetch_all():
    import ccxt, requests
    KEY,SEC = get_api_keys()
    
    # No proxy on Render
    ex_config = {
        'apiKey': KEY, 'secret': SEC,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
        'timeout': 30000
    }
    
    ex = ccxt.huobi(ex_config)

    doge = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=CANDLES_LIMIT)
    df = pd.DataFrame(doge,columns=['ts','open','high','low','close','volume'])
    df['ts']=pd.to_datetime(df['ts'],unit='ms'); df.set_index('ts',inplace=True)

    # BTC
    btc_r1=0; btc_r5=0
    try:
        btc=ex.fetch_ohlcv('BTC/USDT:USDT',TIMEFRAME,limit=6)
        bc=[b[4] for b in btc]
        if len(bc)>1: btc_r1=(bc[-1]-bc[-2])/bc[-2]
        if len(bc)>5: btc_r5=(bc[-1]-bc[-6])/bc[-6]
    except: pass

    # Funding
    funding=0
    try:
        fr=ex.fetch_funding_rate(SYMBOL)
        funding=float(fr.get('fundingRate',0))
    except: pass

    # OI
    oi_chg=0; oi_diverge=0
    try:
        r=ex.contractPublicGetLinearSwapApiV1SwapHisOpenInterest({'contract_code':'DOGE-USDT','period':'60min','amount_type':1,'size':5})
        ticks=r.get('data',{}).get('tick',[])
        if ticks:
            ov=[float(t['value']) for t in ticks[-5:]]
            if len(ov)>4: oi_chg=(ov[-1]-ov[0])/ov[0]
            if len(ov)>1 and len(df)>1:
                last_d=df['close'].iloc[-1]/df['close'].iloc[-2]-1
                oi_d=ov[-1]/ov[-2]-1
                oi_diverge=1 if np.sign(oi_d)!=np.sign(last_d) else 0
    except: pass

    # F&G
    fg=50
    try:
        r=requests.get('https://api.alternative.me/fng/?limit=1',headers={'User-Agent':'Mozilla/5.0'},timeout=10)
        fg=int(r.json()['data'][0]['value'])
    except: pass

    return df,btc_r1,btc_r5,funding,oi_chg,oi_diverge,fg,ex

def get_dynamic_leverage(drawdown_pct):
    """Dynamic leverage: reduce on drawdown for capital preservation"""
    if drawdown_pct > 40:
        return 3
    elif drawdown_pct > 20:
        return 5
    return 10

def get_position(ex):
    try:
        pos=ex.fetch_positions([SYMBOL])
        for p in pos:
            contracts=float(p.get('contracts',0) or 0)
            if contracts>0:
                side=p.get('side','')
                entry=float(p.get('entryPrice',0))
                return side,entry
        return None,0
    except Exception as e:
        logger.warning(f"Position fetch: {e}")
        return None,0

def get_unrealized_pnl(ex, cur_side, entry_price, current_price, leverage):
    """Estimate unrealized PnL as a multiplier (1.0 = breakeven)"""
    if cur_side is None or entry_price == 0:
        return 1.0
    if cur_side == 'long':
        pnl_pct = (current_price - entry_price) / entry_price * leverage
    else:
        pnl_pct = (entry_price - current_price) / entry_price * leverage
    return 1.0 + pnl_pct

def close_position(ex, side, leverage):
    try:
        if side=='long':
            ex.create_order(SYMBOL,'market','sell',ORDER_SIZE,None,{'direction':'sell','offset':'close','lever_rate':leverage})
        else:
            ex.create_order(SYMBOL,'market','buy',ORDER_SIZE,None,{'direction':'buy','offset':'close','lever_rate':leverage})
        logger.info(f"Closed {side}")
    except Exception as e:
        logger.error(f"Close failed: {e}")

def open_position(ex, side, leverage):
    try:
        ex.set_leverage(leverage, SYMBOL)
        if side=='long':
            ex.create_order(SYMBOL,'market','buy',ORDER_SIZE,None,{'direction':'buy','offset':'open','lever_rate':leverage})
        else:
            ex.create_order(SYMBOL,'market','sell',ORDER_SIZE,None,{'direction':'sell','offset':'open','lever_rate':leverage})
        logger.info(f"Opened {side} @ {leverage}x")
    except Exception as e:
        logger.error(f"Open failed: {e}")

def main():
    import lightgbm as lgb
    logger.info("=== QuantBot v5 (Classifier) Starting ===")

    df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg, ex = fetch_all()
    X = compute_features(df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg)
    
    # Binary target: 1 if next bar goes up
    c = df['close'].values
    y_binary = np.zeros(len(c))
    y_binary[:-1] = (c[1:] > c[:-1]).astype(int)
    y_binary[-1] = np.nan
    
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_binary)
    Xt = X[valid]
    yt = y_binary[valid]
    
    if len(Xt) < 300:
        logger.error(f"Insufficient data: {len(Xt)}")
        return
    
    # Train classifier
    train_size = min(INITIAL_TRAIN_BARS, len(Xt) - 2)
    train_valid = ~np.isnan(yt[:train_size])
    
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.03,
        num_leaves=20, reg_lambda=3, min_child_samples=30,
        random_state=42, verbose=-1, force_row_wise=True,
        class_weight='balanced'
    )
    model.fit(Xt[:train_size][train_valid], yt[:train_size][train_valid])
    logger.info(f"Initial training: {train_valid.sum()} samples, {len(df)} candles, {df.index[0]}~{df.index[-1]}")
    
    # Check current position
    cur_side, entry_price = get_position(ex)
    logger.info(f"Current position: {cur_side or 'none'}, entry: {entry_price}")
    
    # State tracking
    first_bar = True
    last_retrain_size = len(df)
    cumulative_return = 1.0
    peak_return = 1.0
    
    while True:
        try:
            df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg, ex = fetch_all()
            X = compute_features(df, btc_r1, btc_r5, funding, oi_chg, oi_diverge, fg)
            
            # Retrain every RETRAIN_EVERY new bars
            if len(df) - last_retrain_size >= RETRAIN_EVERY:
                c = df['close'].values
                y_binary = np.zeros(len(c))
                y_binary[:-1] = (c[1:] > c[:-1]).astype(int)
                y_binary[-1] = np.nan
                valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_binary)
                Xt_rt = X[valid]
                yt_rt = y_binary[valid]
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
            
            # Predict
            prob_up = float(model.predict_proba(X[-1:])[0][1])
            
            cur_side, entry_price = get_position(ex)
            price = float(df['close'].iloc[-1])
            
            # Estimate drawdown for dynamic leverage
            unrealized = get_unrealized_pnl(ex, cur_side, entry_price, price, 10)
            estimated_equity = cumulative_return * unrealized
            dd_pct = (1 - estimated_equity / max(peak_return, cumulative_return)) * 100
            leverage = get_dynamic_leverage(max(dd_pct, 0))
            
            # First run: enter position
            if first_bar and cur_side is None:
                init_side = "long" if prob_up > 0.5 else "short"
                open_position(ex, init_side, leverage)
                logger.info(f"Initial entry: {init_side} prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} dd={dd_pct:.1f}%")
                first_bar = False
                time.sleep(LOOP_INTERVAL)
                continue
            first_bar = False
            
            # Flip logic: prob > 0.5 = long, prob < 0.5 = short
            should_long = prob_up > 0.5
            should_short = prob_up < 0.5
            
            if should_long and cur_side == "short":
                # Compute realized PnL from closing short
                realized = get_unrealized_pnl(ex, "short", entry_price, price, 10)
                cumulative_return *= realized
                peak_return = max(peak_return, cumulative_return)
                
                close_position(ex, "short", leverage)
                time.sleep(1)
                open_position(ex, "long", leverage)
                logger.info(f"Flip LONG: prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} cum_ret={cumulative_return:.3f}x dd={dd_pct:.1f}%")
            
            elif should_short and cur_side == "long":
                realized = get_unrealized_pnl(ex, "long", entry_price, price, 10)
                cumulative_return *= realized
                peak_return = max(peak_return, cumulative_return)
                
                close_position(ex, "long", leverage)
                time.sleep(1)
                open_position(ex, "short", leverage)
                logger.info(f"Flip SHORT: prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} cum_ret={cumulative_return:.3f}x dd={dd_pct:.1f}%")
            
            else:
                logger.info(f"Holding {cur_side}: prob={prob_up:.4f} F&G={fg} lev={leverage}x price={price:.4f} cum_ret={cumulative_return:.3f}x dd={dd_pct:.1f}%")
        
        except Exception as e:
            logger.error(f"Loop error: {e}")
            import traceback; traceback.print_exc()
        
        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    main()
