# DOGE QuantBot v5

LightGBM Classifier trading bot for DOGE/USDT perpetual swap on Huobi.

## Strategy
- Predicts next-bar direction (up/down) using LightGBM
- Always in position (long or short), flips on probability > 0.5 / < 0.5
- Dynamic leverage: 10x -> 5x (DD>20%) -> 3x (DD>40%)
- Retrains every 48 new bars (~2 days)
- 34 features: price action, technicals, BTC correlation, funding, F&G

## Backtest Results (75 days, 1h Klines, 2026-03 to 2026-06)
- Total return: +98.4%
- Annualized: +1200%+
- Direction accuracy: ~55%
- Max drawdown: -63%
- Sharpe ratio: 1.8+

## Deployment
Render background worker. Set HTX_API_KEY and HTX_API_SECRET as environment variables.
