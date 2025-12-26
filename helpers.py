# helpers.py
# nrsy mmin ofqs rudr -- app password

import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Try import nsepython (may require installation)
try:
    from nsepython import nse_fetch
    NSEPY_AVAILABLE = True
except Exception:
    NSEPY_AVAILABLE = False


def fetch_us(ticker, period_str):
    """Fetch history for US tickers using yfinance"""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period_str, auto_adjust=False)
    df = df.reset_index()
    df.rename(columns={"Date": "Datetime"}, inplace=True)
    return df


def fetch_india(ticker, period_str):
    """Fetch Indian data — try yfinance first (many NSE tickers are available as <TICKER>.NS)
    fallback: nsepython if installed."""
    # try yfinance with .NS suffix
    try:
        sym = ticker if ticker.endswith('.NS') else f"{ticker}.NS"
        df = fetch_us(sym, period_str)
        if not df.empty:
            return df
    except Exception:
        pass

    if NSEPY_AVAILABLE:
        try:
            # nsepython has many helper functions; this is a simple attempt to fetch OHLC
            out = nse_fetch(ticker)
            # If nse_fetch returns historical data, convert to df — user may need to adapt
            return pd.DataFrame(out)
        except Exception:
            pass

    raise RuntimeError("Failed to fetch India data. Install nsepython or use Yahoo suffix .NS")


def fetch_history(market, asset_type, ticker, period_str):
    if market == 'US':
        return fetch_us(ticker, period_str)
    else:
        return fetch_india(ticker, period_str)


# Simple utility for formatting period strings for yfinance
# def period_from_inputs(years=0, months=0, days=0):
#     # yfinance accepts periods like '5y', '6mo', '30d'
#     if years > 0 and months == 0 and days == 0:
#         return f"{years}y"
#     if months > 0 and years == 0 and days == 0:
#         return f"{months}mo"
#     if days > 0 and years == 0 and months == 0:
#         return f"{days}d"
#     # fallback: compute days
#     total_days = years * 365 + months * 30 + days
#     return f"{total_days}d"


# all converted to one unit (trading days so that same period has same precision (1year = 365days))
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21

def period_from_inputs(years=0, months=0, days=0):
    total_days = (
        years * TRADING_DAYS_PER_YEAR +
        months * TRADING_DAYS_PER_MONTH +
        days
    )
    return f"{max(total_days, 1)}d"


def send_email(smtp_server, smtp_port, username, password, to_email, subject, body):
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = username
    msg['To'] = to_email

    s = smtplib.SMTP(smtp_server, smtp_port)
    s.starttls()
    s.login(username, password)
    s.sendmail(username, [to_email], msg.as_string())
    s.quit()


def beep():
    # cross-platform beep: Windows winsound, else try print('\a')
    try:
        import winsound
        winsound.Beep(1000, 700)
    except Exception:
        print('\a')


def notify_desktop(title, message=""):
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=6)
    except Exception as e:
        print('Desktop notification failed:', e)

