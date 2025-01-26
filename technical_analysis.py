from typing import Dict
from binance.client import Client
import pandas as pd
import ta
import logging
from logging.handlers import TimedRotatingFileHandler

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler('technical_analysis.log', when='midnight', interval=1, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analisar_indicadores_técnicos(symbol: str) -> Dict:
    """Analisar indicadores técnicos para um símbolo específico"""
    try:
        # Obter dados históricos da Binance
        client = Client()
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, "1 day ago UTC")
        
        # Converter dados para DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Converter colunas para tipos numéricos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calcular indicadores técnicos
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        
        # Extrair os últimos valores dos indicadores
        últimos_valores = df.iloc[-1].to_dict()
        últimos_valores['symbol'] = symbol
        
        return últimos_valores
    except Exception as e:
        logger.error(f"Erro ao analisar indicadores técnicos para {symbol}: {str(e)}")
        return {}