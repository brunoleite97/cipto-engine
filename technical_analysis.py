from typing import Dict
from binance.client import Client
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

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
        client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "1 week ago UTC")
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Verificar se o DataFrame está vazio
        if df.empty:
            logger.error(f"Dados históricos vazios para o símbolo {symbol}")
            return {}
        
        # Converter colunas para tipos numéricos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calcular indicadores técnicos
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['rsi'] = rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_diff'] = macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Preencher valores ausentes com 0
        df.fillna(0, inplace=True)
        
        # Retornar as últimas métricas calculadas
        return df.iloc[-1][['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma']].to_dict()
    except Exception as e:
        logger.error(f"Erro ao analisar indicadores técnicos para {symbol}: {str(e)}")
        return {}

def rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """Calcular o RSI (Relative Strength Index)"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close_prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> (pd.Series, pd.Series, pd.Series):
    """Calcular o MACD (Moving Average Convergence Divergence)"""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_diff = macd_line - macd_signal_line
    return macd_line, macd_signal_line, macd_diff

def bollinger_bands(close_prices: pd.Series, window: int = 20, num_std: int = 2) -> (pd.Series, pd.Series, pd.Series):
    """Calcular as Bandas de Bollinger"""
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band