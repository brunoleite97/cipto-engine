import sqlite3
import pandas as pd
import logging
import hashlib
from datetime import datetime
from typing import List, Dict
from logging.handlers import TimedRotatingFileHandler

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler('db_manager.log', when='midnight', interval=1, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GerenciadorDeBancoDeDados:
    def __init__(self):
        self.con = sqlite3.connect('trading_bot.db', check_same_thread=False)
        self._criar_tabelas()

    def _criar_tabelas(self):
        """Criar tabelas no banco de dados SQLite"""
        try:
            with self.con:
                self.con.execute('''
                    CREATE TABLE IF NOT EXISTS preços (
                        timestamp DATETIME,
                        symbol TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        PRIMARY KEY (timestamp, symbol)
                    )
                ''')
                self.con.execute('''
                    CREATE TABLE IF NOT EXISTS métricas (
                        timestamp DATETIME,
                        symbol TEXT,
                        sma_20 REAL,
                        sma_50 REAL,
                        ema_20 REAL,
                        rsi REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_diff REAL,
                        bb_upper REAL,
                        bb_middle REAL,
                        bb_lower REAL,
                        volume_sma REAL,
                        PRIMARY KEY (timestamp, symbol)
                    )
                ''')
                self.con.execute('''
                    CREATE TABLE IF NOT EXISTS notícias (
                        hash TEXT PRIMARY KEY,
                        symbol TEXT,
                        title TEXT,
                        description TEXT,
                        url TEXT,
                        timestamp DATETIME,
                        sentiment REAL
                    )
                ''')
                self.con.execute('''
                    CREATE TABLE IF NOT EXISTS recomendações (
                        timestamp DATETIME,
                        symbol TEXT,
                        recomendação TEXT,
                        confiança REAL,
                        nível_de_risco TEXT,
                        razão TEXT,
                        PRIMARY KEY (timestamp, symbol)
                    )
                ''')
                self.con.execute('''
                    CREATE TABLE IF NOT EXISTS negociações (
                        timestamp DATETIME,
                        symbol TEXT,
                        side TEXT,
                        quantity REAL,
                        price REAL,
                        total_value REAL,
                        ai_confidence REAL,
                        technical_score REAL,
                        news_sentiment REAL,
                        status TEXT,
                        profit REAL,
                        PRIMARY KEY (timestamp, symbol)
                    )
                ''')
        except Exception as e:
            logger.error(f"Erro ao criar tabelas: {str(e)}")
            raise

    def armazenar_dados_de_preco(self, df_preços: pd.DataFrame, symbol: str):
        """Armazenar dados de preços no banco de dados"""
        try:
            df_preços.to_sql('preços', self.con, if_exists='append', index=False)
            logger.info(f"Dados de preços armazenados com sucesso para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao armazenar dados de preços: {str(e)}")
            raise

    def armazenar_metricas(self, metricas: Dict, symbol: str):
        """Armazenar indicadores técnicos no banco de dados"""
        try:
            df_metricas = pd.DataFrame([metricas])
            df_metricas.to_sql('métricas', self.con, if_exists='append', index=False)
            logger.info(f"Indicadores técnicos armazenados com sucesso para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao armazenar indicadores técnicos: {str(e)}")
            raise

    def armazenar_notícias(self, notícias: List[Dict], symbol: str):
        """Armazenar notícias no banco de dados"""
        try:
            notícias_com_hash = []
            for noticia in notícias:
                # Garantir que a coluna 'timestamp' esteja presente e convertida corretamente
                if 'publishedAt' in noticia:
                    timestamp = pd.to_datetime(noticia['publishedAt'])
                else:
                    timestamp = datetime.now()
                
                # Gerar hash único para cada notícia
                hash_notícia = hashlib.md5(f"{noticia['title']} {noticia['description']} {noticia['url']} {timestamp}".encode()).hexdigest()
                notícias_com_hash.append({
                    'hash': hash_notícia,
                    'symbol': symbol,
                    'title': noticia['title'],
                    'description': noticia['description'],
                    'url': noticia['url'],
                    'timestamp': timestamp,
                    'sentiment': noticia['sentiment']
                })
            
            df_notícias = pd.DataFrame(notícias_com_hash)
            df_notícias.to_sql('notícias', self.con, if_exists='append', index=False)
            logger.info(f"Notícias armazenadas com sucesso para {symbol}")
        except sqlite3.IntegrityError as e:
            logger.warning(f"Notícia já existente para {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao armazenar notícias: {str(e)}")
            raise

    def armazenar_recomendação(self, recomendação: Dict, symbol: str):
        """Armazenar recomendação no banco de dados"""
        try:
            df_recomendação = pd.DataFrame([recomendação])
            df_recomendação.to_sql('recomendações', self.con, if_exists='append', index=False)
            logger.info(f"Recomendação armazenada com sucesso para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao armazenar recomendação: {str(e)}")
            raise

    def armazenar_negociação(self, dados_de_negociação: Dict):
        """Armazenar detalhes da negociação no banco de dados"""
        try:
            # Calcular lucro
            dados_de_negociação['profit'] = dados_de_negociação['total_value'] - (dados_de_negociação['quantity'] * dados_de_negociação['price'])
            
            df_negociação = pd.DataFrame([dados_de_negociação])
            df_negociação.to_sql('negociações', self.con, if_exists='append', index=False)
            logger.info(f"Negociação armazenada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao armazenar negociação: {str(e)}")
            raise

    def obter_negociações(self) -> pd.DataFrame:
        """Obter dados de negociações do banco de dados"""
        try:
            query = 'SELECT * FROM negociações'
            logger.info(f"Executando consulta SQL: {query}")
            df_negociações = pd.read_sql_query(query, self.con)
            return df_negociações
        except Exception as e:
            logger.error(f"Erro ao obter negociações: {str(e)}")
            raise

    def obter_notícias(self) -> pd.DataFrame:
        """Obter dados de notícias do banco de dados"""
        try:
            query = 'SELECT * FROM notícias'
            logger.info(f"Executando consulta SQL: {query}")
            df_notícias = pd.read_sql_query(query, self.con)
            return df_notícias
        except Exception as e:
            logger.error(f"Erro ao obter notícias: {str(e)}")
            raise

    def obter_métricas(self) -> pd.DataFrame:
        """Obter indicadores técnicos do banco de dados"""
        try:
            query = 'SELECT * FROM métricas'
            logger.info(f"Executando consulta SQL: {query}")
            df_métricas = pd.read_sql_query(query, self.con)
            return df_métricas
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {str(e)}")
            raise

    def obter_preços(self) -> pd.DataFrame:
        """Obter dados de preços do banco de dados"""
        try:
            query = 'SELECT * FROM preços'
            logger.info(f"Executando consulta SQL: {query}")
            df_preços = pd.read_sql_query(query, self.con)
            return df_preços
        except Exception as e:
            logger.error(f"Erro ao obter preços: {str(e)}")
            raise

    def obter_preços_passados(self, symbol: str, intervalo: str = '1d', periodo: str = '1 week ago UTC') -> pd.DataFrame:
        """Obter dados de preços passados do banco de dados"""
        try:
            query = f"SELECT * FROM preços WHERE symbol = ? AND timestamp >= ?"
            logger.info(f"Executando consulta SQL: {query}")
            df_preços = pd.read_sql_query(query, self.con, params=(symbol, periodo))
            return df_preços
        except Exception as e:
            logger.error(f"Erro ao obter preços passados: {str(e)}")
            raise

    def contar_registros(self, tabela: str) -> int:
        """Contar o número de registros em uma tabela específica"""
        try:
            with self.con:
                count = self.con.execute(f'SELECT COUNT(*) FROM {tabela}').fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Erro ao contar registros na tabela {tabela}: {str(e)}")
            raise