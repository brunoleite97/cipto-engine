import pandas as pd
import logging
import hashlib
from datetime import datetime
from typing import List, Dict
from logging.handlers import TimedRotatingFileHandler
from pymongo import MongoClient
import json

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
        # Conectar ao MongoDB
        self.client = MongoClient("mongodb://bruno:18fd65f33baff46f91da@easypanell.meupcbleite97.shop:27017/?tls=false")
        self.db = self.client["trading_bot"]
        # Criar coleções (equivalente às tabelas no SQLite)
        self.precos = self.db["precos"]
        self.metricas = self.db["metricas"]
        self.noticias = self.db["noticias"]
        self.recomendacoes = self.db["recomendacoes"]
        self.negociacoes = self.db["negociacoes"]

    def _criar_tabelas(self):
        """Criar índices nas coleções do MongoDB"""
        try:
            # Criar índices compostos para as coleções
            self.precos.create_index([('timestamp', 1), ('symbol', 1)], unique=True)
            self.metricas.create_index([('timestamp', 1), ('symbol', 1)], unique=True)
            self.noticias.create_index('hash', unique=True)
            self.recomendacoes.create_index([('timestamp', 1), ('symbol', 1)], unique=True)
            self.negociacoes.create_index([('timestamp', 1), ('symbol', 1)], unique=True)
            logger.info("Índices criados com sucesso no MongoDB")
        except Exception as e:
            logger.error(f"Erro ao criar índices no MongoDB: {str(e)}")
            raise

    def armazenar_dados_de_preco(self, df_preços: pd.DataFrame, symbol: str):
        """Armazenar dados de preços no MongoDB"""
        try:
            # Converter DataFrame para lista de dicionários
            registros = json.loads(df_preços.to_json(orient='records'))
            # Inserir registros no MongoDB com upsert para evitar duplicatas
            for registro in registros:
                # Garantir que timestamp seja um objeto datetime
                if 'timestamp' in registro and isinstance(registro['timestamp'], str):
                    registro['timestamp'] = datetime.fromisoformat(registro['timestamp'].replace('Z', '+00:00'))
                # Definir critério de busca para upsert
                filtro = {'timestamp': registro['timestamp'], 'symbol': symbol}
                # Inserir ou atualizar documento
                self.precos.update_one(filtro, {'$set': registro}, upsert=True)
            logger.info(f"Dados de preços armazenados com sucesso para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao armazenar dados de preços: {str(e)}")
            raise

    def armazenar_metricas(self, metricas: Dict, symbol: str):
        """Armazenar indicadores técnicos no MongoDB"""
        try:
            # Adicionar o símbolo ao dicionário de métricas
            metricas['symbol'] = symbol
            # Garantir que timestamp seja um objeto datetime
            if 'timestamp' in metricas and isinstance(metricas['timestamp'], str):
                metricas['timestamp'] = datetime.fromisoformat(metricas['timestamp'].replace('Z', '+00:00'))
            # Definir critério de busca para upsert
            filtro = {'timestamp': metricas['timestamp'], 'symbol': symbol}
            # Inserir ou atualizar documento
            self.metricas.update_one(filtro, {'$set': metricas}, upsert=True)
            logger.info(f"Indicadores técnicos armazenados com sucesso para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao armazenar indicadores técnicos: {str(e)}")
            raise

    def armazenar_notícias(self, notícias: List[Dict], symbol: str):
        """Armazenar notícias no MongoDB"""
        try:
            for noticia in notícias:
                # Garantir que a coluna 'timestamp' esteja presente e convertida corretamente
                if 'publishedAt' in noticia:
                    timestamp = pd.to_datetime(noticia['publishedAt'])
                else:
                    timestamp = datetime.now()
                
                # Gerar hash único para cada notícia
                hash_notícia = hashlib.md5(f"{noticia['title']} {noticia['description']} {noticia['url']} {timestamp}".encode()).hexdigest()
                
                # Preparar documento para MongoDB
                documento = {
                    'hash': hash_notícia,
                    'symbol': symbol,
                    'title': noticia['title'],
                    'description': noticia['description'],
                    'url': noticia['url'],
                    'timestamp': timestamp,
                    'sentiment': noticia['sentiment']
                }
                
                # Inserir ou atualizar documento
                self.noticias.update_one({'hash': hash_notícia}, {'$set': documento}, upsert=True)
            
            logger.info(f"Notícias armazenadas com sucesso para {symbol}")
        except Exception as e:
            if "duplicate key" in str(e):
                logger.warning(f"Notícia já existente para {symbol}: {str(e)}")
            else:
                logger.error(f"Erro ao armazenar notícias: {str(e)}")
                raise

    def armazenar_recomendação(self, recomendação: Dict, symbol: str):
        """Armazenar recomendação no MongoDB"""
        try:
            # Adicionar o símbolo ao dicionário de recomendação
            recomendação['symbol'] = symbol
            # Garantir que timestamp seja um objeto datetime
            if 'timestamp' in recomendação and isinstance(recomendação['timestamp'], str):
                recomendação['timestamp'] = datetime.fromisoformat(recomendação['timestamp'].replace('Z', '+00:00'))
            # Definir critério de busca para upsert
            filtro = {'timestamp': recomendação['timestamp'], 'symbol': symbol}
            # Inserir ou atualizar documento
            self.recomendacoes.update_one(filtro, {'$set': recomendação}, upsert=True)
            logger.info(f"Recomendação armazenada com sucesso para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao armazenar recomendação: {str(e)}")
            raise

    def armazenar_negociação(self, dados_de_negociação: Dict):
        """Armazenar detalhes da negociação no MongoDB"""
        try:
            # Calcular lucro
            dados_de_negociação['profit'] = dados_de_negociação['total_value'] - (dados_de_negociação['quantity'] * dados_de_negociação['price'])
            
            # Garantir que timestamp seja um objeto datetime
            if 'timestamp' in dados_de_negociação and isinstance(dados_de_negociação['timestamp'], str):
                dados_de_negociação['timestamp'] = datetime.fromisoformat(dados_de_negociação['timestamp'].replace('Z', '+00:00'))
            
            # Definir critério de busca para upsert
            filtro = {'timestamp': dados_de_negociação['timestamp'], 'symbol': dados_de_negociação['symbol']}
            # Inserir ou atualizar documento
            self.negociacoes.update_one(filtro, {'$set': dados_de_negociação}, upsert=True)
            logger.info(f"Negociação armazenada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao armazenar negociação: {str(e)}")
            raise

    def obter_negociações(self) -> pd.DataFrame:
        """Obter dados de negociações do MongoDB"""
        try:
            logger.info("Obtendo todas as negociações do MongoDB")
            # Buscar todos os documentos da coleção de negociações
            documentos = list(self.negociacoes.find({}))
            # Converter para DataFrame
            if documentos:
                df_negociações = pd.DataFrame(documentos)
                # Remover o campo _id gerado pelo MongoDB
                if '_id' in df_negociações.columns:
                    df_negociações = df_negociações.drop('_id', axis=1)
                return df_negociações
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter negociações: {str(e)}")
            raise

    def obter_notícias(self) -> pd.DataFrame:
        """Obter dados de notícias do MongoDB"""
        try:
            logger.info("Obtendo todas as notícias do MongoDB")
            # Buscar todos os documentos da coleção de notícias
            documentos = list(self.noticias.find({}))
            # Converter para DataFrame
            if documentos:
                df_notícias = pd.DataFrame(documentos)
                # Remover o campo _id gerado pelo MongoDB
                if '_id' in df_notícias.columns:
                    df_notícias = df_notícias.drop('_id', axis=1)
                return df_notícias
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter notícias: {str(e)}")
            raise

    def obter_métricas(self) -> pd.DataFrame:
        """Obter indicadores técnicos do MongoDB"""
        try:
            logger.info("Obtendo todas as métricas do MongoDB")
            # Buscar todos os documentos da coleção de métricas
            documentos = list(self.metricas.find({}))
            # Converter para DataFrame
            if documentos:
                df_métricas = pd.DataFrame(documentos)
                # Remover o campo _id gerado pelo MongoDB
                if '_id' in df_métricas.columns:
                    df_métricas = df_métricas.drop('_id', axis=1)
                return df_métricas
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {str(e)}")
            raise

    def obter_preços(self) -> pd.DataFrame:
        """Obter dados de preços do MongoDB"""
        try:
            logger.info("Obtendo todos os preços do MongoDB")
            # Buscar todos os documentos da coleção de preços
            documentos = list(self.precos.find({}))
            # Converter para DataFrame
            if documentos:
                df_preços = pd.DataFrame(documentos)
                # Remover o campo _id gerado pelo MongoDB
                if '_id' in df_preços.columns:
                    df_preços = df_preços.drop('_id', axis=1)
                return df_preços
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter preços: {str(e)}")
            raise

    def obter_preços_passados(self, symbol: str, intervalo: str = '1d', periodo: str = '1 week ago UTC') -> pd.DataFrame:
        """Obter dados de preços passados do MongoDB"""
        try:
            # Converter periodo para datetime
            data_limite = pd.to_datetime(periodo)
            logger.info(f"Obtendo preços passados para {symbol} desde {data_limite}")
            
            # Buscar documentos que correspondem aos critérios
            filtro = {
                'symbol': symbol,
                'timestamp': {'$gte': data_limite}
            }
            documentos = list(self.precos.find(filtro).sort('timestamp', 1))
            
            # Converter para DataFrame
            if documentos:
                df_preços = pd.DataFrame(documentos)
                # Remover o campo _id gerado pelo MongoDB
                if '_id' in df_preços.columns:
                    df_preços = df_preços.drop('_id', axis=1)
                return df_preços
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter preços passados: {str(e)}")
            raise

    def contar_registros(self, colecao: str) -> int:
        """Contar o número de registros em uma coleção específica"""
        try:
            # Mapear nome da coleção para o atributo correspondente
            mapeamento_colecoes = {
                'preços': self.precos,
                'precos': self.precos,
                'métricas': self.metricas,
                'metricas': self.metricas,
                'notícias': self.noticias,
                'noticias': self.noticias,
                'recomendações': self.recomendacoes,
                'recomendacoes': self.recomendacoes,
                'negociações': self.negociacoes,
                'negociacoes': self.negociacoes
            }
            
            if colecao in mapeamento_colecoes:
                count = mapeamento_colecoes[colecao].count_documents({})
                return count
            else:
                logger.error(f"Coleção {colecao} não encontrada")
                return 0
        except Exception as e:
            logger.error(f"Erro ao contar registros na coleção {colecao}: {str(e)}")
            raise