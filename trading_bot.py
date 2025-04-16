from typing import Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from model_training import NeuralNetwork
import torch
import torch.nn as nn
from db_manager import GerenciadorDeBancoDeDados
from binance.client import Client
from technical_analysis import analisar_indicadores_técnicos
from news_analyzer import AnalisadorDeNotícias
from datetime import datetime, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler('trading_bot.log', when='midnight', interval=1, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lista de símbolos de criptomoedas para análise
symbols = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'DOGEUSDT', 'ANIMEUSDT', 
    'TRUMPUSDT', 'MEMEUSDT', 'SHIBUSDT', 'SOLUSDT', 'EURIUSDT', 'VANAUSDT', 
    'STPTUSDT', 'SXPUSDT', 'COSUSDT', 'CATIUSDT', 'SLFUSDT', 'ACHUSDT',
    'DOTUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 
    'ATOMUSDT', 'AAVEUSDT', 'ALGOUSDT', 'VETUSDT', 'XTZUSDT',
    'UNIUSDT', 'FILUSDT', 'TRXUSDT', 'HBARUSDT', 'FTMUSDT', 'SANDUSDT', 
    'MANAUSDT', 'ICPUSDT', 'ZECUSDT', 'DASHUSDT', 'EOSUSDT', 'MKRUSDT', 
    'NEOUSDT', 'BCHUSDT'
]

class ModeloDeAprendizadoDeMáquina:
    def __init__(self, db: GerenciadorDeBancoDeDados):
        self.db = db
        # Não armazenar a chave da API como atributo da classe
        # Verificar apenas se a variável de ambiente existe
        if not os.getenv('GROQ_API_KEY'):
            logger.error("GROQ_API_KEY não encontrada nas variáveis de ambiente.")
            raise ValueError("GROQ_API_KEY não configurada. Verifique o arquivo .env")
        logger.info("Inicializado ModeloDeAprendizadoDeMáquina com API Groq.")

    def prever_resultado(self, technical_score: float, news_sentiment: float, ai_confidence: float, sma_20: float, sma_50: float, ema_20: float, rsi: float, macd: float, macd_signal: float, macd_diff: float, bb_upper: float, bb_middle: float, bb_lower: float, volume_sma: float, sentiment_notícias: float) -> int:
        """Prever resultado de uma recomendação de negociação usando a API do Groq com fallback para análise baseada em regras"""
        # Verificar se a chave da API Groq está disponível e válida
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            logger.warning("GROQ_API_KEY não encontrada. Usando método de fallback para previsão.")
            return self._prever_com_regras(technical_score, rsi, macd, macd_signal, macd_diff)
            
        try:
            import requests
            import json
            import time
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Preparar os dados para enviar à API do Groq
            dados_mercado = {
                'technical_score': technical_score,
                'news_sentiment': news_sentiment,
                'ai_confidence': ai_confidence,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_20': ema_20,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_diff': macd_diff,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'volume_sma': volume_sma,
                'sentiment_notícias': sentiment_notícias
            }
            
            # Construir o prompt para a API do Groq
            prompt = f"""Analise os seguintes dados de mercado de criptomoedas e determine se é recomendável comprar (1) ou não comprar (0):
            
            Dados técnicos:
            - Pontuação técnica: {technical_score}
            - Sentimento de notícias: {news_sentiment}
            - Confiança da IA: {ai_confidence}
            - SMA 20: {sma_20}
            - SMA 50: {sma_50}
            - EMA 20: {ema_20}
            - RSI: {rsi}
            - MACD: {macd}
            - Sinal MACD: {macd_signal}
            - Diferença MACD: {macd_diff}
            - Banda de Bollinger Superior: {bb_upper}
            - Banda de Bollinger Média: {bb_middle}
            - Banda de Bollinger Inferior: {bb_lower}
            - Volume SMA: {volume_sma}
            - Sentimento de notícias: {sentiment_notícias}
            
            Baseado nesses dados, responda apenas com o número 1 (comprar) ou 0 (não comprar).
            """
            
            # Configurar a requisição para a API do Groq com retry e backoff exponencial
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            # Configurar sessão com retry
            session = requests.Session()
            retry_strategy = Retry(
                total=3,  # número máximo de tentativas
                backoff_factor=1,  # fator de backoff exponencial
                status_forcelist=[429, 500, 502, 503, 504],  # códigos de status para retry
                allowed_methods=["POST"]  # métodos permitidos para retry
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            
            # Obter e verificar a chave da API diretamente da variável de ambiente
            api_key = os.getenv('GROQ_API_KEY', '').strip()
            if not api_key.startswith("gsk_"):
                logger.warning("Formato da chave GROQ_API_KEY parece inválido. Verifique se a chave começa com 'gsk_'.")
            
            # Configurar headers com a chave limpa
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Configurar payload
            payload = {
                "model": "llama-3.3-70b-versatile",  # Modelo LLM do Groq mais avançado
                "messages": [
                    {"role": "system", "content": "Você é um assistente especializado em análise de mercado de criptomoedas. Responda apenas com 0 ou 1."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Baixa temperatura para respostas mais determinísticas
                "max_tokens": 10  # Resposta curta
            }
            
            # Log detalhado para diagnóstico (sem mostrar partes da chave)
            logger.info("Enviando requisição para API Groq")
            
            # Fazer a requisição à API do Groq com a sessão configurada
            response = session.post(url, headers=headers, json=payload, timeout=10)
            
            # Log detalhado da resposta
            logger.info(f"Resposta da API Groq - Status: {response.status_code}, Headers: {response.headers}")
            
            # Verificar se houve erro na requisição
            response.raise_for_status()
            
            # Processar a resposta
            result = response.json()
            resposta_texto = result['choices'][0]['message']['content'].strip()
            logger.info(f"Resposta da API Groq: {resposta_texto}")
            
            # Extrair o valor numérico (0 ou 1) da resposta
            if '1' in resposta_texto:
                return 1
            else:
                return 0
                
        except requests.exceptions.HTTPError as http_err:
            status_code = getattr(http_err.response, 'status_code', None)
            error_detail = ""
            try:
                error_json = http_err.response.json()
                error_detail = f", Detalhes: {error_json}"
            except:
                error_detail = f", Resposta: {http_err.response.text if hasattr(http_err.response, 'text') else 'Não disponível'}"
                
            logger.error(f"Erro HTTP {status_code} ao acessar API Groq{error_detail}")
            
            if status_code == 401 or status_code == 403:
                logger.error("Erro de autenticação com a API Groq. Verifique se a chave API está correta e ativa.")
            elif status_code == 429:
                logger.error("Limite de taxa excedido na API Groq. Aguarde antes de fazer novas solicitações.")
            
            logger.info("Usando método de fallback para previsão devido a erro na API Groq.")
            return self._prever_com_regras(technical_score, rsi, macd, macd_signal, macd_diff)
            
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Erro de conexão com a API Groq: {str(conn_err)}")
            logger.info("Usando método de fallback para previsão devido a erro na API Groq.")
            return self._prever_com_regras(technical_score, rsi, macd, macd_signal, macd_diff)
            
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout ao conectar com a API Groq: {str(timeout_err)}")
            logger.info("Usando método de fallback para previsão devido a erro na API Groq.")
            return self._prever_com_regras(technical_score, rsi, macd, macd_signal, macd_diff)
            
        except Exception as e:
            logger.error(f"Erro ao prever resultado com API Groq: {str(e)}")
            logger.info("Usando método de fallback para previsão devido a erro na API Groq.")
            return self._prever_com_regras(technical_score, rsi, macd, macd_signal, macd_diff)
    
    def _prever_com_regras(self, technical_score: float, rsi: float, macd: float, macd_signal: float, macd_diff: float) -> int:
        """Método de fallback para previsão baseada em regras quando a API Groq não está disponível"""
        # Implementação de regras básicas para análise técnica
        sinais_positivos = 0
        
        # Verificar pontuação técnica
        if technical_score > 60:
            sinais_positivos += 1
        
        # Verificar RSI (Índice de Força Relativa)
        # RSI < 30 indica sobrevendido (potencial de compra)
        # RSI > 70 indica sobrecomprado (potencial de venda)
        if rsi < 30:
            sinais_positivos += 1
        elif rsi > 70:
            sinais_positivos -= 1
        
        # Verificar MACD (Moving Average Convergence Divergence)
        # MACD acima do sinal indica tendência de alta
        if macd > macd_signal:
            sinais_positivos += 1
        else:
            sinais_positivos -= 1
            
        # Verificar diferença do MACD
        # Diferença positiva e crescente indica força na tendência de alta
        if macd_diff > 0:
            sinais_positivos += 1
        
        logger.info(f"Análise de fallback: {sinais_positivos} sinais positivos encontrados")
        
        # Decisão final baseada na contagem de sinais positivos
        return 1 if sinais_positivos > 1 else 0

class TradingBot:
    def __init__(self, symbols: List[str]):
        self.db = GerenciadorDeBancoDeDados()
        self.analisador_de_notícias = AnalisadorDeNotícias(
            os.getenv('NEWS_API_KEY'),
            os.getenv('APITUBE_API_KEY')
        )
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
        self.sincronizar_tempo()
        self.trading_threads = {}
        self.última_análise = {symbol: datetime.now() - timedelta(seconds=25) for symbol in symbols}
        self.model = ModeloDeAprendizadoDeMáquina(self.db)
        self.lock = threading.Lock()  # Adicionar um bloqueio para sincronização

    def sincronizar_tempo(self):
        """Sincronizar o relógio do sistema com o servidor da Binance"""
        try:
            server_time = self.client.get_server_time()
            server_timestamp = server_time['serverTime']
            local_timestamp = int(time.time() * 1000)
            time_difference = server_timestamp - local_timestamp
            
            if abs(time_difference) > 1000:
                logger.warning(f"Diferença de tempo significativa: {time_difference}ms. Ajustando timestamp...")
                self.client.timestamp_offset = time_difference
        except Exception as e:
            logger.error(f"Erro ao sincronizar tempo: {str(e)}")
            raise

    def analisar_mercado(self, symbol: str) -> Dict:
        """Análise abrangente do mercado"""
        try:
            # Verificar se já passou 25 segundos desde a última análise
            if (datetime.now() - self.última_análise[symbol]).total_seconds() < 25:
                logger.warning(f"Análise para {symbol} foi feita recentemente. Aguardando 25 segundos...")
                return {}
            
            # Adquirir o bloqueio para garantir que nenhuma análise de mercado seja feita durante o treinamento
            with self.lock:
                # Execução paralela de tarefas de análise
                with ThreadPoolExecutor() as executor:
                    análise_técnica_futuro = executor.submit(analisar_indicadores_técnicos, symbol)
                    notícias_futuro = executor.submit(self.analisador_de_notícias.obter_notícias_crypto, symbol)
                    dados_de_mercado_futuro = executor.submit(self._obter_dados_de_mercado, symbol)
                    dados_de_mercado_passados_futuro = executor.submit(self._obter_dados_de_mercado_passados, symbol)

                dados_técnicos = análise_técnica_futuro.result()
                notícias = notícias_futuro.result()
                dados_de_mercado = dados_de_mercado_futuro.result()
                dados_de_mercado_passados = dados_de_mercado_passados_futuro.result()

                # Verificar se os dados técnicos estão vazios
                if not dados_técnicos:
                    logger.error(f"Dados técnicos vazios para o símbolo {symbol}")
                    return {}

                # Verificar se as notícias estão vazias
                if not notícias:
                    logger.warning(f"Nenhuma notícia encontrada para o símbolo {symbol}")
                    notícias = []

                # Calcular confiança da negociação
                confiança = self._calcular_confiança_da_negociação(dados_técnicos, notícias, dados_de_mercado, dados_de_mercado_passados)
                
                # Combinar todas as análises
                análise = {
                    'técnica': dados_técnicos,
                    'notícias': notícias,
                    'mercado': dados_de_mercado,
                    'mercado_passado': dados_de_mercado_passados,
                    'confiança': confiança,
                    'timestamp': datetime.now()
                }

                # Armazenar resultados da análise
                self._armazenar_análise(análise, symbol)

                # Atualizar a última análise
                self.última_análise[symbol] = datetime.now()

                return análise
        except Exception as e:
            logger.error(f"Erro na análise do mercado: {str(e)}")
            return {}

    def dividir_valor_em_carteira(self, symbols: List[str], percentual_usdt: float = 0.1) -> Dict:
        """Dividir o valor em carteira entre as moedas especificadas e manter um percentual em USDT"""
        try:
            conta = self.client.get_account()
            saldo_usdt = float(next(filter(lambda x: x['asset'] == 'USDT', conta['balances']))['free'])
            
            # Calcular o valor a ser investido em criptomoedas
            valor_investimento = saldo_usdt * (1 - percentual_usdt)
            
            # Dividir o valor investido igualmente entre as moedas
            valor_por_moeda = valor_investimento / len(symbols)
            
            return {symbol: valor_por_moeda for symbol in symbols}
        except Exception as e:
            logger.error(f"Erro ao dividir valor em carteira: {str(e)}")
            return {}

    def executar_negociação(self, symbol: str, quantidade: float, tipo_ordem: str) -> Optional[Dict]:
        """Executar uma ordem de compra ou venda"""
        try:
            if tipo_ordem == 'comprar':
                ordem = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantidade
                )
            elif tipo_ordem == 'vender':
                ordem = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantidade
                )
            else:
                logger.error(f"Tipo de ordem inválido: {tipo_ordem}")
                return None

            return ordem
        except Exception as e:
            logger.error(f"Erro ao executar ordem para {symbol}: {str(e)}")
            return None

    def _obter_dados_de_mercado(self, symbol: str) -> Dict:
        """Obter dados de mercado atuais"""
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return {
                'current_price': float(ticker['lastPrice']),
                'volume': float(ticker['volume']),
                'price_change': float(ticker['priceChangePercent'])
            }
        except Exception as e:
            logger.error(f"Erro ao buscar dados de mercado: {str(e)}")
            return {}

    def _obter_dados_de_mercado_passados(self, symbol: str) -> List[Dict]:
        """Obter dados de mercado passados"""
        try:
            # Exemplo de obtenção de dados de mercado passados
            # Aqui você pode adicionar a lógica para obter dados históricos
            klines = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "1 week ago UTC")
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Erro ao buscar dados de mercado passados: {str(e)}")
            return {}

    def _calcular_confiança_da_negociação(self, dados_técnicos: Dict, notícias: List[Dict], dados_de_mercado: Dict, dados_de_mercado_passados: List[Dict]) -> Dict:
        """Calcular confiança da negociação com base na análise técnica e de notícias"""
        try:
            # Calcular pontuação técnica
            pontuação_técnica = (dados_técnicos['rsi'] + dados_técnicos['macd_diff']) / 2
            
            # Calcular pontuação de sentimento das notícias
            if notícias:
                pontuação_de_sentimento = np.mean([artigo['sentiment'] for artigo in notícias])
            else:
                pontuação_de_sentimento = 0.0
            
            # Combinar pontuações
            pontuação_combinada = (pontuação_técnica + pontuação_de_sentimento) / 2
            
            # Determinar recomendação e nível de risco
            if pontuação_combinada > 0.7:
                recomendação_final = 'comprar'
                nível_de_risco = 'médio'
            elif pontuação_combinada < 0.3:
                recomendação_final = 'vender'
                nível_de_risco = 'alto'
            else:
                recomendação_final = 'segurar'
                nível_de_risco = 'baixo'
            
            confiança = pontuação_combinada
            
            # Refinar recomendação com modelo de aprendizado de máquina
            features = {
                'technical_score': pontuação_combinada,
                'news_sentiment': pontuação_de_sentimento,
                'ai_confidence': confiança,
                'sma_20': dados_técnicos['sma_20'],
                'sma_50': dados_técnicos['sma_50'],
                'ema_20': dados_técnicos['ema_20'],
                'rsi': dados_técnicos['rsi'],
                'macd': dados_técnicos['macd'],
                'macd_signal': dados_técnicos['macd_signal'],
                'macd_diff': dados_técnicos['macd_diff'],
                'bb_upper': dados_técnicos['bb_upper'],
                'bb_middle': dados_técnicos['bb_middle'],
                'bb_lower': dados_técnicos['bb_lower'],
                'volume_sma': dados_técnicos['volume_sma'],
                'sentiment_notícias': pontuação_de_sentimento
            }
            
            resultado_previsto = self.model.prever_resultado(**features)
            if resultado_previsto == 0 and recomendação_final == 'comprar':
                recomendação_final = 'segurar'
            elif resultado_previsto == 1 and recomendação_final == 'vender':
                recomendação_final = 'segurar'
            
            return {
                'score': confiança,
                'nível_de_risco': nível_de_risco,
                'recomendação': recomendação_final
            }
        except Exception as e:
            logger.error(f"Erro ao calcular confiança da negociação: {str(e)}")
            return {'score': 0.0, 'nível_de_risco': 'médio', 'recomendação': 'segurar'}

    def _calcular_tamanho_da_posição(self, symbol: str, nível_de_risco: str) -> Dict:
        """Calcular tamanho da posição com base em regras de gerenciamento de risco"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            preço = float(ticker['price'])
            
            # Definir porcentagem de risco com base no nível de risco
            porcentagem_de_risco = {
                'baixo': 0.01,
                'médio': 0.02,
                'alto': 0.03
            }.get(nível_de_risco, 0.02)
            
            # Calcular valor de risco
            portfolio = float(self.client.get_asset_balance(asset='USDT')['free'])
            valor_de_risco = portfolio * porcentagem_de_risco
            
            # Calcular tamanho da posição em relação ao stop loss
            porcentagem_de_stop_loss = 0.02
            tamanho_da_posição = valor_de_risco / (preço * porcentagem_de_stop_loss)
            
            # Arredondar para casas decimais apropriadas
            info = self.client.get_symbol_info(symbol)
            filtro_de_tamanho_de_lote = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))
            tamanho_de_passo = float(filtro_de_tamanho_de_lote['stepSize'])
            tamanho_minimo = float(filtro_de_tamanho_de_lote['minQty'])
            
            # Garantir que a posição seja maior que o tamanho mínimo
            if tamanho_da_posição < tamanho_minimo:
                tamanho_da_posição = tamanho_minimo
            
            return {
                'quantity': round(tamanho_da_posição / tamanho_de_passo) * tamanho_de_passo
            }
        except Exception as e:
            logger.error(f"Erro ao calcular tamanho da posição: {str(e)}")
            return {}

    def _verificar_notional_minimo(self, symbol: str, quantity: float) -> bool:
        """Verificar se a ordem atende ao requisito mínimo de notional"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            preço = float(ticker['price'])
            
            notional = preço * quantity
            
            info = self.client.get_symbol_info(symbol)
            filtro_de_notional = next(filter(lambda x: x['filterType'] == 'MIN_NOTIONAL', info['filters']))
            notional_minimo = float(filtro_de_notional['minNotional'])
            
            return notional >= notional_minimo
        except Exception as e:
            logger.error(f"Erro ao verificar notional mínimo: {str(e)}")
            return False

    def _ajustar_tamanho_da_posição(self, symbol: str, quantity: float) -> Optional[Dict]:
        """Ajustar tamanho da posição para atender ao requisito mínimo de notional"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            preço = float(ticker['price'])
            
            info = self.client.get_symbol_info(symbol)
            filtro_de_notional = next(filter(lambda x: x['filterType'] == 'MIN_NOTIONAL', info['filters']))
            notional_minimo = float(filtro_de_notional['minNotional'])
            
            filtro_de_tamanho_de_lote = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))
            tamanho_de_passo = float(filtro_de_tamanho_de_lote['stepSize'])
            tamanho_minimo = float(filtro_de_tamanho_de_lote['minQty'])
            
            # Calcular a quantidade mínima necessária para atender ao notional mínimo
            quantidade_minima = notional_minimo / preço
            
            # Arredondar para o tamanho de passo mínimo
            quantidade_minima = round(quantidade_minima / tamanho_de_passo) * tamanho_de_passo
            
            # Garantir que a quantidade mínima seja maior que o tamanho mínimo
            if quantidade_minima < tamanho_minimo:
                quantidade_minima = tamanho_minimo
            
            return {
                'quantity': quantidade_minima
            }
        except Exception as e:
            logger.error(f"Erro ao ajustar tamanho da posição: {str(e)}")
            return None

    def _armazenar_análise(self, análise: Dict, symbol: str):
        """Armazenar resultados da análise no banco de dados"""
        try:
            dados_técnicos = análise['técnica']
            notícias = análise['notícias']
            dados_de_mercado = análise['mercado']
            dados_de_mercado_passados = análise['mercado_passado']
            confiança = análise['confiança']
            timestamp = análise['timestamp']
            
            # Armazenar dados de preços
            df_preços = pd.DataFrame([{
                'timestamp': timestamp,
                'open': dados_técnicos['open'],
                'high': dados_técnicos['high'],
                'low': dados_técnicos['low'],
                'close': dados_técnicos['close'],
                'volume': dados_técnicos['volume']
            }])
            df_preços['symbol'] = symbol
            self.db.armazenar_dados_de_preco(df_preços, symbol)
            
            # Armazenar indicadores técnicos
            self.db.armazenar_metricas({
                'timestamp': timestamp,
                'symbol': symbol,
                'sma_20': dados_técnicos['sma_20'],
                'sma_50': dados_técnicos['sma_50'],
                'ema_20': dados_técnicos['ema_20'],
                'rsi': dados_técnicos['rsi'],
                'macd': dados_técnicos['macd'],
                'macd_signal': dados_técnicos['macd_signal'],
                'macd_diff': dados_técnicos['macd_diff'],
                'bb_upper': dados_técnicos['bb_upper'],
                'bb_middle': dados_técnicos['bb_middle'],
                'bb_lower': dados_técnicos['bb_lower'],
                'volume_sma': dados_técnicos['volume_sma']
            }, symbol)
            
            # Armazenar notícias recentes
            if notícias:
                self.db.armazenar_notícias(notícias, symbol)
            
            # Armazenar confiança e recomendação
            self.db.armazenar_recomendação({
                'timestamp': timestamp,
                'symbol': symbol,
                'recomendação': confiança['recomendação'],
                'confiança': confiança['score'],
                'nível_de_risco': confiança['nível_de_risco'],
                'razão': "Análise baseada em indicadores técnicos e sentimentos de notícias"
            }, symbol)
        except Exception as e:
            logger.error(f"Erro ao armazenar análise: {str(e)}")
            raise

    def _armazenar_negociação(self, ordem: Dict, confiança: Dict, análise: Dict):
        """Armazenar detalhes da negociação no banco de dados"""
        try:
            # Extrair detalhes da ordem
            timestamp = datetime.now()
            symbol = ordem['symbol']
            side = ordem['side']
            quantity = float(ordem['executedQty'])
            price = float(ordem['fills'][0]['price'])
            total_value = quantity * price
            status = ordem['status']
            
            # Extrair pontuações de confiança e análise técnica
            ai_confidence = confiança['score']
            technical_score = análise['técnica']['rsi']  # Exemplo de pontuação técnica
            news_sentiment = np.mean([artigo['sentiment'] for artigo in análise['notícias']]) if análise['notícias'] else 0.0
            
            # Criar dicionário de dados da negociação
            dados_de_negociação = {
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'total_value': total_value,
                'ai_confidence': ai_confidence,
                'technical_score': technical_score,
                'news_sentiment': news_sentiment,
                'status': status
            }
            
            # Adicionar indicadores técnicos ao dicionário de dados da negociação
            dados_técnicos = análise['técnica']
            dados_de_negociação.update({
                'sma_20': dados_técnicos['sma_20'],
                'sma_50': dados_técnicos['sma_50'],
                'ema_20': dados_técnicos['ema_20'],
                'rsi': dados_técnicos['rsi'],
                'macd': dados_técnicos['macd'],
                'macd_signal': dados_técnicos['macd_signal'],
                'macd_diff': dados_técnicos['macd_diff'],
                'bb_upper': dados_técnicos['bb_upper'],
                'bb_middle': dados_técnicos['bb_middle'],
                'bb_lower': dados_técnicos['bb_lower'],
                'volume_sma': dados_técnicos['volume_sma'],
                'sentiment_notícias': news_sentiment
            })
            
            # Armazenar negociação no banco de dados
            self.db.armazenar_negociação(dados_de_negociação)
        except Exception as e:
            logger.error(f"Erro ao armazenar negociação: {str(e)}")
            raise

    def iniciar_trading(self, symbol: str):
        """Iniciar o trading para um símbolo específico"""
        try:
            if symbol in self.trading_threads:
                logger.warning(f"Trading já está ativo para {symbol}")
                return
            
            logger.info(f"Iniciando trading para {symbol}")
            thread = threading.Thread(target=self._executar_trading, args=(symbol,))
            thread.daemon = True
            thread.stop = False
            thread.start()
            self.trading_threads[symbol] = thread
        except Exception as e:
            logger.error(f"Erro ao iniciar trading para {symbol}: {str(e)}")
            raise

    def parar_trading(self, symbol: str):
        """Parar o trading para um símbolo específico"""
        try:
            if symbol not in self.trading_threads:
                logger.warning(f"Trading não está ativo para {symbol}")
                return
            
            logger.info(f"Parando trading para {symbol}")
            self.trading_threads[symbol].stop = True
            self.trading_threads.pop(symbol)
        except Exception as e:
            logger.error(f"Erro ao parar trading para {symbol}: {str(e)}")
            raise

    def _executar_trading(self, symbol: str):
        """Executar trading para um símbolo específico"""
        try:
            while not self.trading_threads[symbol].stop:
                try:
                    análise = self.analisar_mercado(symbol)
                    if análise:
                        confiança = self._calcular_confiança_da_negociação(análise['técnica'], análise['notícias'], análise['mercado'], análise['mercado_passado'])
                        if confiança['score'] > 0.7:
                            posição = self._calcular_tamanho_da_posição(
                                symbol,
                                confiança['nível_de_risco']
                            )
                            
                            # Verificar se a posição atende ao requisito mínimo de notional
                            if not self._verificar_notional_minimo(symbol, posição['quantity']):
                                logger.warning(f"Posição para {symbol} não atende ao requisito mínimo de notional. Ajustando...")
                                posição = self._ajustar_tamanho_da_posição(symbol, posição['quantity'])
                                if not posição:
                                    logger.error(f"Não foi possível ajustar a posição para {symbol}. Negociação cancelada.")
                                    continue
                            
                            # Executar negociação com mecanismo de repetição
                            for tentativa in range(3):
                                try:
                                    ordem = self.client.create_order(
                                        symbol=symbol,
                                        side=confiança['recomendação'],
                                        type='MARKET',
                                        quantity=posição['quantity']
                                    )
                                    
                                    # Armazenar detalhes da negociação
                                    self._armazenar_negociação(ordem, confiança, análise)
                                    
                                    logger.info(f"Negociação executada com sucesso para {symbol}: {ordem}")
                                    break
                                except Exception as e:
                                    if tentativa == 2:  # Última tentativa
                                        logger.error(f"Erro ao executar ordem para {symbol}: {str(e)}")
                                    else:
                                        logger.warning(f"Tentativa {tentativa + 1} falhou para {symbol}. Retentando...")
                                        time.sleep(1)  # Aguardar antes de repetir
                except Exception as e:
                    logger.error(f"Erro ao analisar e executar negociação para {symbol}: {str(e)}")
                time.sleep(2)  # Aguardar 15 segundos antes da próxima análise
        except Exception as e:
            logger.error(f"Erro no loop de trading para {symbol}: {str(e)}")
            raise

    def iniciar_coleta_de_dados(self, symbols: List[str]):
        """Iniciar a coleta de dados em um loop contínuo"""
        try:
            logger.info("Iniciando coleta de dados...")
            while True:
                for symbol in symbols:
                    try:
                        análise = self.analisar_mercado(symbol)
                        if análise:
                            logger.info(f"Análise armazenada com sucesso para {symbol}")
                    except Exception as e:
                        logger.error(f"Erro ao analisar e armazenar dados para {symbol}: {str(e)}")
                time.sleep(3)
        except Exception as e:
            logger.error(f"Erro na coleta de dados: {str(e)}")
            raise

# Inicialização do bot
bot = TradingBot(symbols)