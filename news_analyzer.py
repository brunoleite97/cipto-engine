import requests
import logging
from typing import List, Dict
from logging.handlers import TimedRotatingFileHandler

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler('news_analyzer.log', when='midnight', interval=1, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnalisadorDeNotícias:
    def __init__(self, news_api_key: str, apitube_api_key: str):
        self.news_api_key = news_api_key
        self.apitube_api_key = apitube_api_key
        self.rate_limit_exceeded = False  # Flag para indicar se o limite de taxa foi excedido

    def obter_notícias_crypto(self, symbol: str) -> List[Dict]:
        """Obter notícias relacionadas a criptomoedas"""
        if self.rate_limit_exceeded:
            logger.warning(f"Limite de taxa excedido para {symbol}. Não obtendo notícias.")
            return []
        
        try:
            # Exemplo de chamada à API de notícias
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}"
            response = requests.get(url)
            response.raise_for_status()  # Lança uma exceção para códigos de status HTTP 4xx e 5xx
            noticias = response.json().get('articles', [])
            
            # Analisar sentimentos das notícias (exemplo simplificado)
            notícias_analisadas = []
            for noticia in noticias:
                sentiment = self.analisar_sentimento(noticia['title'] + " " + noticia['description'])
                notícias_analisadas.append({
                    'title': noticia['title'],
                    'description': noticia['description'],
                    'url': noticia['url'],
                    'publishedAt': noticia['publishedAt'],
                    'sentiment': sentiment
                })
            
            return notícias_analisadas
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                logger.warning(f"Limite de taxa atingido para {symbol}. Não obtendo notícias.")
                self.rate_limit_exceeded = True
                return []
            else:
                logger.error(f"Erro HTTP ao obter notícias para {symbol}: {http_err}")
        except Exception as e:
            logger.error(f"Erro ao obter notícias para {symbol}: {str(e)}")
        return []

    def analisar_sentimento(self, texto: str) -> float:
        """Analisar sentimentos de um texto (exemplo simplificado)"""
        try:
            # Exemplo de chamada à API de sentimentos
            # Substitua isso por uma API real de sentimentos
            url = f"https://api.apitube.io/v1/news?limit=1&query={texto}&apiKey={self.apitube_api_key}"
            response = requests.get(url)
            response.raise_for_status()  # Lança uma exceção para códigos de status HTTP 4xx e 5xx
            data = response.json()
            
            # Supondo que a API retorne um sentimento entre -1 e 1
            sentiment = data.get('sentiment', 0.0)
            return sentiment
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Erro HTTP ao analisar sentimento: {http_err}")
        except Exception as e:
            logger.error(f"Erro ao analisar sentimento: {str(e)}")
        return 0.0