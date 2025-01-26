import os
import logging
from typing import List, Dict
import groq
import time
import re

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgenteLLM:
    def __init__(self):
        self.client = groq.Client(api_key=os.getenv('GROQ_API_KEY'))
        self.model = "llama-3.1-8b-instant"  # Usar o modelo llama-3.2 para análise de condições do mercado

    def analisar_condições_do_mercado(self, 
                                    dados_técnicos: Dict, 
                                    dados_de_mercado: Dict, 
                                    noticias_recentes: List[Dict]) -> Dict:
        """Obter análise do LLM das condições do mercado"""
        try:
            # Gerar prompt para o LLM
            noticias_texto = '\n'.join([f"- {noticia['title']} (Sentimento: {noticia['sentiment']})" for noticia in noticias_recentes])
            prompt = f"""
            Como um especialista em negociação de criptomoedas, analise os seguintes dados do mercado e forneça orientações de negociação:

            Indicadores Técnicos:
            - SMA 20: {dados_técnicos['sma_20']}
            - SMA 50: {dados_técnicos['sma_50']}
            - EMA 20: {dados_técnicos['ema_20']}
            - RSI: {dados_técnicos['rsi']}
            - MACD: {dados_técnicos['macd']}
            - Sinal MACD: {dados_técnicos['macd_signal']}
            - Diferença MACD: {dados_técnicos['macd_diff']}
            - Banda de Bollinger Superior: {dados_técnicos['bb_upper']}
            - Banda de Bollinger Média: {dados_técnicos['bb_middle']}
            - Banda de Bollinger Inferior: {dados_técnicos['bb_lower']}
            - Volume SMA: {dados_técnicos['volume_sma']}

            Dados do Mercado:
            - Preço Atual: {dados_de_mercado['current_price']}
            - Volume 24h: {dados_de_mercado['volume']}
            - Mudança de Preço: {dados_de_mercado['price_change']}

            Notícias Recentes:
            {noticias_texto}

            Forneça uma análise estruturada no seguinte formato:
            recomendação: comprar/vender/segurar
            confiança: 0-1
            nível_de_risco: alto/baixo/médio
            razão: breve explicação da recomendação
            """

            # Obter resposta do LLM
            resposta = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            # Analisar a resposta
            análise = resposta.choices[0].message.content
            
            # Extrair dados estruturados da resposta usando regex
            resultado = self._extrair_dados_da_resposta(análise)
            
            # Validar modelo de saída
            if not self._validar_resposta(resultado):
                logger.error("Resposta do LLM não segue o modelo esperado")
                return self._resposta_padrao()
            
            return resultado
        except groq.GroqError as e:
            if e.status_code == 429:
                retry_after = e.response.headers.get('retry-after', 10)
                logger.warning(f"Rate limit reached. Retrying in {retry_after} seconds...")
                time.sleep(int(retry_after))
                return self.analisar_condições_do_mercado(dados_técnicos, dados_de_mercado, noticias_recentes)
            else:
                logger.error(f"Erro ao analisar condições do mercado com LLM: {str(e)}")
                return self._resposta_padrao()
        except Exception as e:
            logger.error(f"Erro ao analisar condições do mercado com LLM: {str(e)}")
            return self._resposta_padrao()

    def _extrair_dados_da_resposta(self, análise: str) -> Dict:
        """Extrair dados estruturados da resposta do LLM usando regex"""
        try:
            resultado = {}
            
            # Usar regex para extrair as chaves e valores
            recomendação_match = re.search(r'recomendação:\s*(comprar|vender|segurar)', análise, re.IGNORECASE)
            confiança_match = re.search(r'confiança:\s*([\d.]+)', análise, re.IGNORECASE)
            nível_de_risco_match = re.search(r'nível_de_risco:\s*(alto|baixo|médio)', análise, re.IGNORECASE)
            razão_match = re.search(r'razão:\s*(.*)', análise, re.IGNORECASE)
            
            if recomendação_match:
                resultado['recomendação'] = recomendação_match.group(1).lower()
            if confiança_match:
                resultado['confiança'] = float(confiança_match.group(1).replace(',', '.'))
            if nível_de_risco_match:
                resultado['nível_de_risco'] = nível_de_risco_match.group(1).lower()
            if razão_match:
                resultado['razão'] = razão_match.group(1).strip()
            
            # Verificar se todas as chaves estão presentes
            if not all(key in resultado for key in ['recomendação', 'confiança', 'nível_de_risco', 'razão']):
                logger.error("Resposta do LLM não contém todas as chaves esperadas")
                return {}
            
            return resultado
        except Exception as e:
            logger.error(f"Erro ao extrair dados da resposta do LLM: {str(e)}")
            return {}

    def _validar_resposta(self, resposta: Dict) -> bool:
        """Validar se a resposta do LLM segue o modelo esperado"""
        try:
            return (
                'recomendação' in resposta and
                'confiança' in resposta and
                'nível_de_risco' in resposta and
                'razão' in resposta and
                resposta['recomendação'] in ['comprar', 'vender', 'segurar'] and
                isinstance(resposta['confiança'], (float, int)) and
                0.0 <= resposta['confiança'] <= 1.0 and
                resposta['nível_de_risco'] in ['baixo', 'médio', 'alto'] and
                isinstance(resposta['razão'], str)
            )
        except Exception as e:
            logger.error(f"Erro ao validar resposta do LLM: {str(e)}")
            return False

    def _resposta_padrao(self) -> Dict:
        """Fornecer uma resposta padrão se a resposta do LLM não for válida"""
        return {
            "recomendação": "segurar",
            "confiança": 0.5,
            "nível_de_risco": "médio",
            "razão": "Resposta do LLM não foi válida ou não seguiu o formato esperado"
        }