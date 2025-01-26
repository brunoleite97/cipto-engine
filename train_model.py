from db_manager import GerenciadorDeBancoDeDados
from model_training import ModeloDeAprendizadoDeMáquina
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import time  # Importar o módulo time

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler('train_model.log', when='midnight', interval=1, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    db = GerenciadorDeBancoDeDados()
    model = ModeloDeAprendizadoDeMáquina(db)
    
    while True:
        try:
            model.treinar_modelo()
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {str(e)}")
        
        time.sleep(1800)  # Treinar a cada 30 minutos

if __name__ == "__main__":
    logger.info("Iniciando treinamento do modelo...")
    main()