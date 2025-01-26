import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from db_manager import GerenciadorDeBancoDeDados

# Configuração aprimorada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler('model_training.log', when='midnight', interval=1, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModeloDeAprendizadoDeMáquina:
    def __init__(self, db: GerenciadorDeBancoDeDados):
        self.db = db
        self.model = self._criar_modelo_neural()
        self.scaler = StandardScaler()
        self.treinar_modelo()  # Treinar o modelo imediatamente ao inicializar

    def _criar_modelo_neural(self):
        """Criar um modelo de rede neural"""
        model = Sequential()
        model.add(Dense(64, input_dim=15, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def treinar_modelo(self):
        """Treinar um modelo de aprendizado de máquina com dados históricos"""
        try:
            # Contar registros em todas as tabelas
            tabelas = ['preços', 'métricas', 'notícias', 'recomendações', 'negociações']
            contagens = {tabela: self.db.contar_registros(tabela) for tabela in tabelas}
            for tabela, count in contagens.items():
                logger.info(f"Número de registros na tabela {tabela}: {count}")
                if count < 100 and tabela == 'negociações':
                    logger.warning("Não há dados suficientes para treinar o modelo com negociações.")
            
            # Obter dados de negociações
            df_negociações = self.db.obter_negociações()
            
            # Log para verificar o conteúdo do DataFrame de negociações
            logger.info(f"Primeiras linhas do DataFrame de negociações:\n{df_negociações.head()}")
            
            # Obter dados de notícias
            df_notícias = self.db.obter_notícias()
            
            # Obter dados de métricas
            df_métricas = self.db.obter_métricas()
            
            # Obter dados de preços
            df_preços = self.db.obter_preços()
            
            # Unir dados de métricas com notícias
            # Agrupar notícias por timestamp e symbol e calcular a média do sentimento
            df_notícias_agrupadas = df_notícias.groupby(['timestamp', 'symbol'])['sentiment'].mean().reset_index()
            df_métricas_com_notícias = pd.merge(df_métricas, df_notícias_agrupadas, on=['timestamp', 'symbol'], how='left', suffixes=('', '_notícias'))
            
            # Preencher valores ausentes com 0
            df_métricas_com_notícias = df_métricas_com_notícias.fillna(0)
            
            # Log para verificar o conteúdo do DataFrame após a união
            logger.info(f"Primeiras linhas do DataFrame após a união de métricas e notícias:\n{df_métricas_com_notícias.head()}")
            
            # Verificar se há dados suficientes após a união
            if len(df_métricas_com_notícias) < 100:
                logger.warning("Não há dados suficientes após a união de métricas e notícias para treinar o modelo.")
                return
            
            # Se a tabela de negociações estiver vazia, criar um DataFrame sintético
            if len(df_negociações) == 0:
                logger.warning("A tabela de negociações está vazia. Criando um DataFrame sintético.")
                # Criar um DataFrame sintético com base nas métricas, notícias e preços
                df_sintético = pd.merge(df_preços, df_métricas_com_notícias, on=['timestamp', 'symbol'], how='left')
                
                # Preencher valores ausentes com 0
                df_sintético = df_sintético.fillna(0)
                
                # Adicionar colunas necessárias para o treinamento
                df_sintético['side'] = 'buy'  # Exemplo de valor padrão
                df_sintético['quantity'] = 1.0  # Exemplo de valor padrão
                df_sintético['price'] = df_sintético['close']  # Usar o preço de fechamento como exemplo
                df_sintético['total_value'] = df_sintético['quantity'] * df_sintético['price']
                df_sintético['ai_confidence'] = 0.5  # Exemplo de valor padrão
                df_sintético['news_sentiment'] = df_sintético['sentiment']  # Usar o sentimento das notícias como exemplo
                df_sintético['status'] = 'open'  # Exemplo de valor padrão
                
                # Calcular o profit sintético
                df_sintético['profit'] = df_sintético['total_value'] - (df_sintético['quantity'] * df_sintético['price'])
                df_sintético['profit'] = df_sintético['profit'].apply(lambda x: 1 if x > 0 else 0)
                
                # Renomear colunas para corresponder às colunas de negociações
                df_sintético.rename(columns={'sentiment': 'sentiment_notícias'}, inplace=True)
                
                # Adicionar colunas faltantes com valores padrão
                colunas_necessárias = ['technical_score', 'news_sentiment', 'ai_confidence', 'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma', 'sentiment_notícias', 'profit']
                for coluna in colunas_necessárias:
                    if coluna not in df_sintético.columns:
                        df_sintético[coluna] = 0  # Adicionar coluna com valor padrão 0
                
                # Garantir que o DataFrame sintético tenha uma distribuição de classes adequada
                if len(df_sintético['profit'].unique()) == 1:
                    logger.warning("O DataFrame sintético contém apenas uma classe para o target. Criando uma distribuição de classes sintética.")
                    # Criar uma distribuição de classes sintética
                    num_registros = len(df_sintético)
                    num_positivos = num_registros // 2
                    num_negativos = num_registros - num_positivos
                    
                    # Selecionar aleatoriamente registros para serem positivos e negativos
                    indices_positivos = np.random.choice(df_sintético.index, num_positivos, replace=False)
                    indices_negativos = np.setdiff1d(df_sintético.index, indices_positivos)
                    
                    df_sintético.loc[indices_positivos, 'profit'] = 1
                    df_sintético.loc[indices_negativos, 'profit'] = 0
                
                df_negociações = df_sintético
            else:
                # Unir dados de negociações com métricas e notícias
                df_negociações = pd.merge(df_negociações, df_métricas_com_notícias, on=['timestamp', 'symbol'], how='left')
                
                # Preencher valores ausentes com 0
                df_negociações = df_negociações.fillna(0)
                
                # Log para verificar o conteúdo do DataFrame após a união
                logger.info(f"Primeiras linhas do DataFrame após a união:\n{df_negociações.head()}")
                
                # Verificar se há dados suficientes após a união
                if len(df_negociações) < 100:
                    logger.warning("Não há dados suficientes após a união para treinar o modelo.")
                    return
            
            # Selecionar features e target
            features = df_negociações[['technical_score', 'news_sentiment', 'ai_confidence', 'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma', 'sentiment_notícias']]
            target = df_negociações['profit']
            
            # Normalizar features
            features = self.scaler.fit_transform(features)
            
            # Dividir dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            
            # Treinar modelo
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
            
            # Avaliar modelo
            loss, accuracy = self.model.evaluate(X_test, y_test)
            logger.info(f"Modelo treinado com sucesso. Acurácia: {accuracy}")
            
            # Validar modelo com cross-validation
            # Note: Keras não suporta diretamente cross_val_score, então precisamos implementar manualmente
            cv_scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(features):
                X_train_cv, X_test_cv = features[train_index], features[test_index]
                y_train_cv, y_test_cv = target[train_index], target[test_index]
                
                # Resetar o modelo para cada fold
                self.model = self._criar_modelo_neural()
                
                self.model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                loss, accuracy_cv = self.model.evaluate(X_test_cv, y_test_cv, verbose=0)
                cv_scores.append(accuracy_cv)
            
            logger.info(f"Cross-validation scores: {cv_scores}, Média: {np.mean(cv_scores)}")
            
            # Salvar o modelo e o scaler
            self.model.save('model.h5')
            np.save('scaler_mean.npy', self.scaler.mean_)
            np.save('scaler_scale.npy', self.scaler.scale_)
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {str(e)}")
            raise

# Inicialização do treinamento do modelo
if __name__ == "__main__":
    db = GerenciadorDeBancoDeDados()
    modelo = ModeloDeAprendizadoDeMáquina(db)