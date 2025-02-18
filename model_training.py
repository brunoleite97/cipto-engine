import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(15, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ModeloDeAprendizadoDeMáquina:
    def __init__(self, db: GerenciadorDeBancoDeDados):
        self.db = db
        self.model = self._criar_modelo_neural()
        self.scaler = StandardScaler()
        self.treinar_modelo()  # Treinar o modelo imediatamente ao inicializar

    def _criar_modelo_neural(self):
        """Criar um modelo de rede neural"""
        model = NeuralNetwork()
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
                df_sintético = pd.merge(df_preços, df_métricas_com_notícias, on=['timestamp', 'symbol'], how='left')
                df_sintético = df_sintético.fillna(0)
                df_sintético['side'] = 'buy'
                df_sintético['quantity'] = 1.0
                df_sintético['price'] = df_sintético['close']
                df_sintético['total_value'] = df_sintético['quantity'] * df_sintético['price']
                df_sintético['ai_confidence'] = 0.5
                df_sintético['news_sentiment'] = df_sintético['sentiment']
                df_sintético['status'] = 'open'
                df_sintético['profit'] = df_sintético['total_value'] - (df_sintético['quantity'] * df_sintético['price'])
                df_sintético['profit'] = df_sintético['profit'].apply(lambda x: 1 if x > 0 else 0)
                df_sintético.rename(columns={'sentiment': 'sentiment_notícias'}, inplace=True)
                
                colunas_necessárias = ['technical_score', 'news_sentiment', 'ai_confidence', 'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_sma', 'sentiment_notícias', 'profit']
                for coluna in colunas_necessárias:
                    if coluna not in df_sintético.columns:
                        df_sintético[coluna] = 0
                
                if len(df_sintético['profit'].unique()) == 1:
                    logger.warning("O DataFrame sintético contém apenas uma classe para o target. Criando uma distribuição de classes sintética.")
                    num_registros = len(df_sintético)
                    num_positivos = num_registros // 2
                    num_negativos = num_registros - num_positivos
                    indices_positivos = np.random.choice(df_sintético.index, num_positivos, replace=False)
                    indices_negativos = np.setdiff1d(df_sintético.index, indices_positivos)
                    df_sintético.loc[indices_positivos, 'profit'] = 1
                    df_sintético.loc[indices_negativos, 'profit'] = 0
                
                df_negociações = df_sintético
            else:
                # Unir dados de negociações com métricas e notícias
                df_negociações = pd.merge(df_negociações, df_métricas_com_notícias, on=['timestamp', 'symbol'], how='left')
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
            
            # Converter para tensores PyTorch
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
            
            # Criar DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Treinar modelo
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            for epoch in range(100):  # Número fixo de épocas
                self.model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(test_loader)
                
                logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            
            # Avaliar modelo
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test)
                loss = criterion(outputs, y_test)
                accuracy = ((outputs > 0.5).int() == y_test.int()).float().mean().item()
                logger.info(f"Modelo treinado com sucesso. Acurácia: {accuracy}")
            
            # Salvar o modelo e o scaler
            torch.save(self.model.state_dict(), 'model.pt')
            np.save('scaler_mean.npy', self.scaler.mean_)
            np.save('scaler_scale.npy', self.scaler.scale_)
        
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {str(e)}")
            raise

# Inicialização do treinamento do modelo
if __name__ == "__main__":
    db = GerenciadorDeBancoDeDados()
    modelo = ModeloDeAprendizadoDeMáquina(db)