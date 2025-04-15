# Cripto Engine

Este projeto é um sistema de análise de criptomoedas com capacidade de treinamento de IA para previsão de mercado.

## Configuração com Docker

### Pré-requisitos

- Docker e Docker Compose instalados
- Chaves de API configuradas (Binance, News API, APITube)

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```
NEWS_API_KEY=sua_chave_aqui
APITUBE_API_KEY=sua_chave_aqui
BINANCE_API_KEY=sua_chave_aqui
BINANCE_API_SECRET=sua_chave_secreta_aqui
```

### Iniciar o Serviço

Para iniciar o serviço de análise, execute:

```bash
docker-compose up -d
```

O serviço estará disponível em `http://localhost:5000`.

## Endpoints Disponíveis

### Análise de Moedas

```
GET /api/analise
```

Retorna análise de todas as moedas configuradas.

### Iniciar Treinamento da IA

```
POST /api/iniciar_treinamento
```

Inicia o processo de treinamento do modelo de IA em segundo plano.

### Outros Endpoints

- `GET /api/portfolio` - Obtém o portfolio atual
- `GET /api/dividir_carteira` - Divide o valor em carteira entre as moedas
- `POST /api/executar_ordem` - Executa uma ordem de compra ou venda
- `POST /api/iniciar_trading` - Inicia o trading para um símbolo específico
- `POST /api/parar_trading` - Para o trading para um símbolo específico

## Construção Manual da Imagem Docker

Se preferir construir a imagem manualmente:

```bash
docker build -t cripto-engine .
docker run -p 5000:5000 --env-file .env cripto-engine
```