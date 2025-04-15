FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

# Expor a porta que o Flask usa
EXPOSE 5000

# Criar um script de inicialização para gerenciar os serviços
RUN echo '#!/bin/bash\n\
# Iniciar o serviço de análise em segundo plano\n\
python app.py &\n\
APP_PID=$!\n\
\
# Aguardar o serviço iniciar\n\
sleep 5\n\
\
echo "Serviço de análise iniciado na porta 5000"\n\
echo "Para iniciar o treinamento da IA, acesse: http://localhost:5000/api/iniciar_treinamento"\n\
\
# Manter o container rodando\n\
wait $APP_PID\n' > /app/start.sh

# Tornar o script executável
RUN chmod +x /app/start.sh

# Configuração do ambiente concluída

# Comando para iniciar os serviços

# Comando para iniciar os serviços
CMD ["/bin/bash", "/app/start.sh"]