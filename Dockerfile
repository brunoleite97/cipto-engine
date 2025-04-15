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

# Adicionar endpoint para iniciar o treinamento
RUN echo '\
@app.route("/api/iniciar_treinamento", methods=["POST"])\n\
def iniciar_treinamento():\n\
    """Iniciar o treinamento do modelo de IA"""\n\
    try:\n\
        # Iniciar o treinamento em uma thread separada\n\
        threading.Thread(target=lambda: os.system("python train_model.py"), daemon=True).start()\n\
        return jsonify({"message": "Treinamento do modelo iniciado com sucesso"}), 200\n\
    except Exception as e:\n\
        logger.error(f"Erro ao iniciar treinamento: {str(e)}")\n\
        return jsonify({"error": str(e)}), 500\n' >> /app/endpoint.py

# Adicionar o novo endpoint ao app.py
RUN sed -i '/if __name__ == \'__main__\':/i \# Endpoint para iniciar treinamento\n@app.route("/api/iniciar_treinamento", methods=["POST"])\ndef iniciar_treinamento():\n    """Iniciar o treinamento do modelo de IA"""\n    try:\n        # Iniciar o treinamento em uma thread separada\n        threading.Thread(target=lambda: os.system("python train_model.py"), daemon=True).start()\n        return jsonify({"message": "Treinamento do modelo iniciado com sucesso"}), 200\n    except Exception as e:\n        logger.error(f"Erro ao iniciar treinamento: {str(e)}")\n        return jsonify({"error": str(e)}), 500\n' /app/app.py

# Comando para iniciar os serviços
CMD ["/bin/bash", "/app/start.sh"]