from flask import Flask, jsonify, request
from trading_bot import bot, symbols
import threading

app = Flask(__name__)

# Adicionar suporte a CORS
from flask_cors import CORS
CORS(app)

@app.route('/api/analise', methods=['GET'])
def analise_moedas():
    """Analisar todas as moedas especificadas"""
    try:
        resultados = [bot.analisar_mercado(symbol) for symbol in symbols]
        return jsonify(resultados)
    except Exception as e:
        logger.error(f"Erro ao analisar moedas: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Obter o portfolio atual"""
    try:
        conta = bot.client.get_account()
        balances = [balance for balance in conta['balances'] if float(balance['free']) > 0]
        return jsonify(balances)
    except Exception as e:
        logger.error(f"Erro ao obter portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dividir_carteira', methods=['GET'])
def dividir_carteira():
    """Dividir o valor em carteira entre as moedas especificadas e manter um percentual em USDT"""
    try:
        percentual_usdt = float(request.args.get('percentual_usdt', 0.1))
        divisão = bot.dividir_valor_em_carteira(symbols, percentual_usdt)
        return jsonify(divisão)
    except Exception as e:
        logger.error(f"Erro ao dividir carteira: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/executar_ordem', methods=['POST'])
def executar_ordem():
    """Executar uma ordem de compra ou venda"""
    try:
        dados = request.json
        symbol = dados.get('symbol')
        quantidade = dados.get('quantidade')
        tipo_ordem = dados.get('tipo_ordem')

        if not symbol or not quantidade or not tipo_ordem:
            return jsonify({'error': 'Dados incompletos'}), 400

        ordem = bot.executar_negociação(symbol, quantidade, tipo_ordem)
        if not ordem:
            return jsonify({'error': 'Erro ao executar ordem'}), 500

        return jsonify(ordem)
    except Exception as e:
        logger.error(f"Erro ao executar ordem: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/iniciar_trading', methods=['POST'])
def iniciar_trading():
    """Iniciar o trading para um símbolo específico"""
    try:
        dados = request.json
        symbol = dados.get('symbol')

        if not symbol:
            return jsonify({'error': 'Símbolo não fornecido'}), 400

        bot.iniciar_trading(symbol)
        return jsonify({'message': f'Trading iniciado para {symbol}'}), 200
    except Exception as e:
        logger.error(f"Erro ao iniciar trading: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parar_trading', methods=['POST'])
def parar_trading():
    """Parar o trading para um símbolo específico"""
    try:
        dados = request.json
        symbol = dados.get('symbol')

        if not symbol:
            return jsonify({'error': 'Símbolo não fornecido'}), 400

        bot.parar_trading(symbol)
        return jsonify({'message': f'Trading parado para {symbol}'}), 200
    except Exception as e:
        logger.error(f"Erro ao parar trading: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Iniciar coleta de dados em uma thread separada
    threading.Thread(target=bot.iniciar_coleta_de_dados, args=(symbols,), daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)