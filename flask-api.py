from flask import Flask, jsonify, request
from flask_jsonpify import jsonpify
import json
import main

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"

@app.route('/game', methods=["GET"])
def recommend_game():
    game = request.args.get('Game')
    game_recommendation = main.gameRec(game)
    # convertimos la salida a JSON    
    return jsonify({'game': json.dumps(game_recommendation['similarity'].to_dict())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
   