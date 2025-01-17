# api/server.py

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class WebInterface:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        
    def start(self):
        @app.route('/api/trading-status')
        def get_status():
            return jsonify({
                'soundEnabled': self.trading_bot.keyboard_controller.state.sound_enabled,
                'newTradesEnabled': self.trading_bot.keyboard_controller.state.new_trades_enabled
            })
            
        @app.route('/api/trigger-action/<action>', methods=['POST'])
        def trigger_action(action):
            if action == 'close-profitable':
                self.trading_bot.keyboard_controller._close_profitable_positions()
            elif action == 'close-losing':
                self.trading_bot.keyboard_controller._close_losing_positions()
            # Add other actions...
            return jsonify({'success': True})
            
        app.run(port=5000)
