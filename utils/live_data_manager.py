import threading
import time
import pandas as pd
import requests
import ccxt
from datetime import datetime

class CryptoDataFeed:
    def __init__(self):
        self.binance = ccxt.binance()
        self.coingecko_url = "https://api.coingecko.com/api/v3"
    
    def get_live_price(self, symbol):
        """Отримати поточну ціну"""
        try:
            ticker = self.binance.fetch_ticker(symbol + '/USDT')
            return ticker['last']
        except:
            try:
                response = requests.get(f"{self.coingecko_url}/simple/price?ids={symbol.lower()}&vs_currencies=usd", timeout=10)
                data = response.json()
                if symbol.lower() in data:
                    return data[symbol.lower()]['usd']
                return None
            except:
                return None

class SentimentData:
    def get_crypto_fear_index(self):
        """Crypto Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            return response.json()
        except:
            return {'value': 50, 'value_classification': 'Neutral'}

class LiveDataManager:
    def __init__(self):
        self.live_data = {}
        self.is_running = False
        self.update_interval = 60
        self.crypto_feed = CryptoDataFeed()
        self.sentiment_feed = SentimentData()
    
    def start_live_feed(self, symbols):
        """Запуск отримання live даних"""
        self.is_running = True
        thread = threading.Thread(target=self._update_loop, args=(symbols,))
        thread.daemon = True
        thread.start()
    
    def _update_loop(self, symbols):
        """Фоновий процес оновлення даних"""
        while self.is_running:
            for symbol in symbols:
                try:
                    self.live_data[symbol] = self._fetch_all_data(symbol)
                except Exception as e:
                    print(f"Помилка оновлення {symbol}: {e}")
            time.sleep(self.update_interval)
    
    def _fetch_all_data(self, symbol):
        """Отримати всі live дані для символу"""
        return {
            'price': self.crypto_feed.get_live_price(symbol),
            'fear_index': self.sentiment_feed.get_crypto_fear_index(),
            'timestamp': pd.Timestamp.now()
        }
    
    def stop_feed(self):
        """Зупинити оновлення даних"""
        self.is_running = False