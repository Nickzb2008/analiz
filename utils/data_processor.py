import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataProcessor:
    @staticmethod
    def normalize_data(data, method='minmax'):
        """
        Нормалізація даних
        """
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Метод нормалізації повинен бути 'minmax' або 'standard'")
        
        return scaler.fit_transform(data), scaler

    @staticmethod
    def create_sequences(data, lookback=60):
        """
        Створення послідовностей для часових рядів
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    @staticmethod
    def prepare_features_for_ml(data):
        """
        Підготовка ознак для ML моделей
        """
        df = data.copy()
        
        # Базові технічні індикатори
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Цільова змінна - ціна наступного дня
        df['Target'] = df['Close'].shift(-1)
        
        # Видалення NaN
        df = df.dropna()
        
        return df

    @staticmethod
    def split_data(X, y, test_size=0.2, shuffle=False):
        """
        Розділення даних на тренувальні та тестові
        """
        split_index = int(len(X) * (1 - test_size))
        
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        return X_train, X_test, y_train, y_test
    

    



# data_processor.py - додаємо нові функції
    def calculate_advanced_indicators(self, data):
        """Розширені технічні індикатори"""
        df = data.copy()
        
        # Базові цінові показники
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Ковзні середні різних періодів
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Волатильність
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_50'] = df['Returns'].rolling(window=50).std()
        
        # RSI різних періодів
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
        
        # MACD різних налаштувань
        macd_fast = [8, 12, 16]
        macd_slow = [26, 21, 32]
        for fast, slow in zip(macd_fast, macd_slow):
            macd, signal = self.calculate_macd(df['Close'], fast, slow)
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}'] = signal
        
        # Стохастичний осцилятор
        df['Stochastic_K_14'], df['Stochastic_D_14'] = self.calculate_stochastic(
            df['High'], df['Low'], df['Close'], 14, 3
        )
        
        # Смуги Боллінджера
        for window in [20, 50]:
            upper, middle, lower = self.calculate_bollinger_bands(df['Close'], window)
            df[f'Bollinger_Upper_{window}'] = upper
            df[f'Bollinger_Middle_{window}'] = middle
            df[f'Bollinger_Lower_{window}'] = lower
            df[f'Bollinger_Width_{window}'] = (upper - lower) / middle
        
        # ATR (Average True Range)
        df['ATR_14'] = self.calculate_atr(df['High'], df['Low'], df['Close'], 14)
        df['ATR_21'] = self.calculate_atr(df['High'], df['Low'], df['Close'], 21)
        
        # Індикатори об'єму
        if 'Volume' in df.columns:
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # Цінові дивергенції
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_20d'] = df['Close'].pct_change(20)
        
        # Відносна сила
        df['Price_MA_Ratio_5_20'] = df['MA_5'] / df['MA_20']
        df['Price_MA_Ratio_20_50'] = df['MA_20'] / df['MA_50']
        
        return df

    def prepare_advanced_features(self, data):
        """Підготовка розширених ознак для ML"""
        df = self.calculate_advanced_indicators(data)
        
        # Додаємо часові ознаки
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        
        # Цільова змінна - ціна через n днів
        df['Target_5'] = df['Close'].shift(-5)
        df['Target_10'] = df['Close'].shift(-10)
        df['Target_30'] = df['Close'].shift(-30)
        
        # Видалення NaN
        df = df.dropna()
        
        return df

# Додайте ці методи в клас DataProcessor у файлі data_processor.py

    def calculate_rsi(self, prices, period=14):
        """Розрахунок RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Розрахунок MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Розрахунок смуг Боллінджера"""
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Розрахунок стохастичного осцилятора"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    def calculate_atr(self, high, low, close, period=14):
        """Розрахунок Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def calculate_advanced_indicators_v2(self, data, use_rsi=True, use_macd=True, use_bollinger=True, 
                                    use_atr=False, use_volume_indicators=False):
        """Розширені технічні індикатори з опціями"""
        df = data.copy()
        
        # RSI
        if use_rsi and 'Close' in df.columns:
            df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        if use_macd and 'Close' in df.columns:
            macd, signal = self.calculate_macd(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Histogram'] = macd - signal
        
        # Bollinger Bands
        if use_bollinger and 'Close' in df.columns:
            df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = self.calculate_bollinger_bands(df['Close'])
            df['Bollinger_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['Bollinger_Middle']
        
        # ATR
        if use_atr and all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        if use_volume_indicators and 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
        
        return df

    def add_advanced_time_features(self, data, use_seasonal=False, use_cyclical=False):
        """Додавання розширених часових ознак"""
        df = data.copy()
        
        if use_seasonal:
            df['Year'] = df.index.year
            df['Quarter'] = df.index.quarter
            df['Season'] = (df.index.month % 12 + 3) // 3
        
        if use_cyclical:
            # Циклічні ознаки для дня тижня та місяця
            df['Day_of_Week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['Day_of_Week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        return df

    def calculate_trading_indicators(self, data):
        """Розширені технічні індикатори для торгівлі"""
        df = data.copy()
        
        # Базові цінові показники
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Волатильність
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['ATR_14'] = self.calculate_atr(df['High'], df['Low'], df['Close'], 14)
        
        # Трендові індикатори
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Momentum індикатори
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['Stochastic_K'], df['Stochastic_D'] = self.calculate_stochastic(
            df['High'], df['Low'], df['Close'], 14, 3
        )
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Об'ємні індикатори
        if 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # Цінові паттерни
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Support/Resistance
        df['Support_Level'] = df['Close'].rolling(window=20).min()
        df['Resistance_Level'] = df['Close'].rolling(window=20).max()
        df['Support_Distance'] = (df['Close'] - df['Support_Level']) / df['Support_Level']
        df['Resistance_Distance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        return df.dropna()

    def calculate_obv(self, close, volume):
        """On-Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)

    def add_market_indicators(self, df, symbol):
        """Додавання ринкових показників"""
        # Відносна сила до BTC (для альткойнів)
        if symbol != 'BTC' and 'BTC_data.csv' in os.listdir('data'):
            try:
                btc_data = pd.read_csv('data/BTC_data.csv', index_col=0)
                df['BTC_Ratio'] = df['Close'] / btc_data['Close']
                df['BTC_Correlation_20'] = df['Returns'].rolling(window=20).corr(btc_data['Returns'])
            except:
                pass
        
        # Середня ринкова волатильність
        market_volatility = self.calculate_market_volatility()
        if market_volatility is not None:
            df['Market_Volatility_Ratio'] = df['Volatility_20'] / market_volatility
        
        return df

    def add_time_features(self, df):
        """Додавання часових ознак для торгівлі"""
        # Часові компоненти
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Циклічні ознаки
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        
        # Торгові сесії
        df['Asian_Session'] = ((df['Hour'] >= 0) & (df['Hour'] <= 8)).astype(int)
        df['European_Session'] = ((df['Hour'] >= 7) & (df['Hour'] <= 16)).astype(int)
        df['US_Session'] = ((df['Hour'] >= 13) & (df['Hour'] <= 22)).astype(int)
        
        # Вихідні та святкові дні (спрощено)
        df['Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
        
        return df

    def prepare_trading_targets(self, df, forecast_horizons=[1, 5, 10, 20]):
        """Підготовка цільових змінних для торгівлі"""
        for horizon in forecast_horizons:
            # Прогноз ціни
            df[f'Target_Price_{horizon}d'] = df['Close'].shift(-horizon)
            
            # Прогноз доходності
            df[f'Target_Return_{horizon}d'] = df['Close'].pct_change(horizon).shift(-horizon)
            
            # Бінарні цілі для класифікації
            df[f'Target_Up_{horizon}d'] = (df[f'Target_Return_{horizon}d'] > 0).astype(int)
            
            # Волатильність
            df[f'Target_Volatility_{horizon}d'] = df['Returns'].rolling(horizon).std().shift(-horizon)
        
        # Торгові сигнали
        df['Target_Signal_5d'] = np.where(
            df['Target_Return_5d'] > 0.02, 1, np.where(df['Target_Return_5d'] < -0.02, -1, 0)
        )
        
        return df.dropna()

    def add_risk_metrics(self, df):
        """Додавання метрик ризик-менеджменту"""
        # Value at Risk (VaR)
        df['VaR_95_1d'] = df['Returns'].rolling(window=20).quantile(0.05)
        df['VaR_99_1d'] = df['Returns'].rolling(window=20).quantile(0.01)
        
        # Expected Shortfall
        df['ES_95_1d'] = df['Returns'].rolling(window=20).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        
        # Максимальна просадка
        df['Rolling_Max'] = df['Close'].rolling(window=20).max()
        df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
        df['Max_Drawdown_20'] = df['Drawdown'].rolling(window=20).min()
        
        # Sharpe Ratio
        risk_free_rate = 0.0001  # Припущення
        df['Sharpe_Ratio_20'] = (df['Returns'].rolling(window=20).mean() - risk_free_rate) / df['Returns'].rolling(window=20).std()
        
        # Sortino Ratio
        downside_returns = df['Returns'].where(df['Returns'] < 0, 0)
        df['Sortino_Ratio_20'] = (df['Returns'].rolling(window=20).mean() - risk_free_rate) / downside_returns.rolling(window=20).std()
        
        return df

    def prepare_for_trading(self, data, symbol, include_all_indicators=True):
        """Повна підготовка даних для торгівельного навчання"""
        df = data.copy()
        
        if include_all_indicators:
            # Технічні індикатори
            df = self.calculate_trading_indicators(df)
            
            # Ринкові показники
            df = self.add_market_indicators(df, symbol)
            
            # Часові ознаки
            df = self.add_time_features(df)
            
            # Ризик-менеджмент
            df = self.add_risk_metrics(df)
        
        # Цільові змінні
        df = self.prepare_trading_targets(df)
        
        # Видаляємо NaN
        df = df.dropna()
        
        return df

    def enrich_with_live_data(self, historical_data, symbol, live_data):
        """Збагачення історичних даних live-даними"""
        df = historical_data.copy()
        
        if live_data and 'price' in live_data and live_data['price'] is not None:
            # Додаємо індикатори страху/жадібності
            if 'fear_index' in live_data and 'value' in live_data['fear_index']:
                df['Fear_Greed_Index'] = live_data['fear_index']['value']
        
        return df

    def prepare_for_real_trading(self, data, symbol, live_data=None):
        """Спеціальна підготовка даних для реальної торгівлі"""
        df = data.copy()
        
        # Збагачуємо live даними якщо вони є
        if live_data:
            df = self.enrich_with_live_data(df, symbol, live_data)
        
        # Додаємо розширені торгові індикатори
        df = self.calculate_trading_indicators(df)
        
        # Додаємо ринкові метрики
        df = self.add_market_indicators(df, symbol)
        
        # Додаємо часові ознаки
        df = self.add_time_features(df)
        
        # Додаємо метрики ризику
        df = self.add_risk_metrics(df)
        
        # Готуємо цільові змінні для торгівлі
        df = self.prepare_trading_targets(df)
        
        return df.dropna()











