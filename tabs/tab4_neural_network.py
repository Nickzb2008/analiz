import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import os
import threading
from datetime import datetime
from utils.data_validator import DataValidator
from utils.file_selector import FileSelector
from utils.model_manager import ModelManager
# Додайте цей імпорт, якщо його немає


class NeuralNetworkTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.model = None
        self.scaler = MinMaxScaler()
        self.current_symbol = "Unknown"  # Додайте цей рядок
        self.model_manager = ModelManager()  # Додаємо менеджер моделей
        self.current_models = []  # Список навчених моделей у поточній сесії
        self.setup_ui()
    
    def setup_ui(self):
        """Налаштування інтерфейсу нейромережі"""
        # Лівий фрейм для налаштувань
        left_frame = ttk.LabelFrame(self.parent, text="Налаштування нейромережі")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Правий фрейм для графіків
        right_frame = ttk.Frame(self.parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Налаштування параметрів
        ttk.Label(left_frame, text="Кількість епох:").pack(pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Entry(left_frame, textvariable=self.epochs_var).pack(pady=2)
        
        ttk.Label(left_frame, text="Розмір батча:").pack(pady=2)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(left_frame, textvariable=self.batch_size_var).pack(pady=2)
        
        ttk.Label(left_frame, text="Розмір вікна (lookback):").pack(pady=2)
        self.lookback_var = tk.IntVar(value=60)
        ttk.Entry(left_frame, textvariable=self.lookback_var).pack(pady=2)
        
        ttk.Label(left_frame, text="Розмір тестової вибірки (%):").pack(pady=2)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(left_frame, textvariable=self.test_size_var).pack(pady=2)
        
        ttk.Label(left_frame, text="Learning Rate:").pack(pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(left_frame, textvariable=self.lr_var).pack(pady=2)
        
        # Кнопки
        ttk.Button(left_frame, text="Навчити модель", 
                  command=self.train_model).pack(pady=10, fill=tk.X)
        ttk.Button(left_frame, text="Прогнозувати", 
                  command=self.predict).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Зберегти модель", 
                  command=self.save_model).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Завантажити модель", 
                  command=self.load_model).pack(pady=5, fill=tk.X)
        
        ttk.Button(left_frame, text="Показати історію навчання", 
                  command=self.show_training_history).pack(pady=5, fill=tk.X)
        
            # Додайте нові кнопки
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)
        
            


        # Графік
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Змінна для зберігання історії навчання
        self.training_history = None
    
    def create_advanced_lstm_model(self, input_shape):
        """Покращена LSTM модель з додатковими параметрами"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape,
                            recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.LSTM(75, return_sequences=True, 
                            recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.LSTM(50, recurrent_dropout=0.2,
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(50, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Правильне використання Huber loss
        huber_loss = tf.keras.losses.Huber(delta=1.0)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_var.get()),
            loss=huber_loss,  # Правильний спосіб
            metrics=['mae', 'mse']
        )
        
        return model
    
    def plot_training_history(self, history):
        """Візуалізація історії навчання"""
        if history is None:
            return
        
        history_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Графік втрат
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Історія втрат моделі')
        ax1.set_xlabel('Епоха')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Графік MAE
        ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title('Історія MAE моделі')
        ax2.set_xlabel('Епоха')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_training_history(self):
        """Показати історію навчання"""
        if self.training_history:
            self.plot_training_history(self.training_history)
        else:
            messagebox.showinfo("Інформація", "Спочатку навчіть модель")
    
    
    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        if self.status_callback:
            self.status_callback(message)
    
    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        if self.progress_callback:
            self.progress_callback(value)
        
    
    def save_model(self):
        """Збереження моделі"""
        if self.model:
            try:
                self.status_callback("Збереження моделі...")
                self.progress_callback(50)
                
                if not os.path.exists('data'):
                    os.makedirs('data')
                self.model.save('data/lstm_model.h5')
                
                self.status_callback("Модель збережена")
                self.progress_callback(100)
                
            except Exception as e:
                self.status_callback("Помилка збереження")
                self.progress_callback(0)
                messagebox.showerror("Помилка", f"Не вдалося зберегти модель: {str(e)}")
    
    def load_model(self):
        """Завантаження моделі"""
        try:
            self.status_callback("Завантаження моделі...")
            self.progress_callback(30)
            
            if os.path.exists('data/lstm_model.h5'):
                self.model = tf.keras.models.load_model('data/lstm_model.h5')
                
                # Спроба визначити символ з назви файлу (якщо збережено)
                # Можна додати логіку для визначення current_symbol
                # Наприклад, зберігати його разом з моделлю
                
                self.status_callback("Модель завантажена")
                self.progress_callback(100)
                
            else:
                self.status_callback("Файл моделі не знайдено")
                self.progress_callback(0)
                messagebox.showwarning("Увага", "Файл моделі не знайдено")
                
        except Exception as e:
            self.status_callback("Помилка завантаження")
            self.progress_callback(0)
            messagebox.showerror("Помилка", f"Не вдалося завантажити модель: {str(e)}")
    
    def prepare_simple_data(self, data, lookback=60):
        """Спрощена та стабільна версія підготовки даних для LSTM"""
        try:
            self.safe_status_callback("Підготовка даних...")
            
            # Перевіряємо наявність даних
            if data is None or data.empty:
                raise ValueError("Немає даних для обробки")
            
            # Використовуємо тільки ціни закриття
            if 'Close' not in data.columns:
                raise ValueError("Відсутня колонка 'Close' в даних")
            
            prices = data[['Close']].values
            
            # Перевіряємо достатню кількість даних
            if len(prices) < lookback + 10:  # +10 для тестових даних
                raise ValueError(f"Замало даних. Потрібно мінімум {lookback + 10} точок, маємо {len(prices)}")
            
            # Нормалізація даних
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(prices)
            
            # Створення послідовностей
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])  # lookback точок
                y.append(scaled_data[i, 0])  # Наступна точка
                
            X = np.array(X)
            y = np.array(y)
            
            # Reshape для LSTM: [samples, timesteps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            self.safe_status_callback(f"Створено {len(X)} послідовностей по {lookback} точок")
            return X, y
            
        except Exception as e:
            self.safe_status_callback(f"Помилка підготовки даних: {str(e)}")
            # Повторно викидаємо помилку для обробки у викликаючому коді
            raise

    def create_lstm_model(self, input_shape):
        """Створення стабільної LSTM моделі для прогнозування цін"""
        try:
            self.safe_status_callback("Створення LSTM моделі...")
            
            model = tf.keras.Sequential([
                # Перший LSTM шар
                tf.keras.layers.LSTM(
                    units=64,
                    return_sequences=True,  # Повертає послідовність для наступного шару
                    input_shape=input_shape,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros'
                ),
                tf.keras.layers.Dropout(0.2),  # Для запобігання перенавчанню
                
                # Другий LSTM шар
                tf.keras.layers.LSTM(
                    units=32,
                    return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal'
                ),
                tf.keras.layers.Dropout(0.2),
                
                # Третій LSTM шар
                tf.keras.layers.LSTM(
                    units=16,
                    return_sequences=False,  # Останній шар не повертає послідовність
                    kernel_initializer='glorot_uniform'
                ),
                tf.keras.layers.Dropout(0.2),
                
                # Повнозв'язні шари
                tf.keras.layers.Dense(
                    units=8,
                    activation='relu',
                    kernel_initializer='he_normal'
                ),
                tf.keras.layers.Dropout(0.1),
                
                # Вихідний шар
                tf.keras.layers.Dense(
                    units=1,  # Один нейрон для регресії
                    activation='linear',
                    kernel_initializer='glorot_uniform'
                )
            ])
            
            # Компіляція моделі
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                ),
                loss='mean_squared_error',  # MSE для регресії
                metrics=[
                    'mae',  # Mean Absolute Error
                    'mse'   # Mean Squared Error
                ]
            )
            
            # Вивід інформації про модель
            #model.summary(print_fn=lambda x: self.safe_status_callback(x))
            
            return model
            
        except Exception as e:
            self.safe_status_callback(f"Помилка створення моделі: {str(e)}")
            raise

    def calculate_rsi(self, prices, period=14):
        """Розрахунок RSI з правильним обчисленням"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        for i in range(period, len(prices)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Розрахунок MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Розрахунок смуг Боллінджера"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_atr(self, high, low, close, period=14):
        """Розрахунок Average True Range"""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_stochastic(self, high, low, close, period=14):
        """Розрахунок Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3).mean()
        return stoch_k, stoch_d

    def calculate_obv(self, close, volume):
        """Розрахунок On Balance Volume"""
        price_change = close.diff()
        obv = volume.copy()
        obv[price_change < 0] = -obv[price_change < 0]
        obv[price_change == 0] = 0
        return obv.cumsum()

    def train_model(self):
        """Навчання нейромережі з вибором файлу"""
        def train_thread():
            try:
                self.safe_status_callback("Перевірка даних...")
                self.progress_callback(10)
                
                # Отримуємо список файлів
                from utils.file_selector import FileSelector
                files = FileSelector.get_sorted_files()
                
                if not files:
                    messagebox.showwarning("Увага", "Спочатку завантажте дані криптовалют")
                    return
                
                # Діалог вибору файлу
                selected_file = FileSelector.ask_user_to_select_file(
                    self.parent, 
                    files,
                    title="Оберіть криптовалюту для навчання",
                    prompt="Оберіть криптовалюту для навчання нейромережі:"
                )
                
                if not selected_file:
                    self.safe_status_callback("Вибір скасовано")
                    self.progress_callback(0)
                    return
                    
                # Зберігаємо символ для подальшого використання
                self.current_symbol = selected_file.replace('_data.csv', '')
                self.safe_status_callback(f"Завантаження {self.current_symbol}...")
                self.progress_callback(30)
                
                data = pd.read_csv(f'data/{selected_file}', index_col=0, parse_dates=True)
                
                # Перевірка даних
                from utils.data_validator import DataValidator
                DataValidator.check_data_requirements(data, self.safe_status_callback)
                
                # Підготовка даних
                lookback = self.lookback_var.get()
                X, y = self.prepare_simple_data(data, lookback)
                
                # Розділення на train/test
                self.safe_status_callback("Розділення даних...")
                self.progress_callback(50)
                
                test_size = int(len(X) * self.test_size_var.get())
                X_train, X_test = X[:-test_size], X[-test_size:]
                y_train, y_test = y[:-test_size], y[-test_size:]
                
                # Створення моделі
                self.model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
                
                # Навчання
                self.safe_status_callback(f"Навчання {self.current_symbol}...")
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs_var.get(),
                    batch_size=self.batch_size_var.get(),
                    validation_data=(X_test, y_test),
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.LambdaCallback(
                            on_epoch_end=lambda epoch, logs: self.progress_callback(
                                60 + (epoch + 1) / self.epochs_var.get() * 35
                            )
                        )
                    ]
                )
                
                # Зберігаємо історію навчання
                self.training_history = history
                
                # Прогнозування та візуалізація
                self.safe_status_callback("Прогнозування...")
                self.progress_callback(95)
                
                predictions = self.model.predict(X_test)
                predictions = self.scaler.inverse_transform(predictions)
                y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Візуалізація
                self.ax.clear()
                self.ax.plot(y_test_actual, label='Фактичні ціни', color='blue', linewidth=2)
                self.ax.plot(predictions, label='Прогноз LSTM', color='red', linewidth=2, alpha=0.8)
                self.ax.set_title(f'Прогноз цін {self.current_symbol} (LSTM)', fontsize=14, fontweight='bold')
                self.ax.set_xlabel('Час')
                self.ax.set_ylabel('Ціна (USD)')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                
                # Метрики
                mse = mean_squared_error(y_test_actual, predictions)
                mae = mean_absolute_error(y_test_actual, predictions)
                
                metrics_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}'
                self.ax.text(0.02, 0.98, metrics_text, transform=self.ax.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                self.canvas.draw()
                
                self.safe_status_callback(f"Навчання {self.current_symbol} завершено! MSE: {mse:.4f}")
                self.progress_callback(100)
                
            except Exception as e:
                error_msg = f"Помилка навчання: {str(e)}"
                self.safe_status_callback(error_msg)
                self.progress_callback(0)
                messagebox.showerror("Помилка", error_msg)
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()

    def predict(self):
        """Прогнозування на майбутнє для навченої криптовалюти"""
        def predict_thread():
            if self.model is None:
                messagebox.showwarning("Увага", "Спочатку навчіть або завантажте модель")
                return
            
            # Перевіряємо, чи є інформація про навчену криптовалюту
            if not hasattr(self, 'current_symbol') or not self.current_symbol:
                messagebox.showwarning("Увага", "Спочатку навчіть модель на конкретній криптовалюті")
                return
            
            try:
                self.safe_status_callback(f"Завантаження даних {self.current_symbol} для прогнозу...")
                self.progress_callback(20)
                
                # Використовуємо той самий файл, на якому навчалися
                selected_file = f"{self.current_symbol}_data.csv"
                file_path = f'data/{selected_file}'
                
                if not os.path.exists(file_path):
                    messagebox.showwarning("Увага", f"Файл даних для {self.current_symbol} не знайдено")
                    self.safe_status_callback(f"Файл {selected_file} не знайдено")
                    return
                
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Підготовка даних для прогнозу
                self.safe_status_callback("Підготовка даних...")
                self.progress_callback(40)
                
                lookback = self.lookback_var.get()
                prices = data[['Close']].values
                
                # Використовуємо той же scaler, що і при навчанні
                scaled_prices = self.scaler.transform(prices)
                
                # Беремо останню послідовність
                last_sequence = scaled_prices[-lookback:]
                
                # Прогноз на 30 днів вперед
                self.safe_status_callback(f"Прогнозування {self.current_symbol} на 30 днів...")
                self.progress_callback(60)
                
                future_predictions = []
                current_sequence = last_sequence.copy()
                
                for i in range(30):
                    self.progress_callback(60 + (i + 1) / 30 * 30)
                    next_pred = self.model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    
                    # Оновлення послідовності
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = next_pred
                
                # Перетворення назад до нормальних значень
                future_predictions = self.scaler.inverse_transform(
                    np.array(future_predictions).reshape(-1, 1)
                )
                
                # Візуалізація прогнозу
                self.ax.clear()
                
                # Історичні дані (останні 100 днів)
                historical_prices = prices[-100:]
                historical_dates = data.index[-100:]
                
                # Майбутні дати для прогнозу
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
                
                self.ax.plot(historical_dates, historical_prices, label='Історичні дані', color='blue', linewidth=2)
                self.ax.plot(future_dates, future_predictions, label='Прогноз на 30 днів', color='red', linewidth=2)
                
                # Додаємо вертикальну лінію для розділення історії та прогнозу
                self.ax.axvline(x=last_date, color='green', linestyle='--', alpha=0.7, label='Початок прогнозу')
                
                self.ax.set_title(f'Прогноз цін {self.current_symbol} на 30 днів', fontsize=14, fontweight='bold')
                self.ax.set_xlabel('Дата')
                self.ax.set_ylabel('Ціна (USD)')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                
                # Форматування дат на осі X
                self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                self.ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
                
                self.canvas.draw()
                
                # Інформація про прогноз
                last_actual_price = prices[-1][0]
                first_predicted_price = future_predictions[0][0]
                last_predicted_price = future_predictions[-1][0]
                
                price_change = last_predicted_price - last_actual_price
                percent_change = (price_change / last_actual_price) * 100
                
                info_text = f"""Прогноз для {self.current_symbol}:
    Поточна ціна: ${last_actual_price:.2f}
    Прогноз через 30 днів: ${last_predicted_price:.2f}
    Зміна: ${price_change:+.2f} ({percent_change:+.2f}%)
    """
                
                self.safe_status_callback(f"Прогноз {self.current_symbol} завершено")
                self.progress_callback(100)
                
                # Показуємо інформацію у вікні
                messagebox.showinfo("Результати прогнозу", info_text)
                
            except Exception as e:
                self.safe_status_callback(f"Помилка прогнозування {self.current_symbol}")
                self.progress_callback(0)
                messagebox.showerror("Помилка", f"Помилка прогнозування: {str(e)}")
        
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()



    def train_multiple_models(self):
        """Навчання моделей для кількох криптовалют"""
        def train_thread():
            try:
                self.safe_status_callback("Пошук файлів даних...")
                self.progress_callback(5)
                
                from utils.file_selector import FileSelector
                files = FileSelector.get_all_files()
                
                if not files:
                    messagebox.showwarning("Увага", "Спочатку завантажте дані криптовалют")
                    return
                
                # Діалог вибору кількох файлів
                selected_files = FileSelector.ask_user_to_select_multiple_files(
                    self.parent, 
                    files,
                    title="Оберіть криптовалюті для навчання",
                    prompt="Оберіть криптовалюті для навчання нейромережі:"
                )
                
                if not selected_files:
                    self.safe_status_callback("Вибір скасовано")
                    self.progress_callback(0)
                    return
                
                self.current_models = []  # Очищаємо список поточних моделей
                
                total_files = len(selected_files)
                for i, selected_file in enumerate(selected_files):
                    symbol = selected_file.replace('_data.csv', '')
                    self.safe_status_callback(f"Обробка {symbol} ({i+1}/{total_files})...")
                    self.progress_callback((i / total_files) * 100)
                    
                    try:
                        # Навчання моделі для кожної валюти
                        success = self.train_single_model(selected_file)
                        if success:
                            self.current_models.append(symbol)
                        
                    except Exception as e:
                        self.safe_status_callback(f"Помилка навчання {symbol}: {str(e)}")
                        continue
                
                # Показуємо результати
                if self.current_models:
                    result_text = f"Успішно навчені моделі для: {', '.join(self.current_models)}"
                    self.safe_status_callback(result_text)
                    messagebox.showinfo("Результат", result_text)
                else:
                    self.safe_status_callback("Не вдалося навчити жодну модель")
                
                self.progress_callback(100)
                
            except Exception as e:
                error_msg = f"Помилка навчання моделей: {str(e)}"
                self.safe_status_callback(error_msg)
                self.progress_callback(0)
                messagebox.showerror("Помилка", error_msg)
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()

    def train_single_model(self, selected_file):
        """Навчання однієї моделі"""
        symbol = selected_file.replace('_data.csv', '')
        
        data = pd.read_csv(f'data/{selected_file}', index_col=0, parse_dates=True)
        
        # Перевірка даних
        from utils.data_validator import DataValidator
        DataValidator.check_data_requirements(data, lambda msg: None)
        
        # Підготовка даних
        lookback = self.lookback_var.get()
        X, y = self.prepare_simple_data(data, lookback)
        
        # Розділення на train/test
        test_size = int(len(X) * self.test_size_var.get())
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Reshape для LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Створення та навчання моделі
        model = self.create_lstm_model((X_train.shape[1], 1))
        
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs_var.get(),
            batch_size=self.batch_size_var.get(),
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Прогнозування та метрики
        predictions = model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        
        # Збереження моделі
        metrics = {'mse': mse, 'mae': mae, 'timestamp': datetime.now()}
        self.model_manager.save_model(symbol, model, self.scaler, metrics)
        
        return True
    



    def show_models(self):
        """Перегляд навчених моделей"""
        self.model_manager.load_all_models()
        available_models = self.model_manager.get_available_models()
        
        if not available_models:
            messagebox.showinfo("Інформація", "Немає навчених моделей")
            return
        
        # Створюємо вікно зі списком моделей
        models_window = tk.Toplevel(self.parent)
        models_window.title("Навчені моделі")
        models_window.geometry("500x400")
        
        ttk.Label(models_window, text="Навчені моделі:", font=('Arial', 12)).pack(pady=10)
        
        tree = ttk.Treeview(models_window, columns=('Symbol', 'MSE', 'MAE', 'Date'), show='headings')
        tree.heading('Symbol', text='Криптовалюта')
        tree.heading('MSE', text='MSE')
        tree.heading('MAE', text='MAE')
        tree.heading('Date', text='Дата навчання')
        
        tree.column('Symbol', width=100)
        tree.column('MSE', width=80)
        tree.column('MAE', width=80)
        tree.column('Date', width=120)
        
        for symbol in available_models:
            metrics = self.model_manager.get_model_metrics(symbol)
            tree.insert('', 'end', values=(
                symbol, 
                f"{metrics.get('mse', 0):.4f}", 
                f"{metrics.get('mae', 0):.4f}",
                metrics.get('timestamp', 'N/A')
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def compare_models(self):
        """Порівняння моделей"""
        comparison_df = self.model_manager.compare_models()
        
        if comparison_df.empty:
            messagebox.showinfo("Інформація", "Немає даних для порівняння")
            return
        
        # Візуалізація порівняння
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Графік MSE
        ax1.bar(comparison_df['symbol'], comparison_df['mse'])
        ax1.set_title('Порівняння MSE')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # Графік MAE
        ax2.bar(comparison_df['symbol'], comparison_df['mae'])
        ax2.set_title('Порівняння MAE')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    

