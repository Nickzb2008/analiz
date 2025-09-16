import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
from utils.data_validator import DataValidator
from utils.file_selector import FileSelector

class TechnicalAnalysisTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.data = None
        self.current_symbol = "Unknown"  # Додайте цей рядок
        self.setup_ui()
    
    def setup_ui(self):
        """Налаштування інтерфейсу технічного аналізу"""
        # Основний фрейм
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Лівий фрейм для індикаторів
        left_frame = ttk.LabelFrame(main_frame, text="Технічні індикатори")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Правий фрейм для графіків
        right_frame = ttk.LabelFrame(main_frame, text="Графіки")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вибір індикаторів
        indicators = [
            "SMA", "EMA", "RSI", "MACD", "Bollinger Bands", 
            "Stochastic Oscillator", "OBV", "ATR"
        ]
        
        self.indicator_vars = {}
        for indicator in indicators:
            var = tk.BooleanVar()
            self.indicator_vars[indicator] = var
            ttk.Checkbutton(left_frame, text=indicator, variable=var).pack(anchor=tk.W, pady=2)
        
        # Періоди для індикаторів
        ttk.Label(left_frame, text="Період (для SMA/EMA/RSI):").pack(pady=(10, 2))
        self.period_var = tk.IntVar(value=14)
        ttk.Entry(left_frame, textvariable=self.period_var).pack(pady=2)
        
        # Кнопки
        ttk.Button(left_frame, text="Завантажити дані", 
                  command=self.load_data).pack(pady=10, fill=tk.X)
        ttk.Button(left_frame, text="Побудувати графіки", 
                  command=self.plot_indicators).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Аналіз сигналів", 
                  command=self.analyze_signals).pack(pady=5, fill=tk.X)
        
        # Графік
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Текстове поле для результатів
        self.result_text = tk.Text(right_frame, height=8)
        self.result_text.pack(fill=tk.X, pady=5)
    
    
    
    def calculate_sma(self, data, period):
        """Розрахунок Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data, period):
        """Розрахунок Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        """Розрахунок Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Розрахунок MACD"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Розрахунок Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, high, low, close, period=14):
        """Розрахунок Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3).mean()
        return stoch_k, stoch_d
    
    def calculate_obv(self, close, volume):
        """Розрахунок On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_atr(self, high, low, close, period=14):
        """Розрахунок Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_indicators(self):
        """Розрахунок технічних індикаторів"""
        if self.data is None:
            messagebox.showwarning("Увага", "Спочатку завантажте дані")
            return None
        
        self.status_callback("Розрахунок індикаторів...")
        self.progress_callback(30)
        
        results = {'price': self.data['Close']}
        period = self.period_var.get()
        
        # SMA
        if self.indicator_vars["SMA"].get():
            results['SMA'] = self.calculate_sma(self.data['Close'], period)
        
        # EMA
        if self.indicator_vars["EMA"].get():
            results['EMA'] = self.calculate_ema(self.data['Close'], period)
        
        # RSI
        if self.indicator_vars["RSI"].get():
            results['RSI'] = self.calculate_rsi(self.data['Close'], period)
        
        # MACD
        if self.indicator_vars["MACD"].get():
            macd, macd_signal, macd_hist = self.calculate_macd(self.data['Close'])
            results['MACD'] = macd
            results['MACD_Signal'] = macd_signal
            results['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        if self.indicator_vars["Bollinger Bands"].get():
            upper, middle, lower = self.calculate_bollinger_bands(self.data['Close'], period)
            results['BB_Upper'] = upper
            results['BB_Middle'] = middle
            results['BB_Lower'] = lower
        
        # Stochastic Oscillator
        if self.indicator_vars["Stochastic Oscillator"].get():
            slowk, slowd = self.calculate_stochastic(
                self.data['High'], self.data['Low'], self.data['Close'], period
            )
            results['Stoch_K'] = slowk
            results['Stoch_D'] = slowd
        
        # OBV
        if self.indicator_vars["OBV"].get():
            results['OBV'] = self.calculate_obv(self.data['Close'], self.data['Volume'])
        
        # ATR
        if self.indicator_vars["ATR"].get():
            results['ATR'] = self.calculate_atr(
                self.data['High'], self.data['Low'], self.data['Close'], period
            )
        
        self.progress_callback(70)
        return results
    
    
    

   



    def load_data(self):
        """Завантаження даних для технічного аналізу з вибором файлу"""
        try:
            self.safe_status_callback("Пошук файлів даних...")
            self.safe_progress_callback(10)
            
            # Отримуємо список файлів БЕЗ сортування
            from utils.file_selector import FileSelector
            files = FileSelector.get_all_files()
            
            if not files:
                messagebox.showwarning("Увага", "Спочатку завантажте дані криптовалют на вкладці 'Завантаження даних'")
                self.safe_status_callback("❌ Файли даних не знайдено")
                self.safe_progress_callback(0)
                return False
            
            # Діалог вибору файлу
            selected_file = FileSelector.ask_user_to_select_file(
                self.parent,
                files,
                title="Оберіть криптовалюту для технічного аналізу",
                prompt="Оберіть криптовалюту для технічного аналізу:"
            )
            
            if not selected_file:
                self.safe_status_callback("Вибір скасовано")
                self.safe_progress_callback(0)
                return False
                
            # Використовуємо саме обраний файл
            self.current_symbol = selected_file.replace('_data.csv', '')
            self.safe_status_callback(f"Завантаження {self.current_symbol}...")
            self.safe_progress_callback(30)
            
            # Завантаження даних
            file_path = f'data/{selected_file}'
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            self.safe_status_callback(f"Завантажено {len(data)} рядків даних")
            self.safe_progress_callback(50)
            
            # Перевірка якості даних для технічного аналізу
            from utils.data_validator import DataValidator
            DataValidator.validate_data_for_technical_analysis(data, self.safe_status_callback)
            
            # Отримуємо інформацію про дані
            available_columns = DataValidator.get_available_columns(data)
            date_range = f"{data.index.min().strftime('%Y-%m-%d')} до {data.index.max().strftime('%Y-%m-%d')}"
            
            self.safe_status_callback(f"Дані за період: {date_range}")
            self.safe_progress_callback(70)
            
            # Зберігаємо дані для подальшого використання
            self.data = data
            
            # Оновлюємо інтерфейс
            self.update_data_info()
            
            self.safe_status_callback(f"✅ Дані {self.current_symbol} готові для технічного аналізу")
            self.safe_progress_callback(100)
            
            # Оновлюємо список доступних індикаторів
            self.update_available_indicators()
            
            return True
            
        except Exception as e:
            self.safe_status_callback("❌ Помилка завантаження даних")
            self.safe_progress_callback(0)
            messagebox.showerror("Помилка", f"Помилка завантаження даних: {str(e)}")
            return False




    
    def update_data_info(self):
        """Оновлення інформації про завантажені дані"""
        if self.data is not None and hasattr(self, 'current_symbol'):
            # Отримуємо базову статистику
            close_prices = self.data['Close']
            price_change = close_prices.iloc[-1] - close_prices.iloc[0]
            percent_change = (price_change / close_prices.iloc[0]) * 100
            
            info_text = f"""
    Символ: {self.current_symbol}
    Період: {self.data.index.min().strftime('%Y-%m-%d')} - {self.data.index.max().strftime('%Y-%m-%d')}
    Кількість днів: {len(self.data)}
    Початкова ціна: ${close_prices.iloc[0]:.2f}
    Кінцева ціна: ${close_prices.iloc[-1]:.2f}
    Зміна: ${price_change:+.2f} ({percent_change:+.2f}%)

    Доступні колонки: {', '.join(self.data.columns)}
            """
            
            # Оновлюємо текстове поле або label
            if hasattr(self, 'info_text'):
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, info_text)
            elif hasattr(self, 'info_label'):
                self.info_label.config(text=info_text)

    def update_available_indicators(self):
        """Оновлення списку доступних індикаторів на основі даних"""
        if self.data is None:
            return
        
        # Вмикаємо тільки ті індикатори, для яких є необхідні дані
        if not all(col in self.data.columns for col in ['High', 'Low', 'Close']):
            # Якщо немає High/Low, вимикаємо відповідні індикатори
            for indicator in ['Bollinger Bands', 'Stochastic Oscillator', 'ATR']:
                if indicator in self.indicator_vars:
                    self.indicator_vars[indicator].set(False)
            
            self.safe_status_callback("Увага: Відсутні High/Low дані, деякі індикатори недоступні")
        
        if 'Volume' not in self.data.columns:
            # Якщо немає Volume, вимикаємо volume-індикатори
            for indicator in ['OBV']:
                if indicator in self.indicator_vars:
                    self.indicator_vars[indicator].set(False)
            
            self.safe_status_callback("Увага: Відсутні Volume дані, OBV недоступний")

    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        if self.status_callback:
            self.status_callback(message)

    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        if self.progress_callback:
            self.progress_callback(value)







    def plot_indicators(self):
        """Побудова графіків індикаторів з назвою криптовалюти"""
        def plot_thread():
            try:
                if self.data is None or not hasattr(self, 'current_symbol'):
                    messagebox.showwarning("Увага", "Спочатку завантажте дані")
                    return
                
                self.safe_status_callback("Побудова графіків...")
                self.safe_progress_callback(20)
                
                indicators = self.calculate_indicators()
                if indicators is None:
                    return
                
                self.safe_status_callback("Візуалізація даних...")
                self.safe_progress_callback(80)
                
                self.ax.clear()
                
                # Основний графік цін
                if 'price' in indicators:
                    self.ax.plot(indicators['price'], label='Ціна закриття', color='black', linewidth=2)
                
                # Додаткові індикатори
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                color_idx = 0
                
                for key, values in indicators.items():
                    if key != 'price':
                        self.ax.plot(values, label=key, color=colors[color_idx % len(colors)], alpha=0.7)
                        color_idx += 1
                
                self.ax.set_title(f'Технічні індикатори - {self.current_symbol}', fontsize=14, fontweight='bold')
                self.ax.set_xlabel('Дата')
                self.ax.set_ylabel('Значення')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                
                # Форматування дат
                self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                self.ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
                
                self.canvas.draw()
                
                self.safe_status_callback(f"Графіки для {self.current_symbol} побудовано")
                self.safe_progress_callback(100)
                
            except Exception as e:
                self.safe_status_callback("Помилка побудови")
                self.safe_progress_callback(0)
                messagebox.showerror("Помилка", f"Помилка побудови графіків: {str(e)}")
        
        thread = threading.Thread(target=plot_thread)
        thread.daemon = True
        thread.start()

    def analyze_signals(self):
        """Аналіз торгових сигналів з назвою криптовалюти"""
        def analyze_thread():
            try:
                if self.data is None or not hasattr(self, 'current_symbol'):
                    messagebox.showwarning("Увага", "Спочатку завантажте дані")
                    return
                
                self.safe_status_callback("Аналіз сигналів...")
                self.safe_progress_callback(30)
                
                indicators = self.calculate_indicators()
                if indicators is None:
                    return
                
                self.safe_status_callback("Обробка сигналів...")
                self.safe_progress_callback(70)
                
                signals = []
                
                # Аналіз RSI
                if 'RSI' in indicators:
                    rsi = indicators['RSI'].dropna()
                    if len(rsi) > 0:
                        last_rsi = rsi.iloc[-1]
                        if last_rsi > 70:
                            signals.append(f"RSI ({last_rsi:.1f}) - Перекупленість ⚠️")
                        elif last_rsi < 30:
                            signals.append(f"RSI ({last_rsi:.1f}) - Перепроданість ⚠️")
                        elif last_rsi > 65:
                            signals.append(f"RSI ({last_rsi:.1f}) - Наближення до перекупленості")
                        elif last_rsi < 35:
                            signals.append(f"RSI ({last_rsi:.1f}) - Наближення до перепродanoсті")
                
                # Аналіз MACD
                if 'MACD' in indicators and 'MACD_Signal' in indicators:
                    macd = indicators['MACD'].dropna()
                    macd_signal = indicators['MACD_Signal'].dropna()
                    if len(macd) > 1 and len(macd_signal) > 1:
                        if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
                            signals.append("MACD - Бичачий перехрест 🐂")
                        elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
                            signals.append("MACD - Ведмежий перехрест 🐻")
                
                # Виведення результатів
                self.result_text.delete(1.0, tk.END)
                
                if signals:
                    result_text = f"Торгові сигнали для {self.current_symbol}:\n\n"
                    result_text += "\n".join([f"• {signal}" for signal in signals])
                    result_text += f"\n\nЗагалом сигналів: {len(signals)}"
                else:
                    result_text = f"Для {self.current_symbol} не виявлено сильних торгових сигналів\n\n"
                    result_text += "Рекомендація: Чекайте на чіткіші сигнали"
                
                self.result_text.insert(tk.END, result_text)
                
                self.safe_status_callback(f"Аналіз {self.current_symbol} завершено")
                self.safe_progress_callback(100)
                
            except Exception as e:
                self.safe_status_callback("Помилка аналізу")
                self.safe_progress_callback(0)
                messagebox.showerror("Помилка", f"Помилка аналізу сигналів: {str(e)}")
        
        thread = threading.Thread(target=analyze_thread)
        thread.daemon = True
        thread.start()



