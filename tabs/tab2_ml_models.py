import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import threading
from utils.data_validator import DataValidator
from utils.file_selector import FileSelector

class MLModelsTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.model = None
        self.scaler = StandardScaler()
        self.current_symbol = "Unknown"  # Додайте цей рядок
        self.setup_ui()
    
    def setup_ui(self):
        """Налаштування інтерфейсу ML моделей"""
        # Основний фрейм
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Лівий фрейм для налаштувань
        left_frame = ttk.LabelFrame(main_frame, text="Вибір моделі та параметрів")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Центральний фрейм для кнопок
        center_frame = ttk.Frame(left_frame)
        center_frame.pack(pady=10, fill=tk.X)
        
        # Правий фрейм для графіків
        right_frame = ttk.LabelFrame(main_frame, text="Результати")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вибір моделі
        ttk.Label(center_frame, text="Оберіть модель:").pack(pady=2)
        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(center_frame, textvariable=self.model_var, 
                                  values=["Random Forest", "Gradient Boosting", "SVR", "Linear Regression"])
        model_combo.pack(pady=2, fill=tk.X)
        
        # Параметри для Random Forest
        self.rf_frame = ttk.LabelFrame(center_frame, text="Параметри Random Forest")
        ttk.Label(self.rf_frame, text="Кількість дерев:").pack(pady=2)
        self.n_estimators_var = tk.IntVar(value=100)
        ttk.Entry(self.rf_frame, textvariable=self.n_estimators_var).pack(pady=2)
        self.rf_frame.pack(pady=5, fill=tk.X)
        
        # Параметри для SVR
        self.svr_frame = ttk.LabelFrame(center_frame, text="Параметри SVR")
        ttk.Label(self.svr_frame, text="Ядро:").pack(pady=2)
        self.kernel_var = tk.StringVar(value="rbf")
        ttk.Combobox(self.svr_frame, textvariable=self.kernel_var, 
                    values=["rbf", "linear", "poly"]).pack(pady=2)
        self.svr_frame.pack_forget()  # Приховати за замовчуванням
        
        # Кнопки
        ttk.Button(center_frame, text="Навчити модель", 
                  command=self.train_model).pack(pady=10, fill=tk.X)
        ttk.Button(center_frame, text="Прогнозувати", 
                  command=self.predict).pack(pady=5, fill=tk.X)
        ttk.Button(center_frame, text="Зберегти модель", 
                  command=self.save_model).pack(pady=5, fill=tk.X)
        ttk.Button(center_frame, text="Завантажити модель", 
                  command=self.load_model).pack(pady=5, fill=tk.X)
        
        # Обробник зміни вибору моделі
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Графік
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Текстове поле для результатів
        self.result_text = tk.Text(right_frame, height=10)
        self.result_text.pack(fill=tk.X, pady=5)
    
    def on_model_change(self, event):
        """Обробник зміни вибору моделі"""
        model_type = self.model_var.get()
        
        # Приховати всі фрейми параметрів
        self.rf_frame.pack_forget()
        self.svr_frame.pack_forget()
        
        # Показати потрібний фрейм
        if model_type == "Random Forest":
            self.rf_frame.pack(pady=5, fill=tk.X)
        elif model_type == "SVR":
            self.svr_frame.pack(pady=5, fill=tk.X)
    
    def prepare_features(self, data, for_prediction=False):
        """
        Підготовка ознак для ML моделей
        
        Args:
            data: DataFrame з даними
            for_prediction: Чи використовується для прогнозу (без цільової змінної)
        """
        self.safe_status_callback("Підготовка ознак...")
        
        # Створюємо копію даних
        df = data.copy()
        
        # Базові технічні індикатори
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['Std_5'] = df['Close'].rolling(window=5).std()
        df['Std_20'] = df['Close'].rolling(window=20).std()
        
        # Додаємо volume indicators якщо доступні
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Цільова змінна - ціна закриття наступного дня (тільки для навчання)
        if not for_prediction:
            df['Target'] = df['Close'].shift(-1)
        
        # Видаляємо NaN значення
        df = df.dropna()
        
        if for_prediction:
            # Для прогнозу повертаємо тільки ознаки
            feature_columns = ['Close', 'Returns', 'Price_Change', 'MA_5', 'MA_20', 'MA_50', 'Std_5', 'Std_20']
            if 'Volume' in df.columns:
                feature_columns.extend(['Volume_Change', 'Volume_MA_20'])
            
            # Фільтруємо тільки існуючі колонки
            existing_columns = [col for col in feature_columns if col in df.columns]
            return df[existing_columns]
        else:
            # Для навчання повертаємо ознаки та цільову змінну
            feature_columns = ['Close', 'Returns', 'Price_Change', 'MA_5', 'MA_20', 'MA_50', 'Std_5', 'Std_20']
            if 'Volume' in df.columns:
                feature_columns.extend(['Volume_Change', 'Volume_MA_20'])
            
            existing_columns = [col for col in feature_columns if col in df.columns]
            X = df[existing_columns]
            y = df['Target']
            
            return X, y

    def train_model(self):
        """Навчання ML моделі з вибором файлу"""
        def train_thread():
            try:
                self.safe_status_callback("Пошук файлів даних...")
                self.safe_progress_callback(10)
                
                # Отримуємо список файлів БЕЗ сортування
                from utils.file_selector import FileSelector
                files = FileSelector.get_all_files()
                
                if not files:
                    messagebox.showwarning("Увага", "Спочатку завантажте дані криптовалют на вкладці 'Завантаження даних'")
                    return
                
                # Діалог вибору файлу
                selected_file = FileSelector.ask_user_to_select_file(
                    self.parent, 
                    files,
                    title="Оберіть криптовалюту для ML аналізу",
                    prompt="Оберіть криптовалюту для навчання ML моделі:"
                )
                
                if not selected_file:
                    self.safe_status_callback("Вибір скасовано")
                    self.safe_progress_callback(0)
                    return
                    
                # Використовуємо саме обраний файл
                self.current_symbol = selected_file.replace('_data.csv', '')
                self.safe_status_callback(f"Завантаження {self.current_symbol}...")
                self.safe_progress_callback(30)
                
                # Завантаження даних
                data = pd.read_csv(f'data/{selected_file}', index_col=0, parse_dates=True)
                
                # Перевірка даних
                from utils.data_validator import DataValidator
                DataValidator.validate_data_for_ml(data, self.safe_status_callback)
                
                # Підготовка ознак
                self.safe_status_callback("Підготовка ознак...")
                self.safe_progress_callback(50)
                X, y = self.prepare_features(data)
                
                # Розділення на train/test
                self.safe_status_callback("Розділення даних...")
                self.safe_progress_callback(60)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Масштабування ознак
                self.safe_status_callback("Масштабування ознак...")
                self.safe_progress_callback(70)
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Вибір та навчання моделі
                model_type = self.model_var.get()
                
                self.safe_status_callback(f"Створення {model_type} моделі...")
                self.safe_progress_callback(80)
                
                if model_type == "Random Forest":
                    self.model = RandomForestRegressor(
                        n_estimators=self.n_estimators_var.get(),
                        random_state=42
                    )
                elif model_type == "Gradient Boosting":
                    self.model = GradientBoostingRegressor(random_state=42)
                elif model_type == "SVR":
                    self.model = SVR(kernel=self.kernel_var.get())
                elif model_type == "Linear Regression":
                    self.model = LinearRegression()
                
                # Навчання моделі
                self.safe_status_callback(f"Навчання {model_type}...")
                self.safe_progress_callback(90)
                self.model.fit(X_train_scaled, y_train)
                
                # Прогнозування
                self.safe_status_callback("Прогнозування...")
                self.safe_progress_callback(95)
                y_pred = self.model.predict(X_test_scaled)
                
                # Розрахунок метрик
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Візуалізація
                self.ax.clear()
                self.ax.plot(y_test.values, label='Фактичні ціни', color='blue', alpha=0.7, linewidth=2)
                self.ax.plot(y_pred, label='Прогноз', color='red', alpha=0.7, linewidth=2)
                self.ax.set_title(f'Прогнозування цін {self.current_symbol} ({model_type})', fontsize=14, fontweight='bold')
                self.ax.set_xlabel('Час')
                self.ax.set_ylabel('Ціна (USD)')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                self.canvas.draw()
                
                # Виведення результатів
                result_text = f"""Результати навчання ({model_type}) для {self.current_symbol}:
    MSE: {mse:.4f}
    MAE: {mae:.4f}
    R²: {r2:.4f}

    Фактична ціна: {y_test.iloc[-1]:.2f}
    Прогнозована ціна: {y_pred[-1]:.2f}
    Похибка: {abs(y_test.iloc[-1] - y_pred[-1]):.2f}
    """
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, result_text)
                
                self.safe_status_callback(f"Навчання {model_type} для {self.current_symbol} завершено. R²: {r2:.4f}")
                self.safe_progress_callback(100)
                
            except Exception as e:
                error_msg = f"Помилка навчання ML моделі: {str(e)}"
                self.safe_status_callback(error_msg)
                self.safe_progress_callback(0)
                messagebox.showerror("Помилка", error_msg)
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
    
    def predict(self):
        """Прогнозування на майбутнє з вибором файлу"""
        def predict_thread():
            if self.model is None:
                messagebox.showwarning("Увага", "Спочатку навчіть модель")
                return
            
            try:
                self.safe_status_callback("Пошук файлів даних...")
                self.safe_progress_callback(10)
                
                # Отримуємо список файлів
                from utils.file_selector import FileSelector
                files = FileSelector.get_all_files()
                
                if not files:
                    messagebox.showwarning("Увага", "Немає даних для прогнозу")
                    return
                
                # Використовуємо той самий файл, що і для навчання, або даємо вибір
                if hasattr(self, 'current_symbol') and self.current_symbol != "Unknown":
                    selected_file = f"{self.current_symbol}_data.csv"
                    if selected_file not in files:
                        # Якщо файл не знайдено, пропонуємо вибрати
                        selected_file = FileSelector.ask_user_to_select_file(
                            self.parent, 
                            files,
                            title="Оберіть дані для прогнозу",
                            prompt="Файл для навченої моделі не знайдено. Оберіть інший файл:"
                        )
                else:
                    # Якщо модель навчена без вибору файлу, пропонуємо вибрати
                    selected_file = FileSelector.ask_user_to_select_file(
                        self.parent, 
                        files,
                        title="Оберіть дані для прогнозу",
                        prompt="Оберіть дані для прогнозу ML моделлю:"
                    )
                
                if not selected_file:
                    self.safe_status_callback("Вибір скасовано")
                    self.safe_progress_callback(0)
                    return
                    
                symbol = selected_file.replace('_data.csv', '')
                self.safe_status_callback(f"Завантаження {symbol}...")
                self.safe_progress_callback(30)
                
                # Завантаження даних
                data = pd.read_csv(f'data/{selected_file}', index_col=0, parse_dates=True)
                
                # Перевірка даних
                from utils.data_validator import DataValidator
                DataValidator.validate_data_for_ml(data, self.safe_status_callback)
                
                # Підготовка ознак для прогнозу
                self.safe_status_callback("Підготовка ознак...")
                self.safe_progress_callback(50)
                
                # Використовуємо ту саму функцію підготовки, що і для навчання
                df_prepared = self.prepare_features(data, for_prediction=True)
                X = df_prepared.drop('Target', axis=1, errors='ignore')
                
                # Масштабування ознак
                X_scaled = self.scaler.transform(X)
                
                # Прогноз
                self.safe_status_callback("Прогнозування...")
                self.safe_progress_callback(70)
                predictions = self.model.predict(X_scaled)
                
                # Додавання прогнозу до даних
                data['Prediction'] = np.nan
                # Прогноз починається з другого дня (бо ми прогнозуємо наступний день)
                data.iloc[1:len(predictions)+1, data.columns.get_loc('Prediction')] = predictions
                
                # Візуалізація
                self.ax.clear()
                
                # Останні 100 днів історії + прогноз
                plot_data = data.tail(100)
                
                self.ax.plot(plot_data.index, plot_data['Close'], label='Фактичні ціни', color='blue', linewidth=2)
                self.ax.plot(plot_data.index, plot_data['Prediction'], label='Прогноз ML', color='red', linewidth=2, alpha=0.8)
                
                self.ax.set_title(f'Прогноз цін {symbol} (ML)', fontsize=14, fontweight='bold')
                self.ax.set_xlabel('Дата')
                self.ax.set_ylabel('Ціна (USD)')
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
                
                # Форматування дат
                self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                self.ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
                
                self.canvas.draw()
                
                # Останній прогноз
                last_prediction = predictions[-1] if len(predictions) > 0 else 0
                last_actual = data['Close'].iloc[-1]
                
                # Знаходимо останній дійсний прогноз (не NaN)
                valid_predictions = data['Prediction'].dropna()
                if not valid_predictions.empty:
                    last_valid_prediction = valid_predictions.iloc[-1]
                    last_valid_actual = data.loc[valid_predictions.index[-1], 'Close']
                    
                    price_change = last_valid_prediction - last_valid_actual
                    percent_change = (price_change / last_valid_actual) * 100
                    
                    result_text = f"""Прогноз для {symbol}:
    Останній прогноз: ${last_valid_prediction:.2f}
    Фактична ціна: ${last_valid_actual:.2f}
    Зміна: ${price_change:+.2f} ({percent_change:+.2f}%)

    Точність моделі:
    - Прогнозує ціну на наступний день
    - На основі технічних індикаторів
    """
                else:
                    result_text = f"""Не вдалося отримати прогноз для {symbol}
    Модель потребує більше даних для прогнозування"""
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, result_text)
                
                self.safe_status_callback(f"Прогноз для {symbol} завершено")
                self.safe_progress_callback(100)
                
            except Exception as e:
                error_msg = f"Помилка прогнозування: {str(e)}"
                self.safe_status_callback(error_msg)
                self.safe_progress_callback(0)
                messagebox.showerror("Помилка", error_msg)
        
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()


    def save_model(self):
        """Збереження моделі"""
        if self.model:
            try:
                self.status_callback("Збереження моделі...")
                self.progress_callback(50)
                
                if not os.path.exists('data'):
                    os.makedirs('data')
                
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'model_type': self.model_var.get()
                }
                
                with open('data/ml_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                
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
            
            if os.path.exists('data/ml_model.pkl'):
                with open('data/ml_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.model_var.set(model_data['model_type'])
                
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



    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        if self.status_callback:
            self.status_callback(message)

    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        if self.progress_callback:
            self.progress_callback(value)














