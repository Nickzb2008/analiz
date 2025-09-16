import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
import os
import threading
from datetime import datetime, timedelta
import json
from utils.model_manager import ModelManager
from utils.file_selector import FileSelector
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class TrainedNetworksGraphsTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.model_manager = ModelManager()
        self.current_symbol = None
        self.predictions = {}
        self.setup_ui()
        self.refresh_models()
    
    def setup_ui(self):
        """Налаштування інтерфейсу графіків навчених нейромереж"""
        # Основні фрейми
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Лівий фрейм - кнопки управління
        left_frame = ttk.LabelFrame(main_frame, text="Управління", width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        # Центральний фрейм - список моделей з прокрутками
        center_frame = ttk.LabelFrame(main_frame, text="Навчені моделі", width=300)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        center_frame.pack_propagate(False)
        
        # Правий фрейм - графіки
        right_frame = ttk.LabelFrame(main_frame, text="Графіки")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Кнопки управління
        ttk.Button(left_frame, text="Оновити моделі", command=self.refresh_models).pack(pady=10, fill=tk.X)
        ttk.Button(left_frame, text="Прогнозувати обрану", command=self.predict_selected).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Прогноз для всіх", command=self.predict_all_dialog).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Видалити модель", command=self.delete_selected_model).pack(pady=5, fill=tk.X)
        ttk.Button(left_frame, text="Експорт результатів", command=self.export_results).pack(pady=10, fill=tk.X)
        
        # Інформація про моделі
        ttk.Label(left_frame, text="Загальна інформація:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        self.models_count_var = tk.StringVar(value="Моделей: 0")
        ttk.Label(left_frame, textvariable=self.models_count_var).pack(pady=2)
        
        self.last_update_var = tk.StringVar(value="Оновлено: -")
        ttk.Label(left_frame, textvariable=self.last_update_var).pack(pady=2)
        
        # Прогресбар для відображення прогресу прогнозування
        ttk.Label(left_frame, text="Прогрес прогнозування:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        self.progress_bar = ttk.Progressbar(left_frame, mode='determinate')
        self.progress_bar.pack(pady=5, fill=tk.X, padx=10)
        
        # Список моделей з горизонтальною та вертикальною прокруткою
        tree_frame = ttk.Frame(center_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Створюємо Treeview з горизонтальною прокруткою
        self.models_tree = ttk.Treeview(tree_frame, columns=('Symbol', 'MSE', 'MAE', 'R2', 'Date', 'Type'), 
                                    show='headings', height=15, selectmode='browse')
        
        # Налаштування колонок
        self.models_tree.heading('Symbol', text='Криптовалюта')
        self.models_tree.heading('MSE', text='MSE')
        self.models_tree.heading('MAE', text='MAE')
        self.models_tree.heading('R2', text='R²')
        self.models_tree.heading('Date', text='Дата навчання')
        self.models_tree.heading('Type', text='Тип')
        
        self.models_tree.column('Symbol', width=100, minwidth=80)
        self.models_tree.column('MSE', width=80, minwidth=60)
        self.models_tree.column('MAE', width=80, minwidth=60)
        self.models_tree.column('R2', width=60, minwidth=50)
        self.models_tree.column('Date', width=120, minwidth=100)
        self.models_tree.column('Type', width=80, minwidth=60)
        
        # Вертикальна прокрутка
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Горизонтальна прокрутка
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.models_tree.xview)
        self.models_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Розміщення елементів
        self.models_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Налаштування розтягування
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Прокрутка мишею
        self.models_tree.bind("<Enter>", lambda e: self.bind_mouse_scroll(self.models_tree))
        self.models_tree.bind("<Leave>", lambda e: self.unbind_mouse_scroll(self.models_tree))
        
        self.models_tree.bind('<<TreeviewSelect>>', self.on_model_select)
        
        # Контекстне меню для моделей
        self.context_menu = tk.Menu(self.models_tree, tearoff=0)
        self.context_menu.add_command(label="Прогнозувати", command=self.predict_selected)
        self.context_menu.add_command(label="Видалити модель", command=self.delete_selected_model)
        self.context_menu.add_command(label="Переглянути деталі", command=self.show_model_details)
        
        self.models_tree.bind("<Button-3>", self.show_context_menu)
        
        # Вкладки для графіків
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладки
        self.tab_main = ttk.Frame(self.notebook)
        self.tab_10d = ttk.Frame(self.notebook)
        self.tab_20d = ttk.Frame(self.notebook)
        self.tab_30d = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_main, text="Основний графік")
        self.notebook.add(self.tab_10d, text="10 днів")
        self.notebook.add(self.tab_20d, text="20 днів")
        self.notebook.add(self.tab_30d, text="30 днів")
        
        # Графіки для кожної вкладки
        self.fig_main, self.ax_main = plt.subplots(figsize=(10, 6))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, self.tab_main)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_10d, self.ax_10d = plt.subplots(figsize=(10, 6))
        self.canvas_10d = FigureCanvasTkAgg(self.fig_10d, self.tab_10d)
        self.canvas_10d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_20d, self.ax_20d = plt.subplots(figsize=(10, 6))
        self.canvas_20d = FigureCanvasTkAgg(self.fig_20d, self.tab_20d)
        self.canvas_20d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_30d, self.ax_30d = plt.subplots(figsize=(10, 6))
        self.canvas_30d = FigureCanvasTkAgg(self.fig_30d, self.tab_30d)
        self.canvas_30d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_progress(self, value):
        """Оновлення прогресбару"""
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            self.progress_bar['value'] = value
            self.parent.update_idletasks()

    def bind_mouse_scroll(self, widget):
        """Прив'язка прокрутки мишею"""
        widget.bind_all("<MouseWheel>", lambda e: widget.yview_scroll(int(-1*(e.delta/120)), "units"))
        widget.bind_all("<Shift-MouseWheel>", lambda e: widget.xview_scroll(int(-1*(e.delta/120)), "units"))
    
    def unbind_mouse_scroll(self, widget):
        """Відв'язка прокрутки мишею"""
        widget.unbind_all("<MouseWheel>")
        widget.unbind_all("<Shift-MouseWheel>")
    
    def refresh_models(self):
        """Оновлення списку моделей"""
        try:
            self.status_callback("Оновлення списку моделей...")
            self.models_tree.delete(*self.models_tree.get_children())
            
            # Очищаємо кеш менеджера моделей
            self.model_manager.models.clear()
            self.model_manager.metrics.clear()
            
            # Отримуємо доступні моделі безпосередньо з папки
            available_models = self.get_available_models_from_folder()
            
            if not available_models:
                self.status_callback("Навчені моделі не знайдені")
                self.models_count_var.set("Моделей: 0")
                self.last_update_var.set("Оновлено: " + datetime.now().strftime('%H:%M:%S'))
                return
            
            # Завантажуємо метрики для кожної моделі
            loaded_count = 0
            for symbol in available_models:
                try:
                    # Завантажуємо метрики з JSON файлу
                    metrics_path = os.path.join('models', f'{symbol}_metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        
                        # Отримуємо тип навчання і перекладаємо його
                        training_type = metrics.get('training_type', 'basic')
                        training_type_display = self.get_training_type_display(training_type)
                        
                        # Конвертуємо рядок дати назад у datetime, якщо потрібно
                        timestamp = metrics.get('timestamp')
                        if isinstance(timestamp, str):
                            try:
                                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            except:
                                timestamp = timestamp  # Залишаємо як рядок, якщо не вдається конвертувати
                        
                        # Додаємо до Treeview
                        date_str = timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
                        
                        self.models_tree.insert('', 'end', values=(
                            symbol,  # Тільки назва моделі без розширення
                            f"{metrics.get('mse', 0):.6f}",
                            f"{metrics.get('mae', 0):.6f}",
                            f"{metrics.get('r2', 0):.4f}",
                            date_str,
                            training_type_display  # Відображуваний тип навчання
                        ), tags=(symbol,))
                        
                        # Зберігаємо метрики в менеджері
                        self.model_manager.metrics[symbol] = metrics
                        loaded_count += 1
                    
                except Exception as e:
                    self.status_callback(f"Помилка завантаження метрик {symbol}: {str(e)}")
                    continue
            
            self.models_count_var.set(f"Моделей: {loaded_count}")
            self.last_update_var.set("Оновлено: " + datetime.now().strftime('%H:%M:%S'))
            self.status_callback(f"Знайдено {loaded_count} навчених моделей")
            
        except Exception as e:
            self.status_callback(f"Помилка оновлення моделей: {str(e)}")
            messagebox.showerror("Помилка", f"Не вдалося оновити список моделей: {str(e)}")

    def get_available_models_from_folder(self):
        """Отримання списку доступних моделей безпосередньо з папки"""
        models = []
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
            return models
        
        try:
            # Шукаємо файли моделей у папці models (тільки .h5)
            for file in os.listdir('models'):
                if file.endswith('_model.h5'):
                    symbol = file.replace('_model.h5', '')  # Видаляємо розширення
                    models.append(symbol)
            
            return sorted(models)
        except Exception as e:
            self.status_callback(f"Помилка читання папки models: {str(e)}")
            return []
    
    def show_context_menu(self, event):
        """Показати контекстне меню"""
        item = self.models_tree.identify_row(event.y)
        if item:
            self.models_tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def delete_selected_model(self):
        """Видалення обраної моделі"""
        selected = self.models_tree.selection()
        if not selected:
            messagebox.showwarning("Увага", "Оберіть модель для видалення")
            return
        
        symbol = self.models_tree.item(selected[0], 'values')[0]
        
        if messagebox.askyesno("Підтвердження", f"Видалити модель {symbol}?\n\nЦя дія незворотна!"):
            try:
                success = self.model_manager.delete_model(symbol)
                if success:
                    self.status_callback(f"Модель {symbol} видалена")
                    self.refresh_models()
                    messagebox.showinfo("Успіх", f"Модель {symbol} успішно видалена")
                else:
                    self.status_callback(f"Помилка видалення моделі {symbol}")
                    messagebox.showerror("Помилка", f"Не вдалося видалити модель {symbol}")
            except Exception as e:
                error_msg = f"Помилка видалення: {str(e)}"
                self.status_callback(error_msg)
                messagebox.showerror("Помилка", error_msg)
    
    def show_model_details(self):
        """Перегляд деталей моделі"""
        selected = self.models_tree.selection()
        if not selected:
            return
        
        symbol = self.models_tree.item(selected[0], 'values')[0]
        metrics = self.model_manager.get_model_metrics(symbol)
        
        # Отримуємо відображуване значення типу навчання
        training_type = metrics.get('training_type', 'basic')
        training_type_display = self.get_training_type_display(training_type)
        
        details_window = tk.Toplevel(self.parent)
        details_window.title(f"Деталі моделі {symbol}")
        details_window.geometry("500x400")
        
        text_widget = tk.Text(details_window, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(details_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        details_text = f"""ДЕТАЛІ МОДЕЛІ: {symbol}
    ========================

    ЗАГАЛЬНА ІНФОРМАЦІЯ:
    -------------------
    Дата навчання: {metrics.get('timestamp', 'Невідомо')}
    Тип навчання: {training_type_display}
    Кількість ознак: {metrics.get('feature_count', 'Невідомо')}
    Кількість зразків: {metrics.get('samples_count', 'Невідомо')}

    МЕТРИКИ ЯКОСТІ:
    --------------
    MSE: {metrics.get('mse', 0):.6f}
    MAE: {metrics.get('mae', 0):.6f}
    R²: {metrics.get('r2', 0):.4f}
    Найкраща епоха: {metrics.get('best_epoch', 'Невідомо')}

    ФАЙЛИ МОДЕЛІ:
    ------------
    Модель: models/{symbol}_model.h5
    Метрики: models/{symbol}_metrics.json

    ВИКОРИСТОВУВАНІ ОЗНАКИ:
    ----------------------
    {', '.join(metrics.get('features', ['Невідомо']))}
    """
        
        text_widget.insert(tk.END, details_text)
        text_widget.config(state=tk.DISABLED)

    def on_model_select(self, event):
        """Обробник вибору моделі"""
        selected = self.models_tree.selection()
        if not selected:
            return
        
        symbol = self.models_tree.item(selected[0], 'values')[0]
        self.current_symbol = symbol
        self.status_callback(f"Обрано: {symbol}")
        
        # Оновлення основного графіка
        self.update_main_graph(symbol)
    
    def update_main_graph(self, symbol):
        """Оновлення основного графіка"""
        try:
            # Завантаження даних
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                self.status_callback(f"Файл даних {file_path} не знайдено")
                return
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Відображення графіка
            self.ax_main.clear()
            self.ax_main.plot(data.index, data['Close'], label='Фактичні ціни', linewidth=2, color='blue')
            
            # Додаємо середні ковзні
            if len(data) > 20:
                data['MA20'] = data['Close'].rolling(window=20).mean()
                self.ax_main.plot(data.index, data['MA20'], label='MA20', linewidth=1, color='orange', alpha=0.7)
            
            if len(data) > 50:
                data['MA50'] = data['Close'].rolling(window=50).mean()
                self.ax_main.plot(data.index, data['MA50'], label='MA50', linewidth=1, color='red', alpha=0.7)
            
            self.ax_main.set_title(f'Графік цін {symbol}', fontsize=14, fontweight='bold')
            self.ax_main.set_xlabel('Дата')
            self.ax_main.set_ylabel('Ціна (USD)')
            self.ax_main.legend()
            self.ax_main.grid(True, alpha=0.3)
            
            # Форматування дат
            self.ax_main.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            self.ax_main.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.setp(self.ax_main.xaxis.get_majorticklabels(), rotation=45)
            
            self.canvas_main.draw()
            
        except Exception as e:
            self.status_callback(f"Помилка оновлення графіка: {str(e)}")
    
    def predict_selected(self):
        """Прогнозування для обраної моделі"""
        if not self.current_symbol:
            messagebox.showwarning("Увага", "Оберіть модель зі списку")
            return
        
        self.predict_for_symbol(self.current_symbol)
    
    def predict_all_dialog(self):
        """Діалог прогнозування для всіх моделей"""
        available_models = self.get_available_models_from_folder()
        if not available_models:
            messagebox.showwarning("Увага", "Немає навчених моделей")
            return
        
        # Використовуємо FileSelector для вибору моделей
        selected_models = FileSelector.ask_user_to_select_multiple_files(
            self.parent,
            available_models,
            title="Оберіть моделі для прогнозування",
            prompt="Оберіть моделі для прогнозування:"
        )
        
        if selected_models:
            self.predict_multiple(selected_models)
    
    def predict_multiple(self, symbols):
        """Прогнозування для кількох моделей"""
        def predict_thread():
            try:
                total_symbols = len(symbols)
                self.status_callback(f"Прогнозування для {total_symbols} моделей...")
                self.update_progress(0)
                
                for i, symbol in enumerate(symbols):
                    if self.current_symbol != symbol:
                        continue  # Пропускаємо, якщо не обрано
                    
                    progress_value = (i / total_symbols) * 100
                    self.update_progress(progress_value)
                    self.status_callback(f"Прогнозування {symbol} ({i+1}/{total_symbols})")
                    
                    try:
                        self.predict_for_symbol(symbol)
                        # Невелика затримка для оновлення UI
                        import time
                        time.sleep(0.1)
                    except Exception as e:
                        self.status_callback(f"Помилка прогнозування {symbol}: {str(e)}")
                        continue
                
                self.status_callback("Прогнозування завершено")
                self.update_progress(100)
                
            except Exception as e:
                self.status_callback(f"Помилка прогнозування: {str(e)}")
                self.update_progress(0)
        
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()

    def predict_for_symbol(self, symbol):
        """Прогнозування для конкретного символу з урахуванням типу навчання"""
        try:
            # Оновлюємо прогресбар на початку
            self.update_progress(10)
            
            # Завантаження моделі БЕЗПЕЧНИМ методом (без метрик)
            model = self.model_manager.load_model_safe(symbol)
            if model is None:
                self.status_callback(f"Модель {symbol} не знайдена або пошкоджена")
                self.update_progress(0)
                return
            
            self.update_progress(20)
            
            # Завантаження метрик для отримання інформації про тип навчання
            metrics = self.model_manager.get_model_metrics(symbol)
            training_type = metrics.get('training_type', 'basic')
            
            self.update_progress(30)
            
            # Завантаження даних
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                self.status_callback(f"Дані для {symbol} не знайдені")
                self.update_progress(0)
                return
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            self.update_progress(40)
            
            # Підготовка даних для прогнозу в залежності від типу навчання
            if training_type == 'basic':
                predictions = self._predict_basic(model, data, symbol)
            elif training_type == 'advanced':
                predictions = self._predict_advanced(model, data, symbol)
            else:  # expert
                predictions = self._predict_expert(model, data, symbol)
            
            self.update_progress(80)
            
            if predictions:
                self.predictions[symbol] = predictions
                # Оновлення графіків
                self.update_prediction_graphs(symbol, data, predictions)
                self.status_callback(f"Прогноз для {symbol} завершено")
                self.update_progress(100)
            else:
                self.status_callback(f"Не вдалося зробити прогноз для {symbol}")
                self.update_progress(0)
            
        except Exception as e:
            self.status_callback(f"Помилка прогнозування {symbol}: {str(e)}")
            self.update_progress(0)
            import traceback
            traceback.print_exc()

    def _predict_basic(self, model, data, symbol):
        """Прогнозування для базового типу навчання"""
        try:
            # Базове навчання - тільки ціни закриття
            prices = data[['Close']].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(prices)
            lookback = 60
            
            if len(scaled_data) < lookback:
                self.status_callback(f"Замало даних для прогнозу ({len(scaled_data)} < {lookback})")
                return None
            
            last_sequence = scaled_data[-lookback:]
            
            # Прогноз на різні періоди
            horizons = [10, 20, 30]
            predictions = {}
            
            total_steps = len(horizons) * max(horizons)
            current_step = 0
            
            for horizon in horizons:
                future_predictions = []
                current_sequence = last_sequence.copy()
                
                for i in range(horizon):
                    # Оновлюємо прогрес для кожного кроку прогнозу
                    current_step += 1
                    step_progress = 40 + (current_step / total_steps) * 40
                    self.update_progress(step_progress)
                    
                    # Прогнозуємо наступне значення
                    next_pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    
                    # Оновлення послідовності
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = next_pred
                
                # Зворотнє перетворення
                future_predictions = scaler.inverse_transform(
                    np.array(future_predictions).reshape(-1, 1)
                )
                predictions[horizon] = future_predictions.flatten()
            
            return predictions
            
        except Exception as e:
            self.status_callback(f"Помилка базового прогнозу для {symbol}: {str(e)}")
            return None

    def _predict_advanced(self, model, data, symbol):
        """Прогнозування для розширеного типу навчання"""
        try:
            from utils.data_processor import DataProcessor
            processor = DataProcessor()
            
            # Підготовка даних як при навчанні
            df = processor.prepare_features_for_ml(data)
            features = ['Close', 'Returns', 'MA_5', 'MA_20', 'Volatility']
            available_features = [f for f in features if f in df.columns]
            feature_data = df[available_features].values
            
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_data)
            lookback = 60
            feature_count = len(available_features)
            
            if len(scaled_data) < lookback:
                self.status_callback(f"Замало даних для прогнозу ({len(scaled_data)} < {lookback})")
                return None
            
            last_sequence = scaled_data[-lookback:]
            
            # Прогноз на різні періоди
            horizons = [10, 20, 30]
            predictions = {}
            
            total_steps = len(horizons) * max(horizons)
            current_step = 0
            
            for horizon in horizons:
                future_predictions = []
                current_sequence = last_sequence.copy()
                
                for i in range(horizon):
                    # Оновлюємо прогрес для кожного кроку прогнозу
                    current_step += 1
                    step_progress = 40 + (current_step / total_steps) * 40
                    self.update_progress(step_progress)
                    
                    # Прогнозуємо наступне значення
                    next_pred = model.predict(current_sequence.reshape(1, lookback, feature_count), verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    
                    # Оновлення послідовності - створюємо новий вектор
                    current_sequence = np.roll(current_sequence, -1, axis=0)
                    
                    # Для прогнозування майбутніх значень створюємо реалістичний вектор
                    new_vector = current_sequence[-1].copy()
                    new_vector[0] = next_pred[0, 0]  # Оновлюємо тільки ціну закриття
                    
                    # Оновлюємо інші ознаки на основі нової ціни
                    if 'Returns' in available_features:
                        prev_close = current_sequence[-2, 0] if len(current_sequence) > 1 else current_sequence[-1, 0]
                        new_vector[available_features.index('Returns')] = (next_pred[0, 0] - prev_close) / prev_close
                    
                    current_sequence[-1] = new_vector
                
                # Зворотнє перетворення - створюємо тимчасові дані
                temp_data = np.zeros((len(future_predictions), feature_count))
                temp_data[:, 0] = future_predictions  # Тільки прогнозовані ціни
                future_predictions = scaler.inverse_transform(temp_data)[:, 0]
                
                predictions[horizon] = future_predictions.flatten()
            
            return predictions
            
        except Exception as e:
            self.status_callback(f"Помилка розширеного прогнозу для {symbol}: {str(e)}")
            return None
    
    def _predict_expert(self, model, data, symbol):
        """Прогнозування для експертного типу навчання"""
        try:
            # Спочатку отримуємо інформацію про модель
            metrics = self.model_manager.get_model_metrics(symbol)
            feature_count = metrics.get('feature_count', 11)
            
            # Використовуємо тільки базові індикатори
            df = data.copy()
            
            if 'Close' in df.columns:
                # Тільки основні індикатори
                df['Returns'] = df['Close'].pct_change()
                df['MA_5'] = df['Close'].rolling(window=5).mean()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility'] = df['Close'].rolling(window=20).std()
                
                # Часові ознаки
                df['Day_of_Week'] = df.index.dayofweek
                df['Month'] = df.index.month
            
            # Видаляємо NaN
            df = df.dropna()
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Обмежуємо кількість ознак
            if len(numeric_columns) > feature_count:
                numeric_columns = numeric_columns[:feature_count]
            
            feature_data = df[numeric_columns].values
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            lookback = 60
            
            if len(scaled_data) < lookback:
                self.status_callback(f"Замало даних для прогнозу ({len(scaled_data)} < {lookback})")
                return None
            
            last_sequence = scaled_data[-lookback:]
            
            # Перевіряємо розмірність
            if last_sequence.shape[1] != feature_count:
                self.status_callback(f"Невідповідність розмірності: модель очікує {feature_count} ознак, дані мають {last_sequence.shape[1]}")
                return None
            
            # Прогноз на різні періоди
            horizons = [10, 20, 30]
            predictions = {}
            
            total_steps = len(horizons) * max(horizons)
            current_step = 0
            
            for horizon in horizons:
                future_predictions = []
                current_sequence = last_sequence.copy()
                
                for i in range(horizon):
                    # Оновлюємо прогрес для кожного кроку прогнозу
                    current_step += 1
                    step_progress = 40 + (current_step / total_steps) * 40
                    self.update_progress(step_progress)
                    
                    # Прогнозуємо наступне значення
                    next_pred = model.predict(current_sequence.reshape(1, lookback, feature_count), verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    
                    # Оновлення послідовності - створюємо новий вектор
                    current_sequence = np.roll(current_sequence, -1, axis=0)
                    
                    # Створюємо новий вектор на основі прогнозованої ціни
                    new_vector = current_sequence[-1].copy()
                    new_vector[0] = next_pred[0, 0]  # Оновлюємо ціну закриття
                    
                    # Оновлюємо пов'язані ознаки
                    if 'Returns' in numeric_columns:
                        returns_idx = numeric_columns.index('Returns')
                        prev_close = current_sequence[-2, 0] if len(current_sequence) > 1 else current_sequence[-1, 0]
                        new_vector[returns_idx] = (next_pred[0, 0] - prev_close) / prev_close
                    
                    current_sequence[-1] = new_vector
                
                # Зворотнє перетворення
                temp_data = np.zeros((len(future_predictions), feature_count))
                temp_data[:, 0] = future_predictions  # Тільки прогнозовані ціни
                future_predictions = scaler.inverse_transform(temp_data)[:, 0]
                
                predictions[horizon] = future_predictions.flatten()
            
            return predictions
            
        except Exception as e:
            self.status_callback(f"Помилка експертного прогнозу для {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_simple_rsi(self, prices, period=14):
        """Спрощений розрахунок RSI"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss.replace(0, 0.0001)  # Уникаємо ділення на нуль
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Заповнюємо NaN середнім значенням
        except:
            # У разі помилки повертаємо серію з 50 (нейтральне значення RSI)
            return pd.Series(50, index=prices.index)

    def _calculate_simple_macd(self, prices, fast=12, slow=26):
        """Спрощений розрахунок MACD"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            return macd
        except:
            # У разі помилки повертаємо серію з нулів
            return pd.Series(0, index=prices.index)
    
    
    def _debug_model_info(self, model, symbol):
        """Відлагоджувальна інформація про модель"""
        try:
            print(f"\n=== ДЕБАГ ІНФОРМАЦІЯ ДЛЯ {symbol} ===")
            print(f"Вхідна форма моделі: {model.input_shape}")
            print(f"Вихідна форма моделі: {model.output_shape}")
            print(f"Кількість шарів: {len(model.layers)}")
            
            for i, layer in enumerate(model.layers):
                print(f"Шар {i}: {layer.name}, {type(layer).__name__}")
                if hasattr(layer, 'units'):
                    print(f"  Юнітів: {layer.units}")
                if hasattr(layer, 'input_shape'):
                    print(f"  Вхідна форма: {layer.input_shape}")
                if hasattr(layer, 'output_shape'):
                    print(f"  Вихідна форма: {layer.output_shape}")
                    
        except Exception as e:
            print(f"Помилка відлагодження: {e}")
    
    def update_prediction_graphs(self, symbol, historical_data, predictions):
        """Оновлення графіків прогнозу"""
        try:
            # Останні 100 днів історичних даних
            last_100_days = historical_data[-100:]
            last_date = last_100_days.index[-1]
            
            # Оновлення графіків для кожного горизонту
            horizons = [10, 20, 30]
            figs = [self.fig_10d, self.fig_20d, self.fig_30d]
            axes = [self.ax_10d, self.ax_20d, self.ax_30d]
            canvases = [self.canvas_10d, self.canvas_20d, self.canvas_30d]
            
            for horizon, fig, ax, canvas in zip(horizons, figs, axes, canvases):
                if horizon in predictions:
                    ax.clear()
                    
                    # Історичні дані (останні 60 днів для кращого вигляду)
                    historical_to_show = historical_data[-60:]
                    ax.plot(historical_to_show.index, historical_to_show['Close'], 
                           label='Історичні дані', color='blue', linewidth=2)
                    
                    # Прогнозовані дані
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1), 
                        periods=horizon, 
                        freq='D'
                    )
                    ax.plot(future_dates, predictions[horizon], 
                           label=f'Прогноз на {horizon} днів', color='red', linewidth=2, linestyle='--')
                    
                    # Вертикальна лінія розділення
                    ax.axvline(x=last_date, color='green', linestyle='--', 
                              alpha=0.7, label='Початок прогнозу')
                    
                    ax.set_title(f'Прогноз {symbol} на {horizon} днів', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Дата')
                    ax.set_ylabel('Ціна (USD)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Форматування дат
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
                    canvas.draw()
        
        except Exception as e:
            self.status_callback(f"Помилка оновлення графіків прогнозу: {str(e)}")
    
    def export_results(self):
        """Експорт результатів прогнозу"""
        if not self.predictions:
            messagebox.showwarning("Увага", "Немає результатів для експорту")
            return
        
        try:
            # Створення DataFrame з прогнозами
            results = []
            for symbol, preds in self.predictions.items():
                for horizon, values in preds.items():
                    for i, value in enumerate(values, 1):
                        results.append({
                            'Symbol': symbol,
                            'Horizon': f'{horizon} днів',
                            'Day': i,
                            'Predicted_Price': value,
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
            
            df = pd.DataFrame(results)
            
            # Збереження
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'predictions_{timestamp}.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            
            self.status_callback(f"Результати збережено у {filename}")
            messagebox.showinfo("Успіх", f"Результати збережено у {filename}")
            
        except Exception as e:
            self.status_callback(f"Помилка експорту: {str(e)}")
            messagebox.showerror("Помилка", f"Помилка експорту: {str(e)}")
    
    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        if self.status_callback:
            self.status_callback(message)
    
    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        if self.progress_callback:
            self.progress_callback(value)



    def get_training_type_display(self, training_type):
        """Повертає відображуване значення типу навчання"""
        type_mapping = {
            'basic': 'Базова',
            'advanced': 'Розширена',
            'expert': 'Експертна',
            'Базове': 'Базова',
            'Розширене': 'Розширена',
            'Експертне': 'Експертна'
        }
        return type_mapping.get(training_type, training_type)

    def update_progress(self, value):
        """Оновлення прогресбару"""
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            self.progress_bar['value'] = value
            self.parent.update_idletasks()
