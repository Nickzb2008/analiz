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
from utils.trading_engine import AdvancedTradingEngine
from tkinter import scrolledtext  # ДОДАЄМО ЦЕ
import csv  # Додаємо для експорту

class MultiNeuralNetworkTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.model_manager = ModelManager()
        self.trained_models = []
        self.current_symbol = None
        self.training_log_window = None
        self.training_log_text = None
        self.show_training_log = tk.BooleanVar(value=False)  # ДОДАЄМО ЦЕ
        self.setup_ui()
        self.load_existing_models()
    
    def setup_ui(self):
        """Налаштування інтерфейсу для множинного навчання"""
        # Основний фрейм
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Лівий фрейм для налаштувань та керування з прокруткою
        left_container = ttk.Frame(main_frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5)
        
        left_frame = ttk.LabelFrame(left_container, text="Керування множинним навчанням")
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        # Встановлюємо фіксовану ширину
        left_container.config(width=300)
        left_container.pack_propagate(False)
        
        # Створюємо Canvas та Scrollbar для прокрутки
        self.left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.left_canvas.yview)
        
        # Фрейм для вмісту
        self.scrollable_frame = ttk.Frame(self.left_canvas)
        
        # Функція для оновлення області прокрутки
        def _on_frame_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        
        self.scrollable_frame.bind("<Configure>", _on_frame_configure)
        
        # Створюємо вікно в Canvas
        self.canvas_window = self.left_canvas.create_window(
            (0, 0), 
            window=self.scrollable_frame, 
            anchor="nw",
            width=290
        )
        
        # Функція для оновлення ширини вікна при зміні розміру Canvas
        def _on_canvas_configure(event):
            self.left_canvas.itemconfig(self.canvas_window, width=event.width - 20)
        
        self.left_canvas.bind("<Configure>", _on_canvas_configure)
        self.left_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Упаковка Canvas та Scrollbar
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 1), pady=1)
        
        # Додаємо обробку прокрутки мишею
        def _on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.left_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Центральний фрейм для інформації
        center_frame = ttk.LabelFrame(main_frame, text="Інформація про моделі")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правий фрейм для графіків
        right_frame = ttk.LabelFrame(main_frame, text="Візуалізація")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Фрейм для радіокнопок періодів
        period_frame = ttk.Frame(right_frame)
        period_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        ttk.Label(period_frame, text="Період:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Змінна для радіокнопок періодів
        self.period_var = tk.StringVar(value="1_month")
        
        # Радіокнопки для вибору періоду
        periods = [
            ("1 тиждень", "1_week"),
            ("1 місяць", "1_month"), 
            ("3 місяці", "3_months"),
            ("6 місяців", "6_months"),
            ("12 місяців", "12_months")
        ]
        
        for text, value in periods:
            ttk.Radiobutton(
                period_frame, 
                text=text, 
                variable=self.period_var, 
                value=value,
                command=self.on_period_change
            ).pack(side=tk.LEFT, padx=5)
        
        # Створюємо notebook для вкладок
        self.viz_notebook = ttk.Notebook(right_frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Додаємо нову вкладку "Основний графік" першою
        self.viz_frame_main = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.viz_frame_main, text="Основний графік")
        
        # Створюємо графік для основної вкладки
        self.fig_main, self.ax_main = plt.subplots(figsize=(10, 6))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, self.viz_frame_main)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Створюємо фрейми для різних горизонтів прогнозу
        self.viz_frame_10 = ttk.Frame(self.viz_notebook)
        self.viz_frame_20 = ttk.Frame(self.viz_notebook)
        self.viz_frame_30 = ttk.Frame(self.viz_notebook)
        
        self.viz_notebook.add(self.viz_frame_10, text="10 днів")
        self.viz_notebook.add(self.viz_frame_20, text="20 днів")
        self.viz_frame_30 = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.viz_frame_30, text="30 днів")
        
        # Створюємо графіки для прогнозних вкладок
        self.fig_10, self.ax_10 = plt.subplots(figsize=(8, 5))
        self.canvas_10 = FigureCanvasTkAgg(self.fig_10, self.viz_frame_10)
        self.canvas_10.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_20, self.ax_20 = plt.subplots(figsize=(8, 5))
        self.canvas_20 = FigureCanvasTkAgg(self.fig_20, self.viz_frame_20)
        self.canvas_20.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_30, self.ax_30 = plt.subplots(figsize=(8, 5))
        self.canvas_30 = FigureCanvasTkAgg(self.fig_30, self.viz_frame_30)
        self.canvas_30.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # === Налаштування навчання ===
        settings_frame = ttk.LabelFrame(self.scrollable_frame, text="Параметри навчання")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Створюємо стилі для малих елементів
        style = ttk.Style()
        style.configure('Small.TLabel', font=('Arial', 9))
        style.configure('Small.TEntry', font=('Arial', 9))
        style.configure('Small.TCombobox', font=('Arial', 9))
        style.configure('Small.TCheckbutton', font=('Arial', 9))
        style.configure('Small.TButton', font=('Arial', 8), padding=2)
        
        # Елементи налаштувань
        ttk.Label(settings_frame, text="Кількість епох:", style='Small.TLabel').pack(pady=2, anchor='w')
        self.epochs_var = tk.IntVar(value=50)
        epochs_entry = ttk.Entry(settings_frame, textvariable=self.epochs_var, width=15, style='Small.TEntry')
        epochs_entry.pack(pady=2, fill=tk.X, padx=5)
        
        ttk.Label(settings_frame, text="Розмір батча:", style='Small.TLabel').pack(pady=2, anchor='w')
        self.batch_size_var = tk.IntVar(value=32)
        batch_entry = ttk.Entry(settings_frame, textvariable=self.batch_size_var, width=15, style='Small.TEntry')
        batch_entry.pack(pady=2, fill=tk.X, padx=5)
        
        ttk.Label(settings_frame, text="Розмір вікна:", style='Small.TLabel').pack(pady=2, anchor='w')
        self.lookback_var = tk.IntVar(value=60)
        lookback_entry = ttk.Entry(settings_frame, textvariable=self.lookback_var, width=15, style='Small.TEntry')
        lookback_entry.pack(pady=2, fill=tk.X, padx=5)
        
        ttk.Label(settings_frame, text="Тестова вибірка (%):", style='Small.TLabel').pack(pady=2, anchor='w')
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_entry = ttk.Entry(settings_frame, textvariable=self.test_size_var, width=15, style='Small.TEntry')
        test_entry.pack(pady=2, fill=tk.X, padx=5)
        
        # Додаємо нові налаштування
        ttk.Label(settings_frame, text="Тип моделі:", style='Small.TLabel').pack(pady=2, anchor='w')
        self.model_type_var = tk.StringVar(value="advanced")
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_type_var, 
                                values=["basic", "advanced", "ensemble"], 
                                state="readonly", width=15, style='Small.TCombobox')
        model_combo.pack(pady=2, fill=tk.X, padx=5)
        
        self.optimize_hyperparams_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Оптимізувати гіперпараметри", 
                    variable=self.optimize_hyperparams_var, style='Small.TCheckbutton').pack(pady=5, anchor='w')
        
        # Перемикач для логу навчання
        self.show_training_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Показувати лог навчання", 
                    variable=self.show_training_log, style='Small.TCheckbutton').pack(pady=5, anchor='w')
        
        # === Кнопки керування ===
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Кнопки керування
        ttk.Button(button_frame, text="Навчити моделі", 
                command=self.train_multiple_models, style='Small.TButton').pack(pady=3, fill=tk.X)
        ttk.Button(button_frame, text="Оновити список моделей", 
                command=self.load_existing_models, style='Small.TButton').pack(pady=3, fill=tk.X)
        ttk.Button(button_frame, text="Видалити обрану модель", 
                command=self.delete_selected_model, style='Small.TButton').pack(pady=3, fill=tk.X)
        ttk.Button(button_frame, text="Прогноз для обраної", 
                command=self.predict_selected, style='Small.TButton').pack(pady=3, fill=tk.X)
        ttk.Button(button_frame, text="Порівняти всі моделі", 
                command=self.compare_all_models, style='Small.TButton').pack(pady=3, fill=tk.X)
        ttk.Button(button_frame, text="Торгівельні сигнали", 
                command=self.show_trading_signals, style='Small.TButton').pack(pady=3, fill=tk.X)
        
        # === Список навчених моделей ===
        list_frame = ttk.LabelFrame(center_frame, text="Навчені моделі")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview для відображення моделей
        self.models_tree = ttk.Treeview(list_frame, columns=('Symbol', 'MSE', 'MAE', 'Date'), 
                                    show='headings', height=8)
        self.models_tree.heading('Symbol', text='Криптовалюта')
        self.models_tree.heading('MSE', text='MSE')
        self.models_tree.heading('MAE', text='MAE')
        self.models_tree.heading('Date', text='Дата навчання')
        
        self.models_tree.column('Symbol', width=100)
        self.models_tree.column('MSE', width=80)
        self.models_tree.column('MAE', width=80)
        self.models_tree.column('Date', width=120)
        
        # Scrollbar для treeview
        tree_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.models_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Обробник вибору моделі
        self.models_tree.bind('<<TreeviewSelect>>', self.on_model_select)
        
        # === Інформаційний текст ===
        self.info_label = ttk.Label(center_frame, text="Оберіть модель для перегляду деталей")
        self.info_label.configure(font=('Arial', 9))
        self.info_label.pack(pady=5)
        
        # Додаємо обробник зміни розміру
        left_frame.bind('<Configure>', self.on_tab_resize)
        
        
        # Оновлюємо область прокрутки після додавання всіх елементів
        self.parent.after(100, self.on_ui_loaded)  # Змінюємо на новий метод


    def on_ui_loaded(self):
        """Викликається після повного завантаження UI"""
        self.update_scroll_region()
        # Якщо є обрана криптовалюта, відображаємо графік
        if hasattr(self, 'current_symbol') and self.current_symbol:
            self.show_main_chart(self.current_symbol)

    def on_period_change(self):
        """Обробник зміни періоду відображення графіка"""
        if self.current_symbol:
            self.show_main_chart(self.current_symbol)

    def show_main_chart(self, symbol):
        """Відображення основного графіка з обраним періодом"""
        try:
            # Перевірка наявності символу
            if not symbol:
                return
                
            # Завантаження даних
            data_file = f"{symbol}_data.csv"
            data_path = f'data/{data_file}'
            
            if not os.path.exists(data_path):
                self.ax_main.clear()
                self.ax_main.text(0.5, 0.5, 'Дані не знайдено', ha='center', va='center', transform=self.ax_main.transAxes)
                self.canvas_main.draw()
                return
                
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            if data.empty:
                self.ax_main.clear()
                self.ax_main.text(0.5, 0.5, 'Немає даних', ha='center', va='center', transform=self.ax_main.transAxes)
                self.canvas_main.draw()
                return
            
            # Визначення періоду на основі обраної радіокнопки
            period = self.period_var.get()
            days_mapping = {
                "1_week": 7,
                "1_month": 30,
                "3_months": 90,
                "6_months": 180,
                "12_months": 365
            }
            
            days = days_mapping.get(period, 30)
            display_data = data.tail(min(days, len(data)))  # Обмежуємо реальною кількістю даних
            
            # Візуалізація даних
            self.ax_main.clear()
            
            if len(display_data) > 0:
                self.ax_main.plot(display_data.index, display_data['Close'], 
                                label=f'{symbol} - Ціна закриття', color='blue', linewidth=2)
                
                self.ax_main.set_title(f'Графік цін {symbol} ({self.get_period_display_name(period)})', 
                                    fontsize=14, fontweight='bold')
                self.ax_main.set_xlabel('Дата')
                self.ax_main.set_ylabel('Ціна (USD)')
                self.ax_main.legend()
                self.ax_main.grid(True, alpha=0.3)
                
                # Форматування дат на осі X
                self.ax_main.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                self.ax_main.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                plt.setp(self.ax_main.xaxis.get_majorticklabels(), rotation=45)
            else:
                self.ax_main.text(0.5, 0.5, 'Недостатньо даних', ha='center', va='center', transform=self.ax_main.transAxes)
            
            self.canvas_main.draw()
            
            self.safe_status_callback(f"Відображено графік для {symbol} ({self.get_period_display_name(period)})")
            
        except Exception as e:
            error_msg = f"Помилка відображення графіка {symbol}: {str(e)}"
            self.safe_status_callback(error_msg)
            self.ax_main.clear()
            self.ax_main.text(0.5, 0.5, 'Помилка завантаження', ha='center', va='center', transform=self.ax_main.transAxes)
            self.canvas_main.draw()

    def get_period_display_name(self, period_value):
        """Отримання відображуваного імені для періоду"""
        names = {
            "1_week": "1 тиждень",
            "1_month": "1 місяць",
            "3_months": "3 місяці",
            "6_months": "6 місяців",
            "12_months": "12 місяців"
        }
        return names.get(period_value, "1 місяць")



    def on_tab_resize(self, event):
        """Обробка зміни розміру вкладки"""
        try:
            # Оновлюємо ширину вікна в Canvas
            if hasattr(self, 'canvas_window') and hasattr(self, 'left_canvas'):
                self.left_canvas.itemconfig(self.canvas_window, width=event.width - 20)  # Враховуємо scrollbar
            
            self.update_scroll_region()
            
            # Використовуємо безпечний виклик
            try:
                self.update_plot_sizes()
            except tk.TclError as e:
                if "invalid command name" not in str(e):
                    raise e
        
        except Exception as e:
            # Ігноруємо помилки, пов'язані з віджетами
            pass
    
    def update_scroll_region(self):
        """Оновлення області прокрутки"""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
    
    def update_plot_sizes(self):
        """Оновлення розмірів графіків при зміні розміру вікна"""
        try:
            # Оновлюємо розміри основних графіків
            if hasattr(self, 'fig') and hasattr(self, 'canvas'):
                self.fig.tight_layout()
                self.canvas.draw()
            
            # Оновлюємо розміри графіків прогнозів
            if hasattr(self, 'fig_10') and hasattr(self, 'canvas_10'):
                self.fig_10.tight_layout()
                self.canvas_10.draw()
            
            if hasattr(self, 'fig_20') and hasattr(self, 'canvas_20'):
                self.fig_20.tight_layout()
                self.canvas_20.draw()
            
            if hasattr(self, 'fig_30') and hasattr(self, 'canvas_30'):
                self.fig_30.tight_layout()
                self.canvas_30.draw()
            
        except Exception as e:
            # Ігноруємо помилки, щоб не порушувати роботу програми
            pass

    def hide_training_log(self):
        """Приховування вікна логу"""
        if self.training_log_window:
            self.training_log_window.withdraw()

    def clear_training_log(self):
        """Очищення логу навчання"""
        if self.training_log_text:
            self.training_log_text.config(state=tk.NORMAL)
            self.training_log_text.delete(1.0, tk.END)
            self.training_log_text.config(state=tk.DISABLED)

    def add_to_training_log(self, message):
        """Додавання повідомлення до логу навчання"""
        try:
            if (hasattr(self, 'training_log_text') and self.training_log_text and 
                self.show_training_log.get() and 
                hasattr(self, 'training_log_window') and self.training_log_window and 
                self.check_window_exists(self.training_log_window)):
                
                self.training_log_text.config(state=tk.NORMAL)
                self.training_log_text.insert(tk.END, message + "\n")
                self.training_log_text.see(tk.END)
                self.training_log_text.config(state=tk.DISABLED)
                
                # Автоматичне збереження логу при додаванні повідомлень
                self.auto_save_log()
                
        except tk.TclError as e:
            if "invalid command name" not in str(e):
                self.safe_status_callback(f"Помилка додавання до логу: {str(e)}")
        except Exception as e:
            self.safe_status_callback(f"Помилка додавання до логу: {str(e)}")
    
    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        if self.status_callback:
            self.status_callback(message)
    
    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        if self.progress_callback:
            self.progress_callback(value)
    
    def load_existing_models(self):
        """Завантаження існуючих моделей"""
        self.model_manager.load_all_models()
        self.update_models_list()
        self.info_label.config(text="Оберіть модель для перегляду деталей")
    
    def update_models_list(self):
        """Оновлення списку моделей"""
        # Очищаємо treeview
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
        
        # Додаємо навчені моделі
        available_models = self.model_manager.get_available_models()
        self.trained_models = available_models
        
        for symbol in available_models:
            metrics = self.model_manager.get_model_metrics(symbol)
            date_str = metrics.get('timestamp', 'N/A')
            if hasattr(date_str, 'strftime'):
                date_str = date_str.strftime('%Y-%m-%d %H:%M')
            
            self.models_tree.insert('', 'end', values=(
                symbol,
                f"{metrics.get('mse', 0):.6f}",
                f"{metrics.get('mae', 0):.6f}",
                date_str
            ))
        
        self.info_label.config(text=f"Знайдено {len(available_models)} навчених моделей")
    
    def on_model_select(self, event):
        """Обробник вибору моделі зі списку"""
        selection = self.models_tree.selection()
        if selection:
            item = self.models_tree.item(selection[0])
            self.current_symbol = item['values'][0]
            self.show_model_details(self.current_symbol)
            # Відображаємо графік відразу при виборі криптовалюти
            self.show_main_chart(self.current_symbol)

    def show_model_details(self, symbol):
        """Показ деталей обраної моделі"""
        try:
            metrics = self.model_manager.get_model_metrics(symbol)
            info_text = f"""
    Деталі моделі {symbol}:
    MSE: {metrics.get('mse', 0):.6f}
    MAE: {metrics.get('mae', 0):.6f}
    Дата навчання: {metrics.get('timestamp', 'N/A')}
    """
            self.info_label.config(text=info_text)
            
        except Exception as e:
            self.info_label.config(text=f"Помилка завантаження деталей: {str(e)}")

    def show_model_chart(self, symbol):
        """Відображення графіка обраної криптовалюти"""
        try:
            # Завантаження даних
            data_file = f"{symbol}_data.csv"
            data = pd.read_csv(f'data/{data_file}', index_col=0, parse_dates=True)
            
            # Візуалізація історичних даних
            self.ax.clear()
            
            # Відображаємо останні 100 днів
            historical_prices = data['Close'].values[-100:]
            historical_dates = data.index[-100:]
            
            self.ax.plot(historical_dates, historical_prices, label=f'{symbol} - Історичні дані', 
                        color='blue', linewidth=2)
            
            self.ax.set_title(f'Графік цін {symbol}', fontsize=14, fontweight='bold')
            self.ax.set_xlabel('Дата')
            self.ax.set_ylabel('Ціна (USD)')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # Форматування дат на осі X
            self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            self.ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
            
            self.canvas.draw()
            
            self.safe_status_callback(f"Відображено графік для {symbol}")
            
        except Exception as e:
            error_msg = f"Помилка відображення графіка {symbol}: {str(e)}"
            self.safe_status_callback(error_msg)
            # Очищаємо графік у разі помилки
            self.ax.clear()
            self.canvas.draw()
    
    def prepare_data(self, data, lookback=60):
        """Підготовка даних для LSTM"""
        if data is None or data.empty:
            raise ValueError("Немає даних для обробки")
        
        if 'Close' not in data.columns:
            raise ValueError("Відсутня колонка 'Close'")
        
        prices = data[['Close']].values
        
        if len(prices) < lookback + 10:
            raise ValueError(f"Замало даних. Потрібно мінімум {lookback + 10} точок")
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def create_lstm_model(self, input_shape):
        """Створення LSTM моделі"""
        model = tf.keras.Sequential([
            # Перший LSTM шар
            tf.keras.layers.LSTM(
                units=64,
                return_sequences=True,
                input_shape=input_shape,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros'
            ),
            tf.keras.layers.Dropout(0.2),
            
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
                return_sequences=False,
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
                units=1,
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
            loss='mean_squared_error',  # Рядкова назва
            metrics=['mean_absolute_error']  # Тільки одна метрика
        )
        
        return model

    def train_multiple_models(self):
        """Навчання моделей для кількох криптовалют"""
        def train_thread():
            try:
                self.safe_status_callback("Пошук файлів даних...")
                self.safe_progress_callback(5)
                
                # Створюємо вікно логу якщо потрібно
                if self.show_training_log.get():
                    self.create_training_log_window()
                    if self.training_log_window:
                        self.training_log_window.deiconify()
                
                # Використовуємо універсальне діалогове вікно для вибору файлів
                selected_files = self.ask_user_to_select_models(
                    title="Оберіть криптовалюті для навчання",
                    prompt="Оберіть криптовалюті для навчання нейромережі:",
                    mode="files"
                )
                
                if not selected_files:
                    self.safe_status_callback("Вибір скасовано")
                    self.safe_progress_callback(0)
                    return
                
                total_files = len(selected_files)
                successful_models = []
                failed_models = []
                
                for i, selected_file in enumerate(selected_files):
                    symbol = selected_file.replace('_data.csv', '')
                    self.safe_status_callback(f"Навчання {symbol} ({i+1}/{total_files})...")
                    self.add_to_training_log(f"=== Навчання моделі для {symbol} ===")
                    
                    # Додаємо інформацію про тип моделі
                    model_type = self.model_type_var.get()
                    optimize = self.optimize_hyperparams_var.get()
                    self.add_to_training_log(f"Тип моделі: {model_type}")
                    self.add_to_training_log(f"Оптимізація гіперпараметрів: {'Так' if optimize else 'Ні'}")
                    self.add_to_training_log(f"Кількість епох: {self.epochs_var.get()}")
                    self.add_to_training_log(f"Розмір вікна: {self.lookback_var.get()}")
                    
                    # Прогрес для поточного файлу (від 10% до 90%)
                    file_progress_start = 10 + (i / total_files) * 80
                    file_progress_end = 10 + ((i + 1) / total_files) * 80
                    
                    try:
                        # Завантаження даних
                        self.safe_status_callback(f"Завантаження даних {symbol}...")
                        data = pd.read_csv(f'data/{selected_file}', index_col=0, parse_dates=True)
                        
                        # Перевірка даних
                        self.safe_status_callback(f"Перевірка даних {symbol}...")
                        DataValidator.check_data_requirements(data, self.safe_status_callback)
                        
                        # Підготовка даних
                        self.safe_status_callback(f"Підготовка даних {symbol}...")
                        lookback = self.lookback_var.get()
                        X, y, scaler = self.prepare_data(data, lookback)
                        
                        # Розділення на train/test
                        self.safe_status_callback(f"Розділення даних {symbol}...")
                        test_size = int(len(X) * self.test_size_var.get())
                        X_train, X_test = X[:-test_size], X[-test_size:]
                        y_train, y_test = y[:-test_size], y[-test_size:]
                        
                        # Вибір типу моделі
                        self.safe_status_callback(f"Створення моделі {symbol}...")
                        
                        # Записуємо час початку навчання
                        start_time = datetime.now()
                        self.add_to_training_log(f"Час початку навчання: {start_time.strftime('%H:%M:%S')}")
                        
                        # Оптимізація гіперпараметрів (якщо обрано)
                        if self.optimize_hyperparams_var.get():
                            self.safe_status_callback(f"Оптимізація гіперпараметрів {symbol}...")
                            self.add_to_training_log(f"Початок оптимізації гіперпараметрів")
                            model = self.optimize_hyperparameters(X_train, y_train)
                            self.add_to_training_log(f"Оптимізація завершена")
                        else:
                            # Створення обраного типу моделі
                            if model_type == "advanced":
                                model = self.create_advanced_lstm_model((X_train.shape[1], 1))
                                self.add_to_training_log(f"Використана покращена LSTM модель з контролем перенавчання")
                            elif model_type == "ensemble":
                                model = self.create_ensemble_model((X_train.shape[1], 1))
                                self.add_to_training_log(f"Використана ансамблева модель")
                            else:
                                model = self.create_lstm_model((X_train.shape[1], 1))
                                self.add_to_training_log(f"Використана базова LSTM модель")
                        
                        # Клас для відображення прогресу
                        class TrainingCallback(tf.keras.callbacks.Callback):
                            def __init__(self, tab_reference, total_epochs, file_progress_start, file_progress_end, symbol):
                                super().__init__()
                                self.tab_reference = tab_reference
                                self.total_epochs = total_epochs
                                self.file_progress_start = file_progress_start
                                self.file_progress_end = file_progress_end
                                self.symbol = symbol
                            
                            def on_epoch_end(self, epoch, logs=None):
                                # Оновлення прогресу
                                epoch_progress = (epoch + 1) / self.total_epochs
                                current_progress = self.file_progress_start + epoch_progress * (self.file_progress_end - self.file_progress_start)
                                self.tab_reference.safe_progress_callback(current_progress)
                                
                                # Оновлення статусу
                                status_msg = f"Навчання {self.symbol}: епоха {epoch+1}/{self.total_epochs}"
                                self.tab_reference.safe_status_callback(status_msg)
                                
                                # Додавання до логу
                                if self.tab_reference.show_training_log.get():
                                    log_message = (
                                        f"Epoch {epoch+1}/{self.total_epochs}\n"
                                        f"loss: {logs['loss']:.4f} - mae: {logs['mean_absolute_error']:.4f} - "
                                        f"val_loss: {logs['val_loss']:.4f} - val_mae: {logs['val_mean_absolute_error']:.4f}"
                                    )
                                    self.tab_reference.add_to_training_log(log_message)
                        
                        # Створюємо callback з посиланням на поточний об'єкт
                        training_callback = TrainingCallback(
                            self, 
                            self.epochs_var.get(),
                            file_progress_start,
                            file_progress_end,
                            symbol
                        )
                        
                        # Навчання моделі
                        self.safe_status_callback(f"Початок навчання {symbol}...")
                        
                        # Створюємо callback'и для контролю перенавчання
                        training_callbacks = [
                            # Рання зупинка при перенавчанні
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=15,
                                min_delta=0.0001,
                                restore_best_weights=True,
                                verbose=0
                            ),
                            
                            # Зменшення learning rate при застії
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,
                                patience=8,
                                min_lr=0.00001,
                                verbose=0
                            )
                        ]
                        
                        # Додаємо наш callback для прогресу
                        training_callbacks.append(training_callback)
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=self.epochs_var.get(),
                            batch_size=self.batch_size_var.get(),
                            validation_data=(X_test, y_test),
                            verbose=0,
                            callbacks=training_callbacks
                        )
                        
                        # Перевіряємо, чи спрацювала рання зупинка
                        actual_epochs = len(history.history['loss'])
                        if actual_epochs < self.epochs_var.get():
                            self.add_to_training_log(f"Рання зупинка на епосі {actual_epochs} (запобігання перенавчанню)")
                        
                        # Записуємо час закінченна навчання
                        end_time = datetime.now()
                        training_duration = end_time - start_time
                        self.add_to_training_log(f"Час закінчення навчання: {end_time.strftime('%H:%M:%S')}")
                        self.add_to_training_log(f"Загальний час навчання: {training_duration}")
                        self.add_to_training_log(f"Фактичне число епох: {actual_epochs}")
                        
                        # Прогнозування та метрики
                        self.safe_status_callback(f"Прогнозування для {symbol}...")
                        predictions = model.predict(X_test, verbose=0)
                        predictions = scaler.inverse_transform(predictions)
                        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                        
                        mse = mean_squared_error(y_test_actual, predictions)
                        mae = mean_absolute_error(y_test_actual, predictions)
                        
                        # Додаємо фінальні метрики до логу
                        final_metrics = f"\nФінальні метрики для {symbol}: MSE={mse:.6f}, MAE={mae:.6f}"
                        self.add_to_training_log(final_metrics)
                        
                        # Додаємо інформацію про навчання
                        train_loss = history.history['loss'][-1]
                        val_loss = history.history['val_loss'][-1]
                        self.add_to_training_log(f"Фінальний train loss: {train_loss:.6f}")
                        self.add_to_training_log(f"Фінальний val loss: {val_loss:.6f}")
                        
                        # Перевіряємо ознаки перенавчання
                        if len(history.history['loss']) > 10:
                            train_val_ratio = train_loss / val_loss if val_loss != 0 else float('inf')
                            if train_val_ratio < 0.8:
                                self.add_to_training_log("⚡ Можливе перенавчання (train loss значно нижчий за val loss)")
                            elif train_val_ratio > 1.2:
                                self.add_to_training_log("✅ Хороша узагальнююча здатність")
                        
                        self.add_to_training_log("="*50 + "\n")
                        
                        # Збереження моделі
                        self.safe_status_callback(f"Збереження моделі {symbol}...")
                        metrics = {
                            'mse': mse, 
                            'mae': mae, 
                            'timestamp': datetime.now(),
                            'lookback': lookback,
                            'scaler': scaler,
                            'model_type': model_type,
                            'optimized': self.optimize_hyperparams_var.get(),
                            'training_time': str(training_duration),
                            'actual_epochs': actual_epochs,
                            'final_train_loss': train_loss,
                            'final_val_loss': val_loss
                        }
                        self.model_manager.save_model(symbol, model, scaler, metrics)
                        successful_models.append(symbol)
                        
                        self.safe_status_callback(f"Модель {symbol} успішно навчена")
                        
                    except Exception as e:
                        error_msg = f"Помилка навчання {symbol}: {str(e)}"
                        self.safe_status_callback(error_msg)
                        self.add_to_training_log(f"ПОМИЛКА: {error_msg}")
                        failed_models.append(symbol)
                        continue
                
                # Оновлення списку моделей
                self.safe_status_callback("Оновлення списку моделей...")
                self.load_existing_models()
                self.safe_progress_callback(100)
                
                # Показ результатів
                result_text = ""
                if successful_models:
                    result_text += f"Успішно навчені: {', '.join(successful_models)}\n"
                if failed_models:
                    result_text += f"Не вдалося навчити: {', '.join(failed_models)}"
                
                self.safe_status_callback("Навчання завершено")
                self.add_to_training_log("=== НАВЧАННЯ ЗАВЕРШЕНО ===")
                messagebox.showinfo("Результати навчання", result_text)
                
            except Exception as e:
                error_msg = f"Помилка навчання: {str(e)}"
                self.safe_status_callback(error_msg)
                self.add_to_training_log(f"ЗАГАЛЬНА ПОМИЛКА: {error_msg}")
                self.safe_progress_callback(0)
                messagebox.showerror("Помилка", error_msg)
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()

    def predict_selected(self):
        """Прогнозування для обраної моделі на різні горизонти"""
        if not self.current_symbol:
            messagebox.showwarning("Увага", "Спочатку оберіть модель зі списку")
            return
        
        def predict_thread():
            try:
                self.safe_status_callback(f"Прогнозування для {self.current_symbol}...")
                self.safe_progress_callback(20)
                
                # Завантаження моделі
                if not self.model_manager.load_model(self.current_symbol):
                    raise ValueError(f"Модель {self.current_symbol} не знайдена")
                
                model = self.model_manager.models[self.current_symbol]
                scaler = self.model_manager.scalers[self.current_symbol]
                metrics = self.model_manager.metrics[self.current_symbol]
                
                # Завантаження даних
                data_file = f"{self.current_symbol}_data.csv"
                data = pd.read_csv(f'data/{data_file}', index_col=0, parse_dates=True)
                
                self.safe_status_callback("Підготовка даних...")
                self.safe_progress_callback(40)
                
                # Підготовка даних для прогнозу
                lookback = metrics.get('lookback', 60)
                prices = data[['Close']].values
                scaled_prices = scaler.transform(prices)
                
                # Беремо останню послідовність
                last_sequence = scaled_prices[-lookback:]
                
                # Прогноз на різні горизонти
                horizons = [10, 20, 30]
                all_predictions = {}
                
                for horizon in horizons:
                    self.safe_status_callback(f"Прогнозування на {horizon} днів...")
                    self.safe_progress_callback(40 + (horizon / 30) * 50)
                    
                    future_predictions = []
                    current_sequence = last_sequence.copy()
                    
                    for i in range(horizon):
                        next_pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
                        future_predictions.append(next_pred[0, 0])
                        
                        # Оновлення послідовності
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[-1] = next_pred
                    
                    # Перетворення назад
                    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                    all_predictions[horizon] = future_predictions
                
                # Візуалізація для кожного горизонту
                self.visualize_predictions(data, all_predictions)
                
                # Інформація про прогноз
                last_actual_price = prices[-1][0]
                info_text = f"Прогноз для {self.current_symbol}:\nПоточна ціна: ${last_actual_price:.2f}\n\n"
                
                for horizon, predictions in all_predictions.items():
                    predicted_price = predictions[-1][0]
                    price_change = predicted_price - last_actual_price
                    percent_change = (price_change / last_actual_price) * 100
                    info_text += f"Через {horizon} днів: ${predicted_price:.2f} ({percent_change:+.2f}%)\n"
                
                self.info_label.config(text=info_text)
                self.safe_status_callback(f"Прогноз {self.current_symbol} завершено")
                self.safe_progress_callback(100)
                
            except Exception as e:
                error_msg = f"Помилка прогнозування: {str(e)}"
                self.safe_status_callback(error_msg)
                self.safe_progress_callback(0)
                messagebox.showerror("Помилка", error_msg)
        
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()

    def visualize_predictions(self, data, all_predictions):
        """Візуалізація прогнозів для різних горизонтів"""
        historical_prices = data['Close'].values[-100:]
        historical_dates = data.index[-100:]
        last_date = data.index[-1]
        
        for horizon, predictions in all_predictions.items():
            # Вибір відповідного графіка
            if horizon == 10:
                ax = self.ax_10
                fig = self.fig_10
                canvas = self.canvas_10
            elif horizon == 20:
                ax = self.ax_20
                fig = self.fig_20
                canvas = self.canvas_20
            else:
                ax = self.ax_30
                fig = self.fig_30
                canvas = self.canvas_30
            
            # Очищення графіка
            ax.clear()
            
            # Майбутні дати
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
            
            # Візуалізація
            ax.plot(historical_dates, historical_prices, label='Історичні дані', color='blue', linewidth=2)
            ax.plot(future_dates, predictions, label=f'Прогноз на {horizon} днів', color='red', linewidth=2)
            ax.axvline(x=last_date, color='green', linestyle='--', alpha=0.7, label='Початок прогнозу')
            
            ax.set_title(f'Прогноз {self.current_symbol} на {horizon} днів', fontsize=14, fontweight='bold')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Ціна (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматування дат
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Додавання анотацій
            last_actual_price = historical_prices[-1]
            predicted_price = predictions[-1][0]
            ax.annotate(f'Поточна: ${last_actual_price:.2f}',
                    xy=(last_date, last_actual_price), xycoords='data',
                    xytext=(10, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            ax.annotate(f'Прогноз: ${predicted_price:.2f}',
                    xy=(future_dates[-1], predicted_price), xycoords='data',
                    xytext=(-100, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            canvas.draw()

    def delete_selected_model(self):
        """Видалення обраної моделі"""
        if not self.current_symbol:
            messagebox.showwarning("Увага", "Спочатку оберіть модель для видалення")
            return
        
        if messagebox.askyesno("Підтвердження", 
                             f"Ви впевнені, що хочете видалити модель {self.current_symbol}?"):
            try:
                model_file = f"models/{self.current_symbol}_model.pkl"
                if os.path.exists(model_file):
                    os.remove(model_file)
                    
                    # Оновлюємо менеджер моделей
                    if self.current_symbol in self.model_manager.models:
                        del self.model_manager.models[self.current_symbol]
                    if self.current_symbol in self.model_manager.scalers:
                        del self.model_manager.scalers[self.current_symbol]
                    if self.current_symbol in self.model_manager.metrics:
                        del self.model_manager.metrics[self.current_symbol]
                    
                    self.current_symbol = None
                    self.update_models_list()
                    self.ax.clear()
                    self.canvas.draw()
                    self.info_label.config(text="Модель видалена")
                    self.safe_status_callback(f"Модель {self.current_symbol} видалена")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося видалити модель: {str(e)}")
    
    def compare_all_models(self):
        """Порівняння всіх навчених моделей"""
        if not self.trained_models:
            messagebox.showinfo("Інформація", "Немає навчених моделей для порівняння")
            return
        
        comparison_df = self.model_manager.compare_models()
        
        # Створення вікна порівняння
        compare_window = tk.Toplevel(self.parent)
        compare_window.title("Порівняння моделей")
        compare_window.geometry("800x600")
        
        # Графіки
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # MSE
        ax1.bar(comparison_df['symbol'], comparison_df['mse'], color='skyblue')
        ax1.set_title('Порівняння MSE моделей', fontweight='bold')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.bar(comparison_df['symbol'], comparison_df['mae'], color='lightcoral')
        ax2.set_title('Порівняння MAE моделей', fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Canvas для графіків
        canvas = FigureCanvasTkAgg(fig, compare_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Таблиця з метриками
        table_frame = ttk.Frame(compare_window)
        table_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tree = ttk.Treeview(table_frame, columns=('Symbol', 'MSE', 'MAE', 'R2', 'Date'), show='headings', height=5)
        tree.heading('Symbol', text='Криптовалюта')
        tree.heading('MSE', text='MSE')
        tree.heading('MAE', text='MAE')
        tree.heading('R2', text='R²')
        tree.heading('Date', text='Дата')
        
        for _, row in comparison_df.iterrows():
            tree.insert('', 'end', values=(
                row['symbol'],
                f"{row['mse']:.6f}",
                f"{row['mae']:.6f}",
                f"{row.get('r2', 0):.4f}",
                str(row['timestamp'])
            ))
        
        tree.pack(fill=tk.X)

    
    

# tab5_multi_neural_network.py - додаємо нові функції моделей
    def create_advanced_lstm_model(self, input_shape):
        """Покращена LSTM модель з контролем перенавчання"""
        model = tf.keras.Sequential([
            # Перший LSTM шар з підвищеною регуляризацією
            tf.keras.layers.LSTM(
                128, 
                return_sequences=True, 
                input_shape=input_shape,
                recurrent_dropout=0.3,  # Збільшено з 0.2
                dropout=0.3,  # Додано dropout для входу
                kernel_regularizer=tf.keras.regularizers.l2(0.01),  # Збільшено з 0.001
                recurrent_regularizer=tf.keras.regularizers.l2(0.005),
                bias_regularizer=tf.keras.regularizers.l2(0.005)
            ),
            tf.keras.layers.Dropout(0.4),  # Збільшено з 0.3
            
            # Другий LSTM шар
            tf.keras.layers.LSTM(
                64, 
                return_sequences=True,
                recurrent_dropout=0.3,
                dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                recurrent_regularizer=tf.keras.regularizers.l2(0.005)
            ),
            tf.keras.layers.Dropout(0.4),
            
            # Третій LSTM шар
            tf.keras.layers.LSTM(
                32, 
                return_sequences=False,
                recurrent_dropout=0.3,
                dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ),
            tf.keras.layers.Dropout(0.4),
            
            # Повнозв'язні шари
            tf.keras.layers.Dense(
                32, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01),  # Збільшено
                bias_regularizer=tf.keras.regularizers.l2(0.005)
            ),
            tf.keras.layers.Dropout(0.3),  # Збільшено з 0.2
            
            tf.keras.layers.Dense(
                16, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005)
            ),
            tf.keras.layers.Dropout(0.2),
            
            # Вихідний шар
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Компіляція моделі з меншим learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0005,  # Зменшено з 0.001
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0  # Додано обмеження градієнтів
            ),
            loss=tf.keras.losses.Huber(delta=1.0),  # Huber loss менш чутлива до викидів
            metrics=['mean_absolute_error']
        )
        
        return model

    def create_ensemble_model(self, input_shape):
        """Ансамбль різних архітектур"""
        # LSTM модель
        lstm_input = tf.keras.layers.Input(shape=input_shape)
        lstm_layer = tf.keras.layers.LSTM(64)(lstm_input)
        
        # GRU модель
        gru_layer = tf.keras.layers.GRU(64)(lstm_input)
        
        # 1D CNN модель
        cnn_layer = tf.keras.layers.Conv1D(64, 3, activation='relu')(lstm_input)
        cnn_layer = tf.keras.layers.GlobalAveragePooling1D()(cnn_layer)
        
        # Об'єднуємо
        merged = tf.keras.layers.concatenate([lstm_layer, gru_layer, cnn_layer])
        merged = tf.keras.layers.Dense(32, activation='relu')(merged)
        output = tf.keras.layers.Dense(1)(merged)
        
        model = tf.keras.Model(inputs=lstm_input, outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',  # Рядкова назва
            metrics=['mean_absolute_error']  # Тільки одна метрика
        )
        
        return model

    def optimize_hyperparameters(self, X_train, y_train):
        """Оптимізація гіперпараметрів з використанням KerasTuner"""
        try:
            import keras_tuner as kt
            
            def build_model(hp):
                model = tf.keras.Sequential()
                
                # Кількість шарів
                num_layers = hp.Int('num_layers', 1, 3)
                
                for i in range(num_layers):
                    model.add(tf.keras.layers.LSTM(
                        units=hp.Int(f'units_{i}', 32, 128, step=32),
                        return_sequences=i < num_layers - 1,
                        input_shape=(X_train.shape[1], 1) if i == 0 else None
                    ))
                    model.add(tf.keras.layers.Dropout(
                        hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)
                    ))
                
                model.add(tf.keras.layers.Dense(1))
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
                    ),
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )
                
                return model
            
            # Використовуємо Hyperband для швидкості з overwrite=True
            tuner = kt.Hyperband(
                build_model,
                objective=kt.Objective("val_mean_absolute_error", direction="min"),
                max_epochs=30,
                factor=3,
                directory='hyperparameter_tuning',
                project_name='crypto_lstm',
                overwrite=True  # Додаємо це для перезапису існуючого проекту
            )
            
            # Рання зупинка для економії часу
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_mean_absolute_error',
                patience=5,
                restore_best_weights=True
            )
            
            tuner.search(
                X_train, y_train,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stop]
            )
            
            # Отримуємо найкращу модель
            best_model = tuner.get_best_models()[0]
            best_hps = tuner.get_best_hyperparameters()[0]
            
            self.add_to_training_log(f"Найкращі параметри: {best_hps.values}")
            return best_model
            
        except ImportError:
            # Якщо keras_tuner не встановлено, використовуємо просту версію
            self.add_to_training_log("KerasTuner не встановлено. Використовую просту оптимізацію.")
            return self.simple_hyperparameter_optimization(X_train, y_train)
        except Exception as e:
            error_msg = f"Помилка оптимізації: {str(e)}. Використовую модель обраного типу."
            self.add_to_training_log(error_msg)
            # Повертаємо модель відповідного типу замість завжди базової
            model_type = self.model_type_var.get()
            if model_type == "advanced":
                return self.create_advanced_lstm_model((X_train.shape[1], 1))
            elif model_type == "ensemble":
                return self.create_ensemble_model((X_train.shape[1], 1))
            else:
                return self.create_lstm_model((X_train.shape[1], 1))

    def simple_hyperparameter_optimization(self, X_train, y_train):
        """Проста оптимізація без KerasTuner"""
        best_model = None
        best_mae = float('inf')
        
        # Простий пошук по параметрах
        params_to_test = [
            {'units': 64, 'lr': 0.001, 'dropout': 0.2},
            {'units': 128, 'lr': 0.0005, 'dropout': 0.3},
            {'units': 256, 'lr': 0.0001, 'dropout': 0.4}
        ]
        
        for i, params in enumerate(params_to_test):
            try:
                self.safe_status_callback(f"Тестування параметрів {i+1}/{len(params_to_test)}")
                
                # Створюємо модель відповідного типу
                model_type = self.model_type_var.get()
                if model_type == "advanced":
                    model = self.create_advanced_lstm_model_with_params((X_train.shape[1], 1), params)
                elif model_type == "ensemble":
                    model = self.create_ensemble_model_with_params((X_train.shape[1], 1), params)
                else:
                    model = self.create_lstm_model_with_params((X_train.shape[1], 1), params)
                
                # Коротке навчання для оцінки
                history = model.fit(
                    X_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                val_mae = min(history.history['val_mean_absolute_error'])
                
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_model = model
                
                self.add_to_training_log(f"Параметри: {params}, Val MAE: {val_mae:.6f}")
                
            except Exception as e:
                self.add_to_training_log(f"Помилка тестування параметрів: {str(e)}")
                continue
        
        if best_model:
            self.add_to_training_log(f"Найкращий Val MAE: {best_mae:.6f}")
            return best_model
        
        self.add_to_training_log("Не вдалося знайти хороші параметри. Використовую модель обраного типу.")
        model_type = self.model_type_var.get()
        if model_type == "advanced":
            return self.create_advanced_lstm_model((X_train.shape[1], 1))
        elif model_type == "ensemble":
            return self.create_ensemble_model((X_train.shape[1], 1))
        else:
            return self.create_lstm_model((X_train.shape[1], 1))

    def create_lstm_model_with_params(self, input_shape, params):
        """Створення LSTM моделі з параметрами"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(params['units'], return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(params['dropout']),
            tf.keras.layers.LSTM(params['units']//2),
            tf.keras.layers.Dropout(params['dropout']),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model

    def create_advanced_lstm_model_with_params(self, input_shape, params):
        """Створення покращеної LSTM моделі з параметрами"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(params['units'], return_sequences=True, input_shape=input_shape,
                            recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(params['dropout']),
            
            tf.keras.layers.LSTM(params['units']//2, return_sequences=True,
                            recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(params['dropout']),
            
            tf.keras.layers.LSTM(params['units']//4, recurrent_dropout=0.2,
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(params['dropout']),
            
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(params['dropout']/2),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss='huber',
            metrics=['mean_absolute_error']
        )
        
        return model

#====================LOG================================
    def create_training_log_window(self):
        """Створення вікна для відображення логу навчання"""
        if self.training_log_window is None or not self.training_log_window.winfo_exists():
            self.training_log_window = tk.Toplevel(self.parent)
            self.training_log_window.title("Лог навчання нейромережі")
            self.training_log_window.geometry("800x500")
            self.training_log_window.protocol("WM_DELETE_WINDOW", self.hide_training_log)
            
            # Текстове поле з прокруткою
            self.training_log_text = scrolledtext.ScrolledText(
                self.training_log_window, wrap=tk.WORD, width=80, height=25
            )
            self.training_log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.training_log_text.config(state=tk.DISABLED)
            
            # Контекстне меню
            self.context_menu = tk.Menu(self.training_log_text, tearoff=0)
            self.context_menu.add_command(label="Виділити все", command=self.select_all_text)
            self.context_menu.add_command(label="Скопіювати", command=self.copy_text)
            self.context_menu.add_separator()
            self.context_menu.add_command(label="Очистити лог", command=self.clear_training_log)
            
            # Прив'язка контекстного меню
            self.training_log_text.bind("<Button-3>", self.show_context_menu)  # Права кнопка миші
            self.training_log_text.bind("<Control-a>", self.select_all_text)  # Ctrl+A
            self.training_log_text.bind("<Control-c>", self.copy_text)  # Ctrl+C
            
            # Кнопки управління
            button_frame = ttk.Frame(self.training_log_window)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(button_frame, text="Очистити лог", 
                    command=self.clear_training_log).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Зберегти лог", 
                    command=self.save_training_log).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Закрити", 
                    command=self.hide_training_log).pack(side=tk.RIGHT, padx=5)

    def create_training_callbacks(self):
        """Створення callback'ів для контролю навчання"""
        callbacks = [
            # Рання зупинка при перенавчанні
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Збільшено терпеливість
                min_delta=0.0001,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Зменшення learning rate при застії
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Зменшення LR вдвічі
                patience=8,
                min_lr=0.00001,
                verbose=1
            ),
            
            # Збереження найкращих ваг
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model_weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
        ]
        
        return callbacks
    
    def show_context_menu(self, event):
        """Показ контекстного меню"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def select_all_text(self, event=None):
        """Виділення всього тексту"""
        if self.training_log_text:
            self.training_log_text.config(state=tk.NORMAL)
            self.training_log_text.tag_add(tk.SEL, "1.0", tk.END)
            self.training_log_text.config(state=tk.DISABLED)
        return "break"  # Запобігає стандартній обробці

    def copy_text(self, event=None):
        """Копіювання виділеного тексту"""
        if self.training_log_text:
            try:
                # Отримуємо виділений текст
                selected_text = self.training_log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
                self.training_log_window.clipboard_clear()
                self.training_log_window.clipboard_append(selected_text)
            except tk.TclError:
                # Якщо нічого не виділено, копіюємо все
                self.training_log_text.config(state=tk.NORMAL)
                all_text = self.training_log_text.get("1.0", tk.END)
                self.training_log_window.clipboard_clear()
                self.training_log_window.clipboard_append(all_text)
                self.training_log_text.config(state=tk.DISABLED)
        return "break"

    def save_training_log(self):
        """Збереження логу у файл"""
        if self.training_log_text:
            try:
                # Створюємо папку для логів, якщо її немає
                if not os.path.exists('training_logs'):
                    os.makedirs('training_logs')
                
                # Генеруємо ім'я файлу з поточною датою та часом
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'training_logs/training_log_{timestamp}.txt'
                
                # Зберігаємо вміст
                self.training_log_text.config(state=tk.NORMAL)
                log_content = self.training_log_text.get("1.0", tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                self.training_log_text.config(state=tk.DISABLED)
                
                self.safe_status_callback(f"Лог збережено у файл: {filename}")
                messagebox.showinfo("Успіх", f"Лог збережено у файл:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося зберегти лог: {str(e)}")

    def clear_training_log(self):
        """Очищення логу навчання"""
        if self.training_log_text:
            self.training_log_text.config(state=tk.NORMAL)
            self.training_log_text.delete(1.0, tk.END)
            self.training_log_text.config(state=tk.DISABLED)
            self.safe_status_callback("Лог очищено")

    def hide_training_log(self):
        """Приховування вікна логу"""
        if self.training_log_window:
            self.training_log_window.withdraw()

    def add_to_training_log(self, message):
        """Додавання повідомлення до логу навчання"""
        if self.training_log_text and self.show_training_log.get():
            self.training_log_text.config(state=tk.NORMAL)
            self.training_log_text.insert(tk.END, message + "\n")
            self.training_log_text.see(tk.END)
            self.training_log_text.config(state=tk.DISABLED)
            
            # Автоматичне збереження логу при додаванні повідомлень
            self.auto_save_log()

    def auto_save_log(self):
        """Автоматичне збереження логу"""
        try:
            if not hasattr(self, '_log_message_count'):
                self._log_message_count = 0
            
            self._log_message_count += 1
            
            # Зберігаємо кожні 20 повідомлень
            if self._log_message_count >= 20:
                if not os.path.exists('training_logs'):
                    os.makedirs('training_logs')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'training_logs/auto_save_log_{timestamp}.txt'
                
                if self.training_log_text:
                    self.training_log_text.config(state=tk.NORMAL)
                    log_content = self.training_log_text.get("1.0", tk.END)
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(log_content)
                    self.training_log_text.config(state=tk.DISABLED)
                    
                    self._log_message_count = 0
                    
        except Exception as e:
            # Не показуємо помилки автоматичного збереження користувачеві
            pass

#==================================log===============================================
#___________________Торг
# В MultiNeuralNetworkTab додайте ці методи
    def setup_trading_engine(self):
        """Ініціалізація торгового двигуна"""
        self.trading_engine = AdvancedTradingEngine(
            initial_balance=10000,
            risk_per_trade=0.02,
            stop_loss=0.05,
            take_profit=0.08
        )
        
        # Додайте кнопки для торгівлі
        trading_frame = ttk.LabelFrame(self.left_frame, text="Торгівля")
        trading_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(trading_frame, text="Генерувати торгові сигнали", 
                command=self.generate_trading_signals).pack(pady=2, fill=tk.X)
        ttk.Button(trading_frame, text="Виконати обраний сигнал", 
                command=self.execute_selected_signal).pack(pady=2, fill=tk.X)
        ttk.Button(trading_frame, text="Перегляд портфеля", 
                command=self.show_portfolio).pack(pady=2, fill=tk.X)

    def generate_trading_signals(self):
        """Генерація торгових сигналів на основі прогнозів"""
        if not self.current_symbol:
            messagebox.showwarning("Увага", "Спочатку оберіть криптовалюту")
            return
        
        try:
            # Отримання поточних даних
            data_file = f"{self.current_symbol}_data.csv"
            data = pd.read_csv(f'data/{data_file}', index_col=0, parse_dates=True)
            
            # Останні дані для технічного аналізу
            latest_data = data.iloc[-1].to_dict()
            
            # Додаємо технічні показники
            latest_data['rsi'] = self.calculate_rsi(data['Close']).iloc[-1]
            latest_data['volatility'] = data['Close'].pct_change().std() * np.sqrt(365)  # Річна волатильність
            latest_data['volume_ratio'] = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
            
            # Генерація сигналів
            signals = self.trading_engine.generate_advanced_signals(
                self.current_symbol, 
                self.current_predictions,  # Має містити прогнози {10: price, 20: price, 30: price}
                latest_data
            )
            
            self.display_trading_signals(signals)
            
        except Exception as e:
            self.safe_status_callback(f"Помилка генерації сигналів: {str(e)}")

    def display_trading_signals(self, signals):
        """Відображення торгових сигналів"""
        if not signals:
            messagebox.showinfo("Сигнали", "Торгові сигнали відсутні")
            return
        
        # Створення вікна для сигналів
        signals_window = tk.Toplevel(self.parent)
        signals_window.title("Торгові сигнали")
        signals_window.geometry("800x400")
        
        tree = ttk.Treeview(signals_window, columns=('Action', 'Price', 'Amount', 'Confidence', 'Reason'), show='headings')
        tree.heading('Action', text='Дія')
        tree.heading('Price', text='Ціна')
        tree.heading('Amount', text='Кількість')
        tree.heading('Confidence', text='Впевненість')
        tree.heading('Reason', text='Причина')
        
        for signal in signals:
            tree.insert('', 'end', values=(
                signal['action'],
                f"${signal['price']:.2f}",
                f"{signal['amount']:.4f}",
                f"{signal['confidence']:.0%}",
                signal['reason'][:100] + "..." if len(signal['reason']) > 100 else signal['reason']
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.current_signals = signals  # Зберігаємо для подальшого використання

    def show_portfolio(self):
        """Відображення поточного портфеля"""
        portfolio_window = tk.Toplevel(self.parent)
        portfolio_window.title("Портфель")
        portfolio_window.geometry("600x400")
        
        # Відображення балансу та позицій
        total_value = self.trading_engine.get_total_value(self.get_current_price())
        
        info_text = f"""
        Загальна вартість портфеля: ${total_value:,.2f}
        Готівка: ${self.trading_engine.balance:,.2f}
        Вкладено в позиції: ${total_value - self.trading_engine.balance:,.2f}
        Доходність: {(total_value - self.trading_engine.initial_balance) / self.trading_engine.initial_balance * 100:.2f}%
        """
        
        ttk.Label(portfolio_window, text=info_text, font=('Arial', 12)).pack(pady=10)
        
        # Таблиця позицій
        if self.trading_engine.positions:
            tree = ttk.Treeview(portfolio_window, columns=('Symbol', 'Amount', 'Entry', 'Current', 'P/L'), show='headings')
            tree.heading('Symbol', text='Криптовалюта')
            tree.heading('Amount', text='Кількість')
            tree.heading('Entry', text='Ціна входу')
            tree.heading('Current', text='Поточна ціна')
            tree.heading('P/L', text='Прибуток/Збиток')
            
            current_price = self.get_current_price()
            for symbol, position in self.trading_engine.positions.items():
                pl = (current_price - position['entry_price']) * position['amount']
                pl_pct = (current_price / position['entry_price'] - 1) * 100
                
                tree.insert('', 'end', values=(
                    symbol,
                    f"{position['amount']:.4f}",
                    f"${position['entry_price']:.2f}",
                    f"${current_price:.2f}",
                    f"${pl:.2f} ({pl_pct:+.2f}%)"
                ))
            
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def generate_symbol_signals(self, symbol):
        """Генерація сигналів для конкретного символу"""
        try:
            # Перевіряємо наявність даних
            data_file = f"{symbol}_data.csv"
            data_path = f'data/{data_file}'
            
            if not os.path.exists(data_path):
                self.safe_status_callback(f"Файл даних для {symbol} не знайдено")
                return []
            
            # Завантажуємо дані
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            if data.empty or len(data) < 60:
                self.safe_status_callback(f"Недостатньо даних для {symbol} ({len(data)} рядків)")
                return []
            
            # Завантажуємо модель
            if not self.model_manager.load_model(symbol):
                self.safe_status_callback(f"Модель для {symbol} не знайдена")
                return []
            
            # Останні дані
            latest_data = data.iloc[-1]
            current_price = latest_data['Close']
            
            print(f"Аналіз {symbol}: поточна ціна = {current_price}")
            
            # Отримуємо прогнози
            predictions = self.get_predictions_for_symbol(symbol, data)
            
            if not predictions:
                print(f"Не вдалося отримати прогнози для {symbol}")
                return []
            
            print(f"Прогнози для {symbol}: {predictions}")
            
            # Генеруємо сигнали
            signals = self.analyze_predictions(symbol, predictions, current_price, latest_data)
            
            print(f"Згенеровано сигналів для {symbol}: {len(signals)}")
            
            return signals
            
        except Exception as e:
            print(f"Помилка генерації сигналів для {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def get_predictions_for_symbol(self, symbol, data):
        """Отримання прогнозів для символу"""
        try:
            if symbol not in self.model_manager.models:
                print(f"Модель {symbol} не завантажена")
                return {}
            
            model = self.model_manager.models[symbol]
            scaler = self.model_manager.scalers[symbol]
            metrics = self.model_manager.metrics[symbol]
            
            lookback = metrics.get('lookback', 60)
            
            # Перевіряємо достатність даних
            if len(data) < lookback + 5:
                print(f"Недостатньо даних для прогнозу {symbol} (потрібно {lookback + 5}, маємо {len(data)})")
                return {}
            
            prices = data[['Close']].values
            
            # Перевіряємо наявність NaN
            if np.isnan(prices).any():
                print(f"Знайдено NaN значення в даних {symbol}")
                return {}
            
            try:
                scaled_prices = scaler.transform(prices)
            except Exception as e:
                print(f"Помилка нормалізації даних {symbol}: {str(e)}")
                return {}
            
            # Остання послідовність
            last_sequence = scaled_prices[-lookback:]
            
            # Прогноз на різні горизонти
            horizons = [1, 3, 7, 14, 30]
            predictions = {}
            
            for horizon in horizons:
                try:
                    future_predictions = []
                    current_sequence = last_sequence.copy()
                    
                    for i in range(horizon):
                        next_pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
                        future_predictions.append(next_pred[0, 0])
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[-1] = next_pred
                    
                    # Перетворення назад
                    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                    predictions[horizon] = float(future_predictions[-1][0])
                    
                except Exception as e:
                    print(f"Помилка прогнозу на {horizon} днів для {symbol}: {str(e)}")
                    continue
            
            print(f"Успішні прогнози для {symbol}: {predictions}")
            return predictions
            
        except Exception as e:
            print(f"Загальна помилка прогнозу для {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def analyze_predictions(self, symbol, predictions, current_price, latest_data):
        """Аналіз прогнозів та генерація торгових сигналів"""
        signals = []
        
        if not predictions:
            return signals
        
        # Аналіз різних горизонтів
        short_term = predictions.get(1, current_price)
        medium_term = predictions.get(7, current_price)
        long_term = predictions.get(30, current_price)
        
        # Розрахунок змін цін
        short_change = (short_term - current_price) / current_price * 100
        medium_change = (medium_term - current_price) / current_price * 100
        long_change = (long_term - current_price) / current_price * 100
        
        # Визначення сигналів
        confidence = self.calculate_confidence(predictions, current_price)
        
        # Додаткові технічні показники
        rsi = self.calculate_rsi_manual(latest_data)
        volume_ratio = self.calculate_volume_ratio(latest_data)
        
        # BUY сигнали
        if medium_change > 3 and long_change > 5 and confidence > 0.5 and rsi < 70:
            signals.append({
                'symbol': symbol,
                'action': 'BUY_STRONG',
                'current_price': current_price,
                'target_price': medium_term,
                'confidence': confidence,
                'timeframe': '7-30 днів',
                'potential_gain': f"{medium_change:.1f}%",
                'reason': f"Сильний бульish прогноз. ST: {short_change:.1f}%, MT: {medium_change:.1f}%, LT: {long_change:.1f}%. RSI: {rsi:.1f}"
            })
        
        elif medium_change > 2 and confidence > 0.4 and rsi < 65:
            signals.append({
                'symbol': symbol,
                'action': 'BUY_WEAK',
                'current_price': current_price,
                'target_price': medium_term,
                'confidence': confidence,
                'timeframe': '7 днів',
                'potential_gain': f"{medium_change:.1f}%",
                'reason': f"Слабкий бульish прогноз. Очікуваний приріст: {medium_change:.1f}%. RSI: {rsi:.1f}"
            })
        
        # SELL сигнали
        elif medium_change < -3 and confidence > 0.5 and rsi > 70:
            signals.append({
                'symbol': symbol,
                'action': 'SELL_STRONG',
                'current_price': current_price,
                'target_price': medium_term,
                'confidence': confidence,
                'timeframe': '7 днів',
                'potential_loss': f"{abs(medium_change):.1f}%",
                'reason': f"Сильний медвежий прогноз. Очікуваний спад: {medium_change:.1f}%. RSI: {rsi:.1f}"
            })
        
        # HOLD сигнали
        elif abs(medium_change) < 2 and confidence > 0.4:
            signals.append({
                'symbol': symbol,
                'action': 'HOLD',
                'current_price': current_price,
                'confidence': confidence,
                'timeframe': '7 днів',
                'expected_change': f"{medium_change:.1f}%",
                'reason': f"Нейтральний прогноз. Зміна: {medium_change:.1f}%. RSI: {rsi:.1f}"
            })
        
        return signals

    def calculate_rsi_manual(self, latest_data):
        """Спрощений розрахунок RSI"""
        # Для спрощення повертаємо фіксоване значення
        return 50.0  # Нейтральне значення

    def calculate_volume_ratio(self, latest_data):
        """Спрощений розрахунок Volume Ratio"""
        return 1.0  # Нормальний об'єм

    def calculate_confidence(self, predictions, current_price):
        """Розрахунок впевненості в прогнозі"""
        if not predictions:
            return 0.0
        
        confidence = 0.0
        changes = []
        
        # Консистентність прогнозів на різних горизонтах
        for horizon, price in predictions.items():
            if price is not None and current_price > 0:
                change = (price - current_price) / current_price
                changes.append(change)
        
        if not changes:
            return 0.0
        
        # Стабільність прогнозу
        positive_changes = sum(1 for change in changes if change > 0)
        negative_changes = sum(1 for change in changes if change < 0)
        
        if positive_changes == len(changes) or negative_changes == len(changes):
            confidence += 0.3
        
        # Величина змін
        avg_change = sum(changes) / len(changes)
        confidence += min(abs(avg_change) * 3, 0.5)  # Більші зміни = більша впевненість
        
        return min(max(confidence, 0.1), 1.0)  # Мінімальна впевненість 10%

    def display_trading_signals_window(self, all_signals, debug_info=None, selected_models=None):
        """Відображення вікна з власними кнопками управління по центру екрана"""
        # Створення головного вікна
        signals_window = tk.Toplevel(self.parent)
        signals_window.title("Торгівельні сигнали - Детальний аналіз")
        
        # Вимкнення стандартного заголовка Windows
        signals_window.overrideredirect(True)
        signals_window.configure(bg='#2b2b2b')
        
        # Встановлюємо розмір вікна
        screen_width = signals_window.winfo_screenwidth()
        screen_height = signals_window.winfo_screenheight()
        window_width = min(1400, screen_width - 100)
        window_height = min(900, screen_height - 100)
        
        # Центруємо вікно на екрані
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        signals_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Встановлюємо поверх основного вікна
        signals_window.transient(self.parent)  # Робимо вікно залежним від головного
        signals_window.attributes('-topmost', True)  # Поверх усіх вікон
        
        # Додайте атрибут для відстеження стану
        signals_window._is_alive = True
        
        def safe_destroy():
            if hasattr(signals_window, '_is_alive') and signals_window._is_alive:
                signals_window._is_alive = False
                try:
                    signals_window.grab_release()
                    signals_window.destroy()
                except tk.TclError:
                    pass
        
        def is_window_alive():
            try:
                return (hasattr(signals_window, '_is_alive') and 
                        signals_window._is_alive and 
                        signals_window.winfo_exists())
            except tk.TclError:
                return False
        
        # Власний заголовок з кнопками
        title_bar = tk.Frame(signals_window, bg='#2b2b2b', height=30, relief='raised', bd=0)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        
        # Назва вікна в заголовку
        title_label = tk.Label(title_bar, text="Торгівельні сигнали - Детальний аналіз", 
                            bg='#2b2b2b', fg='white', font=('Arial', 10, 'bold'))
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Фрейм для кнопок управління (справа)
        button_frame = tk.Frame(title_bar, bg='#2b2b2b')
        button_frame.pack(side=tk.RIGHT)
        
        # Кнопка згорнути
        min_button = tk.Button(button_frame, text="─", bg='#2b2b2b', fg='white', 
                            relief='flat', font=('Arial', 12), width=3, cursor='hand2',
                            command=lambda: signals_window.iconify() if is_window_alive() else None)
        min_button.pack(side=tk.LEFT, padx=(0, 2))
        
        # Кнопка розгорнути/відновити
        self.maximized = False
        def toggle_maximize():
            if not is_window_alive():
                return
            if self.maximized:
                # Відновлюємо нормальний розмір по центру
                x = (screen_width - window_width) // 2
                y = (screen_height - window_height) // 2
                signals_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
                max_button.config(text="□")
            else:
                # Розгортаємо на весь екран
                signals_window.geometry(f"{screen_width}x{screen_height}+0+0")
                max_button.config(text="❐")
            self.maximized = not self.maximized
        
        max_button = tk.Button(button_frame, text="□", bg='#2b2b2b', fg='white', 
                            relief='flat', font=('Arial', 10), width=3, cursor='hand2',
                            command=toggle_maximize)
        max_button.pack(side=tk.LEFT, padx=(0, 2))
        
        # Кнопка закрити
        close_button = tk.Button(button_frame, text="×", bg='#e81123', fg='white', 
                            relief='flat', font=('Arial', 12), width=3, cursor='hand2',
                            command=safe_destroy)
        close_button.pack(side=tk.LEFT)
        
        # Основний контент
        main_container = tk.Frame(signals_window, bg='white')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Стилізація кнопок при наведенні
        def on_enter_min(e):
            if is_window_alive():
                try:
                    min_button.config(bg='#505050')
                except tk.TclError:
                    pass
        def on_leave_min(e):
            if is_window_alive():
                try:
                    min_button.config(bg='#2b2b2b')
                except tk.TclError:
                    pass
        def on_enter_max(e):
            if is_window_alive():
                try:
                    max_button.config(bg='#505050')
                except tk.TclError:
                    pass
        def on_leave_max(e):
            if is_window_alive():
                try:
                    max_button.config(bg='#2b2b2b')
                except tk.TclError:
                    pass
        def on_enter_close(e):
            if is_window_alive():
                try:
                    close_button.config(bg='#f1707a')
                except tk.TclError:
                    pass
        def on_leave_close(e):
            if is_window_alive():
                try:
                    close_button.config(bg='#e81123')
                except tk.TclError:
                    pass
        
        min_button.bind("<Enter>", on_enter_min)
        min_button.bind("<Leave>", on_leave_min)
        max_button.bind("<Enter>", on_enter_max)
        max_button.bind("<Leave>", on_leave_max)
        close_button.bind("<Enter>", on_enter_close)
        close_button.bind("<Leave>", on_leave_close)
        
        # Функції для переміщення вікна
        def start_move(event):
            if is_window_alive():
                try:
                    signals_window.x = event.x_root
                    signals_window.y = event.y_root
                except tk.TclError:
                    pass
        
        def stop_move(event):
            if is_window_alive():
                try:
                    signals_window.x = None
                    signals_window.y = None
                except tk.TclError:
                    pass
        
        def do_move(event):
            if is_window_alive() and hasattr(signals_window, 'x') and hasattr(signals_window, 'y'):
                try:
                    deltax = event.x_root - signals_window.x
                    deltay = event.y_root - signals_window.y
                    x = signals_window.winfo_x() + deltax
                    y = signals_window.winfo_y() + deltay
                    signals_window.geometry(f"+{x}+{y}")
                    signals_window.x = event.x_root
                    signals_window.y = event.y_root
                except tk.TclError:
                    pass
        
        title_bar.bind("<ButtonPress-1>", start_move)
        title_bar.bind("<ButtonRelease-1>", stop_move)
        title_bar.bind("<B1-Motion>", do_move)
        title_label.bind("<ButtonPress-1>", start_move)
        title_label.bind("<ButtonRelease-1>", stop_move)
        title_label.bind("<B1-Motion>", do_move)
        
        # Контейнер для вмісту
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        title_label_content = ttk.Label(content_frame, text="📊 Детальний аналіз торгівельних сигналів", 
                                    font=('Arial', 16, 'bold'))
        title_label_content.pack(pady=(0, 15))
        
        # Створення notebook для різних типів інформації
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка з усіма сигналами
        all_signals_frame = ttk.Frame(notebook)
        notebook.add(all_signals_frame, text="📈 Всі сигнали")
        
        # Вкладка з прийнятними сигналами
        acceptable_signals_frame = ttk.Frame(notebook)
        notebook.add(acceptable_signals_frame, text="✅ Прийнятні сигнали")
        
        # Вкладка зі статистикою
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📊 Статистика")
        
        # Заповнення вкладок
        self.fill_all_signals_tab(all_signals_frame, all_signals)
        self.fill_acceptable_signals_tab(acceptable_signals_frame, all_signals)
        self.fill_stats_tab(stats_frame, all_signals, debug_info, selected_models)
        
        # Кнопки управління
        button_frame_content = ttk.Frame(content_frame)
        button_frame_content.pack(pady=10)
        
        # Кнопка експорту всіх сигналів
        def safe_export():
            if is_window_alive():
                try:
                    # Викликаємо напряму, щоб messagebox показувався поверх вікна
                    result = self.export_signals(all_signals)
                    if result:
                        # Піднімаємо вікно знову наверх після messagebox
                        signals_window.attributes('-topmost', True)
                        signals_window.after(100, lambda: signals_window.attributes('-topmost', False))
                except Exception as e:
                    # Мовчазно ігноруємо помилки експорту
                    pass
        
        export_button = ttk.Button(button_frame_content, text="💾 Експортувати всі сигнали", 
                                command=safe_export)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка збереження звіту
        def safe_save_report():
            if is_window_alive():
                try:
                    # Викликаємо напряму, щоб messagebox показувався поверх вікна
                    result = self.save_trading_analysis_report(all_signals, debug_info, selected_models)
                    if result:
                        # Піднімаємо вікно знову наверх після messagebox
                        signals_window.attributes('-topmost', True)
                        signals_window.after(100, lambda: signals_window.attributes('-topmost', False))
                except Exception as e:
                    # Мовчазно ігноруємо помилки збереження
                    pass
        
        report_button = ttk.Button(button_frame_content, text="📄 Зберегти повний звіт", 
                                command=safe_save_report)
        report_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка закрити
        close_button_content = ttk.Button(button_frame_content, text="✕ Закрити", 
                                        command=safe_destroy)
        close_button_content.pack(side=tk.LEFT, padx=5)
        
        # Фокусуємося на першій вкладці
        try:
            notebook.select(0)
        except tk.TclError:
            pass
        
        # Піднімаємо вікно на передній план і фокусуємо
        try:
            signals_window.lift()
            signals_window.focus_force()
        except tk.TclError:
            pass
        
            # Після відображення трохи знижуємо пріоритет поверхусності
        def lower_topmost():
            if is_window_alive():
                try:
                    signals_window.attributes('-topmost', False)
                    signals_window.focus_force()
                except tk.TclError:
                    pass
        
        # Зберігаємо звіт ПІСЛЯ відображення вікна
        def save_report_after_display():
            if is_window_alive():
                try:
                    # Використовуємо безпечне збереження без показу messagebox
                    reports_dir = 'trading_analysis_reports'
                    if not os.path.exists(reports_dir):
                        os.makedirs(reports_dir)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{reports_dir}/trading_analysis_{timestamp}.txt"
                    
                    report_content = self.generate_detailed_stats(all_signals, debug_info, selected_models)
                    report_content += "\n\n" + "=" * 80 + "\n"
                    report_content += "ДЕТАЛЬНИЙ ПЕРЕЛІК УСІХ СИГНАЛІВ\n"
                    report_content += "=" * 80 + "\n\n"
                    
                    for i, signal in enumerate(all_signals, 1):
                        status = "✅ ПРИЙНЯТНО" if signal['acceptable'] else "❌ НЕПРИЙНЯТНО"
                        report_content += f"{i}. {signal['symbol']} - {signal['action']} - {status}\n"
                        report_content += f"   Поточна ціна: ${signal['current_price']:.2f}\n"
                        if 'target_price' in signal:
                            report_content += f"   Цільова ціна: ${signal['target_price']:.2f}\n"
                        report_content += f"   Впевненість: {signal['confidence']:.0%}\n"
                        if 'rsi' in signal:
                            report_content += f"   RSI: {signal['rsi']:.1f}\n"
                        report_content += f"   Обґрунтування: {signal['reason']}\n"
                        report_content += "-" * 60 + "\n"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                        
                except Exception as e:
                    # Мовчазно ігноруємо помилки збереження
                    pass
        
        # Відкладаємо збереження на 500 мс після відображення
        signals_window.after(500, save_report_after_display)
        signals_window.after(100, lower_topmost)
        
        # Обробник для клавіші Escape
        def on_escape(event):
            safe_destroy()
        
        try:
            signals_window.bind('<Escape>', on_escape)
        except tk.TclError:
            pass
        
        # Блокуємо головне вікно поки відкрите це вікно
        try:
            signals_window.grab_set()
        except tk.TclError:
            pass
        
        # Обробник закриття вікна
        def on_close():
            safe_destroy()
        
        try:
            signals_window.protocol("WM_DELETE_WINDOW", on_close)
        except tk.TclError:
            pass
        
        # Додайте обробник для події знищення вікна
        def on_destroy(e):
            if hasattr(signals_window, '_is_alive'):
                signals_window._is_alive = False
        
        try:
            signals_window.bind('<Destroy>', on_destroy)
        except tk.TclError:
            pass
        
        return signals_window

    def enable_standard_window_features(self, window):
        """Активація стандартних функцій вікна Windows"""
        # Оновлюємо вікно для активації стандартних кнопок
        window.update_idletasks()
        
        # Додаємо обробник для подвійного кліку по заголовку (максимізація)
        def on_title_double_click(event):
            if window.attributes('-zoomed'):
                window.attributes('-zoomed', False)
            else:
                window.attributes('-zoomed', True)
        
        # Спроба отримати доступ до заголовку вікна
        try:
            # Це може працювати на деяких системах
            window.overrideredirect(False)
        except:
            pass
        
        # Додаємо обробник розміру вікна
        def on_resize(event):
            window.update_idletasks()
        
        window.bind('<Configure>', on_resize)
        
        # Встановлюємо фокус на вікно
        window.focus_force()

    def save_trading_report(self, signals):
        """Збереження детального звіту по сигналам"""
        try:
            if not os.path.exists('trading_reports'):
                os.makedirs('trading_reports')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trading_reports/trading_report_{timestamp}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("ЗВІТ ПО ТОРГІВЕЛЬНИМ СИГНАЛАМ\n")
                f.write("=" * 60 + "\n")
                f.write(f"Дата генерації: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Всього сигналів: {len(signals)}\n")
                f.write("=" * 60 + "\n\n")
                
                # Групуємо сигнали по типах
                signal_types = {
                    'BUY_STRONG': [],
                    'BUY_WEAK': [],
                    'SELL_STRONG': [],
                    'HOLD': []
                }
                
                for signal in signals:
                    signal_types[signal['action']].append(signal)
                
                # Записуємо по типах
                for action_type, type_signals in signal_types.items():
                    if type_signals:
                        f.write(f"\n{self.get_action_display_name(action_type)} ({len(type_signals)}):\n")
                        f.write("-" * 40 + "\n")
                        
                        for signal in type_signals:
                            f.write(f"• {signal['symbol']}: {signal['reason']}\n")
                            if 'potential_gain' in signal:
                                f.write(f"  Прибуток: {signal['potential_gain']}\n")
                            elif 'potential_loss' in signal:
                                f.write(f"  Збиток: {signal['potential_loss']}\n")
                            f.write(f"  Впевненість: {signal['confidence']:.0%}\n")
                            f.write(f"  Час: {signal.get('timeframe', 'N/A')}\n\n")
                
                # Загальна статистика
                f.write("\n" + "=" * 60 + "\n")
                f.write("ЗАГАЛЬНА СТАТИСТИКА:\n")
                f.write("=" * 60 + "\n")
                
                for action_type, type_signals in signal_types.items():
                    if type_signals:
                        f.write(f"{self.get_action_display_name(action_type)}: {len(type_signals)} сигналів\n")
                
                f.write(f"\nЗагальна рекомендация: {self.generate_overall_recommendation(signals)}\n")
            
            messagebox.showinfo("Успіх", f"Звіт збережено у файл: {filename}")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося зберегти звіт: {str(e)}")

    def get_action_display_name(self, action):
        """Отримання відображуваного імені для типу сигналу"""
        names = {
            'BUY_STRONG': '🟢 СИЛЬНІ ПОКУПКИ',
            'BUY_WEAK': '🟡 СЛАБКІ ПОКУПКИ',
            'SELL_STRONG': '🔴 ПРОДАЖІ',
            'HOLD': '⚪ УТРИМАННЯ'
        }
        return names.get(action, action)

    def generate_overall_recommendation(self, signals):
        """Генерація загальної рекомендації"""
        strong_buys = len([s for s in signals if s['action'] == 'BUY_STRONG'])
        weak_buys = len([s for s in signals if s['action'] == 'BUY_WEAK'])
        sells = len([s for s in signals if s['action'] == 'SELL_STRONG'])
        
        if strong_buys > 0:
            return "Сильні покупки - хороша можливість для входу в позиції"
        elif weak_buys > 0 and sells == 0:
            return "Слабкі покупки - можна розглянути обережне входження"
        elif sells > 0:
            return "Переважають продажі - рекомендується обережність"
        else:
            return "Нейтральний ринок - рекомендується очікування"
    
    def fill_all_signals_tab(self, parent, all_signals):
        """Заповнення вкладки з усіма сигналами"""
        if not all_signals:
            ttk.Label(parent, text="📭 Сигнали відсутні", font=('Arial', 14)).pack(pady=50)
            return
        
        # Treeview для всіх сигналів
        columns = ('Symbol', 'Action', 'Status', 'Price', 'Target', 'Change', 'Confidence', 'RSI', 'Timeframe', 'Reason')
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=20)
        
        # Налаштування колонок
        column_widths = {
            'Symbol': 80, 'Action': 80, 'Status': 100, 'Price': 80, 
            'Target': 80, 'Change': 80, 'Confidence': 90, 'RSI': 60,
            'Timeframe': 80, 'Reason': 300
        }
        
        tree.heading('Symbol', text='Криптовалюта')
        tree.heading('Action', text='Дія')
        tree.heading('Status', text='Статус')
        tree.heading('Price', text='Ціна')
        tree.heading('Target', text='Ціль')
        tree.heading('Change', text='Зміна %')
        tree.heading('Confidence', text='Впевненість')
        tree.heading('RSI', text='RSI')
        tree.heading('Timeframe', text='Період')
        tree.heading('Reason', text='Обґрунтування')
        
        for col in columns:
            tree.column(col, width=column_widths[col])
        
        # Додавання даних
        for signal in all_signals:
            status = "✅ ПРИЙНЯТНО" if signal['acceptable'] else "❌ НЕПРИЙНЯТНО"
            change = signal.get('medium_change', 0)
            confidence = f"{signal['confidence']:.0%}"
            rsi = f"{signal.get('rsi', 0):.1f}"
            
            tree.insert('', 'end', values=(
                signal['symbol'],
                signal['action'],
                status,
                f"${signal['current_price']:.2f}",
                f"${signal.get('target_price', signal['current_price']):.2f}",
                f"{change:+.1f}%",
                confidence,
                rsi,
                signal.get('timeframe', 'N/A'),
                signal['reason']
            ))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def fill_acceptable_signals_tab(self, parent, all_signals):
        """Заповнення вкладки з прийнятними сигналами"""
        acceptable_signals = [s for s in all_signals if s['acceptable']]
        
        if not acceptable_signals:
            no_signals_frame = ttk.Frame(parent)
            no_signals_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(no_signals_frame, text="📭 Прийнятні сигнали для торгівлі відсутні", 
                    font=('Arial', 14, 'bold'), foreground='orange').pack(pady=20)
            
            info_text = """
            ℹ️ Жоден з згенерованих сигналів не пройшов критерії прийнятності для торгівлі.
            
            Можливі причини:
            • Недостатня впевненість у прогнозах
            • Занадто малі очікувані зміни цін
            • Несприятливі технічні показники (RSI тощо)
            • Ризиковані умови ринку
            
            💡 Рекомендації:
            • Дочекайтеся кращих торгових можливостей
            • Перевірте інші криптовалюти
            • Оновіть моделі на свіжих даних
            """
            
            info_label = ttk.Label(no_signals_frame, text=info_text, font=('Arial', 11),
                                justify=tk.LEFT)
            info_label.pack(pady=10, padx=20)
            return
        
        # Заголовок з кількістю прийнятних сигналів
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text=f"✅ Знайдено {len(acceptable_signals)} прийнятних сигналів для торгівлі", 
                font=('Arial', 12, 'bold'), foreground='green').pack(anchor=tk.W)
        
        # Рекомендація на основі типів сигналів
        buy_signals = [s for s in acceptable_signals if 'BUY' in s['action']]
        sell_signals = [s for s in acceptable_signals if 'SELL' in s['action']]
        
        if buy_signals and not sell_signals:
            recommendation = "🟢 РЕКОМЕНДАЦІЯ: Переважають сигнали покупки - сприятливий момент для входу в позиції"
        elif sell_signals and not buy_signals:
            recommendation = "🔴 РЕКОМЕНДАЦІЯ: Переважають сигнали продажу - рекомендується обережність"
        elif buy_signals and sell_signals:
            recommendation = "🟡 РЕКОМЕНДАЦІЯ: Змішані сигнали - ретельний аналіз кожної позиції"
        else:
            recommendation = "⚪ РЕКОМЕНДАЦІЯ: Нейтральні сигнали - рекомендується очікування"
        
        ttk.Label(header_frame, text=recommendation, font=('Arial', 11), 
                foreground='blue').pack(anchor=tk.W, pady=(5, 0))
        
        # Treeview для прийнятних сигналів
        tree_container = ttk.Frame(parent)
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Action', 'Current', 'Target', 'Gain/Loss', 'Confidence', 
                'RSI', 'Timeframe', 'Potential', 'Reason')
        
        tree = ttk.Treeview(tree_container, columns=columns, show='headings', height=15)
        
        # Налаштування колонок
        column_widths = {
            'Symbol': 80, 'Action': 90, 'Current': 80, 'Target': 80, 
            'Gain/Loss': 90, 'Confidence': 90, 'RSI': 60,
            'Timeframe': 80, 'Potential': 100, 'Reason': 250
        }
        
        column_texts = {
            'Symbol': 'Криптовалюта',
            'Action': 'Дія',
            'Current': 'Поточна ціна',
            'Target': 'Цільова ціна',
            'Gain/Loss': 'Прибуток/Збиток',
            'Confidence': 'Впевненість',
            'RSI': 'RSI',
            'Timeframe': 'Період',
            'Potential': 'Потенціал',
            'Reason': 'Обґрунтування'
        }
        
        for col in columns:
            tree.heading(col, text=column_texts[col])
            tree.column(col, width=column_widths[col], anchor=tk.CENTER)
        
        # Додавання даних
        for signal in acceptable_signals:
            # Визначаємо прибуток/збиток
            gain_loss = ""
            if 'potential_gain' in signal:
                gain_loss = f"+{signal['potential_gain']}"
                potential = "📈 Високий потенціал" if float(signal['potential_gain'].replace('%', '')) > 5 else "📈 Помірний потенціал"
            elif 'potential_loss' in signal:
                gain_loss = f"-{signal['potential_loss']}"
                potential = "📉 Високий ризик" if abs(float(signal['potential_loss'].replace('%', ''))) > 5 else "📉 Помірний ризик"
            else:
                gain_loss = f"{signal.get('expected_change', '0%')}"
                potential = "➡️ Нейтральний"
            
            # Визначаємо іконку для дії
            action_icon = ""
            if 'BUY_STRONG' in signal['action']:
                action_icon = "🟢 СИЛЬНА ПОКУПКА"
            elif 'BUY_WEAK' in signal['action']:
                action_icon = "🟡 СЛАБКА ПОКУПКА"
            elif 'SELL_STRONG' in signal['action']:
                action_icon = "🔴 СИЛЬНИЙ ПРОДАЖ"
            else:
                action_icon = "⚪ УТРИМАННЯ"
            
            tree.insert('', 'end', values=(
                signal['symbol'],
                action_icon,
                f"${signal['current_price']:.2f}",
                f"${signal.get('target_price', signal['current_price']):.2f}",
                gain_loss,
                f"{signal['confidence']:.0%}",
                f"{signal.get('rsi', 0):.1f}",
                signal.get('timeframe', 'N/A'),
                potential,
                signal['reason']
            ))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Додаткові кнопки для цієї вкладки
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Кнопка для експорту тільки прийнятних сигналів
        export_button = ttk.Button(button_frame, text="💾 Експортувати прийнятні сигнали", 
                                command=lambda: self.export_signals(acceptable_signals))
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка для відображення графіків обраних сигналів
        chart_button = ttk.Button(button_frame, text="📊 Переглянути графіки", 
                                command=lambda: self.show_selected_signals_charts(acceptable_signals))
        chart_button.pack(side=tk.LEFT, padx=5)
        
        # Інформація про критерії прийнятності
        criteria_frame = ttk.Frame(parent)
        criteria_frame.pack(fill=tk.X, pady=(10, 0))
        
        criteria_text = """
        📋 Критерії прийнятності сигналів:
        • 🟢 ПОКУПКА: Впевненість > 50%, RSI < 70, очікуваний приріст > 2%
        • 🔴 ПРОДАЖ: Впевненість > 50%, RSI > 70, очікуваний спад > 3%
        • ⚪ УТРИМАННЯ: Впевненість > 40%, зміна ціни < 2%
        """
        
        criteria_label = ttk.Label(criteria_frame, text=criteria_text, font=('Arial', 9),
                                justify=tk.LEFT, foreground='gray')
        criteria_label.pack(anchor=tk.W)

    def fill_stats_tab(self, parent, all_signals, debug_info=None, selected_models=None):
        """Заповнення вкладки зі статистикою з перевірками на існування вікна"""
        try:
            # Перевіряємо, чи вікно все ще існує
            if not parent.winfo_exists():
                return
                
            # Основний фрейм
            main_frame = ttk.Frame(parent)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Перевіряємо, чи вікно все ще існує після створення фрейму
            if not parent.winfo_exists():
                return
                
            # Створюємо Text widget з прокруткою
            text_widget = tk.Text(main_frame, wrap=tk.WORD, font=('Consolas', 10), padx=15, pady=15)
            scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            # Використовуємо pack з fill та expand
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Генерація детальної статистики
            stats_text = self.generate_detailed_stats(all_signals, debug_info, selected_models)
            
            # Перевіряємо, чи вікно все ще існує перед вставкою тексту
            if parent.winfo_exists():
                text_widget.insert(1.0, stats_text)
                text_widget.config(state=tk.DISABLED)
            
            # Додаємо обробку прокрутки мишею з перевіркою
            def on_mousewheel(event):
                if parent.winfo_exists():
                    text_widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
            if parent.winfo_exists():
                text_widget.bind("<MouseWheel>", on_mousewheel)
                
        except Exception as e:
            # Ігноруємо помилки, пов'язані з закритим вікном
            if "invalid command name" in str(e) or "winfo_exists" in str(e):
                pass
            else:
                print(f"Помилка у fill_stats_tab: {e}")

    def create_signals_table(self, parent, signals, title):
        """Створення таблиці з сигналами"""
        # Заголовок
        ttk.Label(parent, text=title, font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Treeview
        columns = ('Symbol', 'Action', 'Current', 'Target', 'Gain/Loss', 'Timeframe', 'Confidence', 'Reason')
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=min(10, len(signals)))
        
        # Налаштування колонок
        tree.heading('Symbol', text='Криптовалюта')
        tree.heading('Action', text='Дія')
        tree.heading('Current', text='Поточна ціна')
        tree.heading('Target', text='Цільова ціна')
        tree.heading('Gain/Loss', text='Прибуток/Збиток')
        tree.heading('Timeframe', text='Період')
        tree.heading('Confidence', text='Впевненість')
        tree.heading('Reason', text='Обґрунтування')
        
        tree.column('Symbol', width=80)
        tree.column('Action', width=80)
        tree.column('Current', width=80)
        tree.column('Target', width=80)
        tree.column('Gain/Loss', width=100)
        tree.column('Timeframe', width=80)
        tree.column('Confidence', width=80)
        tree.column('Reason', width=300)
        
        # Додавання даних
        for signal in signals:
            gain_loss = ""
            if 'potential_gain' in signal:
                gain_loss = f"+{signal['potential_gain']}"
            elif 'potential_loss' in signal:
                gain_loss = f"-{signal['potential_loss']}"
            elif 'expected_change' in signal:
                gain_loss = signal['expected_change']
            
            tree.insert('', 'end', values=(
                signal['symbol'],
                signal['action'],
                f"${signal['current_price']:.2f}",
                f"${signal.get('target_price', signal['current_price']):.2f}",
                gain_loss,
                signal.get('timeframe', 'N/A'),
                f"{signal['confidence']:.0%}",
                signal['reason']
            ))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Кнопка експорту
        export_button = ttk.Button(parent, text="Експортувати сигнали", 
                                command=lambda: self.export_signals(signals))
        export_button.pack(pady=5)

    def export_signals(self, signals, parent_window=None):
        """Експорт сигналів у CSV файл"""
        try:
            if not signals:
                # Показуємо messagebox у тому ж потоці
                try:
                    if hasattr(self, 'parent') and self.parent.winfo_exists():
                        messagebox.showinfo("Інформація", "Немає сигналів для експорту", parent=parent_window)
                except:
                    pass
                return None
            
            if not os.path.exists('trading_signals'):
                os.makedirs('trading_signals')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trading_signals/signals_export_{timestamp}.csv'
            full_path = os.path.abspath(filename)
            
            # Підготовка даних для експорту
            export_data = []
            for signal in signals:
                export_data.append({
                    'symbol': signal['symbol'],
                    'action': signal['action'],
                    'status': 'ПРИЙНЯТНО' if signal['acceptable'] else 'НЕПРИЙНЯТНО',
                    'current_price': f"{signal['current_price']:.2f}",
                    'target_price': f"{signal.get('target_price', signal['current_price']):.2f}",
                    'confidence': f"{signal['confidence']:.0%}",
                    'rsi': f"{signal.get('rsi', 0):.1f}" if signal.get('rsi') else 'N/A',
                    'timeframe': signal.get('timeframe', 'N/A'),
                    'potential': signal.get('potential_gain', signal.get('potential_loss', signal.get('expected_change', 'N/A'))),
                    'reason': signal['reason'][:200],
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            # Показуємо повідомлення про успіх
            try:
                if hasattr(self, 'parent') and self.parent.winfo_exists():
                    # Використовуємо parent_window для messagebox
                    messagebox.showinfo("Експорт завершено", 
                                    f"✅ Сигнали успішно експортовано!\n\n"
                                    f"📁 Файл: {filename}\n"
                                    f"📂 Папка: {os.path.dirname(full_path)}\n"
                                    f"🔢 Експортовано сигналів: {len(signals)}",
                                    parent=parent_window)
                    
                    # Після закриття messagebox фокусуємося на батьківському вікні
                    if parent_window and parent_window.winfo_exists():
                        parent_window.focus_force()
                        parent_window.lift()
            except:
                pass
            
            # Відкриваємо папку з файлом (але не втрачаємо фокус)
            try:
                # Відкладаємо відкриття папки, щоб не втрачати фокус
                if parent_window and parent_window.winfo_exists():
                    parent_window.after(1000, lambda: os.startfile(os.path.dirname(full_path)))
            except:
                pass
            
            return filename
            
        except Exception as e:
            error_msg = f"❌ Не вдалося експортувати сигнали: {str(e)}"
            try:
                if hasattr(self, 'parent') and self.parent.winfo_exists():
                    messagebox.showerror("Помилка експорту", error_msg, parent=parent_window)
                    if parent_window and parent_window.winfo_exists():
                        parent_window.focus_force()
            except:
                pass
            return None

    def save_trading_analysis_report(self, all_signals, debug_info=None, selected_models=None, parent_window=None):
        """Збереження повного звіту аналізу торгових сигналів"""
        try:
            reports_dir = 'trading_analysis_reports'
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{reports_dir}/trading_analysis_{timestamp}.txt"
            full_path = os.path.abspath(filename)
            
            report_content = self.generate_detailed_stats(all_signals, debug_info, selected_models)
            report_content += "\n\n" + "=" * 80 + "\n"
            report_content += "ДЕТАЛЬНИЙ ПЕРЕЛІК УСІХ СИГНАЛІВ\n"
            report_content += "=" * 80 + "\n\n"
            
            for i, signal in enumerate(all_signals, 1):
                status = "✅ ПРИЙНЯТНО" if signal['acceptable'] else "❌ НЕПРИЙНЯТНО"
                report_content += f"{i}. {signal['symbol']} - {signal['action']} - {status}\n"
                report_content += f"   Поточна ціна: ${signal['current_price']:.2f}\n"
                if 'target_price' in signal:
                    report_content += f"   Цільова ціна: ${signal['target_price']:.2f}\n"
                report_content += f"   Впевненість: {signal['confidence']:.0%}\n"
                if 'rsi' in signal:
                    report_content += f"   RSI: {signal['rsi']:.1f}\n"
                report_content += f"   Обґрунтування: {signal['reason']}\n"
                report_content += "-" * 60 + "\n"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Показуємо повідомлення про успіх
            try:
                if hasattr(self, 'parent') and self.parent.winfo_exists():
                    messagebox.showinfo("Звіт збережено", 
                                    f"Повний звіт аналізу збережено у файл:\n{full_path}",
                                    parent=parent_window)
                    
                    # Після закриття messagebox фокусуємося на батьківському вікні
                    if parent_window and parent_window.winfo_exists():
                        parent_window.focus_force()
                        parent_window.lift()
            except:
                pass
            
            return filename
            
        except Exception as e:
            error_msg = f"Не вдалося зберегти звіт: {str(e)}"
            try:
                if hasattr(self, 'parent') and self.parent.winfo_exists():
                    messagebox.showerror("Помилка збереження", error_msg, parent=parent_window)
                    if parent_window and parent_window.winfo_exists():
                        parent_window.focus_force()
            except:
                pass
            return None
    
    def show_trading_signals(self):
        """Показ торгівельних сигналів для вибраних навчених моделей"""
        if not self.trained_models:
            messagebox.showwarning("Увага", "Спочатку навчіть моделі")
            return
        
        # Очищаємо попередній лог
        self.clear_debug_log()
        
        # Діалог вибору моделей для аналізу
        selected_models = self.ask_user_to_select_models(
            title="Оберіть моделі для аналізу",
            prompt="Оберіть криптовалюті для генерації торгових сигналів:",
            mode="models"
        )
        
        if not selected_models:
            self.safe_status_callback("Вибір скасовано")
            return
        
        # Записуємо початок аналізу в лог
        self.log_debug_info("=" * 80)
        self.log_debug_info("ПОЧАТОК АНАЛІЗУ ТОРГІВЕЛЬНИХ СИГНАЛІв")
        self.log_debug_info("=" * 80)
        self.log_debug_info(f"Обрано моделей для аналізу: {len(selected_models)}")
        self.log_debug_info(f"Моделі: {', '.join(selected_models)}")
        self.log_debug_info("")
        
        # Використовуємо lambda для передачі контексту
        def generate_signals_thread():
            try:
                self.safe_status_callback("Генерація торгівельних сигналів...")
                self.safe_progress_callback(10)
                
                all_signals = []
                debug_info = []
                
                for i, symbol in enumerate(selected_models):
                    self.safe_status_callback(f"Аналіз {symbol} ({i+1}/{len(selected_models)})...")
                    self.safe_progress_callback(10 + (i / len(selected_models)) * 80)
                    
                    try:
                        signals, symbol_debug = self.generate_symbol_signals_with_debug(symbol)
                        debug_info.append(symbol_debug)
                        
                        if signals:
                            all_signals.extend(signals)
                            self.log_debug_info(f"✅ Знайдено {len(signals)} сигналів для {symbol}")
                        else:
                            self.log_debug_info(f"❌ Немає сигналів для {symbol}")
                            
                    except Exception as e:
                        error_msg = f"Помилка аналізу {symbol}: {str(e)}"
                        self.log_debug_info(f"❌ {error_msg}")
                        debug_info.append({'symbol': symbol, 'error': error_msg})
                        continue
                
                self.safe_progress_callback(95)
                
                # Виводимо детальну налагоджувальну інформацію у лог
                self.log_debug_info("")
                self.log_debug_info("ДЕТАЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ:")
                self.log_debug_info("-" * 60)
                self.print_debug_info_to_log(debug_info, selected_models, all_signals)
                
                # ВІДОБРАЖЕННЯ СИГНАЛІВ - ВИКЛИКАЄМО ЗАВЖДИ, НАВІТЬ ЯКЩО СИГНАЛІВ НЕМАЄ
                try:
                    self.display_trading_signals_window(all_signals, debug_info, selected_models)
                    self.log_debug_info("✅ Вікно з сигналами успішно відображено")
                except Exception as e:
                    self.log_debug_info(f"❌ Помилка відображення вікна: {str(e)}")
                
                self.safe_status_callback("Генерація сигналів завершена")
                self.safe_progress_callback(100)
                
                # Записуємо завершення аналізу
                self.log_debug_info("")
                self.log_debug_info("=" * 80)
                self.log_debug_info("АНАЛІЗ ЗАВЕРШЕНО")
                self.log_debug_info("=" * 80)
                
            except Exception as e:
                error_msg = f"Помилка генерації сигналів: {str(e)}"
                self.safe_status_callback(error_msg)
                self.safe_progress_callback(0)
                self.log_debug_info(f"🔥 КРИТИЧНА ПОМИЛКА: {error_msg}")
                
                # Показуємо messagebox тільки якщо головне вікно існує
                try:
                    if self.parent.winfo_exists():
                        messagebox.showerror("Помилка", f"Не вдалося згенерувати сигнали: {str(e)}")
                except:
                    pass
        
        # Використовуємо правильний спосіб створення потоку
        thread = threading.Thread(target=generate_signals_thread)
        thread.daemon = True
        thread.start()

    def generate_symbol_signals_with_debug(self, symbol):
        """Генерація сигналів з детальним налагоджувальним виводом"""
        debug_info = {
            'symbol': symbol,
            'has_data': False,
            'data_rows': 0,
            'has_model': False,
            'predictions': {},
            'current_price': 0,
            'price_changes': {},
            'all_signals_count': 0,  # Змінено з signals_generated
            'acceptable_signals_count': 0,  # Додано нове поле
            'reasons': [],
            'issues': []
        }
        
        try:
            self.log_debug_info(f"🔍 Аналіз символу: {symbol}")
            
            # Перевіряємо наявність даних
            data_file = f"{symbol}_data.csv"
            data_path = f'data/{data_file}'
            
            if not os.path.exists(data_path):
                debug_info['issues'].append("Файл даних не знайдено")
                self.log_debug_info(f"   ❌ Файл даних не знайдено: {data_path}")
                return [], debug_info
            
            # Завантажуємо дані
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            if data.empty:
                debug_info['issues'].append("Файл даних порожній")
                self.log_debug_info("   ❌ Файл даних порожній")
                return [], debug_info
            
            debug_info['has_data'] = True
            debug_info['data_rows'] = len(data)
            self.log_debug_info(f"   ✅ Дані знайдено: {len(data)} рядків")
            
            if len(data) < 60:
                debug_info['issues'].append(f"Недостатньо даних для аналізу (потрібно ≥60, маємо {len(data)})")
                self.log_debug_info(f"   ⚠️ Недостатньо даних: {len(data)} рядків (потрібно ≥60)")
                return [], debug_info
            
            # Перевіряємо наявність колонки Close
            if 'Close' not in data.columns:
                debug_info['issues'].append("Відсутня колонка 'Close' в даних")
                self.log_debug_info("   ❌ Відсутня колонка 'Close' в даних")
                return [], debug_info
            
            # Завантажуємо модель
            if not self.model_manager.load_model(symbol):
                debug_info['issues'].append("Модель не знайдена або не завантажена")
                self.log_debug_info("   ❌ Модель не знайдена або не завантажена")
                return [], debug_info
            
            debug_info['has_model'] = True
            self.log_debug_info("   ✅ Модель успішно завантажена")
            
            # Останні дані
            latest_data = data.iloc[-1]
            current_price = latest_data['Close']
            
            if pd.isna(current_price) or current_price <= 0:
                debug_info['issues'].append(f"Некоректна поточна ціна: {current_price}")
                self.log_debug_info(f"   ❌ Некоректна поточна ціна: {current_price}")
                return [], debug_info
            
            debug_info['current_price'] = current_price
            self.log_debug_info(f"   💰 Поточна ціна: ${current_price:.2f}")
            
            # Отримуємо прогнози
            predictions = self.get_predictions_for_symbol(symbol, data)
            
            if not predictions:
                debug_info['issues'].append("Не вдалося отримати прогнози")
                self.log_debug_info("   ❌ Не вдалося отримати прогнози")
                return [], debug_info
            
            debug_info['predictions'] = predictions
            
            # Розраховуємо зміни цін
            for horizon, predicted_price in predictions.items():
                if predicted_price is not None and current_price > 0:
                    change = ((predicted_price - current_price) / current_price) * 100
                    debug_info['price_changes'][horizon] = change
                    self.log_debug_info(f"   📈 Прогноз {horizon} дн.: ${predicted_price:.2f} ({change:+.2f}%)")
            
            # Генеруємо ВСІ сигнали (не фільтруємо)
            all_signals = self.generate_all_signals(symbol, predictions, current_price, latest_data)
            debug_info['all_signals_count'] = len(all_signals)  # Оновлено поле
            
            # Фільтруємо сигнали за критеріями прийнятності
            acceptable_signals = self.filter_signals(all_signals)
            debug_info['acceptable_signals_count'] = len(acceptable_signals)  # Додано нове поле
            
            # Збираємо причини для сигналів
            for signal in all_signals:
                debug_info['reasons'].append(signal['reason'])
                status = "✅ ПРИЙНЯТНО" if signal['acceptable'] else "❌ НЕПРИЙНЯТНО"
                self.log_debug_info(f"   📨 Сигнал: {signal['action']} - {status} - {signal['reason']}")
            
            self.log_debug_info(f"   ✅ Всіх сигналів: {len(all_signals)}")
            self.log_debug_info(f"   ✅ Прийнятних сигналів: {len(acceptable_signals)}")
            
            return all_signals, debug_info  # Повертаємо ВСІ сигнали, не тільки відфільтровані
            
        except Exception as e:
            error_msg = f"Загальна помилка: {str(e)}"
            debug_info['issues'].append(error_msg)
            self.log_debug_info(f"   🔥 Помилка: {error_msg}")
            import traceback
            debug_info['traceback'] = traceback.format_exc()
            return [], debug_info

    def generate_all_signals(self, symbol, predictions, current_price, latest_data):
        """Генерація всіх можливих сигналів без фільтрації"""
        signals = []
        
        if not predictions:
            return signals
        
        # Аналіз різних горизонтів
        short_term = predictions.get(1, current_price)
        medium_term = predictions.get(7, current_price)
        long_term = predictions.get(30, current_price)
        
        # Розрахунок змін цін
        short_change = (short_term - current_price) / current_price * 100
        medium_change = (medium_term - current_price) / current_price * 100
        long_change = (long_term - current_price) / current_price * 100
        
        # Додаткові технічні показники
        rsi = self.calculate_rsi_manual(latest_data)
        volume_ratio = self.calculate_volume_ratio(latest_data)
        
        # Генерація всіх можливих сигналів
        signals.extend(self.generate_buy_signals(symbol, predictions, current_price, short_change, medium_change, long_change, rsi))
        signals.extend(self.generate_sell_signals(symbol, predictions, current_price, short_change, medium_change, long_change, rsi))
        signals.extend(self.generate_hold_signals(symbol, predictions, current_price, short_change, medium_change, long_change, rsi))
        
        return signals

    def generate_buy_signals(self, symbol, predictions, current_price, short_change, medium_change, long_change, rsi):
        """Генерація buy сигналів"""
        signals = []
        confidence = self.calculate_confidence(predictions, current_price)
        
        # Сильний buy сигнал
        strong_buy = {
            'symbol': symbol,
            'action': 'BUY_STRONG',
            'current_price': current_price,
            'target_price': predictions.get(7, current_price),
            'confidence': confidence,
            'short_change': short_change,
            'medium_change': medium_change,
            'long_change': long_change,
            'rsi': rsi,
            'timeframe': '7-30 днів',
            'potential_gain': f"{medium_change:.1f}%",
            'reason': f"Сильний бульish прогноз. ST: {short_change:.1f}%, MT: {medium_change:.1f}%, LT: {long_change:.1f}%. RSI: {rsi:.1f}",
            'acceptable': medium_change > 3 and long_change > 5 and confidence > 0.5 and rsi < 70
        }
        signals.append(strong_buy)
        
        # Слабкий buy сигнал
        weak_buy = {
            'symbol': symbol,
            'action': 'BUY_WEAK',
            'current_price': current_price,
            'target_price': predictions.get(7, current_price),
            'confidence': confidence,
            'short_change': short_change,
            'medium_change': medium_change,
            'long_change': long_change,
            'rsi': rsi,
            'timeframe': '7 днів',
            'potential_gain': f"{medium_change:.1f}%",
            'reason': f"Слабкий бульish прогноз. Очікуваний приріст: {medium_change:.1f}%. RSI: {rsi:.1f}",
            'acceptable': medium_change > 2 and confidence > 0.4 and rsi < 65
        }
        signals.append(weak_buy)
        
        return signals

    def generate_sell_signals(self, symbol, predictions, current_price, short_change, medium_change, long_change, rsi):
        """Генерація sell сигналів"""
        signals = []
        confidence = self.calculate_confidence(predictions, current_price)
        
        # Сильний sell сигнал
        strong_sell = {
            'symbol': symbol,
            'action': 'SELL_STRONG',
            'current_price': current_price,
            'target_price': predictions.get(7, current_price),
            'confidence': confidence,
            'short_change': short_change,
            'medium_change': medium_change,
            'long_change': long_change,
            'rsi': rsi,
            'timeframe': '7 днів',
            'potential_loss': f"{abs(medium_change):.1f}%",
            'reason': f"Сильний медвежий прогноз. Очікуваний спад: {medium_change:.1f}%. RSI: {rsi:.1f}",
            'acceptable': medium_change < -3 and confidence > 0.5 and rsi > 70
        }
        signals.append(strong_sell)
        
        return signals

    def generate_hold_signals(self, symbol, predictions, current_price, short_change, medium_change, long_change, rsi):
        """Генерація hold сигналів"""
        signals = []
        confidence = self.calculate_confidence(predictions, current_price)
        
        # Hold сигнал
        hold = {
            'symbol': symbol,
            'action': 'HOLD',
            'current_price': current_price,
            'confidence': confidence,
            'short_change': short_change,
            'medium_change': medium_change,
            'long_change': long_change,
            'rsi': rsi,
            'timeframe': '7 днів',
            'expected_change': f"{medium_change:.1f}%",
            'reason': f"Нейтральний прогноз. Зміна: {medium_change:.1f}%. RSI: {rsi:.1f}",
            'acceptable': abs(medium_change) < 2 and confidence > 0.4
        }
        signals.append(hold)
        
        return signals

    def filter_signals(self, signals):
        """Фільтрація сигналів за критеріями прийнятності"""
        return [signal for signal in signals if signal['acceptable']]
    
    def generate_detailed_stats(self, all_signals, debug_info=None, selected_models=None):
        """Генерація детальної статистики аналізу"""
        stats_text = "=" * 80 + "\n"
        stats_text += "📊 ДЕТАЛЬНА СТАТИСТИКА АНАЛІЗУ ТОРГІВЕЛЬНИХ СИГНАЛІВ\n"
        stats_text += "=" * 80 + "\n\n"
        
        # Загальна інформація
        stats_text += f"📅 Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        stats_text += f"🔢 Всього проаналізовано моделей: {len(debug_info) if debug_info else 0}\n"
        stats_text += f"📈 Всього згенеровано сигналів: {len(all_signals)}\n\n"
        
        # Статистика по типах сигналів
        if all_signals:
            strong_buy = len([s for s in all_signals if s['action'] == 'BUY_STRONG'])
            weak_buy = len([s for s in all_signals if s['action'] == 'BUY_WEAK'])
            strong_sell = len([s for s in all_signals if s['action'] == 'SELL_STRONG'])
            hold = len([s for s in all_signals if s['action'] == 'HOLD'])
            
            acceptable = len([s for s in all_signals if s['acceptable']])
            unacceptable = len(all_signals) - acceptable
            
            stats_text += "🎯 СТАТИСТИКА ПО ТИПАХ СИГНАЛІВ:\n"
            stats_text += "-" * 50 + "\n"
            stats_text += f"• 🟢 СИЛЬНІ ПОКУПКИ: {strong_buy} ({strong_buy/len(all_signals)*100:.1f}%)\n"
            stats_text += f"• 🟡 СЛАБКІ ПОКУПКИ: {weak_buy} ({weak_buy/len(all_signals)*100:.1f}%)\n"
            stats_text += f"• 🔴 СИЛЬНІ ПРОДАЖІ: {strong_sell} ({strong_sell/len(all_signals)*100:.1f}%)\n"
            stats_text += f"• ⚪ УТРИМАННЯ: {hold} ({hold/len(all_signals)*100:.1f}%)\n\n"
            
            stats_text += f"• ✅ ПРИЙНЯТНІ СИГНАЛИ: {acceptable} ({acceptable/len(all_signals)*100:.1f}%)\n"
            stats_text += f"• ❌ НЕПРИЙНЯТНІ СИГНАЛИ: {unacceptable} ({unacceptable/len(all_signals)*100:.1f}%)\n\n"
        
        # Статистика по криптовалютах
        if all_signals:
            symbols_stats = {}
            for signal in all_signals:
                symbol = signal['symbol']
                if symbol not in symbols_stats:
                    symbols_stats[symbol] = {'total': 0, 'acceptable': 0}
                symbols_stats[symbol]['total'] += 1
                if signal['acceptable']:
                    symbols_stats[symbol]['acceptable'] += 1
            
            stats_text += "💰 СТАТИСТИКА ПО КРИПТОВАЛЮТАХ:\n"
            stats_text += "-" * 50 + "\n"
            for symbol, stats in symbols_stats.items():
                percentage = stats['acceptable'] / stats['total'] * 100 if stats['total'] > 0 else 0
                stats_text += f"• {symbol}: {stats['acceptable']}/{stats['total']} прийнятних ({percentage:.1f}%)\n"
            stats_text += "\n"
        
        # Детальна інформація про моделі
        if debug_info:
            stats_text += "🤖 ДЕТАЛЬНИЙ АНАЛІЗ МОДЕЛЕЙ:\n"
            stats_text += "-" * 50 + "\n"
            
            successful_models = 0
            for info in debug_info:
                status = "✅" if info.get('acceptable_signals_count', 0) > 0 else "❌"
                stats_text += f"{status} {info['symbol']}: {info.get('acceptable_signals_count', 0)}/{info.get('all_signals_count', 0)} прийнятних сигналів"
                
                if info.get('issues'):
                    stats_text += f" - Проблеми: {', '.join(info['issues'][:2])}"
                    if len(info['issues']) > 2:
                        stats_text += f"... (+{len(info['issues'])-2} more)"
                
                stats_text += "\n"
                
                if info.get('acceptable_signals_count', 0) > 0:
                    successful_models += 1
            
            stats_text += f"\n📈 Ефективність моделей: {successful_models}/{len(debug_info)} ({successful_models/len(debug_info)*100:.1f}%)\n\n"
        
        # Рекомендації
        stats_text += "💡 РЕКОМЕНДАЦІЇ:\n"
        stats_text += "-" * 50 + "\n"
        
        if not all_signals:
            stats_text += "📭 Сигнали відсутні. Рекомендується:\n"
            stats_text += "• Перевірити наявність даних\n"
            stats_text += "• Оновити моделі на свіжих даних\n"
            stats_text += "• Аналізувати інші криптовалюти\n"
        else:
            acceptable_signals = [s for s in all_signals if s['acceptable']]
            if not acceptable_signals:
                stats_text += "⚠️ Прийнятні сигнали відсутні. Рекомендується:\n"
                stats_text += "• Дочекатися кращих торгових умов\n"
                stats_text += "• Переглянути критерії прийнятності\n"
                stats_text += "• Аналізувати інші активи\n"
            else:
                buy_signals = [s for s in acceptable_signals if 'BUY' in s['action']]
                sell_signals = [s for s in acceptable_signals if 'SELL' in s['action']]
                
                if buy_signals and not sell_signals:
                    stats_text += "🟢 СИЛЬНІ ПОКУПКИ! Рекомендується:\n"
                    stats_text += f"• Розглянути {len(buy_signals)} сигналів покупки\n"
                    stats_text += "• Поступове входження в позиції\n"
                    stats_text += "• Встановлення стоп-лоссів\n"
                elif sell_signals and not buy_signals:
                    stats_text += "🔴 СИЛЬНІ ПРОДАЖІ! Рекомендується:\n"
                    stats_text += f"• Розглянути {len(sell_signals)} сигналів продажу\n"
                    stats_text += "• Захист існуючих позицій\n"
                    stats_text += "• Обережність при нових інвестиціях\n"
                else:
                    stats_text += "🟡 ЗМІШАНІ СИГНАЛИ. Рекомендується:\n"
                    stats_text += "• Ретельний аналіз кожної позиції\n"
                    stats_text += "• Диверсифікація портфеля\n"
                    stats_text += "• Обережність у торгівлі\n"
        
        stats_text += "\n" + "=" * 80 + "\n"
        stats_text += "КІНЕЦЬ ЗВІТУ\n"
        stats_text += "=" * 80
        
        return stats_text

    
    def print_debug_info(self, debug_info, selected_models, all_signals):
        """Вивід детальної налагоджувальної інформації"""
        print("\n" + "="*80)
        print("ДЕТАЛЬНИЙ АНАЛІЗ ГЕНЕРАЦІЇ ТОРГІВЕЛЬНИХ СИГНАЛІВ")
        print("="*80)
        
        print(f"📋 Обрано моделей для аналізу: {len(selected_models)}")
        print(f"📈 Знайдено сигналів: {len(all_signals)}")
        print(f"🔍 Детальний аналіз по кожній моделі:")
        print("-" * 80)
        
        for i, info in enumerate(debug_info, 1):
            print(f"\n{i}. 📊 {info['symbol']}:")
            print(f"   ✅ Дані: {'Так' if info['has_data'] else 'Ні'} ({info.get('data_rows', 0)} рядків)")
            print(f"   🤖 Модель: {'Так' if info['has_model'] else 'Ні'}")
            print(f"   💰 Поточна ціна: ${info.get('current_price', 0):.2f}")
            
            if info['predictions']:
                print("   📊 Прогнози та зміни цін:")
                for horizon, price in info['predictions'].items():
                    change = info['price_changes'].get(horizon, 0)
                    change_symbol = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                    print(f"     {horizon:2d} дн.: ${price:8.2f} {change_symbol} {change:+.2f}%")
            
            print(f"   📨 Сигналів згенеровано: {info['signals_generated']}")
            
            if info['reasons']:
                print("   📝 Причини сигналів:")
                for reason in info['reasons']:
                    print(f"     • {reason}")
            
            if info['issues']:
                print("   ❗ Проблеми:")
                for issue in info['issues']:
                    print(f"     ⚠️  {issue}")
            
            if 'traceback' in info:
                print("   🔍 Stack trace:")
                print(f"     {info['traceback']}")
            
            print("-" * 80)
        
        # Аналіз ефективності моделей
        successful_models = sum(1 for info in debug_info if info['signals_generated'] > 0)
        failed_models = len(debug_info) - successful_models
        
        print(f"\n📊 ЗВОДКА:")
        print(f"   ✅ Успішні моделі: {successful_models}")
        print(f"   ❌ Моделі без сигналів: {failed_models}")
        print(f"   📈 Загальна ефективність: {successful_models/len(debug_info)*100:.1f}%")
        print("="*80)

    def show_detailed_no_signals_message(self, debug_info, selected_models):
        """Показ детального повідомлення про відсутність сигналів"""
        # Аналізуємо конкретні причини
        issues_counter = {
            'no_data': 0,
            'insufficient_data': 0,
            'no_model': 0,
            'no_predictions': 0,
            'invalid_price': 0,
            'small_changes': 0,
            'other': 0
        }
        
        for info in debug_info:
            if not info['has_data']:
                issues_counter['no_data'] += 1
            elif info.get('data_rows', 0) < 60:
                issues_counter['insufficient_data'] += 1
            elif not info['has_model']:
                issues_counter['no_model'] += 1
            elif not info['predictions']:
                issues_counter['no_predictions'] += 1
            elif info.get('current_price', 0) <= 0:
                issues_counter['invalid_price'] += 1
            elif info['signals_generated'] == 0 and info['predictions']:
                # Аналізуємо величину змін
                max_change = max([abs(change) for change in info['price_changes'].values()]) if info['price_changes'] else 0
                if max_change < 2.0:  # Дуже маленькі зміни
                    issues_counter['small_changes'] += 1
                else:
                    issues_counter['other'] += 1
            else:
                issues_counter['other'] += 1
        
        # Формуємо детальне повідомлення
        message = "📊 Торгові сигнали відсутні для обраних моделей\n\n"
        message += "🔍 Детальний аналіз причин:\n\n"
        
        if issues_counter['no_data'] > 0:
            message += f"• 📁 Відсутні файли даних: {issues_counter['no_data']} модель(ей)\n"
        if issues_counter['insufficient_data'] > 0:
            message += f"• 📉 Недостатньо історичних даних: {issues_counter['insufficient_data']} модель(ей)\n"
        if issues_counter['no_model'] > 0:
            message += f"• 🤖 Відсутні навчені моделі: {issues_counter['no_model']} модель(ей)\n"
        if issues_counter['no_predictions'] > 0:
            message += f"• 📉 Не вдалося отримати прогнози: {issues_counter['no_predictions']} модель(ей)\n"
        if issues_counter['invalid_price'] > 0:
            message += f"• 💰 Некоректні ціни: {issues_counter['invalid_price']} модель(ей)\n"
        if issues_counter['small_changes'] > 0:
            message += f"• 📏 Малі зміни цін у прогнозах (<2%): {issues_counter['small_changes']} модель(ей)\n"
        if issues_counter['other'] > 0:
            message += f"• 🔍 Інші причини: {issues_counter['other']} модель(ей)\n"
        
        message += f"\n📈 Загальна статистика:"
        message += f"\n   • Обрано моделей: {len(selected_models)}"
        message += f"\n   • Проаналізовано: {len(debug_info)}"
        message += f"\n   • Успішно: {sum(1 for info in debug_info if info['signals_generated'] > 0)}"
        message += f"\n   • Не вдалося: {sum(1 for info in debug_info if info['signals_generated'] == 0)}"
        
        message += "\n\n💡 Рекомендації:"
        message += "\n• Перевірте наявність файлів у папці 'data/'"
        message += "\n• Переконайтесь, що моделі успішно навчені"
        message += "\n• Перевірте якість та кількість даних"
        message += "\n• Дивіться консоль для детального аналізу"
        
        messagebox.showinfo("Аналіз завершено - Сигнали відсутні", message)

    
    def safe_destroy_window(self, window):
        """Безпечне закриття вікна з обробкою помилок"""
        try:
            if window and window.winfo_exists():
                # Звільняємо grab перед закриттям
                try:
                    window.grab_release()
                except:
                    pass
                window.destroy()
        except Exception as e:
            print(f"Помилка закриття вікна: {e}")



    def show_no_signals_message(self, debug_info):
        """Показ детального повідомлення про відсутність сигналів"""
        # Аналізуємо причини
        issues_summary = {
            'no_data': 0,
            'no_model': 0,
            'no_predictions': 0,
            'small_changes': 0,
            'other': 0
        }
        
        for info in debug_info:
            if not info['has_data']:
                issues_summary['no_data'] += 1
            elif not info['has_model']:
                issues_summary['no_model'] += 1
            elif not info['predictions']:
                issues_summary['no_predictions'] += 1
            elif info['signals_generated'] == 0 and info['predictions']:
                # Аналізуємо величину змін у прогнозах
                changes = []
                for horizon, price in info['predictions'].items():
                    if info['current_price'] > 0:
                        change = abs((price - info['current_price']) / info['current_price'] * 100)
                        changes.append(change)
                
                if changes and max(changes) < 2.0:  # Дуже маленькі зміни
                    issues_summary['small_changes'] += 1
                else:
                    issues_summary['other'] += 1
        
        # Формуємо детальне повідомлення
        message = "Торгові сигнали відсутні для обраних моделей.\n\n"
        message += "Детальний аналіз причин:\n\n"
        
        if issues_summary['no_data'] > 0:
            message += f"• Відсутні дані: {issues_summary['no_data']} модель(ей)\n"
        if issues_summary['no_model'] > 0:
            message += f"• Відсутні моделі: {issues_summary['no_model']} модель(ей)\n"
        if issues_summary['no_predictions'] > 0:
            message += f"• Не вдалося отримати прогнози: {issues_summary['no_predictions']} модель(ей)\n"
        if issues_summary['small_changes'] > 0:
            message += f"• Малі зміни цін у прогнозах: {issues_summary['small_changes']} модель(ей)\n"
        if issues_summary['other'] > 0:
            message += f"• Інші причини: {issues_summary['other']} модель(ей)\n"
        
        message += "\nРекомендації:\n"
        message += "• Перевірте наявність даних у папці 'data/'\n"
        message += "• Переконайтесь, що моделі успішно навчені\n"
        message += "• Спробуйте навчити моделі на більшій кількості даних\n"
        
        messagebox.showinfo("Сигнали відсутні", message)

    


    def log_debug_info(self, message, log_file='trading_signals_debug.log'):
        """Запис налагоджувальної інформації у файл логу"""
        try:
            # Створюємо папку для логів, якщо її немає
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            log_path = os.path.join(log_dir, log_file)
            
            # Додаємо timestamp до повідомлення
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] {message}\n"
            
            # Записуємо у файл
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_message)
                
        except Exception as e:
            print(f"Помилка запису в лог: {str(e)}")

    def clear_debug_log(self, log_file='trading_signals_debug.log'):
        """Очищення файлу логу"""
        try:
            log_dir = 'logs'
            log_path = os.path.join(log_dir, log_file)
            if os.path.exists(log_path):
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write("")  # Очищаємо файл
        except Exception as e:
            print(f"Помилка очищення логу: {str(e)}")
    
    def print_debug_info_to_log(self, debug_info, selected_models, all_signals):
        """Вивід детальної налагоджувальної інформації у файл логу"""
        self.log_debug_info(f"📋 Обрано моделей для аналізу: {len(selected_models)}")
        self.log_debug_info(f"📈 Знайдено сигналів: {len(all_signals)}")
        self.log_debug_info("")
        self.log_debug_info("🔍 ДЕТАЛЬНИЙ АНАЛІЗ ПО КОЖНІЙ МОДЕЛІ:")
        self.log_debug_info("-" * 60)
        
        for i, info in enumerate(debug_info, 1):
            self.log_debug_info(f"{i}. 📊 {info['symbol']}:")
            self.log_debug_info(f"   ✅ Дані: {'Так' if info['has_data'] else 'Ні'} ({info.get('data_rows', 0)} рядків)")
            self.log_debug_info(f"   🤖 Модель: {'Так' if info['has_model'] else 'Ні'}")
            
            if info.get('current_price', 0) > 0:
                self.log_debug_info(f"   💰 Поточна ціна: ${info['current_price']:.2f}")
            
            if info['predictions']:
                self.log_debug_info("   📊 Прогнози та зміни цін:")
                for horizon, price in info['predictions'].items():
                    change = info['price_changes'].get(horizon, 0)
                    change_symbol = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                    self.log_debug_info(f"     {horizon:2d} дн.: ${price:8.2f} {change_symbol} {change:+.2f}%")
            
            # ОНОВЛЕНО: Використовуємо нові поля
            self.log_debug_info(f"   📨 Всіх сигналів: {info.get('all_signals_count', 0)}")
            self.log_debug_info(f"   ✅ Прийнятних сигналів: {info.get('acceptable_signals_count', 0)}")
            
            if info['reasons']:
                self.log_debug_info("   📝 Причини сигналів:")
                for reason in info['reasons']:
                    self.log_debug_info(f"     • {reason}")
            
            if info['issues']:
                self.log_debug_info("   ❗ Проблеми:")
                for issue in info['issues']:
                    self.log_debug_info(f"     ⚠️  {issue}")
            
            self.log_debug_info("-" * 60)
        
        # Аналіз ефективності моделей
        successful_models = sum(1 for info in debug_info if info.get('acceptable_signals_count', 0) > 0)
        failed_models = len(debug_info) - successful_models
        
        self.log_debug_info("")
        self.log_debug_info("📊 ЗВОДКА:")
        self.log_debug_info(f"   ✅ Моделі з прийнятними сигналами: {successful_models}")
        self.log_debug_info(f"   ❌ Моделі без прийнятних сигналів: {failed_models}")
        self.log_debug_info(f"   📈 Загальна ефективність: {successful_models/len(debug_info)*100:.1f}%")
        
        # Детальний аналіз причин
        if failed_models > 0:
            self.log_debug_info("")
            self.log_debug_info("🔎 АНАЛІЗ ПРИЧИН ВІДСУТНОСТІ ПРИЙНЯТНИХ СИГНАЛІВ:")
            
            issues_counter = {
                'no_data': 0,
                'insufficient_data': 0,
                'no_model': 0,
                'no_predictions': 0,
                'invalid_price': 0,
                'no_acceptable_signals': 0,
                'other': 0
            }
            
            for info in debug_info:
                if info.get('acceptable_signals_count', 0) == 0:
                    if not info['has_data']:
                        issues_counter['no_data'] += 1
                    elif info.get('data_rows', 0) < 60:
                        issues_counter['insufficient_data'] += 1
                    elif not info['has_model']:
                        issues_counter['no_model'] += 1
                    elif not info['predictions']:
                        issues_counter['no_predictions'] += 1
                    elif info.get('current_price', 0) <= 0:
                        issues_counter['invalid_price'] += 1
                    elif info.get('all_signals_count', 0) > 0:
                        issues_counter['no_acceptable_signals'] += 1
                    else:
                        issues_counter['other'] += 1
            
            for issue_type, count in issues_counter.items():
                if count > 0:
                    issue_name = {
                        'no_data': 'Відсутні файли даних',
                        'insufficient_data': 'Недостатньо даних',
                        'no_model': 'Відсутні моделі',
                        'no_predictions': 'Не вдалося отримати прогнози',
                        'invalid_price': 'Некоректні ціни',
                        'no_acceptable_signals': 'Немає прийнятних сигналів',
                        'other': 'Інші причини'
                    }.get(issue_type, issue_type)
                    
                    self.log_debug_info(f"   • {issue_name}: {count} модель(ей)")
    
    def ask_user_to_select_models(self, title="Оберіть моделі", prompt="Оберіть моделі:", mode="models"):
        """
        Універсальне діалогове вікно для вибору моделей або файлів
        mode: "models" - для вибору навчених моделей, "files" - для вибору файлів для навчання
        """
        selection_window = tk.Toplevel(self.parent)
        selection_window.title(title)
        selection_window.transient(self.parent)
        selection_window.grab_set()
        
        # Налаштування стилів
        style = ttk.Style()
        style.configure('Accent.TButton', foreground='white', background='#0078D7')
        style.configure('Treeview', font=('Arial', 9))
        style.configure('Treeview.Heading', font=('Arial', 9, 'bold'))
        
        # Отримуємо дані в залежності від режиму
        if mode == "models":
            items = self.model_manager.get_available_models()
            columns = ('Select', 'Symbol', 'MSE', 'MAE', 'Date')
            column_texts = {
                'Select': '✅', 
                'Symbol': 'Криптовалюта', 
                'MSE': 'MSE', 
                'MAE': 'MAE', 
                'Date': 'Дата навчання'
            }
            column_widths = {'Select': 60, 'Symbol': 120, 'MSE': 80, 'MAE': 80, 'Date': 120}
        else:  # mode == "files"
            items = self.get_all_files()
            columns = ('Select', 'Symbol', 'File')
            column_texts = {
                'Select': '✅', 
                'Symbol': 'Криптовалюта', 
                'File': 'Файл даних'
            }
            column_widths = {'Select': 60, 'Symbol': 120, 'File': 300}
        
        num_items = len(items)
        
        # Адаптивна висота
        if num_items <= 5:
            height = 450
        elif num_items <= 10:
            height = 550
        else:
            height = 650
        
        selection_window.geometry(f"700x{height}")
        selection_window.minsize(650, 400)
        
        selected_items = []
        
        # Головний контейнер
        main_container = ttk.Frame(selection_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Заголовок
        ttk.Label(main_container, text=prompt, font=('Arial', 11, 'bold')).pack(pady=(0, 15))
        
        # Treeview з прокруткою
        tree_container = ttk.Frame(main_container)
        tree_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tree = ttk.Treeview(tree_container, columns=columns, show='headings', height=min(15, num_items))
        
        # Налаштування колонок
        for col in columns:
            tree.heading(col, text=column_texts[col])
            tree.column(col, width=column_widths[col], minwidth=column_widths[col] - 10)
        
        # Прокрутка
        v_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Розміщення
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
        
        # Заповнення даними
        if mode == "models":
            for symbol in items:
                metrics = self.model_manager.get_model_metrics(symbol)
                date_str = metrics.get('timestamp', 'N/A')
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_str)[:10]
                
                tree.insert('', 'end', values=(
                    '☐', 
                    symbol, 
                    f"{metrics.get('mse', 0):.6f}", 
                    f"{metrics.get('mae', 0):.6f}",
                    date_str
                ))
        else:  # mode == "files"
            for file in items:
                symbol = file.replace('_data.csv', '')
                tree.insert('', 'end', values=('☐', symbol, file))
        
        # Обробник вибору
        def on_select_click(event):
            if tree.identify_region(event.x, event.y) == "cell":
                col = tree.identify_column(event.x)
                if col == "#1":  # Колонка Select
                    item = tree.identify_row(event.y)
                    if item:
                        current = tree.set(item, 'Select')
                        tree.set(item, 'Select', '☑' if current == '☐' else '☐')
        
        tree.bind('<Button-1>', on_select_click)
        
        # Функції для кнопок
        def select_all():
            for item in tree.get_children():
                tree.set(item, 'Select', '☑')
        
        def deselect_all():
            for item in tree.get_children():
                tree.set(item, 'Select', '☐')
        
        def on_confirm():
            if mode == "models":
                selected = [tree.set(item, 'Symbol') for item in tree.get_children() 
                        if tree.set(item, 'Select') == '☑']
            else:  # mode == "files"
                selected = [tree.set(item, 'File') for item in tree.get_children() 
                        if tree.set(item, 'Select') == '☑']
            
            if selected:
                selection_window.result = selected
                selection_window.destroy()
            else:
                messagebox.showwarning("Увага", "Оберіть хоча б один елемент")
        
        def on_cancel():
            selection_window.result = []
            selection_window.destroy()
        
        # Фрейм для кнопок керування вибором
        control_button_frame = ttk.Frame(main_container)
        control_button_frame.pack(fill=tk.X, pady=(15, 10))
        
        control_buttons = [
            ("Обрати все", select_all),
            ("Скасувати все", deselect_all)
        ]
        
        for i, (text, command) in enumerate(control_buttons):
            btn = ttk.Button(control_button_frame, text=text, command=command, width=15)
            btn.grid(row=0, column=i, padx=5, sticky='ew')
            control_button_frame.grid_columnconfigure(i, weight=1)
        
        # Фрейм для кнопок дій
        action_button_frame = ttk.Frame(main_container)
        action_button_frame.pack(fill=tk.X, pady=(0, 5))
        
        action_buttons = [
            ("Підтвердити", on_confirm, 'Accent.TButton'),
            ("Скасувати", on_cancel, 'TButton')
        ]
        
        for i, (text, command, btn_style) in enumerate(action_buttons):
            btn = ttk.Button(action_button_frame, text=text, command=command, 
                            width=15, style=btn_style)
            btn.grid(row=0, column=i, padx=5, sticky='ew')
            action_button_frame.grid_columnconfigure(i, weight=1)
        
        # Додаємо обробник прокрутки мишею
        def on_mousewheel(event):
            tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        tree.bind("<MouseWheel>", on_mousewheel)
        
        # Центрування вікна
        selection_window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - selection_window.winfo_width()) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - selection_window.winfo_height()) // 2
        selection_window.geometry(f"+{x}+{y}")
        
        # Фокусуємося на вікні
        selection_window.focus_force()
        
        # Активуємо перший елемент
        if tree.get_children():
            tree.focus(tree.get_children()[0])
        
        # Очікуємо вибору
        selection_window.wait_window()
        
        return getattr(selection_window, 'result', [])
    
    def get_all_files(self):
        """Отримання списку всіх файлів даних"""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return []
        
        files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
        return sorted(files)
    
    #---------------------торг

    def safe_tk_command(self, func):
        """Декоратор для безпечного виконання tkinter команд"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tk.TclError as e:
                if "invalid command name" in str(e):
                    # Ігноруємо помилки неіснуючих команд
                    return None
                else:
                    # Перенаправляємо інші помилки в статус
                    self.safe_status_callback(f"Tkinter error: {str(e)}")
                    return None
        return wrapper

    def check_window_exists(self, window):
        """Перевіряє, чи існує вікно"""
        try:
            return window.winfo_exists()
        except (tk.TclError, AttributeError):
            return False

    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        try:
            if self.status_callback and hasattr(self, 'parent') and self.parent.winfo_exists():
                self.status_callback(message)
        except (tk.TclError, AttributeError):
            # Ігноруємо помилки, пов'язані з знищеними віджетами
            pass

    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        try:
            if (self.progress_callback and hasattr(self, 'parent') and 
                self.parent.winfo_exists()):
                self.progress_callback(value)
        except (tk.TclError, AttributeError):
            # Ігноруємо помилки, пов'язані з знищеними віджетами
            pass

    def bring_window_to_front(self, window):
        """Піднімає вікно на передній план"""
        try:
            if window and window.winfo_exists():
                window.attributes('-topmost', True)
                window.after(100, lambda: window.attributes('-topmost', False))
                window.focus_force()
        except tk.TclError:
            pass




