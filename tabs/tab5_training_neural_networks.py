import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import os
import threading
import time
from datetime import datetime
import json
import sys
from io import StringIO
from utils.data_validator import DataValidator
from utils.file_selector import FileSelector
from utils.model_manager import ModelManager
from utils.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# Вимкнення зайвих попереджень
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warnings, 3 = errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class TrainingProgressCallback(Callback):
    """Кастомний callback для відстеження прогресу навчання"""
    def __init__(self, status_callback, progress_callback, total_epochs, log_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self.log_callback = log_callback
        self.best_epoch = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
    
    def on_train_begin(self, logs=None):
        if self.log_callback:
            self.log_callback(f"🏁 Початок навчання. Заплановано епох: {self.total_epochs}\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        current_epoch = epoch + 1
        message = f"Епоха {current_epoch}/{self.total_epochs}..."
        self.status_callback(message)
        if self.log_callback:
            self.log_callback(f"🔹 {message}\n")
    
    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        progress = (current_epoch / self.total_epochs) * 100
        self.progress_callback(min(progress, 100))
        
        # Відстеження найкращої епохи
        current_loss = logs.get('val_loss', float('inf'))
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            message = f"🎯 Найкращий результат на епосі {self.best_epoch}: val_loss={current_loss:.6f}"
            self.status_callback(message)
            if self.log_callback:
                self.log_callback(f"✅ {message}\n")
    
    def on_train_end(self, logs=None):
        training_time = time.time() - self.start_time
        if self.log_callback:
            self.log_callback(f"⏱️ Час навчання: {training_time:.1f} секунд\n")
            self.log_callback(f"🏆 Найкраща епоха: {self.best_epoch} (val_loss={self.best_loss:.6f})\n\n")

class TrainingNeuralNetworksTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.model_manager = ModelManager()
        self.current_training_thread = None
        self.training_stop_flag = False
        self.log_window = None
        self.log_text = None
        self.original_stdout = None
        self.original_stderr = None
        self.setup_ui()
    
    def setup_ui(self):
        """Налаштування інтерфейсу навчання нейромереж"""
        # Основні фрейми
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Лівий фрейм - параметри навчання з фіксованою шириною
        left_frame_container = ttk.Frame(main_frame, width=300)
        left_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame_container.pack_propagate(False)  # Фіксуємо ширину
        
        # Створюємо Canvas і Scrollbar для прокрутки
        canvas = tk.Canvas(left_frame_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame_container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Додаємо прокрутку мишею
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Центральний фрейм - вибір даних
        center_frame = ttk.LabelFrame(main_frame, text="Вибір даних")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правий фрейм - графік навчання
        right_frame = ttk.LabelFrame(main_frame, text="Графік навчання")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Параметри навчання (тепер у scrollable_frame)
        ttk.Label(scrollable_frame, text="Тип навчання:", font=('Arial', 10, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=10)
        self.training_type_var = tk.StringVar(value="basic")
        training_types = [("Базове", "basic"), ("Розширене", "advanced"), ("Експертне", "expert")]
        for text, value in training_types:
            ttk.Radiobutton(scrollable_frame, text=text, variable=self.training_type_var, value=value).pack(anchor=tk.W, padx=20)
        
        ttk.Label(scrollable_frame, text="Основні параметри:", font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)
        
        ttk.Label(scrollable_frame, text="Кількість епох:").pack(anchor=tk.W, padx=20)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(scrollable_frame, textvariable=self.epochs_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        ttk.Label(scrollable_frame, text="Розмір батча:").pack(anchor=tk.W, padx=20)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(scrollable_frame, textvariable=self.batch_size_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        ttk.Label(scrollable_frame, text="Розмір вікна:").pack(anchor=tk.W, padx=20)
        self.lookback_var = tk.IntVar(value=60)
        ttk.Entry(scrollable_frame, textvariable=self.lookback_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        ttk.Label(scrollable_frame, text="Розмір тестової вибірки (%):").pack(anchor=tk.W, padx=20)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(scrollable_frame, textvariable=self.test_size_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        ttk.Label(scrollable_frame, text="Learning Rate:").pack(anchor=tk.W, padx=20)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(scrollable_frame, textvariable=self.lr_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        ttk.Label(scrollable_frame, text="Параметри ранньої зупинки:", font=('Arial', 10, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=10)
        
        ttk.Label(scrollable_frame, text="Patience (терпіння):").pack(anchor=tk.W, padx=20)
        self.patience_var = tk.IntVar(value=10)
        ttk.Entry(scrollable_frame, textvariable=self.patience_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        ttk.Label(scrollable_frame, text="Мінімальне покращення:").pack(anchor=tk.W, padx=20)
        self.min_delta_var = tk.DoubleVar(value=0.0001)
        ttk.Entry(scrollable_frame, textvariable=self.min_delta_var, width=15).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # Додаткові опції
        ttk.Label(scrollable_frame, text="Додаткові опції:", font=('Arial', 10, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=10)
        
        self.use_technical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Технічні індикатори", 
                    variable=self.use_technical_var).pack(anchor=tk.W, padx=20)
        
        self.use_time_features_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Часові ознаки", 
                    variable=self.use_time_features_var).pack(anchor=tk.W, padx=20)
        
        ttk.Label(scrollable_frame, text="Покращені параметри навчання:", font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)

        # Dropout rates
        ttk.Label(scrollable_frame, text="Dropout Rate (початковий):").pack(anchor=tk.W, padx=20)
        self.dropout_start_var = tk.DoubleVar(value=0.2)
        ttk.Entry(scrollable_frame, textvariable=self.dropout_start_var, width=20).pack(anchor=tk.W, padx=20, pady=(0, 5))

        ttk.Label(scrollable_frame, text="Dropout Rate (кінцевий):").pack(anchor=tk.W, padx=20)
        self.dropout_end_var = tk.DoubleVar(value=0.4)
        ttk.Entry(scrollable_frame, textvariable=self.dropout_end_var, width=20).pack(anchor=tk.W, padx=20, pady=(0, 5))

        # Кількість LSTM шарів
        ttk.Label(scrollable_frame, text="Кількість LSTM шарів:").pack(anchor=tk.W, padx=20)
        self.lstm_layers_var = tk.IntVar(value=2)
        ttk.Entry(scrollable_frame, textvariable=self.lstm_layers_var, width=20).pack(anchor=tk.W, padx=20, pady=(0, 5))

        # Розмір LSTM юнітів
        ttk.Label(scrollable_frame, text="Розмір LSTM юнітів (початковий):").pack(anchor=tk.W, padx=20)
        self.lstm_units_start_var = tk.IntVar(value=64)
        ttk.Entry(scrollable_frame, textvariable=self.lstm_units_start_var, width=20).pack(anchor=tk.W, padx=20, pady=(0, 5))

        ttk.Label(scrollable_frame, text="Розмір LSTM юнітів (кінцевий):").pack(anchor=tk.W, padx=20)
        self.lstm_units_end_var = tk.IntVar(value=32)
        ttk.Entry(scrollable_frame, textvariable=self.lstm_units_end_var, width=20).pack(anchor=tk.W, padx=20, pady=(0, 5))

        # Додаткові технічні індикатори
        ttk.Label(scrollable_frame, text="Розширені технічні індикатори:", font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)

        self.use_rsi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="RSI (Relative Strength Index)", 
                    variable=self.use_rsi_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_macd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="MACD", 
                    variable=self.use_macd_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_bollinger_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Смуги Боллінджера", 
                    variable=self.use_bollinger_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_atr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="ATR (Average True Range)", 
                    variable=self.use_atr_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_volume_indicators_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Вольюм індикатори", 
                    variable=self.use_volume_indicators_var, width=25).pack(anchor=tk.W, padx=20)

        # Покращені часові ознаки
        ttk.Label(scrollable_frame, text="Покращені часові ознаки:", font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)

        self.use_seasonal_features_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Сезонність (рік/квартал)", 
                    variable=self.use_seasonal_features_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_cyclical_features_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Циклічні ознаки (sin/cos)", 
                    variable=self.use_cyclical_features_var, width=25).pack(anchor=tk.W, padx=20)

        # Методи регуляризації
        ttk.Label(scrollable_frame, text="Методи регуляризації:", font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)

        self.use_batch_norm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Batch Normalization", 
                    variable=self.use_batch_norm_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_l2_reg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="L2 Regularization", 
                    variable=self.use_l2_reg_var, width=25).pack(anchor=tk.W, padx=20)

        self.use_early_stopping_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Рання зупинка", 
                    variable=self.use_early_stopping_var, width=25).pack(anchor=tk.W, padx=20)
        
        
        ttk.Label(scrollable_frame, text="Торгові індикатори:", font=('Arial', 10, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=10)

        self.use_volatility_indicators_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Волатильність (ATR, Volatility)", 
                    variable=self.use_volatility_indicators_var).pack(anchor=tk.W, padx=20)

        self.use_momentum_indicators_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Momentum (RSI, Stochastic, MACD)", 
                    variable=self.use_momentum_indicators_var).pack(anchor=tk.W, padx=20)

        self.use_volume_indicators_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Об'ємні індикатори (OBV, Volume)", 
                    variable=self.use_volume_indicators_var).pack(anchor=tk.W, padx=20)

        self.use_market_indicators_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Ринкові показники (BTC correlation)", 
                    variable=self.use_market_indicators_var).pack(anchor=tk.W, padx=20)

        self.use_risk_metrics_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Метрики ризику (VaR, Drawdown)", 
                    variable=self.use_risk_metrics_var).pack(anchor=tk.W, padx=20)

        # Цільові змінні для торгівлі
        ttk.Label(scrollable_frame, text="Цільові змінні:", font=('Arial', 10, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=10)

        self.forecast_horizon_var = tk.StringVar(value="5")
        ttk.Label(scrollable_frame, text="Горизонт прогнозу (днів):").pack(anchor=tk.W, padx=20)
        ttk.Entry(scrollable_frame, textvariable=self.forecast_horizon_var, width=10).pack(anchor=tk.W, padx=20, pady=(0, 10))

        self.use_price_target_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Прогноз ціни", 
                    variable=self.use_price_target_var).pack(anchor=tk.W, padx=20)

        self.use_return_target_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Прогноз доходності", 
                    variable=self.use_return_target_var).pack(anchor=tk.W, padx=20)

        self.use_signal_target_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Торгові сигнали", 
                    variable=self.use_signal_target_var).pack(anchor=tk.W, padx=20)
        
        
            # Додаємо нову секцію для вибору ознак
        ttk.Label(scrollable_frame, text="Вибір ознак для навчання:", 
                font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)
        
        # Основні цінові ознаки
        self.use_close_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Ціна закриття (Close)", 
                        variable=self.use_close_var).pack(anchor=tk.W, padx=20)
        
        self.use_high_low_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="High/Low ціни", 
                        variable=self.use_high_low_var).pack(anchor=tk.W, padx=20)
        
        self.use_open_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Ціна відкриття (Open)", 
                        variable=self.use_open_var).pack(anchor=tk.W, padx=20)
        
        self.use_volume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Об'єм торгів (Volume)", 
                        variable=self.use_volume_var).pack(anchor=tk.W, padx=20)
        
        # Технічні індикатори
        ttk.Label(scrollable_frame, text="Технічні індикатори:", 
                font=('Arial', 9, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=20)
        
        self.use_returns_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Денна доходність (Returns)", 
                        variable=self.use_returns_var).pack(anchor=tk.W, padx=30)
        
        self.use_ma_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Ковзні середні (MA)", 
                        variable=self.use_ma_var).pack(anchor=tk.W, padx=30)
        
        self.use_volatility_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Волатильність", 
                        variable=self.use_volatility_var).pack(anchor=tk.W, padx=30)
        
        self.use_rsi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="RSI", 
                        variable=self.use_rsi_var).pack(anchor=tk.W, padx=30)
        
        self.use_macd_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="MACD", 
                        variable=self.use_macd_var).pack(anchor=tk.W, padx=30)
        
        # Часові ознаки
        ttk.Label(scrollable_frame, text="Часові ознаки:", 
                font=('Arial', 9, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=20)
        
        self.use_day_of_week_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="День тижня", 
                        variable=self.use_day_of_week_var).pack(anchor=tk.W, padx=30)
        
        self.use_month_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Місяць", 
                        variable=self.use_month_var).pack(anchor=tk.W, padx=30)
        
        self.use_quarter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Квартал", 
                        variable=self.use_quarter_var).pack(anchor=tk.W, padx=30)
        


                # Додаємо після чекбоксів у setup_ui
        ttk.Label(scrollable_frame, text="Автоматичний вибір:", 
                font=('Arial', 9, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=20)

        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        ttk.Button(button_frame, text="Мінімальний", 
                command=lambda: self.auto_select_features("minimal")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Торгівельний", 
                command=lambda: self.auto_select_features("trading")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Повний", 
                command=lambda: self.auto_select_features("comprehensive")).pack(side=tk.LEFT, padx=2)

        ttk.Label(scrollable_frame, text="Автоматичний підбір LR:").pack(anchor=tk.W, padx=20)
        self.auto_lr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="Автоматичний підбір Learning Rate", 
                    variable=self.auto_lr_var).pack(anchor=tk.W, padx=20)

        ttk.Label(scrollable_frame, text="Мінімальна кореляція:").pack(anchor=tk.W, padx=20)
        self.min_correlation_var = tk.DoubleVar(value=0.15)
        ttk.Entry(scrollable_frame, textvariable=self.min_correlation_var, width=10).pack(anchor=tk.W, padx=20)
        
        # Чекбокс для показу логу
        self.show_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scrollable_frame, text="Показувати лог навчання", 
                    variable=self.show_log_var, command=self.toggle_log_window).pack(anchor=tk.W, padx=20, pady=(0, 15))
        
        
        
        
        # Кнопки управління даними
        ttk.Label(scrollable_frame, text="Управління даними:", font=('Arial', 10, 'bold')).pack(pady=(10, 5), anchor=tk.W, padx=10)
        
        ttk.Button(scrollable_frame, text="Оновити список", command=self.refresh_data_list, width=20).pack(pady=5, padx=20)
        ttk.Button(scrollable_frame, text="Вибрати всі", command=self.select_all, width=20).pack(pady=2, padx=20)
        ttk.Button(scrollable_frame, text="Скасувати вибір", command=self.deselect_all, width=20).pack(pady=2, padx=20)
        
        ttk.Label(scrollable_frame, text="Навчання моделей:", font=('Arial', 10, 'bold')).pack(pady=(15, 5), anchor=tk.W, padx=10)
        
        ttk.Button(scrollable_frame, text="Навчити обрані", command=self.train_selected, width=20).pack(pady=5, padx=20)
        ttk.Button(scrollable_frame, text="Навчити всі", command=self.train_all, width=20).pack(pady=2, padx=20)
        ttk.Button(scrollable_frame, text="Зупинити навчання", command=self.stop_training, width=20).pack(pady=2, padx=20)
        ttk.Button(scrollable_frame, text="Очистити моделі", command=self.cleanup_models, width=20).pack(pady=5, padx=20)
        
        # Вибір даних
        tree_frame = ttk.Frame(center_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.data_tree = ttk.Treeview(tree_frame, columns=('Symbol', 'Records', 'Last Update'), 
                                    show='headings', height=15, selectmode='extended')
        self.data_tree.heading('Symbol', text='Криптовалюта')
        self.data_tree.heading('Records', text='Записів')
        self.data_tree.heading('Last Update', text='Оновлено')
        
        self.data_tree.column('Symbol', width=100)
        self.data_tree.column('Records', width=80)
        self.data_tree.column('Last Update', width=100)
        
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Додаємо прокрутку мишею для Treeview
        def _on_tree_mousewheel(event):
            self.data_tree.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.data_tree.bind("<Enter>", lambda e: self.data_tree.bind_all("<MouseWheel>", _on_tree_mousewheel))
        self.data_tree.bind("<Leave>", lambda e: self.data_tree.unbind_all("<MouseWheel>"))
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Графік навчання
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Інформаційний текст
        info_frame = ttk.LabelFrame(right_frame, text="Інформація про навчання")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=50)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Статусна інформація
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Статус:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Готовий до навчання", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Завантажуємо список даних
        self.refresh_data_list()
        
        # Оновлюємо область прокрутки
        self.parent.after(100, self.update_scroll_region)

    def update_scroll_region(self):
        """Оновлює область прокрутки після додавання елементів"""
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'configure'):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def refresh_data_list(self):
        """Оновлює список доступних даних"""
        self.data_tree.delete(*self.data_tree.get_children())
        files = FileSelector.get_sorted_files()
        
        for file in files:
            try:
                symbol = file.replace('_data.csv', '')
                file_path = f'data/{file}'
                
                if os.path.exists(file_path):
                    # Отримуємо кількість рядків
                    df = pd.read_csv(file_path)
                    records = len(df)
                    
                    # Отримуємо дату останнього оновлення
                    mod_time = os.path.getmtime(file_path)
                    last_update = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
                    
                    self.data_tree.insert('', 'end', values=(symbol, records, last_update))
            except Exception as e:
                self.status_callback(f"Помилка завантаження {file}: {str(e)}")
    
    def select_all(self):
        """Вибирає всі елементи"""
        for item in self.data_tree.get_children():
            self.data_tree.selection_add(item)
    
    def deselect_all(self):
        """Скасовує вибір всіх елементів"""
        self.data_tree.selection_remove(self.data_tree.selection())
    
    def get_selected_symbols(self):
        """Повертає список вибраних символів"""
        selected_items = self.data_tree.selection()
        return [self.data_tree.item(item, 'values')[0] for item in selected_items]
    
    def train_selected(self):
        """Навчає вибрані моделі"""
        selected_symbols = self.get_selected_symbols()
        if not selected_symbols:
            messagebox.showwarning("Увага", "Оберіть хоча б один символ для навчання")
            return
        self.start_training(selected_symbols)
    
    def train_all(self):
        """Навчає всі моделі"""
        all_symbols = [self.data_tree.item(item, 'values')[0] for item in self.data_tree.get_children()]
        if not all_symbols:
            messagebox.showwarning("Увага", "Немає доступних символів для навчання")
            return
        self.start_training(all_symbols)
    
    def start_training(self, symbols):
        """Запуск навчання"""
        if self.current_training_thread and self.current_training_thread.is_alive():
            messagebox.showwarning("Увага", "Навчання вже виконується")
            return
        
        # Перевірка параметрів перед запуском
        training_params = self.get_training_parameters()
        validation_errors = self.validate_training_parameters(training_params)
        
        if validation_errors:
            error_msg = "Помилка в параметрах навчання:\n\n" + "\n".join(validation_errors)
            messagebox.showerror("Помилка параметрів", error_msg)
            return
        
        self.training_stop_flag = False
        self.current_training_thread = threading.Thread(
            target=self.training_worker, args=(symbols,)
        )
        self.current_training_thread.daemon = True
        self.current_training_thread.start()

    def training_worker(self, symbols):
        """Робоча функція навчання для багатьох моделей"""
        try:
            total_symbols = len(symbols)
            trained_count = 0
            failed_count = 0
            
            # Отримуємо параметри з UI
            training_params = self.get_training_parameters()
            
            # Перевірка валідності параметрів
            validation_errors = self.validate_training_parameters(training_params)
            if validation_errors:
                error_msg = "❌ Помилка в параметрах навчання:\n" + "\n".join(validation_errors)
                self.update_info_text(error_msg)
                self.status_callback("Помилка в параметрах навчання")
                return
            
            self.update_info_text(f"🚀 Початок навчання {total_symbols} моделей...\n")
            
            for i, symbol in enumerate(symbols):
                if self.training_stop_flag:
                    self.update_info_text("⏹️ Навчання зупинено користувачем")
                    break
                
                # Оновлення прогресу для загального процесу
                progress = (i / total_symbols) * 100
                self.update_progress(progress)
                self.status_callback(f"Навчання {symbol} ({i+1}/{total_symbols})...")
                
                try:
                    success = self.train_single_model(symbol, training_params)
                    if success:
                        trained_count += 1
                    else:
                        failed_count += 1
                        
                    # Невелика пауза між навчанням моделей
                    time.sleep(1)
                        
                except Exception as e:
                    error_msg = f"❌ Помилка навчання {symbol}: {str(e)}"
                    self.status_callback(error_msg)
                    self.update_info_text(error_msg)
                    failed_count += 1
                    continue
            
            # Результати
            result_msg = f"✅ Навчання завершено. Успішно: {trained_count}, Не вдалося: {failed_count}"
            self.status_callback(result_msg)
            self.update_info_text(f"\n{result_msg}")
            self.update_progress(100)
            
            if trained_count > 0:
                messagebox.showinfo("Результат", result_msg)
                
        except Exception as e:
            error_msg = f"❌ Критична помилка навчання: {str(e)}"
            self.status_callback(error_msg)
            self.update_info_text(error_msg)
            self.update_progress(0)

    def train_single_model(self, symbol, training_params=None):
        """Навчання однієї моделі з урахуванням усіх параметрів"""
        if training_params is None:
            training_params = self.get_training_parameters()
        
        file_path = f'data/{symbol}_data.csv'
        if not os.path.exists(file_path):
            self.update_info_text(f"❌ Файл {file_path} не знайдено")
            self.add_log_message(f"❌ Файл {file_path} не знайдено\n")
            return False

        try:
            self.add_log_message(f"\n{'='*60}\n")
            self.add_log_message(f"=== ПОЧАТОК НАВЧАННЯ {symbol} ===\n")
            self.add_log_message(f"{'='*60}\n\n")
            
            # Завантаження даних
            self.add_log_message("📥 Завантаження даних...\n")
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.add_log_message(f"✅ Завантажено {len(data)} записів\n")
            self.add_log_message(f"📅 Діапазон даних: {data.index[0]} до {data.index[-1]}\n\n")

            # Перевірка даних
            self.add_log_message("🔍 Перевірка якості даних...\n")
            try:
                DataValidator.check_data_requirements(data, lambda msg: self.add_log_message(f"   {msg}\n"))
                self.add_log_message("✅ Дані відповідають вимогам\n")
            except Exception as e:
                self.add_log_message(f"❌ Помилка перевірки даних: {str(e)}\n")
                return False

            # Класифікація криптовалюти та волатильності
            crypto_type = self.classify_crypto_type(symbol, data)
            volatility_type = self.classify_asset_volatility(data, symbol)
            
            self.add_log_message(f"🔍 Класифікація {symbol}: Тип - {crypto_type}, Волатильність - {volatility_type}\n")
            
            # Отримуємо UI параметри
            ui_params = {
                'epochs': self.epochs_var.get(),
                'batch_size': self.batch_size_var.get(),
                'lookback': self.lookback_var.get(),
                'learning_rate': self.lr_var.get(),
                'patience': self.patience_var.get(),
                'min_delta': self.min_delta_var.get(),
                'test_size': self.test_size_var.get(),
                'use_early_stopping': self.use_early_stopping_var.get(),
                'auto_lr': self.auto_lr_var.get()
            }
            
            # Отримуємо профіль навчання з урахуванням типу криптовалюти
            training_profile = self.get_training_profile(crypto_type, volatility_type, ui_params)
            
            # Оновлюємо параметри навчання
            training_params.update(training_profile)
            
            self.add_log_message("⚙️ ПАРАМЕТРИ НАВЧАННЯ:\n")
            self.add_log_message(f"  Тип криптовалюти: {crypto_type}\n")
            self.add_log_message(f"  Тип волатильності: {volatility_type}\n")
            self.add_log_message(f"  Епохи: {training_params['epochs']}\n")
            self.add_log_message(f"  Розмір батча: {training_params['batch_size']}\n")
            self.add_log_message(f"  Розмір вікна: {training_params['lookback']}\n")
            self.add_log_message(f"  Learning Rate: {training_params['learning_rate']}\n")
            self.add_log_message(f"  Розмір тестової вибірки: {training_params['test_size']}\n")
            self.add_log_message(f"  Патінс: {training_params['patience']}\n")
            self.add_log_message(f"  Мін. дельта: {training_params['min_delta']}\n")
            self.add_log_message(f"  Макс. ознак: {training_params['max_features']}\n")
            self.add_log_message(f"  Мін. кореляція: {training_params['min_correlation']}\n")
            self.add_log_message(f"  Рання зупинка: {training_params['use_early_stopping']}\n")
            self.add_log_message(f"  Авто LR: {training_params['auto_lr']}\n")
            
            if training_params['use_early_stopping']:
                self.add_log_message(f"  Patience: {training_params['patience']}\n")
                self.add_log_message(f"  Min Delta: {training_params['min_delta']}\n")
            
            self.add_log_message("\n")

            # Аналіз кореляції ознак
            self.add_log_message("📈 АНАЛІЗ КОРЕЛЯЦІЇ ОЗНАК...\n")
            processor = DataProcessor()
            
            # Розрахунок індикаторів
            df_with_indicators = processor.calculate_advanced_indicators(data)
            
            # Додаємо часові ознаки
            df_with_indicators = self.add_time_features(df_with_indicators)
            
            # Видаляємо NaN значення
            df_with_indicators = df_with_indicators.dropna()
            
            # ВІДБІР ОЗНАК ЗА ТИПОМ КРИПТОВАЛЮТИ ТА ВОЛАТИЛЬНОСТІ
            selected_features = self.prepare_features_by_crypto_type(df_with_indicators, crypto_type, volatility_type)
            
            # Додатковий відбір за кореляцією
            selected_features = self.filter_features_by_correlation(
                df_with_indicators, selected_features, training_params['min_correlation']
            )
            
            # ОБМЕЖЕННЯ КІЛЬКОСТІ ОЗНАК
            max_features = training_params['max_features']
            if len(selected_features) > max_features:
                selected_features = selected_features[:max_features]
            
            self.add_log_message(f"🎯 ВІДІБРАНО {len(selected_features)} ОЗНАК ДЛЯ {crypto_type} ({volatility_type}):\n")
            for i, feature in enumerate(selected_features):
                self.add_log_message(f"  {i+1:2d}. {feature}\n")
            
            if len(selected_features) == 0:
                self.add_log_message("❌ НЕ ВДАЛОСЯ ВІДІБРАТИ ОЗНАКИ!\n")
                return False

            # Аналіз кореляції для логу
            correlation = self.analyze_feature_correlation(df_with_indicators[selected_features])

            # Підготовка даних з вибраними ознаками
            self.add_log_message("🔄 Підготовка даних з обраними ознаками...\n")
            feature_data = df_with_indicators[selected_features].values
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            lookback = training_params['lookback']
            X, y = [], []
            
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                close_idx = selected_features.index('Close')
                y.append(scaled_data[i, close_idx])
            
            X = np.array(X)
            y = np.array(y)

            if X is None or len(X) == 0:
                self.add_log_message("❌ Недостатньо даних для навчання\n")
                return False

            self.add_log_message(f"✅ Дані підготовлено: {X.shape[0]} samples, {X.shape[2]} features\n")
            
            # Детальна інформація про ознаки
            feature_analysis = self.analyze_features(selected_features)
            self.add_log_message("📊 ВИКОРИСТАНІ ОЗНАКИ:\n")
            for group, features in feature_analysis.items():
                self.add_log_message(f"  {group}: {len(features)} ознак\n")
            self.add_log_message(f"  Загалом: {len(selected_features)} ознак\n\n")

            # Розділення на train/test
            test_size = int(len(X) * training_params['test_size'])
            X_train, X_test = X[:-test_size], X[-test_size:]
            y_train, y_test = y[:-test_size], y[-test_size:]

            self.add_log_message("📊 РОЗПОДІЛ ДАНИХ:\n")
            self.add_log_message(f"  Train: {len(X_train)} samples ({((1-training_params['test_size'])*100):.1f}%)\n")
            self.add_log_message(f"  Test: {len(X_test)} samples ({((training_params['test_size'])*100):.1f}%)\n")
            self.add_log_message(f"  Загалом: {len(X_train) + len(X_test)} samples\n\n")

            # Аналіз важливості ознак
            self.add_log_message("🎯 АНАЛІЗ ВАЖЛИВОСТІ ОЗНАК...\n")
            if len(selected_features) > 0:
                try:
                    feature_indices, importance_scores = self.analyze_feature_importance_correct(
                        X_train, 
                        y_train, 
                        selected_features
                    )
                except Exception as e:
                    self.add_log_message(f"⚠️ Помилка аналізу важливості ознак: {str(e)}\n")
                    feature_indices, importance_scores = np.array([]), np.array([])
            else:
                self.add_log_message("⚠️ Немає ознак для аналізу важливості\n")
                feature_indices, importance_scores = np.array([]), np.array([])

            # Створення моделі за типом криптовалюти та волатильності
            self.add_log_message("🧠 Створення моделі...\n")
            model = self.create_model_by_crypto_type((X_train.shape[1], X_train.shape[2]), crypto_type, volatility_type)
            self.add_log_message(f"✅ Створено модель для {crypto_type} ({volatility_type})\n\n")

            # Автоматичний підбір Learning Rate
            auto_lr = training_params.get('auto_lr', True)
            if auto_lr and len(X_train) > 100:
                self.add_log_message("🔍 АВТОМАТИЧНИЙ ПІДБІР LEARNING RATE...\n")
                try:
                    optimal_lr = self.find_optimal_learning_rate_safe(model, X_train, y_train, X_test, y_test)
                    training_params['learning_rate'] = optimal_lr
                    
                    # Перекомпіляція моделі з оптимальним LR
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=optimal_lr),
                        loss='mse',
                        metrics=['mae', 'mse']
                    )
                    self.add_log_message(f"✅ Встановлено оптимальний Learning Rate: {optimal_lr:.0e}\n\n")
                except Exception as e:
                    self.add_log_message(f"⚠️ Помилка підбору Learning Rate: {str(e)}\n")
                    self.add_log_message("ℹ️ Використовується стандартний Learning Rate\n\n")
            else:
                self.add_log_message("ℹ️ Автоматичний підбір LR пропущено\n\n")

            # Callbacks
            callbacks = []
            if training_params.get('use_early_stopping', True):
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=training_params['patience'],
                    min_delta=training_params['min_delta'],
                    restore_best_weights=True,
                    verbose=1
                ))
                self.add_log_message("✅ Рання зупинка активована\n")

            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(2, training_params['patience'] // 2),
                min_lr=0.00001,
                verbose=1
            ))

            callbacks.append(TrainingProgressCallback(
                self.status_callback,
                self.update_progress,
                training_params['epochs'],
                self.add_log_message
            ))

            # Навчання
            self.add_log_message("🚀 Початок навчання...\n")
            self.add_log_message(f"⏰ Час початку: {datetime.now().strftime('%H:%M:%S')}\n\n")
            self.status_callback(f"Навчання {symbol} ({crypto_type}, {volatility_type})...")

            history = model.fit(
                X_train, y_train,
                epochs=training_params['epochs'],
                batch_size=training_params['batch_size'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )

            # Оцінка моделі
            self.add_log_message("\n📈 ОЦІНКА МОДЕЛІ...\n")
            predictions = model.predict(X_test, verbose=0)
            
            # Основні метрики
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Детальний аналіз результатів
            detailed_metrics = self.analyze_training_results(history, X_test, y_test, model)

            # Оновлення графіка
            self.update_training_plot(history, symbol)

            # Визначення найкращої епохи
            best_epoch = len(history.history['loss'])
            if 'val_loss' in history.history:
                best_epoch = np.argmin(history.history['val_loss']) + 1

            # Аналіз перенавчання
            overfitting_info = ""
            if 'val_loss' in history.history and 'loss' in history.history:
                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 1.0
                overfitting_info = f"\n  Співвідношення перенавчання: {overfitting_ratio:.2f}"
                if overfitting_ratio > 1.2:
                    overfitting_info += " ⚠️ (можливе перенавчання)"
                elif overfitting_ratio < 0.8:
                    overfitting_info += " ⚠️ (можливе недонавчання)"

            # Вивід результатів
            self.add_log_message("📊 РЕЗУЛЬТАТИ НАВЧАННЯ:\n")
            self.add_log_message(f"  MSE: {mse:.6f} (менше = краще)\n")
            self.add_log_message(f"  MAE: {mae:.6f} (менше = краще)\n")
            self.add_log_message(f"  R²: {r2:.4f} (ближче до 1 = краще)\n")
            
            if 'mape' in detailed_metrics and not np.isnan(detailed_metrics['mape']):
                self.add_log_message(f"  MAPE: {detailed_metrics['mape']:.2f}% (менше = краще)\n")
            
            self.add_log_message(f"  Епох виконано: {len(history.history['loss'])}/{training_params['epochs']}\n")
            self.add_log_message(f"  Найкраща епоха: {best_epoch}\n")
            
            if 'val_loss' in history.history:
                best_val_loss = np.min(history.history['val_loss'])
                self.add_log_message(f"  Найкращий val_loss: {best_val_loss:.6f}\n")
            
            self.add_log_message(overfitting_info + "\n\n")

            # Комплексний аналіз якості
            self.add_log_message("📊 ДЕТАЛЬНИЙ АНАЛІЗ ЯКОСТІ...\n")

            quality_analysis = self.analyze_training_quality(
                history, X_test, y_test, model, selected_features, symbol
            )

            # Вивід результатів аналізу
            self.add_log_message(f"🏆 СТАТУС НАВЧАННЯ: {quality_analysis['status']}\n")
            self.add_log_message(f"📈 ЗАГАЛЬНИЙ БАЛ: {quality_analysis['score']}/10\n")

            # Попередження
            if quality_analysis['warnings']:
                self.add_log_message("⚠️  ПОПЕРЕДЖЕННЯ:\n")
                for warning in quality_analysis['warnings']:
                    self.add_log_message(f"   • {warning}\n")

            # Рекомендації
            if quality_analysis['recommendations']:
                self.add_log_message("💡 РЕКОМЕНДАЦІЇ:\n")
                for recommendation in quality_analysis['recommendations']:
                    self.add_log_message(f"   • {recommendation}\n")

            # Детальні метрики
            self.add_log_message("📊 ДЕТАЛЬНІ МЕТРИКИ:\n")
            for metric, value in quality_analysis['metrics'].items():
                self.add_log_message(f"   {metric.upper()}: {value:.6f}\n")

            # Підготовка метрик для збереження
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'crypto_type': crypto_type,
                'volatility_type': volatility_type,
                'feature_count': int(X_train.shape[2]),
                'samples_count': int(X_train.shape[0]),
                'best_epoch': int(best_epoch),
                'timestamp': datetime.now().isoformat(),
                'features': selected_features,
                'training_parameters': training_params,
                'quality_analysis': quality_analysis,
                'correlation_analysis': {
                    'min_correlation': training_params['min_correlation'],
                    'selected_features_count': len(selected_features),
                    'top_correlated_features': dict(correlation.head(10)) if not correlation.empty else {}
                }
            }

            # Додаємо feature_importance
            if len(feature_indices) > 0 and len(importance_scores) > 0:
                metrics['feature_importance'] = {
                    'indices': feature_indices.tolist(),
                    'scores': importance_scores.tolist()
                }

            # Додаємо detailed_metrics
            try:
                serializable_detailed = {}
                for key, value in detailed_metrics.items():
                    if hasattr(value, 'item'):
                        serializable_detailed[key] = value.item()
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        serializable_detailed[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        serializable_detailed[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        serializable_detailed[key] = value.tolist()
                    else:
                        serializable_detailed[key] = value
                metrics['detailed_metrics'] = serializable_detailed
            except Exception as e:
                self.add_log_message(f"⚠️ Помилка конвертації detailed_metrics: {e}\n")

            # Збереження моделі
            self.add_log_message("💾 Збереження моделі...\n")
            success = self.model_manager.save_model(symbol, model, None, metrics)

            if success:
                # Збереження профілю активу
                self.save_asset_profile(symbol, crypto_type, volatility_type, metrics)
                
                # Створення звіту якості
                self.create_quality_report(history, quality_analysis, symbol)
                
                result_message = f"""✅ УСПІШНО ЗАВЕРШЕНО: {symbol} ({crypto_type}, {volatility_type})
            MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}
            Епохи: {len(history.history['loss'])}/{training_params['epochs']}
            Ознаки: {len(selected_features)}
            Статус: {quality_analysis['status']}
            """
                self.add_log_message(result_message + "\n")
                self.add_log_message(f"⏰ Час завершення: {datetime.now().strftime('%H:%M:%S')}\n")
                self.add_log_message(f"{'='*60}\n\n")
                
                self.update_info_text(result_message)
                self.status_callback(f"✅ {symbol} ({crypto_type}, {volatility_type}) навчено успішно! R²: {r2:.4f}")
                
                # Генерація звіту
                self.generate_training_report(symbol, history, metrics, training_params)
                
                return True
            else:
                self.add_log_message("❌ Помилка збереження моделі\n")
                return False

        except Exception as e:
            error_msg = f"❌ КРИТИЧНА ПОМИЛКА НАВЧАННЯ {symbol}:\n{str(e)}\n"
            self.add_log_message(error_msg)
            self.update_info_text(error_msg)
            self.status_callback(f"Помилка навчання {symbol}")
            import traceback
            traceback.print_exc()
            return False

    
    
    
    def prepare_data_advanced(self, data, params):
        """Покращена підготовка даних з аналізом кореляції"""
        processor = DataProcessor()
        
        try:
            # Розрахунок всіх індикаторів
            df = processor.calculate_advanced_indicators(data)
            
            # Додаємо часові ознаки
            df = self.add_time_features(df)
            
            # Видаляємо NaN значення
            df = df.dropna()
            
            # Аналіз кореляції
            correlation = self.analyze_feature_correlation(df)
            
            # Автоматичний вибір ознак на основі кореляції
            selected_features = self.select_features_by_correlation(
                df, min_correlation=0.15, target_column='Close'
            )
            
            # Додаємо обов'язкові ознаки
            essential_features = ['Close', 'Returns', 'Volume_MA_20']
            for feature in essential_features:
                if feature in df.columns and feature not in selected_features:
                    selected_features.append(feature)
            
            # Вибірка даних
            feature_data = df[selected_features].values
            
            # Масштабування
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Створення послідовностей
            lookback = params['lookback']
            X, y = [], []
            
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                close_idx = selected_features.index('Close')
                y.append(scaled_data[i, close_idx])
            
            X = np.array(X)
            y = np.array(y)
            
            return X, y, selected_features
            
        except Exception as e:
            self.add_log_message(f"❌ Помилка підготовки даних: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return None, None, None

    def add_time_features(self, df):
        """Додавання розширених часових ознак"""
        df = df.copy()
        
        # Базові часові компоненти
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        
        # Циклічні ознаки
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Торгові сесії
        df['Asian_Session'] = ((df['Hour'] >= 0) & (df['Hour'] <= 8)).astype(int)
        df['European_Session'] = ((df['Hour'] >= 7) & (df['Hour'] <= 16)).astype(int)
        df['US_Session'] = ((df['Hour'] >= 13) & (df['Hour'] <= 22)).astype(int)
        
        # Вихідні та святкові дні
        df['Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
        
        # Сезонність
        df['Season'] = (df['Month'] % 12 + 3) // 3
        
        return df
    
    def find_optimal_learning_rate_safe(self, model, X_train, y_train, X_val, y_val):
        """Безпечний пошук оптимального learning rate"""
        learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        best_lr = 0.001
        best_loss = float('inf')
        
        for lr in learning_rates:
            try:
                # Створюємо нову модель замість клонування
                input_shape = (X_train.shape[1], X_train.shape[2])
                temp_model = self.create_simple_advanced_model(input_shape, {'learning_rate': lr})
                
                # Коротке навчання
                history = temp_model.fit(
                    X_train[:100], y_train[:100],  # Використовуємо підмножину
                    epochs=2,
                    batch_size=32,
                    validation_data=(X_val[:50], y_val[:50]),
                    verbose=0
                )
                
                val_loss = history.history['val_loss'][-1]
                self.add_log_message(f"  LR: {lr:.0e} -> Val Loss: {val_loss:.6f}\n")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_lr = lr
                    
            except Exception as e:
                self.add_log_message(f"  LR: {lr:.0e} -> Помилка: {str(e)}\n")
                continue
        
        return best_lr

    def create_simple_advanced_model(self, input_shape, params):
        """Спрощена модель для багатьох ознак"""
        model = tf.keras.Sequential()
        
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.LSTM(128, return_sequences=True,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.LSTM(96, return_sequences=True,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.LSTM(64,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Dense(128, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Dense(64, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))
        
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        
        model.add(tf.keras.layers.Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model

    def prepare_filtered_features(self, data, min_correlation=0.15):
        """Відбір тільки якісних ознак"""
        try:
            # Кореляційний аналіз
            correlation = data.corr()['Close'].abs().sort_values(ascending=False)
            
            # Вибір ознак з достатньою кореляцією
            selected_features = correlation[correlation >= min_correlation].index.tolist()
            
            # Видаляємо NaN значення
            selected_features = [f for f in selected_features if not np.isnan(correlation[f])]
            
            # Додаємо обов'язкові ознаки
            essential = ['Close', 'Returns', 'Volume_MA_20']
            for feature in essential:
                if feature in data.columns and feature not in selected_features:
                    selected_features.append(feature)
            
            # Видаляємо дублікати
            selected_features = list(set(selected_features))
            
            return selected_features
            
        except Exception as e:
            self.add_log_message(f"⚠️ Помилка фільтрації ознак: {str(e)}\n")
            # Повертаємо всі ознаки у разі помилки
            return data.columns.tolist()

    def analyze_feature_importance_correct(self, X_train, y_train, feature_names):
        """Аналіз важливості ознак без вирівнювання"""
        try:
            # Використовуємо середнє значення по часових вікнах
            X_mean = np.mean(X_train, axis=1)  # [samples, features]
            
            from sklearn.ensemble import GradientBoostingRegressor
            gb = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=3)
            gb.fit(X_mean, y_train)
            
            importance = gb.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            self.add_log_message("🎯 ВАЖЛИВІСТЬ ОЗНАК (середнє по вікну):\n")
            for i, idx in enumerate(indices):
                if importance[idx] > 0.01 and i < 15:  # Топ-15 з важливістю > 1%
                    feature_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                    self.add_log_message(f"  {i+1:2d}. {feature_name:25s}: {importance[idx]:.4f}\n")
            
            return indices, importance
            
        except Exception as e:
            self.add_log_message(f"⚠️ Помилка аналізу важливості: {str(e)}\n")
            return np.array([]), np.array([])
    
    def classify_asset_volatility(self, data, symbol):
        """Автоматична класифікація активу за волатильністю"""
        returns = data['Close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_daily_change = returns.abs().mean()
        max_daily_change = returns.abs().max()
        
        # Критерії класифікації
        if volatility > 0.04 or max_daily_change > 0.15:
            volatility_type = 'HIGH'
        elif volatility > 0.02:
            volatility_type = 'MEDIUM' 
        else:
            volatility_type = 'LOW'
        
        self.add_log_message(f"📊 КЛАСИФІКАЦІЯ {symbol}:\n")
        self.add_log_message(f"  Волатильність: {volatility:.4f}\n")
        self.add_log_message(f"  Середня зміна: {avg_daily_change:.4f}\n")
        self.add_log_message(f"  Макс зміна: {max_daily_change:.4f}\n")
        self.add_log_message(f"  Тип: {volatility_type}\n")
        
        return volatility_type

    def validate_training_parameters(self, params):
        """Перевірка валідності параметрів навчання"""
        errors = []
        
        # Перевірка основних параметрів
        if params['epochs'] <= 0 or params['epochs'] > 1000:
            errors.append("Кількість епох повинна бути між 1 та 1000")
        
        if params['batch_size'] <= 0 or params['batch_size'] > 256:
            errors.append("Розмір батча повинен бути між 1 та 256")
        
        if params['lookback'] < 5 or params['lookback'] > 200:
            errors.append("Розмір вікна повинен бути між 5 та 200")
        
        if params['learning_rate'] <= 0 or params['learning_rate'] > 0.1:
            errors.append("Learning rate повинен бути між 0.0001 та 0.1")
        
        if params['test_size'] <= 0 or params['test_size'] >= 1:
            errors.append("Розмір тестової вибірки повинен бути між 0 та 1")
        
        if params['patience'] < 1 or params['patience'] > 50:
            errors.append("Патінс повинен бути між 1 та 50")
        
        if params['min_delta'] <= 0 or params['min_delta'] > 0.01:
            errors.append("Мінімальна дельта повинна бути між 0.000001 та 0.01")
        
        # Перевірка додаткових параметрів
        if params['dropout_start'] < 0 or params['dropout_start'] > 0.9:
            errors.append("Початковий dropout повинен бути між 0 та 0.9")
        
        if params['dropout_end'] < 0 or params['dropout_end'] > 0.9:
            errors.append("Кінцевий dropout повинен бути між 0 та 0.9")
        
        if params['lstm_layers'] < 1 or params['lstm_layers'] > 5:
            errors.append("Кількість LSTM шарів повинна бути між 1 та 5")
        
        if params['lstm_units_start'] < 8 or params['lstm_units_start'] > 512:
            errors.append("Початкові LSTM юніти повинні бути між 8 та 512")
        
        if params['lstm_units_end'] < 8 or params['lstm_units_end'] > 512:
            errors.append("Кінцеві LSTM юніти повинні бути між 8 та 512")
        
        if params['min_correlation'] < 0 or params['min_correlation'] > 1:
            errors.append("Мінімальна кореляція повинна бути між 0 та 1")
        
        return errors
    
    def get_training_profile(self, crypto_type, volatility_type, ui_params):
        """Отримання профілю навчання з урахуванням типу криптовалюти"""
        
        # Базові параметри з UI
        profile = {
            'crypto_type': crypto_type,
            'volatility_type': volatility_type,
            'lookback': ui_params['lookback'],
            'epochs': ui_params['epochs'],
            'batch_size': ui_params['batch_size'],
            'test_size': ui_params['test_size'],
            'learning_rate': ui_params['learning_rate'],
            'patience': ui_params['patience'],
            'min_delta': ui_params.get('min_delta', 0.0001),
            'use_early_stopping': ui_params.get('use_early_stopping', True),
            'auto_lr': ui_params.get('auto_lr', True),
        }
        
        # Специфічні налаштування для кожного типу криптовалюти
        crypto_profiles = {
            'STABLECOIN': {
                'max_features': 8,
                'lstm_units': [64, 32],
                'dropout_rates': [0.1, 0.1],
                'required_features': ['Close', 'Returns', 'Volume_MA_20', 'MA_20'],
                'recommended_features': ['MA_50', 'Volatility_20', 'Day_of_Week'],
                'description': 'Консервативна модель для стабільних активів'
            },
            'MEMECOIN': {
                'max_features': 15,
                'lstm_units': [128, 64, 32],
                'dropout_rates': [0.3, 0.4, 0.3],
                'required_features': ['Close', 'Returns', 'Volume', 'RSI_14', 'Volatility_20'],
                'recommended_features': ['Social_Volume', 'Twitter_Sentiment', 'MACD', 'Bollinger_Width'],
                'description': 'Агресивна модель для високоволатильних мемкоінів'
            },
            'ALTCOIN': {
                'max_features': 12,
                'lstm_units': [96, 64, 32],
                'dropout_rates': [0.2, 0.3, 0.2],
                'required_features': ['Close', 'Returns', 'Volume_MA_20', 'RSI_14', 'MA_20'],
                'recommended_features': ['MACD', 'BTC_Correlation', 'Market_Cap_Change', 'Volatility_20'],
                'description': 'Збалансована модель для альткойнів'
            },
            'BLUECHIP': {
                'max_features': 20,
                'lstm_units': [128, 96, 64, 32],
                'dropout_rates': [0.15, 0.2, 0.15, 0.1],
                'required_features': ['Close', 'Returns', 'Volume', 'RSI_14', 'MA_20', 'MA_50'],
                'recommended_features': ['MACD', 'Bollinger_Bands', 'Volatility_50', 'Institutional_Flow'],
                'description': 'Комплексна модель для blue-chip активів'
            },
            'HIGH_VOLATILITY': {
                'max_features': 10,
                'lstm_units': [64, 32],
                'dropout_rates': [0.3, 0.25],
                'required_features': ['Close', 'Returns', 'Volatility_20', 'RSI_14'],
                'recommended_features': ['ATR_14', 'Volume_Spike', 'Price_Change_5d'],
                'description': 'Обережна модель для волатильних активів'
            }
        }
        
        # Отримуємо профіль для конкретного типу
        specific_profile = crypto_profiles.get(crypto_type, crypto_profiles['ALTCOIN'])
        profile.update(specific_profile)
        
        # Додаємо параметри волатильності
        volatility_adjustments = {
            'LOW': {'learning_rate_multiplier': 1.1, 'patience_adjust': -2},
            'MEDIUM': {'learning_rate_multiplier': 1.0, 'patience_adjust': 0},
            'HIGH': {'learning_rate_multiplier': 0.8, 'patience_adjust': +3}
        }
        
        adjustment = volatility_adjustments.get(volatility_type, volatility_adjustments['MEDIUM'])
        profile['learning_rate'] *= adjustment['learning_rate_multiplier']
        profile['patience'] += adjustment['patience_adjust']
        
        return profile

    def prepare_features_by_volatility(self, data, volatility_type):
        """Підготовка ознак за типом волатильності з обов'язковими ознаками"""
        
        # Обов'язкові ознаки для всіх типів
        base_features = ['Close', 'Returns', 'Volume_MA_20', 'Volatility_20', 'ATR_14']
        
        if volatility_type == 'HIGH':
            additional_features = ['RSI_14', 'MACD', 'Bollinger_Width', 'MA_20', 'Price_Change_5d']
        elif volatility_type == 'MEDIUM':
            additional_features = ['RSI_14', 'MACD', 'MACD_Signal', 'Bollinger_Width', 
                                'MA_20', 'EMA_20', 'Price_Change_5d', 'OBV']
        else:  # LOW
            additional_features = ['RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                                'Bollinger_Width', 'MA_20', 'EMA_20', 'MA_50', 'EMA_50',
                                'Price_Change_5d', 'Price_Change_20d', 'OBV']
        
        # Вибір тільки доступних ознак
        available_base = [f for f in base_features if f in data.columns]
        available_additional = [f for f in additional_features if f in data.columns]
        
        return available_base + available_additional
    
    def create_model_by_volatility(self, input_shape, volatility_type):
        """Створення моделі за типом волатильності"""
        
        if volatility_type == 'HIGH':
            # Проста модель для волатильних активів
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
        elif volatility_type == 'MEDIUM':
            # Стандартна модель
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(96, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(48, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
        else:  # LOW
            # Складна модель для стабільних активів
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(96, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        
        return model

    def adaptive_training_parameters(self, data, symbol, base_params):
        """Адаптивне налаштування параметрів"""
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Адаптація параметрів за волатильністю
        adaptive_params = base_params.copy()
        
        # Коефіцієнти адаптації
        volatility_ratio = volatility / 0.02  # Базова волатильність 2%
        
        # Адаптація параметрів
        adaptive_params['lookback'] = int(base_params['lookback'] / volatility_ratio)
        adaptive_params['batch_size'] = int(base_params['batch_size'] / volatility_ratio)
        adaptive_params['learning_rate'] = base_params['learning_rate'] * volatility_ratio
        adaptive_params['min_correlation'] = base_params['min_correlation'] * volatility_ratio
        
        # Обмеження значень
        adaptive_params['lookback'] = max(20, min(adaptive_params['lookback'], 100))
        adaptive_params['batch_size'] = max(8, min(adaptive_params['batch_size'], 64))
        adaptive_params['learning_rate'] = max(0.0001, min(adaptive_params['learning_rate'], 0.01))
        adaptive_params['min_correlation'] = max(0.1, min(adaptive_params['min_correlation'], 0.4))
        
        return adaptive_params

    
    def create_model_by_crypto_type(self, input_shape, crypto_type, volatility_type):
        """Створення моделі за типом криптовалюти"""
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Архітектури для різних типів криптовалют
        if crypto_type == 'STABLECOIN':
            model.add(tf.keras.layers.LSTM(64, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.LSTM(32))
            model.add(tf.keras.layers.Dropout(0.1))
            
        elif crypto_type == 'MEMECOIN':
            model.add(tf.keras.layers.LSTM(128, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.LSTM(64, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.4))
            model.add(tf.keras.layers.LSTM(32))
            model.add(tf.keras.layers.Dropout(0.3))
            
        elif crypto_type == 'ALTCOIN':
            model.add(tf.keras.layers.LSTM(96, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(64, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.LSTM(32))
            model.add(tf.keras.layers.Dropout(0.2))
            
        else:  # BLUECHIP або інші
            model.add(tf.keras.layers.LSTM(128, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.15))
            model.add(tf.keras.layers.LSTM(96, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(64))
            model.add(tf.keras.layers.Dropout(0.15))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
        
        # Фінальні шари
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model

    def save_asset_profile(self, symbol, crypto_type, volatility_type, metrics):
        """Збереження профілю активу з урахуванням типу криптовалюти"""
        profile_path = f'profiles/{symbol}_profile.json'
        
        profile_data = {
            'symbol': symbol,
            'crypto_type': crypto_type,
            'volatility_type': volatility_type,
            'last_trained': datetime.now().isoformat(),
            'best_r2': metrics.get('r2', 0),
            'best_mse': metrics.get('mse', 0),
            'feature_count': metrics.get('feature_count', 0),
            'recommended_features': metrics.get('features', []),
            'training_parameters': metrics.get('training_parameters', {})
        }
        
        os.makedirs('profiles', exist_ok=True)
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        
        return profile_data


    def filter_features_by_correlation(self, data, features, min_correlation=0.15):
        """Фільтрація ознак за мінімальною кореляцією"""
        try:
            if not features:
                return features
                
            # Обчислюємо кореляцію тільки для обраних ознак
            correlation = data[features].corr()['Close'].abs()
            
            # Фільтруємо ознаки за кореляцією
            filtered_features = [
                feature for feature in features 
                if feature in correlation and correlation[feature] >= min_correlation
            ]
            
            # Завжди додаємо Close
            if 'Close' not in filtered_features and 'Close' in features:
                filtered_features.append('Close')
                
            return filtered_features
            
        except Exception as e:
            self.add_log_message(f"⚠️ Помилка фільтрації ознак: {str(e)}\n")
            return features

    def generate_trading_recommendations(self, analysis, crypto_type, symbol):
        """Генерація торгових рекомендацій на основі типу криптовалюти"""
        recommendations = []
        r2 = analysis['metrics']['r2']
        
        # Загальні рекомендації на основі якості
        if r2 > 0.8:
            recommendations.append("✅ Висока якість прогнозу - можна використовувати для торгівлі")
        elif r2 > 0.6:
            recommendations.append("✓ Прийнятна якість прогнозу - обережна торгівля")
        else:
            recommendations.append("⚠️ Низька якість прогнозу - утриматися від торгівлі")
        
        # Специфічні рекомендації для типів криптовалют
        crypto_recommendations = {
            'STABLECOIN': [
                "• Стратегія: Арбітраж між біржами",
                "• Ризик: Низький",
                "• Позиція: До 20% капіталу",
                "• Таймфрейм: Короткострокові угоди"
            ],
            'MEMECOIN': [
                "• Стратегія: Моментум трейдинг",
                "• Ризик: Дуже високий",
                "• Позиція: До 5% капіталу",
                "• Стоп-лосс: 15-20%",
                "• Увага до соціальних сигналів"
            ],
            'ALTCOIN': [
                "• Стратегія: Трендове торгівля",
                "• Ризик: Середній-Високий",
                "• Позиція: До 10% капіталу",
                "• Стоп-лосс: 10-15%",
                "• Моніторинг BTC кореляції"
            ],
            'BLUECHIP': [
                "• Стратегія: Swing trading",
                "• Ризик: Середній",
                "• Позиція: До 15% капіталу",
                "• Стоп-лосс: 8-12%",
                "• Фокус на фундаментальних факторах"
            ]
        }
        
        # Додаємо специфічні рекомендації
        specific_recs = crypto_recommendations.get(crypto_type, [])
        recommendations.extend(specific_recs)
        
        # Додаткові рекомендації на основі волатильності
        if analysis.get('volatility_type') == 'HIGH':
            recommendations.append("⚡ Висока волатильність - зменшіть розмір позиції")
            recommendations.append("📊 Використовуйте ATR для стоп-лосс")
        
        return recommendations

    def classify_crypto_type(self, symbol, data):
        """Класифікація криптовалюти за типом на основі даних"""
        try:
            if len(data) < 100:
                return 'UNKNOWN'
            
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std()
            avg_volume = data['Volume'].mean() if 'Volume' in data.columns else 0
            price = data['Close'].iloc[-1]
            
            # Визначення типу на основі характеристик
            if volatility < 0.02 and price < 10:
                return 'STABLECOIN'
            
            elif volatility > 0.06 or symbol in ['DOGE', 'SHIB', 'PEPE', 'FLOKI']:
                return 'MEMECOIN'
            
            elif volatility > 0.04 or symbol in ['LINK', 'UNI', 'AAVE', 'MATIC', 'SOL', 'DOT']:
                return 'ALTCOIN'
            
            elif symbol in ['BTC', 'ETH']:
                return 'BLUECHIP'
            
            else:
                # Автоматична класифікація на основі даних
                if volatility < 0.03:
                    return 'LOW_VOLATILITY'
                elif volatility > 0.05:
                    return 'HIGH_VOLATILITY'
                else:
                    return 'MEDIUM_VOLATILITY'
                    
        except Exception as e:
            self.add_log_message(f"⚠️ Помилка класифікації {symbol}: {str(e)}\n")
            return 'UNKNOWN'

    def prepare_features_by_crypto_type(self, data, crypto_type, volatility_type):
        """Підготовка ознак за типом криптовалюти"""
        
        processor = DataProcessor()
        df = processor.calculate_advanced_indicators(data)
        
        # Додаємо часові ознаки
        df = self.add_time_features(df)
        
        # Специфічні ознаки для різних типів криптовалют
        crypto_specific_features = {
            'STABLECOIN': [
                'Price_Stability', 'Volume_Consistency', 'MA_20_Deviation',
                'Range_5d', 'Support_Level', 'Resistance_Level'
            ],
            'MEMECOIN': [
                'Social_Volume_Index', 'Twitter_Mentions', 'Community_Growth',
                'Volume_Spike_5d', 'Price_Pump_24h', 'Trend_Strength'
            ],
            'ALTCOIN': [
                'BTC_Correlation_30d', 'Market_Cap_Rank', 'Trading_Pairs',
                'Liquidity_Score', 'Development_Activity', 'Exchange_Inflows'
            ],
            'BLUECHIP': [
                'Institutional_Flow', 'Futures_Open_Interest', 'Options_Volume',
                'Whale_Transactions', 'Network_Growth', 'Hash_Rate'
            ]
        }
        
        # Базові обов'язкові ознаки
        base_features = ['Close', 'Returns', 'Volume_MA_20', 'Volatility_20']
        
        # Отримуємо специфічні ознаки для типу
        specific_features = crypto_specific_features.get(crypto_type, [])
        
        # Вибираємо тільки доступні ознаки
        available_features = []
        for feature in base_features + specific_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # Додаємо технічні індикатори на основі волатильності
        technical_indicators = self.select_technical_indicators(volatility_type)
        for indicator in technical_indicators:
            if indicator in df.columns and indicator not in available_features:
                available_features.append(indicator)
        
        return available_features

    def select_technical_indicators(self, volatility_type):
        """Вибір технічних індикаторів на основі волатильності"""
        if volatility_type == 'HIGH':
            return ['RSI_14', 'ATR_14', 'Bollinger_Width', 'Stochastic_K', 'Price_Change_1d']
        elif volatility_type == 'MEDIUM':
            return ['RSI_14', 'MACD', 'MA_20', 'MA_50', 'Volume_Ratio', 'Price_Change_5d']
        else:  # LOW
            return ['MA_20', 'MA_50', 'EMA_20', 'Volatility_50', 'OBV', 'Price_Change_20d']

    
    def create_advanced_model(self, training_type, input_shape, params):
        """Створення моделі, оптимізованої для багатьох ознак"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Batch Normalization для стабілізації навчання
        model.add(tf.keras.layers.BatchNormalization())
        
        # Перший LSTM шар з більшою кількістю юнітів
        model.add(tf.keras.layers.LSTM(
            units=128, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())
        
        # Другий LSTM шар
        model.add(tf.keras.layers.LSTM(
            units=96,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())
        
        # Третій LSTM шар
        model.add(tf.keras.layers.LSTM(
            units=64,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.BatchNormalization())
        
        # Dense шари з регуляризацією
        model.add(tf.keras.layers.Dense(128, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Dense(64, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.3))
        
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Компіляція з адаптивним learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get('learning_rate', 0.001),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber_loss',  # Краще для фінансових даних
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_model_with_attention(self, input_shape, params):
        """Модель з механізмом уваги для важливих ознак"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Batch Normalization
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        # LSTM layers
        lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        lstm_out = tf.keras.layers.Dropout(0.3)(lstm_out)
        
        # Attention mechanism - ВИПРАВЛЕНО
        attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        attended = tf.keras.layers.Multiply()([lstm_out, attention])
        
        # ВИПРАВЛЕННЯ: Заміна Lambda шару на явний
        attended = tf.keras.layers.GlobalAveragePooling1D()(attended)
        
        # Dense layers
        x = tf.keras.layers.Dense(64, activation='relu')(attended)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='huber_loss',
            metrics=['mae']
        )
        
        return model

    def analyze_feature_importance(self, X_train_flat, y_train, feature_names):
        """Аналіз важливості ознак через Gradient Boosting"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.inspection import permutation_importance
            
            # Перевіряємо розмірності
            n_features_flat = X_train_flat.shape[1]
            n_original_features = len(feature_names)
            
            self.add_log_message(f"📊 АНАЛІЗ ВАЖЛИВОСТІ ОЗНАК:\n")
            self.add_log_message(f"  Розмірність даних: {X_train_flat.shape}\n")
            self.add_log_message(f"  Кількість оригінальних ознак: {n_original_features}\n")
            self.add_log_message(f"  Кількість ознак після вирівнювання: {n_features_flat}\n")
            
            # Якщо розмірності не співпадають, використовуємо generic names
            if n_features_flat != n_original_features:
                self.add_log_message("⚠️  Увага: розмірності не співпадають, використовуються generic names\n")
                feature_names_flat = [f'feature_{i}' for i in range(n_features_flat)]
            else:
                feature_names_flat = feature_names
            
            # Тренуємо Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            gb.fit(X_train_flat, y_train)
            
            # Важливість ознак
            importance = gb.feature_importances_
            
            # Сортуємо ознаки за важливістю
            indices = np.argsort(importance)[::-1]
            
            self.add_log_message("🎯 ВАЖЛИВІСТЬ ОЗНАК (Gradient Boosting):\n")
            for i, idx in enumerate(indices):
                if importance[idx] > 0.001:  # Показуємо тільки важливі ознаки
                    feature_name = feature_names_flat[idx] if idx < len(feature_names_flat) else f'feature_{idx}'
                    self.add_log_message(f"  {i+1:2d}. {feature_name:25s}: {importance[idx]:.4f}\n")
                if i >= 20:  # Обмежуємо вивід
                    remaining = len(indices) - 20
                    self.add_log_message(f"  ... і ще {remaining} ознак з меншою важливістю\n")
                    break
            
            # Permutation importance (обережно - це може бути повільно)
            try:
                if n_features_flat <= 50:  # Робимо тільки для розумної кількості ознак
                    perm_importance = permutation_importance(gb, X_train_flat, y_train, 
                                                        n_repeats=3, random_state=42, n_jobs=-1)
                    
                    self.add_log_message("\n🎯 PERMUTATION IMPORTANCE (топ-10):\n")
                    sorted_idx = perm_importance.importances_mean.argsort()[::-1][:10]
                    
                    for i, idx in enumerate(sorted_idx):
                        if perm_importance.importances_mean[idx] > 0:
                            feature_name = feature_names_flat[idx] if idx < len(feature_names_flat) else f'feature_{idx}'
                            self.add_log_message(f"  {i+1:2d}. {feature_name:25s}: {perm_importance.importances_mean[idx]:.4f}\n")
                else:
                    self.add_log_message("ℹ️  Permutation importance пропущено (забагато ознак)\n")
                    
            except Exception as e:
                self.add_log_message(f"⚠️  Помилка permutation importance: {str(e)}\n")
            
            return indices, importance
            
        except Exception as e:
            self.add_log_message(f"❌ Помилка аналізу важливості ознак: {str(e)}\n")
            # Повертаємо пусті масиви у разі помилки
            return np.array([]), np.array([])

    def prepare_multivariate_timeseries(self, data, lookback=60, forecast_horizon=5):
        """Підготовка багатовимірних часових рядів для всіх ознак"""
        # Всі числові колонки
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Видаляємо цільову змінну з ознак
        features = [col for col in numeric_columns if col != 'Close']
        
        # Масштабування
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(data[features])
        y_scaled = scaler_y.fit_transform(data[['Close']])
        
        # Створення послідовностей
        X, y = [], []
        
        for i in range(lookback, len(X_scaled) - forecast_horizon):
            X.append(X_scaled[i-lookback:i])  # Всі ознаки
            y.append(y_scaled[i+forecast_horizon-1])  # Ціль через forecast_horizon періодів
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, features, scaler_X, scaler_y
    
    def train_with_all_features(self, symbol, training_params):
        """Навчання з використанням всіх ознак"""
        # Завантаження даних
        data = pd.read_csv(f'data/{symbol}_data.csv', index_col=0, parse_dates=True)
        
        # Використовуємо ВСІ числові ознаки
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.add_log_message(f"📊 Використовуємо {len(numeric_columns)} ознак:\n")
        for col in numeric_columns:
            self.add_log_message(f"  • {col}\n")
        
        # Підготовка даних
        X, y, feature_names = self.prepare_data_advanced(data, training_params)
        
        # Розділення на train/test
        test_size = int(len(X) * training_params['test_size'])
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Аналіз важливості ознак
        feature_indices, importance = self.analyze_feature_importance(
            X_train.reshape(X_train.shape[0], -1), 
            y_train, 
            feature_names
        )
        
        # Створення моделі, оптимізованої для багатьох ознак
        model = self.create_advanced_model('expert', (X_train.shape[1], X_train.shape[2]), training_params)
        
        # Рання зупинка з більшим patience
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Більше терпіння
                min_delta=0.00001,  # Менша мінімальна зміна
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Навчання з більшою кількістю епох
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Більше епох
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0,
            shuffle=False  # Для часових рядів
        )
        
        return history, model, X_test, y_test

    def create_ensemble_model(self, input_shape, training_params, num_models=3):
        """Створення ансамблю моделей"""
        models = []
        for i in range(num_models):
            model = self.create_advanced_model('expert', input_shape, training_params)
            models.append(model)
        return models

    def train_ensemble(self, models, X_train, y_train, X_test, y_test, training_params):
        """Навчання ансамблю"""
        histories = []
        predictions = []
        
        for i, model in enumerate(models):
            self.add_log_message(f"🏗️ Навчання моделі {i+1}/{len(models)}...\n")
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=training_params.get('patience', 10),
                    restore_best_weights=True
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=training_params.get('epochs', 50),
                batch_size=training_params.get('batch_size', 32),
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=callbacks
            )
            
            histories.append(history)
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred)
        
        # Усереднення прогнозів
        ensemble_prediction = np.mean(predictions, axis=0)
        return ensemble_prediction, histories
    
    def get_training_parameters(self):
        """Отримання всіх параметрів навчання включаючи вибір ознак"""
        params = {
            'training_type': self.training_type_var.get(),
            'epochs': self.epochs_var.get(),
            'batch_size': self.batch_size_var.get(),
            'lookback': self.lookback_var.get(),
            'test_size': self.test_size_var.get(),
            'learning_rate': self.lr_var.get(),
            'patience': self.patience_var.get(),
            'min_delta': self.min_delta_var.get(),
            'dropout_start': self.dropout_start_var.get(),
            'dropout_end': self.dropout_end_var.get(),
            'lstm_layers': self.lstm_layers_var.get(),
            'lstm_units_start': self.lstm_units_start_var.get(),
            'lstm_units_end': self.lstm_units_end_var.get(),
            'use_technical': self.use_technical_var.get(),
            'use_time_features': self.use_time_features_var.get(),
            'use_batch_norm': self.use_batch_norm_var.get(),
            'use_l2_reg': self.use_l2_reg_var.get(),
            'use_early_stopping': self.use_early_stopping_var.get(),
            
            # Нові параметри для вибору ознак
            'use_close': self.use_close_var.get(),
            'use_high_low': self.use_high_low_var.get(),
            'use_open': self.use_open_var.get(),
            'use_volume': self.use_volume_var.get(),
            'use_returns': self.use_returns_var.get(),
            'use_ma': self.use_ma_var.get(),
            'use_volatility': self.use_volatility_var.get(),
            'use_rsi': self.use_rsi_var.get(),
            'use_macd': self.use_macd_var.get(),
            'use_day_of_week': self.use_day_of_week_var.get(),
            'use_month': self.use_month_var.get(),
            'use_quarter': self.use_quarter_var.get(),
            
            # Додаткові параметри
            'auto_lr': self.auto_lr_var.get(),
            'min_correlation': self.min_correlation_var.get(),
            
            # Існуючі торгові параметри
            'use_volatility_indicators': self.use_volatility_indicators_var.get(),
            'use_momentum_indicators': self.use_momentum_indicators_var.get(),
            'use_volume_indicators': self.use_volume_indicators_var.get(),
            'use_market_indicators': self.use_market_indicators_var.get(),
            'use_risk_metrics': self.use_risk_metrics_var.get(),
            'forecast_horizon': int(self.forecast_horizon_var.get()),
            'use_price_target': self.use_price_target_var.get(),
            'use_return_target': self.use_return_target_var.get(),
            'use_signal_target': self.use_signal_target_var.get()
        }
        
        return params

    def prepare_data(self, data, training_type):
        """Підготовка даних для навчання"""
        processor = DataProcessor()
        
        try:
            if training_type == "basic":
                # Базове навчання - тільки ціни закриття
                prices = data[['Close']].values
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(prices)
                
                lookback = self.lookback_var.get()
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i, 0])
                    y.append(scaled_data[i, 0])
                
                X = np.array(X)
                y = np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                return X, y, ['Close']
                
            elif training_type == "advanced":
                # Розширене навчання - основні технічні індикатори
                df = processor.prepare_features_for_ml(data)
                features = ['Close', 'Returns', 'MA_5', 'MA_20', 'Volatility']
                
                # Вибір тільки доступних ознак
                available_features = [f for f in features if f in df.columns]
                feature_data = df[available_features].values
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(feature_data)
                
                lookback = self.lookback_var.get()
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i])
                    y.append(scaled_data[i, 0])  # Ціль - Close price
                
                X = np.array(X)
                y = np.array(y)
                
                return X, y, available_features
                
            else:  # expert
                # Експертне навчання - спрощена версія
                df = data.copy()
                
                # Додаємо базові технічні індикатори
                if 'Close' in df.columns:
                    df['Returns'] = df['Close'].pct_change()
                    df['MA_5'] = df['Close'].rolling(window=5).mean()
                    df['MA_20'] = df['Close'].rolling(window=20).mean()
                    df['Volatility'] = df['Close'].rolling(window=20).std()
                
                # Видаляємо NaN
                df = df.dropna()
                
                # Вибираємо всі числові колонки
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_data = df[numeric_columns].values
                
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)
                
                lookback = self.lookback_var.get()
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i])
                    y.append(scaled_data[i, numeric_columns.index('Close')])  # Ціль - Close price
                
                X = np.array(X)
                y = np.array(y)
                
                return X, y, numeric_columns
                
        except Exception as e:
            self.info_text.insert(tk.END, f"❌ Помилка підготовки даних: {str(e)}\n")
            return None, None, None
    
    def create_model(self, training_type, input_shape):
        """Створення моделі нейромережі (для зворотної сумісності)"""
        model = tf.keras.Sequential()
        
        # Додаємо Input шар явно
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        if training_type == "basic":
            model.add(tf.keras.layers.LSTM(64, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.LSTM(32))
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
            
        elif training_type == "advanced":
            model.add(tf.keras.layers.LSTM(100, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.LSTM(75, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.LSTM(50))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Dense(25, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
            
        else:  # expert
            model.add(tf.keras.layers.LSTM(128, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.4))
            model.add(tf.keras.layers.LSTM(96, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.4))
            model.add(tf.keras.layers.LSTM(64))
            model.add(tf.keras.layers.Dropout(0.4))
            model.add(tf.keras.layers.Dense(48, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_var.get()),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model

    def update_training_plot(self, history, symbol):
        """Оновлення графіка навчання"""
        self.ax.clear()
        
        if not history or not history.history:
            return
        
        # Графік втрат
        epochs = range(1, len(history.history['loss']) + 1)
        self.ax.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        
        if 'val_loss' in history.history:
            self.ax.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        
        self.ax.set_title(f'Навчання {symbol}', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Епоха')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_progress(self, value):
        """Оновлення прогрес-бару"""
        self.progress_bar['value'] = value
        self.parent.update_idletasks()
    
    def stop_training(self):
        """Зупинка навчання"""
        self.training_stop_flag = True
        self.status_label.config(text="Навчання зупиняється...", foreground="orange")
    
    def cleanup_models(self):
        """Очищення моделей"""
        if messagebox.askyesno("Підтвердження", "Видалити всі навчені моделі?"):
            try:
                models_dir = 'models'
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        file_path = os.path.join(models_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    self.status_callback("Всі моделі видалені")
                    self.info_text.insert(tk.END, "✅ Всі навчені моделі видалені\n")
                else:
                    self.status_callback("Папка models не існує")
                    self.info_text.insert(tk.END, "ℹ️ Папка models не існує\n")
            except Exception as e:
                error_msg = f"❌ Помилка очищення: {str(e)}"
                self.status_callback(error_msg)
                self.info_text.insert(tk.END, f"{error_msg}\n")


    def auto_select_features(self, scenario="trading"):
        """Автоматичний вибір ознак для різних сценаріїв"""
        if scenario == "minimal":
            self.use_close_var.set(True)
            self.use_returns_var.set(True)
            self.use_ma_var.set(False)
            self.use_volatility_var.set(False)
            self.use_rsi_var.set(False)
            self.use_macd_var.set(False)
            self.use_day_of_week_var.set(True)
            self.use_month_var.set(False)
            
        elif scenario == "trading":
            self.use_close_var.set(True)
            self.use_high_low_var.set(True)
            self.use_volume_var.set(True)
            self.use_returns_var.set(True)
            self.use_ma_var.set(True)
            self.use_volatility_var.set(True)
            self.use_rsi_var.set(True)
            self.use_macd_var.set(False)
            self.use_day_of_week_var.set(True)
            self.use_month_var.set(True)
            
        elif scenario == "comprehensive":
            self.use_close_var.set(True)
            self.use_high_low_var.set(True)
            self.use_open_var.set(True)
            self.use_volume_var.set(True)
            self.use_returns_var.set(True)
            self.use_ma_var.set(True)
            self.use_volatility_var.set(True)
            self.use_rsi_var.set(True)
            self.use_macd_var.set(True)
            self.use_day_of_week_var.set(True)
            self.use_month_var.set(True)
            self.use_quarter_var.set(True)

    def analyze_training_quality(self, history, X_test, y_test, model, selected_features, symbol):
        """Комплексний аналіз якості навчання з правильними рекомендаціями"""
        analysis = {
            'score': 0,
            'status': 'UNKNOWN',
            'issues': [],
            'recommendations': [],
            'metrics': {},
            'warnings': []
        }
        
        try:
            # Отримуємо прогнози
            predictions = model.predict(X_test, verbose=0)
            
            # Основні метрики
            analysis['metrics']['mse'] = float(mean_squared_error(y_test, predictions))
            analysis['metrics']['mae'] = float(mean_absolute_error(y_test, predictions))
            analysis['metrics']['r2'] = float(r2_score(y_test, predictions))
            
            # Додаткові метрики
            analysis['metrics']['std_error'] = float(np.std(y_test - predictions.flatten()))
            analysis['metrics']['max_error'] = float(np.max(np.abs(y_test - predictions.flatten())))
            
            # Аналіз R² - ОСНОВНОЙ критерій якості
            if analysis['metrics']['r2'] > 0.9:
                analysis['score'] += 4
                r2_status = "Відмінно"
            elif analysis['metrics']['r2'] > 0.8:
                analysis['score'] += 3
                r2_status = "Дуже добре"
            elif analysis['metrics']['r2'] > 0.7:
                analysis['score'] += 2
                r2_status = "Добре"
            elif analysis['metrics']['r2'] > 0.5:
                analysis['score'] += 1
                r2_status = "Задовільно"
            elif analysis['metrics']['r2'] > 0.3:
                analysis['score'] += 0
                r2_status = "Слабко"
            elif analysis['metrics']['r2'] > 0:
                analysis['score'] -= 1
                r2_status = "Погано"
                analysis['issues'].append('r2_low')
                analysis['warnings'].append(f'R² занадто низький ({analysis["metrics"]["r2"]:.4f})')
            else:
                analysis['score'] -= 3
                r2_status = "Дуже погано"
                analysis['issues'].append('r2_negative')
                analysis['warnings'].append(f'R² негативний ({analysis["metrics"]["r2"]:.4f})')
            
            # Аналіз історії навчання
            if history and len(history.history) > 5:
                train_loss = history.history['loss']
                val_loss = history.history.get('val_loss', [])
                
                if val_loss:
                    # Перевірка перенавчання/недонавчання
                    final_ratio = val_loss[-1] / train_loss[-1] if train_loss[-1] > 0 else 1
                    
                    if final_ratio > 2.0:
                        analysis['issues'].append('overfitting')
                        analysis['warnings'].append(f'Можливе перенавчання (співвідношення: {final_ratio:.2f})')
                        analysis['score'] -= 2
                    elif final_ratio > 1.5:
                        analysis['warnings'].append(f'Незначне перенавчання (співвідношення: {final_ratio:.2f})')
                        analysis['score'] -= 1
                    elif final_ratio < 0.7:
                        analysis['issues'].append('underfitting')
                        analysis['warnings'].append(f'Можливе недонавчання (співвідношення: {final_ratio:.2f})')
                        analysis['score'] -= 1
                    elif final_ratio < 0.9:
                        analysis['warnings'].append(f'Незначне недонавчання (співвідношення: {final_ratio:.2f})')
                        analysis['score'] -= 0.5
                    
                    # Визначення найкращої епохи
                    best_epoch = np.argmin(val_loss) + 1
                    epochs_completed = len(train_loss)
                    
                    if best_epoch < epochs_completed * 0.8:
                        analysis['warnings'].append(f'Рання зупинка на епосі {best_epoch}/{epochs_completed}')
            
            # Аналіз ознак
            if len(selected_features) < 5:
                analysis['issues'].append('few_features')
                analysis['warnings'].append(f'Замало ознак: {len(selected_features)}')
                analysis['score'] -= 1
            elif len(selected_features) > 25:
                analysis['issues'].append('many_features')
                analysis['warnings'].append(f'Забагато ознак: {len(selected_features)}')
                analysis['score'] -= 1
            else:
                analysis['score'] += 1
            
            # Генерація КОРЕКТНИХ рекомендацій
            analysis['recommendations'] = self.generate_correct_recommendations(
                analysis, history, len(selected_features), symbol
            )
            
            # Фінальна оцінка на основі R² як основного критерію
            analysis = self.calculate_final_score_based_on_r2(analysis)
            
        except Exception as e:
            analysis['warnings'].append(f'Помилка аналізу: {str(e)}')
            analysis['status'] = 'ERROR'
        
        return analysis

    def calculate_final_score_based_on_r2(self, analysis):
        """Розрахунок фінальної оцінки на основі R² як основного критерію"""
        r2 = analysis['metrics']['r2']
        
        # R² - основний критерій якості
        if r2 > 0.9:
            analysis['status'] = 'EXCELLENT'
            analysis['score'] = 9 + min(analysis.get('score', 0), 1)  # 9-10 балів
        elif r2 > 0.8:
            analysis['status'] = 'VERY_GOOD'
            analysis['score'] = 8 + min(analysis.get('score', 0), 2)  # 8-10 балів
        elif r2 > 0.7:
            analysis['status'] = 'GOOD'
            analysis['score'] = 7 + min(analysis.get('score', 0), 3)  # 7-10 балів
        elif r2 > 0.6:
            analysis['status'] = 'FAIR'
            analysis['score'] = 6 + min(analysis.get('score', 0), 4)  # 6-10 балів
        elif r2 > 0.4:
            analysis['status'] = 'POOR'
            analysis['score'] = 4 + min(analysis.get('score', 0), 6)  # 4-10 балів
        else:
            analysis['status'] = 'FAILED'
            analysis['score'] = max(0, min(analysis.get('score', 0), 3))  # 0-3 бали
        
        # Обмежуємо score від 0 до 10
        analysis['score'] = min(max(analysis['score'], 0), 10)
        
        return analysis

    def generate_correct_recommendations(self, analysis, history, feature_count, symbol):
        """Генерація коректних рекомендацій на основі аналізу"""
        recommendations = []
        r2 = analysis['metrics']['r2']
        
        # Загальні рекомендації на основі R²
        if r2 > 0.8:
            recommendations.append("✅ Відмінний результат прогнозування!")
            if r2 > 0.9:
                recommendations.append("⭐ Модель має дуже високу точність")
        elif r2 > 0.6:
            recommendations.append("✓ Добрий результат прогнозування")
        else:
            recommendations.append("⚠️ Результат потребує покращення")
        
        # Специфічні рекомендації на основі проблем
        if 'few_features' in analysis['issues']:
            recommendations.append("• Додайте більше технічних індикаторів (MA, RSI, MACD, Bollinger Bands)")
            recommendations.append("• Перевірте наявність часових ознак")
        
        if 'overfitting' in analysis['issues']:
            recommendations.append("• Зменшіть кількість шарів LSTM")
            recommendations.append("• Збільште dropout rate")
            recommendations.append("• Додайте L2 regularization")
        
        if 'underfitting' in analysis['issues']:
            recommendations.append("• Збільшіть кількість епох навчання")
            recommendations.append("• Збільшіть кількість нейронів у шарах")
            recommendations.append("• Зменшіть learning rate")
        
        # Рекомендації щодо зупинки навчання
        if history and 'val_loss' in history.history:
            val_loss = history.history['val_loss']
            best_epoch = np.argmin(val_loss) + 1
            total_epochs = len(val_loss)
            
            if best_epoch < total_epochs * 0.7:
                recommendations.append(f"• Модель досягла оптимуму на епосі {best_epoch} - можна зменшити кількість епох")
            elif best_epoch == total_epochs:
                recommendations.append("• Модель продовжує вчитися - спробуйте збільшити кількість епох")
        
        # Рекомендації для конкретних криптовалют
        if symbol in ['LINK', 'XRP', 'ADA', 'DOT']:  # Альткойни
            recommendations.append("• Для альткойнів додайте кореляцію з BTC")
            recommendations.append("• Використовуйте об'ємні індикатори")
        
        elif symbol in ['BTC', 'ETH']:  # Основні криптовалюти
            recommendations.append("• Для BTC/ETH можна використовувати більш складні моделі")
            recommendations.append("• Додайте ринкові індикатори (Fear & Greed Index)")
        
        # Унікальні рекомендації на основі кількості ознак
        if feature_count < 8:
            recommendations.append("• Мінімальна рекомендована кількість ознак: 8-12")
        
        return recommendations

    
    def calculate_final_score(self, analysis):
        """Розрахунок фінальної оцінки якості"""
        score = analysis['score']
        
        # Додаткові бали за стабільність
        if not analysis['issues']:
            score += 2
        
        # Визначення статусу
        if score >= 7:
            analysis['status'] = 'EXCELLENT'
        elif score >= 5:
            analysis['status'] = 'GOOD'
        elif score >= 3:
            analysis['status'] = 'FAIR'
        elif score >= 0:
            analysis['status'] = 'POOR'
        else:
            analysis['status'] = 'FAILED'
        
        analysis['score'] = min(max(score, 0), 10)  # Обмежуємо 0-10
        return analysis
    
    def generate_recommendations(self, analysis, history, feature_count):
        """Генерація конкретних рекомендацій"""
        recommendations = []
        
        # Рекомендації на основі проблем
        if 'r2_negative' in analysis['issues']:
            recommendations.extend([
                "🚨 СПОЧАТКУ ВИРІШИТЬ ПРОБЛЕМИ!",
                "• Перевірте якість даних на наявність NaN",
                "• Спростіть модель до базового рівня",
                "• Перевірте цільову змінну на викиди"
            ])
        
        elif 'r2_low' in analysis['issues']:
            recommendations.extend([
                "• Збільшіть кількість епох навчання",
                "• Додайте більше технічних індикаторів",
                "• Спробуйте змінити learning rate",
                "• Перевірте кореляцію ознак з ціллю"
            ])
        
        # Рекомендації щодо ознак
        if 'few_features' in analysis['issues']:
            recommendations.append("• Додайте більше ознак (MA, RSI, Volatility)")
        
        if 'many_features' in analysis['issues']:
            recommendations.append("• Зменшіть кількість ознак або використайте feature selection")
        
        # Рекомендації щодо навчання
        if history and len(history.history.get('val_loss', [])) > 10:
            val_loss = history.history['val_loss']
            if min(val_loss) == val_loss[-1]:
                recommendations.append("✅ Модель продовжує покращуватись - збільшіть кількість епох")
        
        # Загальні рекомендації
        if analysis['metrics']['r2'] > 0.7:
            recommendations.extend([
                "✅ Відмінний результат!",
                "• Можете експериментувати з більш складними моделями",
                "• Спробуйте додати часові ознаки"
            ])
        
        return recommendations

    
    def create_quality_report(self, history, quality_analysis, symbol):
        """Створення звіту якості навчання"""
        report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'training_quality': quality_analysis['status'],
            'score': quality_analysis['score'],
            'metrics': quality_analysis['metrics'],
            'issues': quality_analysis['issues'],
            'recommendations': quality_analysis['recommendations'],
            'epochs_trained': len(history.history['loss']) if history else 0,
            'total_epochs': self.epochs_var.get()
        }
        
        # Збереження звіту
        report_path = f'models/{symbol}_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

    def find_optimal_learning_rate(self, model, X_train, y_train, X_val, y_val):
        """Знаходження оптимального learning rate"""
        learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        best_lr = 0.001
        best_loss = float('inf')
        
        self.add_log_message("🔍 Пошук оптимального Learning Rate...\n")
        
        for lr in learning_rates:
            # Клонуємо модель
            model_clone = tf.keras.models.clone_model(model)
            model_clone.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='mse',
                metrics=['mae']
            )
            
            # Коротке навчання
            history = model_clone.fit(
                X_train, y_train,
                epochs=5,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            val_loss = history.history['val_loss'][-1]
            self.add_log_message(f"  LR: {lr:.0e} -> Val Loss: {val_loss:.6f}\n")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_lr = lr
        
        self.add_log_message(f"✅ Оптимальний Learning Rate: {best_lr:.0e}\n")
        return best_lr

    def analyze_feature_correlation(self, data, target_column='Close'):
        """Аналіз кореляції ознак з цільовою змінною"""
        correlation = data.corr()[target_column].sort_values(ascending=False)
        
        self.add_log_message("📊 КОРЕЛЯЦІЯ ОЗНАК З ЦІНОЮ ЗАКРИТТЯ:\n")
        for feature, corr_value in correlation.items():
            if feature != target_column:
                significance = "🚀 ВИСОКА" if abs(corr_value) > 0.3 else "✅ ПОМІРНА" if abs(corr_value) > 0.1 else "⚠️ НИЗЬКА"
                self.add_log_message(f"  {feature:25s}: {corr_value:7.3f} ({significance})\n")
        
        return correlation

    def select_features_by_correlation(self, data, min_correlation=0.1, target_column='Close'):
        """Вибір ознак на основі кореляції"""
        correlation = data.corr()[target_column]
        selected_features = correlation[abs(correlation) >= min_correlation].index.tolist()
        
        # Завжди включаємо цільову змінну
        if target_column not in selected_features:
            selected_features.append(target_column)
        
        self.add_log_message(f"📈 ВІДІБРАНО {len(selected_features)} ОЗНАК З КОРЕЛЯЦІЄЮ ≥ {min_correlation}:\n")
        for feature in selected_features:
            self.add_log_message(f"  • {feature}\n")
        
        return selected_features


    
    def show_log_window(self):
        """Показує вікно логу"""
        if self.log_window is None or not self.log_window.winfo_exists():
            self.log_window = tk.Toplevel(self.parent)
            self.log_window.title("Лог навчання")
            self.log_window.geometry("800x400")
            self.log_window.protocol("WM_DELETE_WINDOW", self.hide_log_window)
            
            self.log_text = scrolledtext.ScrolledText(self.log_window, wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Перенаправляємо stdout/stderr
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            sys.stdout = TextRedirector(self.log_text, "stdout")
            sys.stderr = TextRedirector(self.log_text, "stderr")
    
    def hide_log_window(self):
        """Ховає вікно логу"""
        if self.log_window is not None and self.log_window.winfo_exists():
            # Відновлюємо stdout/stderr
            if self.original_stdout:
                sys.stdout = self.original_stdout
            if self.original_stderr:
                sys.stderr = self.original_stderr
            
            self.log_window.destroy()
            self.log_window = None
            self.log_text = None
            self.show_log_var.set(False)
    
    def show_context_menu(self, event):
        """Показує контекстне меню"""
        self.context_menu.tk_popup(event.x_root, event.y_root)
        
    def train_models_thread(self, symbols, params):
        """Потокова функція для навчання моделей"""
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(symbols):
            if self.training_stop_flag:
                break
            
            self.status_callback(f"Статус: Навчання {symbol} ({i+1}/{len(symbols)})...")
            
            try:
                # Завантажуємо дані
                data_file = self.model_manager.get_data_file(symbol)
                if not data_file:
                    self.status_callback(f"Статус: Файл даних для {symbol} не знайдено")
                    failed += 1
                    continue
                
                df = pd.read_csv(data_file)
                if len(df) < 100:  # Мінімальна кількість записів
                    self.status_callback(f"Статус: Недостатньо даних для {symbol} ({len(df)} записів)")
                    failed += 1
                    continue
                
                # Підготовка даних
                X_train, X_test, y_train, y_test, scaler, feature_names = self.prepare_data(df, params)
                
                if X_train is None or len(X_train) == 0:
                    self.status_callback(f"Статус: Помилка підготовки даних для {symbol}")
                    failed += 1
                    continue
                
                # Створення моделі
                model = self.create_model(params, X_train.shape[1:])
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=params['patience'],
                        min_delta=params['min_delta'],
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=params['patience'] // 2,
                        min_lr=1e-6,
                        verbose=1
                    ),
                    TrainingProgressCallback(
                        self.status_callback,
                        self.update_progress,
                        params['epochs'],
                        self.log_callback if self.log_text else None
                    )
                ]
                
                # Навчання моделі
                history = model.fit(
                    X_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Оцінка моделі
                train_loss = model.evaluate(X_train, y_train, verbose=0)
                test_loss = model.evaluate(X_test, y_test, verbose=0)
                
                # Прогнозування
                y_pred = model.predict(X_test, verbose=0)
                
                # Зворотнє масштабування
                y_test_orig = scaler.inverse_transform(
                    np.concatenate([np.zeros((len(y_test), len(feature_names) - 1)), y_test.reshape(-1, 1)], axis=1)
                )[:, -1]
                
                y_pred_orig = scaler.inverse_transform(
                    np.concatenate([np.zeros((len(y_pred), len(feature_names) - 1)), y_pred.reshape(-1, 1)], axis=1)
                )[:, -1]
                
                # Метрики
                mse = mean_squared_error(y_test_orig, y_pred_orig)
                mae = mean_absolute_error(y_test_orig, y_pred_orig)
                r2 = r2_score(y_test_orig, y_pred_orig)
                
                # Збереження моделі
                model_info = {
                    'symbol': symbol,
                    'training_type': params['training_type'],
                    'metrics': {
                        'train_loss': float(train_loss),
                        'test_loss': float(test_loss),
                        'mse': float(mse),
                        'mae': float(mae),
                        'r2': float(r2)
                    },
                    'parameters': params,
                    'feature_names': feature_names,
                    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data_points': len(df)
                }
                
                self.model_manager.save_model(model, model_info)
                
                # Оновлення графіка
                self.update_training_plot(history, symbol, model_info['metrics'])
                
                successful += 1
                self.status_callback(f"Статус: ✅ {symbol} навчено успішно (R²={r2:.4f})")
                
            except Exception as e:
                failed += 1
                error_msg = f"Статус: Помилка навчання {symbol}: {str(e)}"
                self.status_callback(error_msg)
                print(f"ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
        
        self.status_callback(f"Статус: ✅ Навчання завершено. Успішно: {successful}, Не вдалося: {failed}")
        self.status_label.config(text="Навчання завершено", foreground="green")
        self.progress_bar['value'] = 100
        
    def calculate_basic_indicators(self, df):
        """Розраховує базові технічні індикатори"""
        df = df.copy()
        
        # Returns
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        # Moving Averages
        if 'Close' in df.columns:
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Volatility
        if 'Returns' in df.columns:
            df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df
        
    def log_callback(self, message):
        """Callback для логування"""
        if self.log_text:
            self.log_text.insert(tk.END, message + '\n')
            self.log_text.see(tk.END)
    
    def toggle_log_window(self):
        """Перемикач вікна логу"""
        if self.show_log_var.get():
            self.open_log_window()
        else:
            self.close_log_window()

    def open_log_window(self):
        """Відкриття вікна логу з перенаправленням всіх логів"""
        if self.log_window and self.log_window.winfo_exists():
            self.log_window.lift()
            return
        
        self.log_window = tk.Toplevel(self.parent)
        self.log_window.title("Лог навчання")
        self.log_window.geometry("900x600")
        self.log_window.protocol("WM_DELETE_WINDOW", self.on_log_window_close)
        
        # Головний фрейм
        main_frame = ttk.Frame(self.log_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Текстове поле для логу
        log_frame = ttk.LabelFrame(main_frame, text="Лог навчання (всі повідомлення)")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=100, height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Кнопки управління
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Зберегти лог", command=self.save_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Очистити лог", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Закрити", command=self.close_log_window).pack(side=tk.RIGHT, padx=5)
        
        # Перенаправляємо всі логи
        self.redirect_all_logs()
        
        # Додаємо початкове повідомлення
        self.add_log_message("=== Лог навчання нейромереж ===\n")
        self.add_log_message(f"Час початку: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.add_log_message("=" * 60 + "\n\n")

    def redirect_all_logs(self):
        """Перенаправлення всіх логів у вікно логу"""
        import logging
        from logging import StreamHandler
        
        # Зберігаємо оригінальні обробники
        if not hasattr(self, 'original_handlers'):
            self.original_handlers = {}
        
        # Отримуємо кореневий логер
        root_logger = logging.getLogger()
        
        # Зберігаємо поточні обробники
        self.original_handlers['root'] = root_logger.handlers.copy()
        
        # Створюємо спеціальний обробник для перенаправлення
        class LogRedirectHandler(StreamHandler):
            def __init__(self, training_tab):
                super().__init__()
                self.training_tab = training_tab
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.training_tab.add_log_message(msg + '\n')
                except Exception:
                    self.handleError(record)
        
        # Додаємо наш обробник до кореневого логера
        redirect_handler = LogRedirectHandler(self)
        redirect_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(redirect_handler)
        
        # Також перенаправляємо stdout/stderr
        self.redirect_console_output()
    
    def on_log_window_close(self):
        """Обробник закриття вікна логу"""
        self.show_log_var.set(False)
        self.close_log_window()

    def close_log_window(self):
        """Закриття вікна логу з відновленням логів"""
        if self.log_window:
            try:
                # Відновлюємо оригінальні обробники логів
                self.restore_log_handlers()
                
                # Відновлюємо stdout/stderr
                self.restore_console_output()
                
                self.log_window.destroy()
                self.log_window = None
                self.log_text = None
                
                # Оновлюємо стан чекбоксу
                self.show_log_var.set(False)
                
            except Exception as e:
                print(f"Помилка закриття вікна логу: {e}")

    def restore_log_handlers(self):
        """Відновлення оригінальних обробників логів"""
        if hasattr(self, 'original_handlers'):
            import logging
            root_logger = logging.getLogger()
            
            # Видаляємо всі поточні обробники
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Відновлюємо оригінальні обробники
            for handler in self.original_handlers.get('root', []):
                root_logger.addHandler(handler)
    
    def redirect_console_output(self):
        """Перенаправлення виводу консолі у вікно логу"""
        if self.original_stdout is None:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
        
        sys.stdout = self
        sys.stderr = self

    def restore_console_output(self):
        """Відновлення стандартного виводу консолі"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.original_stdout = None
            self.original_stderr = None

    def write(self, message):
        """Запис повідомлення у лог (для перенаправлення stdout/stderr)"""
        if self.log_text and self.log_window and self.log_window.winfo_exists():
            self.add_log_message(message)
        
        # Також виводимо у оригінальний stdout для консолі
        if self.original_stdout:
            self.original_stdout.write(message)

    def flush(self):
        """Flush метод для сумісності з sys.stdout"""
        if self.original_stdout:
            self.original_stdout.flush()

    def add_log_message(self, message):
        """Додавання повідомлення у лог з фільтрацією зайвих повідомлень"""
        if self.log_text and self.log_window and self.log_window.winfo_exists():
            try:
                # Фільтрація зайвих повідомлень
                skip_messages = [
                    "This TensorFlow binary is optimized to use available CPU instructions",
                    "Do not pass an `input_shape`/`input_dim` argument to a layer",
                    "The `save_format` argument is deprecated",
                    "You are saving your model as an HDF5 file"
                ]
                
                if any(skip_msg in message for skip_msg in skip_messages):
                    return
                    
                # Автоматичне прокручування до кінця
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
                self.log_text.update_idletasks()
            except Exception as e:
                print(f"Помилка додавання повідомлення у лог: {e}")

    def save_log(self):
        """Збереження логу у файл"""
        if not self.log_text:
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_log_{timestamp}.txt"
            
            log_content = self.log_text.get(1.0, tk.END)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(log_content)
            
            self.add_log_message(f"\n✅ Лог збережено у файл: {filename}\n")
            messagebox.showinfo("Успіх", f"Лог збережено у файл: {filename}")
            
        except Exception as e:
            error_msg = f"❌ Помилка збереження логу: {str(e)}"
            self.add_log_message(f"\n{error_msg}\n")
            messagebox.showerror("Помилка", error_msg)

    def clear_log(self):
        """Очищення вмісту логу"""
        if self.log_text:
            if messagebox.askyesno("Підтвердження", "Очистити весь лог?"):
                self.log_text.delete(1.0, tk.END)
                self.add_log_message("=== Лог очищено ===\n")
                self.add_log_message(f"Час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.add_log_message("=" * 50 + "\n\n")
    
    def analyze_training_results(self, history, X_test, y_test, model):
        """Детальний аналіз результатів навчання"""
        results = {}
        
        # Прогнозування
        predictions = model.predict(X_test, verbose=0)
        
        # Конвертація в Python типи
        def to_python_type(value):
            if hasattr(value, 'item'):
                return value.item()
            return value

        # Основні метрики
        results['mse'] = to_python_type(mean_squared_error(y_test, predictions))
        results['mae'] = to_python_type(mean_absolute_error(y_test, predictions))
        results['r2'] = to_python_type(r2_score(y_test, predictions))
        
        # MAPE
        try:
            results['mape'] = to_python_type(mean_absolute_percentage_error(y_test, predictions))
        except:
            try:
                y_test_clean = y_test[y_test != 0]  # Уникаємо ділення на нуль
                predictions_clean = predictions.flatten()[y_test != 0]
                mape = np.mean(np.abs((y_test_clean - predictions_clean) / y_test_clean)) * 100
                results['mape'] = to_python_type(mape)
            except:
                results['mape'] = float('nan')

        # Аналіз залишків
        residuals = y_test - predictions.flatten()
        results['residual_mean'] = to_python_type(np.mean(residuals))
        results['residual_std'] = to_python_type(np.std(residuals))
        results['residual_skew'] = to_python_type(pd.Series(residuals).skew())
        results['residual_kurtosis'] = to_python_type(pd.Series(residuals).kurtosis())
        
        # Статистика навчання
        results['final_train_loss'] = to_python_type(history.history['loss'][-1])
        
        if 'val_loss' in history.history:
            results['final_val_loss'] = to_python_type(history.history['val_loss'][-1])
            results['best_val_loss'] = to_python_type(np.min(history.history['val_loss']))
            results['best_epoch'] = to_python_type(np.argmin(history.history['val_loss']) + 1)
            
            # Аналіз перенавчання
            results['overfitting_ratio'] = to_python_type(results['final_val_loss'] / results['final_train_loss'])
        
        # Час навчання (приблизно)
        results['total_epochs'] = to_python_type(len(history.history['loss']))
        
        return results

    def generate_training_report(self, symbol, history, metrics, params):
        """Генерація звіту про навчання"""
        report = {
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'training_type': self.convert_to_serializable(params.get('training_type', 'basic')),
            'parameters': self.convert_to_serializable(params),
            'metrics': self.convert_to_serializable(metrics),
            'training_history': self.convert_to_serializable({
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None,
                'best_epoch': np.argmin(history.history['val_loss']) + 1 if 'val_loss' in history.history else len(history.history['loss']),
                'total_epochs': len(history.history['loss'])
            })
        }
        
        # Збереження звіту
        report_path = f'models/{symbol}_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

    def convert_to_serializable(self, obj):
        """Конвертує об'єкт у JSON-серіалізований формат"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    def analyze_features(self, feature_names):
        """Аналіз та групування ознак для логування"""
        groups = {
            'Цінові': ['Close', 'Open', 'High', 'Low', 'Price'],
            'Волатильність': ['Volatility', 'ATR', 'VaR', 'Drawdown', 'Std'],
            'Momentum': ['RSI', 'MACD', 'Stochastic', 'Returns', 'Momentum'],
            'Трендові': ['MA', 'EMA', 'SMA', 'Trend'],
            'Об\'єм': ['Volume', 'OBV'],
            'Часові': ['Hour', 'Day', 'Month', 'Week', 'Session'],
            'Ринкові': ['BTC', 'Market', 'Correlation'],
            'Ризик': ['Risk', 'Sharpe', 'Sortino', 'VaR', 'ES']
        }
        
        result = {}
        for feature in feature_names:
            found_group = 'Інші'
            for group_name, keywords in groups.items():
                if any(keyword in feature for keyword in keywords):
                    found_group = group_name
                    break
            
            if found_group not in result:
                result[found_group] = []
            result[found_group].append(feature)
        
        return result
     
    def update_selected_count(self):
        """Оновити лічильник обраних елементів"""
        selected_count = len(self.data_tree.selection())
        self.selected_count_var.set(f"Обрано: {selected_count}")
       
    
    
    def update_info_text(self, text):
        """Оновлення інформаційного тексту"""
        self.info_text.insert(tk.END, text + "\n")
        self.info_text.see(tk.END)
        
    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        if self.status_callback:
            self.status_callback(message)
    
    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        if self.progress_callback:
            self.progress_callback(value)

class TextRedirector:
    """Клас для перенаправлення stdout/stderr у текстове поле"""
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag
    
    def write(self, string):
        self.text_widget.insert(tk.END, string, (self.tag,))
        self.text_widget.see(tk.END)
    
    def flush(self):
        pass



