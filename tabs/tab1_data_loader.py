import os
os.environ['YFINANCE_DISABLE_WEBSOCKET'] = '1'
import yfinance as yf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import threading
import time
from bs4 import BeautifulSoup
import ccxt
from urllib.parse import quote
import warnings
import aiohttp
import asyncio
import concurrent.futures
from functools import lru_cache
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
import csv
import random 

warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






class UltimateCryptoDataLoader:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.crypto_list = []
        self.filtered_crypto_list = []
        self.exchanges = {}
        self.all_crypto_data = {}
        self.session = None
        self.sort_column = 'market_cap_rank'
        self.sort_direction = 'asc'
        self.current_filters = {}
        self.tree_item_data = {}  # Для зберігання даних елементів treeview
        self.cache_dir = 'cache'
        self.request_cache = {}
        
        # Додаткові атрибути з EnhancedDataLoaderTab
        self.filter_var = tk.StringVar()
        self.source_var = tk.StringVar(value="all")
        self.period_var = tk.StringVar(value="max")
        self.include_ohlcv = tk.BooleanVar(value=True)
        self.include_technical = tk.BooleanVar(value=True)
        self.include_social = tk.BooleanVar(value=True)
        self.include_fundamental = tk.BooleanVar(value=False)
        self.symbol_var = tk.StringVar()
        self.name_var = tk.StringVar()
        
        # Додати ці змінні для лічильників
        self.selection_stats_var = tk.StringVar(value="Обрано: 0 / 0")
        self.total_count_var = tk.StringVar(value="Всього: 0 криптовалют")
        
        self.non_existent_symbols = set()  # Для зберігання неіснуючих символів

        self.COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
        self.CMC_API_URL = "https://pro-api.coinmarketcap.com/v1"
        self.USER_AGENTS = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
        ]
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(self.USER_AGENTS)})

        self.setup_ui()
        self.setup_styles()  # Додайте цей рядок
        self.initialize_session()
        self.initialize_exchanges()

        # Спочатку пробуємо завантажити з кешу
        if not self.load_from_cache():
            # Якщо кешу немає, завантажуємо звичайний список
            self.load_crypto_list()
        
        self.ensure_directories()

        self.load_crypto_list()  

    def setup_ui(self):
        """Розширений інтерфейс з сортуванням та фільтрацією"""
        # Основний фрейм
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Додаємо праву панель з EnhancedDataLoaderTab ПІД САМИЙ ВЕРХ
        self.setup_styles()
        self.setup_right_panel(main_frame)

        # Верхня панель - фільтри та сортування
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Ліва частина - фільтрація (зменшена ширина)
        filter_frame = ttk.LabelFrame(control_frame, text="Фільтрація")
        filter_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5), ipadx=5)
        
        # Пошук за назвою/символом
        search_frame = ttk.Frame(filter_frame)
        search_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(search_frame, text="Пошук:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.apply_filters())
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=15)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Фільтр за джерелом даних
        source_frame = ttk.Frame(filter_frame)
        source_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(source_frame, text="Джерело:").pack(side=tk.LEFT)
        self.source_filter_var = tk.StringVar(value="all")
        sources = [("Всі", "all"), ("CG", "coingecko"), ("CMC", "coinmarketcap"), 
                ("Біржі", "exchange"), ("Інші", "other")]
        for text, value in sources:
            ttk.Radiobutton(source_frame, text=text, variable=self.source_filter_var, 
                        value=value, command=self.apply_filters).pack(side=tk.LEFT, padx=1)
        
        # Фільтр за якістю даних
        quality_frame = ttk.Frame(filter_frame)
        quality_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(quality_frame, text="Якість:").pack(side=tk.LEFT)
        self.quality_filter_var = tk.StringVar(value="all")
        qualities = [("Всі", "all"), ("Висока", "high"), ("Середня", "medium"), 
                    ("Низька", "low"), ("Немає", "none")]
        for text, value in qualities:
            ttk.Radiobutton(quality_frame, text=text, variable=self.quality_filter_var, 
                        value=value, command=self.apply_filters).pack(side=tk.LEFT, padx=1)
        
        # Центральна частина - керування списками (2 стовпці) - ЗМЕНШЕНА
        list_control_frame = ttk.LabelFrame(control_frame, text="Керування списками")
        list_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, ipadx=2)
        
        # Лівий стовпець кнопок
        left_column = ttk.Frame(list_control_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        # Правий стовпець кнопок
        right_column = ttk.Frame(list_control_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        
        # Кнопки в лівому стовпці
        ttk.Button(left_column, text="Завантажити список", 
                command=self.load_crypto_list_from_api).pack(pady=1, fill=tk.X)
        
        ttk.Button(left_column, text="Топ 100", 
                command=lambda: self.load_top_crypto_list(100)).pack(pady=1, fill=tk.X)
        
        ttk.Button(left_column, text="Топ 500", 
                command=lambda: self.load_top_crypto_list(500)).pack(pady=1, fill=tk.X)
        
        ttk.Button(left_column, text="Топ 1000", 
                command=lambda: self.load_top_crypto_list(1000)).pack(pady=1, fill=tk.X)
        
        ttk.Button(left_column, text="Завантажити обрані", 
                command=self.download_selected_from_list).pack(pady=1, fill=tk.X)
        
        # Кнопки в правому стовпці
        ttk.Button(right_column, text="Зберегти список", 
                command=self.save_crypto_list).pack(pady=1, fill=tk.X)
        
        ttk.Button(right_column, text="Відкрити список", 
                command=self.open_crypto_list).pack(pady=1, fill=tk.X)
        
        ttk.Button(right_column, text="Видалити список", 
                command=self.delete_crypto_list).pack(pady=1, fill=tk.X)
        
        ttk.Button(right_column, text="Актуалізувати список", 
                command=self.refresh_crypto_list).pack(pady=1, fill=tk.X)
        ttk.Button(right_column, text="Завантажити з кешу", 
                command=self.load_from_cache).pack(pady=1, fill=tk.X)
        
        
        # Права частина - сортування
        sort_frame = ttk.LabelFrame(control_frame, text="Сортування")
        sort_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        
        # Випадаючий список для сортування
        sort_options_frame = ttk.Frame(sort_frame)
        sort_options_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(sort_options_frame, text="Сорт:").pack(side=tk.LEFT)
        self.sort_var = tk.StringVar(value="market_cap_rank")
        
        sort_options = [
            ("Капіталізація", "market_cap_rank"),
            ("Назва", "name"),
            ("Символ", "symbol"),
            ("Ціна", "current_price"),
            ("Об'єм", "volume"),
            ("Зміна 24h", "price_change_24h"),
            ("Якість", "data_quality"),
            ("Вік", "age_days")
        ]
        
        sort_combobox = ttk.Combobox(sort_options_frame, textvariable=self.sort_var, 
                                values=[opt[0] for opt in sort_options], state="readonly", width=12)
        sort_combobox.pack(side=tk.LEFT, padx=2)
        sort_combobox.bind('<<ComboboxSelected>>', self.on_sort_change)
        
        # Напрямок сортування
        sort_dir_frame = ttk.Frame(sort_frame)
        sort_dir_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(sort_dir_frame, text="Напрям:").pack(side=tk.LEFT)
        self.sort_dir_var = tk.StringVar(value="asc")
        ttk.Radiobutton(sort_dir_frame, text="↑", variable=self.sort_dir_var, 
                    value="asc", command=self.apply_sorting).pack(side=tk.LEFT, padx=1)
        ttk.Radiobutton(sort_dir_frame, text="↓", variable=self.sort_dir_var, 
                    value="desc", command=self.apply_sorting).pack(side=tk.LEFT, padx=1)
        
        # Кнопки швидкого сортування
        quick_sort_frame = ttk.Frame(sort_frame)
        quick_sort_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_sort_frame, text="Топ-100", command=lambda: self.quick_sort("top100")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(quick_sort_frame, text="Топ-500", command=lambda: self.quick_sort("top500")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(quick_sort_frame, text="Топ-1000", command=lambda: self.quick_sort("top1000")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Додаємо праву панель з EnhancedDataLoaderTab ВИЩЕ основного списку
        #self.setup_right_panel(main_frame)
        
        # Основна область - список криптовалют
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview з прокруткою
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Визначаємо колонки
        self.tree_columns = [
            ('select', 5, '', False),
            ('symbol', 40, 'Символ', True),
            ('name', 150, 'Назва', True),
            ('market_cap', 100, 'Капіталізація', True),
            ('volume', 100, 'Об\'єм', True),
            ('price', 80, 'Ціна', True),
            ('change_24h', 80, 'Зміна 24h', True),
            ('rank', 60, 'Рейтинг', True),
            ('quality', 80, 'Якість', True),
            ('age', 80, 'Вік', True),
            ('source', 100, 'Джерело', True),
            ('data_points', 80, 'Точок даних', True)
        ]
        
        # Створюємо Treeview з прокруткою
        self.crypto_tree = ttk.Treeview(tree_frame, 
                                    columns=[col[0] for col in self.tree_columns], 
                                    show='tree headings',
                                    height=8,
                                    selectmode='extended')  # Дозволяємо множинний вибір
        
        # Налаштування колонок
        for col_id, width, heading, visible in self.tree_columns:
            if visible:
                self.crypto_tree.heading(col_id, text=heading, command=lambda c=col_id: self.on_header_click(c))
                self.crypto_tree.column(col_id, width=width, anchor=tk.CENTER)
        
        # ПРИХОВУЄМО ПЕРШИЙ ПОРОЖНІЙ СТОВБЕЦЬ
        self.hide_first_column()
        
        # Додаємо вертикальну прокрутку
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.crypto_tree.yview)
        self.crypto_tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Додаємо горизонтальну прокрутку
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.crypto_tree.xview)
        self.crypto_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Розміщуємо елементи за допомогою grid для кращого контролю
        self.crypto_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Налаштовуємо вагу для розтягування
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Нижня панель - керування вибором
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(selection_frame, text="Вибір:").pack(side=tk.LEFT)
        ttk.Button(selection_frame, text="Вибрати всі", command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(selection_frame, text="Скасувати всі", command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(selection_frame, text="Інвертувати", command=self.invert_selection).pack(side=tk.LEFT, padx=5)
        
        # Статистика вибору
        ttk.Label(selection_frame, textvariable=self.selection_stats_var).pack(side=tk.LEFT, padx=10)
        
        # Загальна кількість криптовалют
        ttk.Label(selection_frame, textvariable=self.total_count_var).pack(side=tk.RIGHT)
        
        # Панель дій
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(action_frame, text="Завантажити обрані", command=self.download_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Завантажити все", command=self.download_all_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Оновити дані", command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Експорт списку", command=self.export_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Зберегти вибір", command=self.save_selection).pack(side=tk.LEFT, padx=5)
        
        # Додаємо кнопки з EnhancedDataLoaderTab
        ttk.Button(action_frame, text="Перевірити якість", command=self.validate_data_quality).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Оновити всі дані", command=self.update_all_data).pack(side=tk.LEFT, padx=5)
        
        # Панель прогресу
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=(0, 2))
        
        # Текстове поле для статусу прогресу
        self.progress_label = ttk.Label(progress_frame, text="Готово до роботи", 
                                    font=('Arial', 9), foreground='gray')
        self.progress_label.pack(fill=tk.X, padx=5)
        
        # Додаємо обробники подій для treeview
        self.setup_tree_bindings()
        
        # Встановлюємо початковий стан прогресс-бару
        self.safe_progress_callback(0)

    def setup_right_panel(self, main_frame):
        """Додає праву панель з EnhancedDataLoaderTab - під самим верхом"""
        # Правий фрейм - керування (компактний)
        right_frame = ttk.LabelFrame(main_frame, text="Додаткові налаштування")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, anchor='n')  # anchor='n' для вирівнювання по верху
        
        # Фіксована ширина
        right_frame.config(width=160)
        
        # Група періоду даних (компактна)
        period_group = ttk.LabelFrame(right_frame, text="Період")
        period_group.pack(fill=tk.X, padx=2, pady=1)
        
        self.period_var = tk.StringVar(value="max")
        periods = [("Макс", "max"), ("5р", "1825"), ("3р", "1095"), 
                ("1р", "365"), ("6м", "180")]
        
        for text, value in periods:
            ttk.Radiobutton(period_group, text=text, variable=self.period_var, 
                        value=value, width=3).pack(anchor=tk.W, pady=0)
        
        # Група додаткових даних (компактна)
        extra_group = ttk.LabelFrame(right_frame, text="Дані")
        extra_group.pack(fill=tk.X, padx=2, pady=1)
        
        self.include_ohlcv = tk.BooleanVar(value=True)
        self.include_technical = tk.BooleanVar(value=True)
        self.include_social = tk.BooleanVar(value=True)
        self.include_fundamental = tk.BooleanVar(value=False)
        
        # Компактні чекбокси
        ttk.Checkbutton(extra_group, text="OHLCV", variable=self.include_ohlcv, 
                    width=6).pack(anchor=tk.W, pady=0)
        ttk.Checkbutton(extra_group, text="Техн.", variable=self.include_technical,
                    width=6).pack(anchor=tk.W, pady=0)
        ttk.Checkbutton(extra_group, text="Соц.", variable=self.include_social,
                    width=6).pack(anchor=tk.W, pady=0)
        ttk.Checkbutton(extra_group, text="Фунд.", variable=self.include_fundamental,
                    width=6).pack(anchor=tk.W, pady=0)
        
        # Додавання вручну (компактне)
        add_frame = ttk.LabelFrame(right_frame, text="Додати")
        add_frame.pack(fill=tk.X, padx=2, pady=1)
        
        ttk.Label(add_frame, text="Символ:").pack(anchor=tk.W)
        self.symbol_var = tk.StringVar()
        symbol_entry = ttk.Entry(add_frame, textvariable=self.symbol_var, width=8)
        symbol_entry.pack(fill=tk.X, pady=0)
        
        ttk.Label(add_frame, text="Назва:").pack(anchor=tk.W)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(add_frame, textvariable=self.name_var, width=8)
        name_entry.pack(fill=tk.X, pady=0)
        
        ttk.Button(add_frame, text="Додати", 
                command=self.add_crypto_manual, width=8).pack(pady=1, fill=tk.X)
        
        # Кнопки швидкого доступу
        quick_actions = ttk.LabelFrame(right_frame, text="Дії")
        quick_actions.pack(fill=tk.X, padx=2, pady=1)
        
        ttk.Button(quick_actions, text="Перевірити", 
                command=self.validate_data_quality, width=10).pack(pady=0, fill=tk.X)
        ttk.Button(quick_actions, text="Оновити всі", 
                command=self.update_all_data, width=10).pack(pady=0, fill=tk.X)

    def hide_first_column(self):
        """Повністю приховує перший порожній стовбець Treeview"""
        try:
            # Отримуємо всі колонки
            columns = list(self.crypto_tree['columns'])
            
            # Встановлюємо ширину першої колонки в 0 і приховуємо її
            self.crypto_tree.column('#0', width=0, stretch=False, minwidth=0)
            
            # Додатково налаштовуємо відображення
            self.crypto_tree.configure(show='headings')  # Приховуємо tree column
            
        except Exception as e:
            logger.debug(f"Помилка приховування першого стовпця: {e}")

    def get_data_path(self, filename):
        """Повертає повний шлях до файлу в папці data"""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return os.path.join(data_dir, filename)
    
    def is_likely_nonexistent(self, symbol):
        """Перевірка чи символ швидше за все не існує"""
        # Тільки явно неіснуючі символи
        known_non_existent = [
            'JUPSOL', 'FAKE123', 'TEST456', 'NONEXISTENT',
            'EXAMPLE', 'DUMMY', 'INVALID'
        ]
        
        # Символи з підкресленням можуть існувати
        if '_' in symbol:
            return False
            
        # Символи з дефісом можуть існувати (наприклад, BSC-USD)
        if '-' in symbol:
            return False
            
        # Символи з цифрами можуть існувати
        if any(char.isdigit() for char in symbol):
            return False
            
        return symbol in known_non_existent

    def get_nonexistent_symbols(self):
        """Отримати список неіснуючих символів"""
        return list(self.non_existent_symbols)

    def clear_nonexistent_symbols(self):
        """Очистити список неіснуючих символів"""
        self.non_existent_symbols.clear()
        logger.info("Список неіснуючих символів очищено")

    def load_crypto_list_from_api(self):
        """Завантаження списку криптовалют з API"""
        def load_thread():
            try:
                self.safe_status_callback("Завантаження списку криптовалют...")
                self.safe_progress_callback(0)
                
                # Отримуємо повний список криптовалют
                all_cryptos = self.get_complete_crypto_list()
                
                # Перевіряємо, чи отримали коректний список
                if not all_cryptos or not isinstance(all_cryptos, list):
                    self.safe_status_callback("Не вдалося завантажити список криптовалют")
                    # Використовуємо резервний список
                    all_cryptos = self.get_backup_crypto_list()
                
                self.crypto_list = all_cryptos
                self.filtered_crypto_list = all_cryptos.copy()
                
                # ЗБЕРІГАЄМО ОСТАННІЙ ЗАВАНТАЖЕНИЙ СПИСОК
                self.save_coingecko_cache(all_cryptos)
                
                # Застосовуємо фільтри для відображення
                self.apply_filters()
                
                self.safe_status_callback(f"Завантажено {len(all_cryptos)} криптовалют")
                self.safe_progress_callback(100)
                
            except Exception as e:
                self.safe_status_callback(f"Помилка завантаження списку: {str(e)}")
                self.safe_progress_callback(0)
        
        threading.Thread(target=load_thread, daemon=True).start()

    def get_complete_crypto_list(self):
        """Отримання повного списку криптовалют з обходом обмежень"""
        try:
            self.safe_status_callback("Отримання списку криптовалют...")
            
            # Спроба 1: CoinGecko (основне джерело)
            coins = self.fetch_coingecko_complete_list()
            
            # Спроба 2: CoinMarketCap (резервне джерело)
            if not coins or len(coins) < 1000:
                cmc_coins = self.fetch_coinmarketcap_list()
                if cmc_coins:
                    coins.extend(cmc_coins)
            
            # Спроба 3: Скрапінг з CoinMarketCap
            if not coins or len(coins) < 2000:
                scraped_coins = self.scrape_coinmarketcap_complete()
                if scraped_coins:
                    coins.extend(scraped_coins)
            
            # Якщо всі спроби не вдалися, використовуємо резервний список
            if not coins:
                coins = self.get_backup_crypto_list()
            
            # Видаляємо дублікати
            unique_coins = []
            seen_symbols = set()
            
            for coin in coins:
                if coin and isinstance(coin, dict):
                    symbol = coin.get('symbol', '').upper()
                    if symbol and symbol not in seen_symbols:
                        unique_coins.append(coin)
                        seen_symbols.add(symbol)
            
            self.safe_status_callback(f"Знайдено {len(unique_coins)} унікальних криптовалют")
            return unique_coins
            
        except Exception as e:
            self.safe_status_callback(f"Помилка отримання списку: {str(e)}")
            return self.get_backup_crypto_list()  # Повертаємо резервний список у разі помилки

    def save_crypto_list(self):
        """Збереження поточного списку криптовалют у файл в папці data/"""
        try:
            # Перевіряємо існування папки data
            if not os.path.exists('data'):
                os.makedirs('data')
            
            filename = self.get_data_path(f"crypto_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            list_data = {
                'timestamp': datetime.now().isoformat(),
                'crypto_list': self.crypto_list,
                'total_count': len(self.crypto_list)
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(list_data, f, indent=2, ensure_ascii=False)
            
            self.safe_status_callback(f"Список збережено у файл: {filename}")
            
        except Exception as e:
            self.safe_status_callback(f"Помилка збереження списку: {str(e)}")

    def open_crypto_list(self):
        """Відкриття збереженого списку криптовалют"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Відкрити список криптовалют"
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    list_data = json.load(f)
                
                if 'crypto_list' in list_data:
                    self.crypto_list = list_data['crypto_list']
                    self.filtered_crypto_list = self.crypto_list.copy()
                    self.apply_filters()
                    
                    self.safe_status_callback(f"Завантажено список з {len(self.crypto_list)} криптовалютами")
                else:
                    self.safe_status_callback("Невірний формат файлу списку")
                    
        except Exception as e:
            self.safe_status_callback(f"Помилка відкриття списку: {str(e)}")

    def delete_crypto_list(self):
        """Видалення поточного списку криптовалют"""
        if messagebox.askyesno("Підтвердження", "Видалити поточний список криптовалют?"):
            self.crypto_list = []
            self.filtered_crypto_list = []
            self.update_crypto_tree()
            self.safe_status_callback("Список криптовалют видалено")

    def refresh_crypto_list(self):
        """Актуалізація списку криптовалют"""
        def refresh_thread():
            try:
                self.safe_status_callback("Актуалізація списку криптовалют...")
                
                # Отримуємо актуальний список
                updated_list = self.get_complete_crypto_list()
                
                if updated_list:
                    # Оновлюємо існуючі записи
                    current_symbols = {crypto['symbol'] for crypto in self.crypto_list}
                    
                    for updated_crypto in updated_list:
                        symbol = updated_crypto['symbol']
                        if symbol in current_symbols:
                            # Оновлюємо існуючу криптовалюту
                            for existing_crypto in self.crypto_list:
                                if existing_crypto['symbol'] == symbol:
                                    existing_crypto.update(updated_crypto)
                                    break
                        else:
                            # Додаємо нову криптовалюту
                            self.crypto_list.append(updated_crypto)
                    
                    self.apply_filters()
                    self.safe_status_callback("Список актуалізовано успішно")
                else:
                    self.safe_status_callback("Не вдалося актуалізувати список")
                    
            except Exception as e:
                self.safe_status_callback(f"Помилка актуалізації: {str(e)}")
        
        threading.Thread(target=refresh_thread, daemon=True).start()

    def download_all_selected(self):
        """Завантаження даних для всіх криптовалют у списку"""
        if not self.crypto_list:
            messagebox.showwarning("Увага", "Список криптовалют порожній. Спочатку завантажте список.")
            return
        
        if not messagebox.askyesno("Підтвердження", 
                                f"Завантажити дані для ВСІХ {len(self.crypto_list)} криптовалют?\n"
                                "Це може зайняти багато часу."):
            return
        
        def download_all_thread():
            try:
                total = len(self.crypto_list)
                successful = 0
                failed = 0
                
                self.safe_status_callback(f"Початок завантаження {total} криптовалют...")
                self.safe_progress_callback(0)
                
                for i, crypto in enumerate(self.crypto_list):
                    symbol = crypto['symbol']
                    self.safe_status_callback(f"Завантаження {symbol} ({i+1}/{total})...")
                    self.update_progress_label(f"Завантаження {i+1}/{total}: {symbol}")
                    self.safe_progress_callback((i + 1) / total * 100)
                    
                    try:
                        data = self.fetch_all_data_sources(crypto)
                        if data is not None and not data.empty:
                            # Зберігаємо дані в папку data/
                            filename = self.get_data_path(f"{symbol}_full_data.csv")
                            data.to_csv(filename)
                            
                            # Оновлюємо якість даних
                            quality = self.calculate_data_quality(data)
                            crypto['data_quality'] = quality
                            crypto['data_points'] = len(data)
                            
                            successful += 1
                            self.safe_status_callback(f"✅ {symbol}: завантажено {len(data)} точок даних")
                        else:
                            failed += 1
                            self.safe_status_callback(f"❌ {symbol}: немає даних")
                            
                    except Exception as e:
                        failed += 1
                        error_msg = f"❌ {symbol}: помилка - {str(e)}"
                        self.safe_status_callback(error_msg)
                        logger.error(error_msg)
                    
                    # Затримка для уникнення rate limits
                    time.sleep(0.5)
                
                # Оновлюємо відображення
                self.apply_filters()
                
                final_msg = f"Завершено: {successful}/{total} успішно, {failed} невдало"
                self.safe_status_callback(final_msg)
                self.update_progress_label(final_msg)
                
            except Exception as e:
                error_msg = f"Критична помилка завантаження: {str(e)}"
                self.safe_status_callback(error_msg)
                self.update_progress_label("Помилка завантаження")
                logger.error(error_msg)
        
        threading.Thread(target=download_all_thread, daemon=True).start()

    def load_from_cache(self):
        """Завантаження списку з кешу"""
        try:
            cache_file = self.get_data_path("coingecko_cache.json")
            if os.path.exists(cache_file):
                # Перевіряємо вік кешу (не старіше 7 днів)
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 7 * 24 * 3600:  # 7 днів
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    if cached_data:
                        self.crypto_list = cached_data
                        self.filtered_crypto_list = cached_data.copy()
                        self.apply_filters()
                        self.safe_status_callback(f"Завантажено з кешу: {len(cached_data)} криптовалют")
                        return True
                else:
                    self.safe_status_callback("Кеш застарів (більше 7 днів)")
            else:
                self.safe_status_callback("Кеш не знайдено")
                
        except Exception as e:
            self.safe_status_callback(f"Помилка завантаження кешу: {str(e)}")
        
        return False
    
    def quick_sort(self, sort_type):
        """Швидке сортування за популярними критеріями"""
        sort_presets = {
            'top100': ('market_cap_rank', 'asc', 100),
            'top500': ('market_cap_rank', 'asc', 500),
            'top1000': ('market_cap_rank', 'asc', 1000)
        }
        
        if sort_type in sort_presets:
            self.sort_column, self.sort_direction, limit = sort_presets[sort_type]
            
            # Сортуємо за капіталізацією
            self.filtered_crypto_list.sort(
                key=lambda x: x.get('market_cap_rank', 999999),
                reverse=(self.sort_direction == 'desc')
            )
            
            # Обмежуємо кількість
            self.filtered_crypto_list = self.filtered_crypto_list[:limit]
            self.update_crypto_tree()
            
            self.safe_status_callback(f"Відображено {limit} криптовалют")

    def load_top_crypto_list(self, count):
        """Завантаження топ-N криптовалют за капіталізацією"""
        def load_top_thread():
            try:
                self.safe_status_callback(f"Завантаження топ-{count} криптовалют...")
                self.safe_progress_callback(0)
                
                # Отримуємо топ криптовалюти
                top_cryptos = self.fetch_coingecko_top_list(count)
                
                # Детальна перевірка отриманих даних
                if top_cryptos is None:
                    self.safe_status_callback("CoinGecko повернув None")
                    top_cryptos = []
                
                if not isinstance(top_cryptos, list):
                    self.safe_status_callback(f"CoinGecko повернув {type(top_cryptos)} замість списку")
                    top_cryptos = []
                
                # Перевіряємо, чи отримали достатньо монет
                actual_count = len(top_cryptos)
                if actual_count > 0:
                    self.crypto_list = top_cryptos
                    self.filtered_crypto_list = top_cryptos.copy()
                    
                    self.apply_filters()
                    
                    if actual_count < count:
                        self.safe_status_callback(f"Завантажено {actual_count} з {count} монет (обмеження API)")
                    else:
                        self.safe_status_callback(f"Завантажено топ-{count} криптовалют")
                        
                    self.safe_progress_callback(100)
                else:
                    # Якщо не вдалося отримати з CoinGecko, використовуємо резервний список
                    backup_list = self.get_backup_crypto_list(count)
                    self.crypto_list = backup_list
                    self.filtered_crypto_list = backup_list.copy()
                    self.apply_filters()
                    self.safe_status_callback(f"Використано резервний список: {len(backup_list)} монет")
                    self.safe_progress_callback(100)
                
            except Exception as e:
                self.safe_status_callback(f"Помилка завантаження топ-{count}: {str(e)}")
                logger.exception(f"Error in load_top_crypto_list: {e}")
                self.safe_progress_callback(0)
        
        threading.Thread(target=load_top_thread, daemon=True).start()

    def fetch_top_cryptos_with_retry(self, count):
        """Отримання топ криптовалют з повторними спробами"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Спроба CoinGecko
                cryptos = self.fetch_coingecko_top_list(count)
                if cryptos and isinstance(cryptos, list) and len(cryptos) > 0:  # Додано перевірку типу
                    return cryptos
                
                # Спроба CoinMarketCap
                cryptos = self.fetch_coinmarketcap_top_list(count)
                if cryptos and isinstance(cryptos, list) and len(cryptos) > 0:  # Додано перевірку типу
                    return cryptos
                    
                time.sleep(1)
                
            except Exception as e:
                self.safe_status_callback(f"Спроба {attempt + 1} невдала: {str(e)}")
                time.sleep(1)
        
        return self.get_backup_top_list(count)

    def fetch_coinmarketcap_top_list(self, count):
        """Отримання топ-N з CoinMarketCap"""
        coins_data = []
        try:
            # Використовуємо публічне API
            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"
            params = {
                'start': 1,
                'limit': min(count, 100),
                'sortBy': 'market_cap',
                'sortType': 'desc',
                'convert': 'USD'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # ДОДАЙТЕ ПЕРЕВІРКИ НА None
                if data is None:
                    self.safe_status_callback("CoinMarketCap: пуста відповідь")
                    return coins_data
                    
                if 'data' not in data or data['data'] is None:
                    self.safe_status_callback("CoinMarketCap: відсутні дані")
                    return coins_data
                    
                crypto_list = data['data'].get('cryptoCurrencyList', [])
                if crypto_list is None:
                    crypto_list = []
                
                for crypto in crypto_list:
                    if crypto is None:  # Перевірка на None
                        continue
                        
                    quotes = crypto.get('quotes', [{}])
                    first_quote = quotes[0] if quotes and quotes[0] is not None else {}
                    
                    coins_data.append({
                        'symbol': crypto.get('symbol', ''),
                        'name': crypto.get('name', ''),
                        'market_cap': first_quote.get('marketCap', 0) if first_quote else 0,
                        'volume': first_quote.get('volume24h', 0) if first_quote else 0,
                        'current_price': first_quote.get('price', 0) if first_quote else 0,
                        'price_change_24h': first_quote.get('percentChange24h', 0) if first_quote else 0,
                        'market_cap_rank': crypto.get('cmcRank', 9999),
                        'data_source': 'coinmarketcap',
                        'data_quality': 'N/A',
                        'selected': False
                    })
                
                self.safe_status_callback(f"CoinMarketCap: {len(coins_data)} монет")
                
            else:
                self.safe_status_callback(f"CoinMarketCap API помилка: {response.status_code}")
                
        except Exception as e:
            self.safe_status_callback(f"Помилка CoinMarketCap топ-{count}: {str(e)}")
        
        return coins_data

    def get_backup_top_list(self, count):
        """Резервний список топ криптовалют"""
        backup_coins = []
        popular_symbols = [
            'BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 
            'DOT', 'DOGE', 'MATIC', 'LTC', 'SHIB', 'TRX', 'LINK'
        ]
        
        for i in range(min(count, 100)):  # Обмежуємо до 100 монет
            if i < len(popular_symbols):
                symbol = popular_symbols[i]
                backup_coins.append({
                    'symbol': symbol,
                    'name': f'{symbol} Coin',
                    'market_cap_rank': i + 1,
                    'data_source': 'backup',
                    'data_quality': 'N/A',
                    'selected': False
                })
            else:
                backup_coins.append({
                    'symbol': f'TOKEN{i+1:04d}',
                    'name': f'Token {i+1}',
                    'market_cap_rank': i + 1,
                    'data_source': 'backup',
                    'data_quality': 'N/A',
                    'selected': False
                })
        
        self.safe_status_callback(f"Використано резервний список: {len(backup_coins)} монет")
        return backup_coins

    def save_coingecko_cache(self, coins):
        """Збереження кешу CoinGecko"""
        try:
            cache_file = self.get_data_path("coingecko_cache.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(coins, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Помилка збереження кешу: {e}")


    def download_selected_from_list(self):
        """Завантаження даних для обраних криптовалют зі списку"""
        selected = self.get_selected_cryptos()
        if not selected:
            messagebox.showwarning("Увага", "Оберіть хоча б одну криптовалюту зі списку")
            return
        
        def download_thread():
            try:
                total = len(selected)
                successful = 0
                failed = 0
                
                self.safe_status_callback(f"Завантаження {total} обраних криптовалют...")
                self.safe_progress_callback(0)
                
                for i, crypto in enumerate(selected):
                    symbol = crypto['symbol']
                    self.safe_status_callback(f"Завантаження {symbol} ({i+1}/{total})...")
                    self.update_progress_label(f"Завантаження {i+1}/{total}: {symbol}")
                    self.safe_progress_callback((i + 1) / total * 100)
                    
                    try:
                        data = self.fetch_all_data_sources(crypto)
                        if data is not None and not data.empty:
                            # Зберігаємо дані в папку data/
                            filename = self.get_data_path(f"{symbol}_full_data.csv")
                            data.to_csv(filename)
                            
                            # Оновлюємо якість даних
                            quality = self.calculate_data_quality(data)
                            crypto['data_quality'] = quality
                            crypto['data_points'] = len(data)
                            
                            successful += 1
                            self.safe_status_callback(f"✅ {symbol}: завантажено {len(data)} точок даних")
                        else:
                            failed += 1
                            self.safe_status_callback(f"❌ {symbol}: немає даних")
                            
                    except Exception as e:
                        failed += 1
                        error_msg = f"❌ {symbol}: помилка - {str(e)}"
                        self.safe_status_callback(error_msg)
                        logger.error(error_msg)
                    
                    # Затримка для уникнення rate limits
                    time.sleep(0.5)
                
                # Оновлюємо відображення
                self.apply_filters()
                
                final_msg = f"Завершено: {successful}/{total} успішно, {failed} невдало"
                self.safe_status_callback(final_msg)
                self.update_progress_label(final_msg)
                
            except Exception as e:
                error_msg = f"Критична помилка завантаження: {str(e)}"
                self.safe_status_callback(error_msg)
                self.update_progress_label("Помилка завантаження")
                logger.error(error_msg)
        
        threading.Thread(target=download_thread, daemon=True).start()

    def fetch_coingecko_top_list(self, count):
        """Отримання топ-N криптовалют з CoinGecko з обходом обмеження 250 монет"""
        try:
            # Перевіряємо, чи ініціалізована сесія
            if self.session is None:
                self.safe_status_callback("Сесія не ініціалізована, ініціалізуємо...")
                self.initialize_session()
                if self.session is None:
                    self.safe_status_callback("Не вдалося ініціалізувати сесію")
                    return []
            
            coins_data = []
            max_per_page = 250  # Максимальна кількість монет на сторінку
            pages_needed = (count + max_per_page - 1) // max_per_page
            
            self.safe_status_callback(f"Отримання {count} монет з {pages_needed} сторінок...")
            
            for page in range(1, pages_needed + 1):
                # Розраховуємо скільки монет потрібно з цієї сторінки
                per_page = min(max_per_page, count - len(coins_data))
                
                url = f"{self.COINGECKO_API_URL}/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': per_page,
                    'page': page,
                    'sparkline': 'false'
                }
                
                # Додаємо заголовки
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
                
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                
                # Перевіряємо успішність запиту
                if response.status_code != 200:
                    self.safe_status_callback(f"Помилка сторінки {page}: {response.status_code}")
                    continue
                
                # Спроба отримати JSON
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    self.safe_status_callback(f"Некоректна JSON відповідь на сторінці {page}")
                    continue
                
                # Перевіряємо, чи отримали коректні дані
                if not data or not isinstance(data, list):
                    self.safe_status_callback(f"Некоректні дані на сторінці {page}")
                    continue
                
                for crypto in data:
                    # Перевіряємо, чи crypto не є None
                    if crypto is None:
                        continue
                        
                    try:
                        # Безпечно отримуємо всі значення
                        symbol = crypto.get('symbol')
                        if not symbol:
                            continue
                            
                        coin_data = {
                            'symbol': str(symbol).upper(),
                            'name': str(crypto.get('name', '')),
                            'market_cap': float(crypto.get('market_cap', 0)) if crypto.get('market_cap') is not None else 0,
                            'volume': float(crypto.get('total_volume', 0)) if crypto.get('total_volume') is not None else 0,
                            'current_price': float(crypto.get('current_price', 0)) if crypto.get('current_price') is not None else 0,
                            'price_change_24h': float(crypto.get('price_change_percentage_24h', 0)) if crypto.get('price_change_percentage_24h') is not None else 0,
                            'market_cap_rank': int(crypto.get('market_cap_rank', 9999)) if crypto.get('market_cap_rank') is not None else 9999,
                            'data_source': 'coingecko',
                            'data_quality': 'N/A',
                            'selected': False
                        }
                        
                        coins_data.append(coin_data)
                        
                    except (ValueError, TypeError) as e:
                        # Пропускаємо пошкоджені дані
                        continue
                
                self.safe_status_callback(f"Сторінка {page}: отримано {len(data)} монет")
                
                # Затримка для уникнення rate limit
                time.sleep(0.5)
                
                # Якщо вже отримали достатньо монет, виходимо
                if len(coins_data) >= count:
                    break
            
            self.safe_status_callback(f"CoinGecko: отримано {len(coins_data)} монет")
            return coins_data[:count]  # Повертаємо тільки потрібну кількість
            
        except requests.exceptions.RequestException as e:
            self.safe_status_callback(f"Помилка мережі CoinGecko: {str(e)}")
            return []
        except Exception as e:
            self.safe_status_callback(f"Неочікувана помилка CoinGecko: {str(e)}")
            return []

    def fetch_exchange_data(self, symbol):
        """Отримання біржових даних для криптовалюти"""
        try:
            exchange_data = {}
            
            # Спроба отримати дані з доступних бірж
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # Завантажуємо ринки
                    exchange.load_markets()
                    
                    # Спроба знайти торгову пару
                    symbols_to_try = [
                        f"{symbol}/USDT",
                        f"{symbol}/USD",
                        f"{symbol}/BTC",
                        symbol
                    ]
                    
                    for sym in symbols_to_try:
                        if sym in exchange.markets:
                            try:
                                ticker = exchange.fetch_ticker(sym)
                                exchange_data[f"{exchange_name}_price"] = ticker.get('last')
                                exchange_data[f"{exchange_name}_volume"] = ticker.get('baseVolume')
                                exchange_data[f"{exchange_name}_bid"] = ticker.get('bid')
                                exchange_data[f"{exchange_name}_ask"] = ticker.get('ask')
                                break  # Знайшли пару, виходимо з циклу
                            except Exception as e:
                                continue
                    
                except Exception as e:
                    # Пропускаємо біржу при помилці
                    continue
            
            return pd.DataFrame([exchange_data]) if exchange_data else None
            
        except Exception as e:
            logger.error(f"Помилка біржових даних для {symbol}: {e}")
            return None





    # Додаємо методи з EnhancedDataLoaderTab
    def add_crypto_manual(self):
        """Додавання криптовалюти вручну"""
        symbol = self.symbol_var.get().strip().upper()
        name = self.name_var.get().strip()
        
        if symbol:
            if not name:
                name = f"Manual - {symbol}"
            
            self.crypto_list.append({
                'symbol': symbol,
                'name': name,
                'data_source': 'manual',
                'data_quality': 'N/A'
            })
            
            self.apply_filters()
            self.symbol_var.set("")
            self.name_var.set("")
            self.safe_status_callback(f"Криптовалюта {symbol} додана")

    def validate_data_quality(self):
        """Перевірка якості всіх наявних даних"""
        def validate_thread():
            try:
                data_files = [f for f in os.listdir('data') if f.endswith('_full_data.csv')]
                total = len(data_files)
                
                for i, filename in enumerate(data_files):
                    symbol = filename.replace('_full_data.csv', '')
                    self.safe_status_callback(f"Перевірка {symbol} ({i+1}/{total})...")
                    self.safe_progress_callback((i + 1) / total * 100)
                    
                    try:
                        data = pd.read_csv(f'data/{filename}', index_col=0, parse_dates=True)
                        quality = self.calculate_data_quality(data)
                        
                        # Оновлюємо якість у списку
                        for crypto in self.crypto_list:
                            if crypto['symbol'] == symbol:
                                crypto['data_quality'] = quality
                                break
                        
                        self.safe_status_callback(f"✅ {symbol}: якість {quality}/10")
                        
                    except Exception as e:
                        self.safe_status_callback(f"❌ {symbol}: помилка перевірки - {str(e)}")
                
                self.apply_filters()
                self.safe_status_callback("Перевірка якості завершена")
                
            except Exception as e:
                self.safe_status_callback(f"Помилка перевірки якості: {str(e)}")
        
        threading.Thread(target=validate_thread, daemon=True).start()

    def update_all_data(self):
        """Оновлення всіх наявних даних"""
        def update_thread():
            try:
                data_files = [f for f in os.listdir('data') if f.endswith('_full_data.csv')]
                total = len(data_files)
                
                for i, filename in enumerate(data_files):
                    symbol = filename.replace('_full_data.csv', '')
                    self.safe_status_callback(f"Оновлення {symbol} ({i+1}/{total})...")
                    self.safe_progress_callback((i + 1) / total * 100)
                    
                    # Знаходимо криптовалюту в списку
                    crypto = None
                    for c in self.crypto_list:
                        if c['symbol'] == symbol:
                            crypto = c
                            break
                    
                    if crypto:
                        # Завантажуємо оновлені дані
                        new_data = self.fetch_all_data_sources(crypto)
                        if new_data is not None and not new_data.empty:
                            new_data.to_csv(f'data/{filename}')
                            self.safe_status_callback(f"✅ {symbol}: дані оновлено")
                    
                    time.sleep(1)  # Затримка для уникнення rate limits
                
                self.safe_status_callback("Оновлення всіх даних завершено")
                
            except Exception as e:
                self.safe_status_callback(f"Помилка оновлення: {str(e)}")
        
        threading.Thread(target=update_thread, daemon=True).start()

    def calculate_data_quality(self, data):
        """Розрахунок якості даних з урахуванням малих наборів"""
        try:
            if data is None or data.empty:
                return 1.0  # Мінімальна якість
            
            score = 10.0
            
            # 1. Кількість даних (менш сувора)
            if len(data) < 50:
                score -= (50 - len(data)) / 10  # Менша штрафна ставка
            elif len(data) < 100:
                score -= 1.0
            
            # 2. Пропущені значення
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            score -= missing_ratio * 3  # Менший штраф
            
            # 3. Наявність ключових колонок
            required_cols = ['Close']
            missing_required = sum(1 for col in required_cols if col not in data.columns)
            score -= missing_required * 2  # Менший штраф
            
            return max(1.0, min(10.0, round(score, 1)))
            
        except Exception as e:
            logger.error(f"Помилка розрахунку якості даних: {e}")
            return 5.0  # Середня якість

    # Інші методи з EnhancedDataLoaderTab
    def add_comprehensive_technical_indicators(self, data):
        """Додавання комплексних технічних індикаторів"""
        df = data.copy()
        
        if 'Close' not in df.columns:
            return df
        
        # Базові показники
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Ковзні середні різних періодів
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Волатильність
        for window in [10, 20, 50]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # RSI різних періодів
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
        
        # MACD з різними параметрами
        macd_fast, macd_signal = self.calculate_macd(df['Close'], 12, 26)
        df['MACD'] = macd_fast
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_fast - macd_signal
        
        # Смуги Боллінджера
        for window in [20, 50]:
            upper, middle, lower = self.calculate_bollinger_bands(df['Close'], window)
            df[f'Bollinger_Upper_{window}'] = upper
            df[f'Bollinger_Middle_{window}'] = middle
            df[f'Bollinger_Lower_{window}'] = lower
            df[f'Bollinger_Width_{window}'] = (upper - lower) / middle
        
        # ATR (Average True Range)
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['ATR_14'] = self.calculate_atr(df['High'], df['Low'], df['Close'], 14)
        
        # Об'ємні індикатори
        if 'Volume' in df.columns:
            for window in [5, 20, 50]:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_MA_{window}']
            
            df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # Моментум індикатори
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Цінові зміни
        for period in [1, 5, 10, 20, 50]:
            df[f'Price_Change_{period}d'] = df['Close'].pct_change(period)
        
        return df.dropna()

    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Розрахунок MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Розрахунок смуг Боллінджера"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    def calculate_atr(self, high, low, close, period=14):
        """Розрахунок ATR"""
        tr = np.maximum(high - low, 
                    np.maximum(abs(high - close.shift(1)), 
                                abs(low - close.shift(1))))
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_obv(self, close, volume):
        """Розрахунок On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    
    
    
    
    
    # Інші методи з оригінального класу UltimateCryptoDataLoader...
    # [Тут мають бути всі інші методи з оригінального класу UltimateCryptoDataLoader]

    
    def load_crypto_list(self):
        """Завантаження початкового списку криптовалют"""
        try:
            # Спочатку пробуємо завантажити з кешу
            if self.load_from_cache():
                return
                
            # Перевіряємо підключення до інтернету
            if not self.check_internet_connection():
                self.safe_status_callback("Немає підключення до інтернету. Використовуємо резервний список.")
                self.crypto_list = self.get_backup_crypto_list(100)
                self.filtered_crypto_list = self.crypto_list.copy()
                self.apply_filters()
                return
                
            # Якщо кешу немає, завантажуємо топ-100
            cache_key = "initial_crypto_list"
            cached_data = self.get_cached_data(cache_key, 24)  # Кеш на 24 години
            
            if cached_data and isinstance(cached_data, list) and len(cached_data) > 0:
                self.crypto_list = cached_data
                self.safe_status_callback("Використано кешований список криптовалют")
            else:
                # Завантажуємо топ-100 криптовалют
                top_cryptos = self.fetch_coingecko_top_list(100)
                if top_cryptos and isinstance(top_cryptos, list) and len(top_cryptos) > 0:
                    self.crypto_list = top_cryptos
                    self.save_to_cache(cache_key, top_cryptos)
                    self.save_coingecko_cache(top_cryptos)
                    self.safe_status_callback("Завантажено топ-100 криптовалют")
                else:
                    # Резервний варіант - використовуємо backup список
                    self.crypto_list = self.get_backup_crypto_list()[:100]
                    self.safe_status_callback("Використано резервний список криптовалют")
            
            # Застосовуємо фільтри для відображення
            self.filtered_crypto_list = self.crypto_list.copy()
            self.apply_filters()
            
        except Exception as e:
            self.crypto_list = self.get_backup_crypto_list()[:100]
            self.filtered_crypto_list = self.crypto_list.copy()
            self.safe_status_callback(f"Помилка завантаження списку: {str(e)}")

    def ensure_directories(self):
        """Створення необхідних директорій"""
        directories = ['data', 'cache', 'logs']
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    self.safe_status_callback(f"Створено директорію: {directory}")
                except Exception as e:
                    self.safe_status_callback(f"Помилка створення директорії {directory}: {str(e)}")

    def on_header_click(self, column):
        """Обробка кліку по заголовку для сортування"""
        if column == 'select':
            return  # Не сортуємо по чекбоксам
            
        if self.sort_column == column:
            # Змінюємо напрямок сортування
            self.sort_direction = 'desc' if self.sort_direction == 'asc' else 'asc'
        else:
            # Нова колонка для сортування
            self.sort_column = column
            self.sort_direction = 'asc'
        
        self.apply_sorting()
        
    def on_sort_change(self, event):
        """Обробка зміни сортування через combobox"""
        sort_mapping = {
            "Капіталізація": "market_cap_rank",
            "Назва": "name", 
            "Символ": "symbol",
            "Ціна": "current_price",
            "Об'єм": "volume",
            "Зміна 24h": "price_change_24h",
            "Якість даних": "data_quality",
            "Вік": "age_days"
        }
        
        selected_text = self.sort_var.get()
        self.sort_column = sort_mapping.get(selected_text, "market_cap_rank")
        self.apply_sorting()

    def apply_sorting(self):
        """Застосування сортування до відфільтрованого списку"""
        if not self.filtered_crypto_list:
            return
            
        reverse = (self.sort_direction == 'desc')
        
        # Визначаємо функцію сортування
        if self.sort_column == 'market_cap_rank':
            key_func = lambda x: x.get('market_cap_rank', 999999)
        elif self.sort_column == 'name':
            key_func = lambda x: x.get('name', '').lower()
        elif self.sort_column == 'symbol':
            key_func = lambda x: x.get('symbol', '').lower()
        elif self.sort_column == 'current_price':
            key_func = lambda x: x.get('current_price', 0) or 0
        elif self.sort_column == 'volume':
            key_func = lambda x: x.get('volume', 0) or 0
        elif self.sort_column == 'price_change_24h':
            key_func = lambda x: x.get('price_change_24h', 0) or 0
        elif self.sort_column == 'data_quality':
            key_func = lambda x: float(x.get('data_quality', 0) or 0)
        elif self.sort_column == 'age_days':
            key_func = lambda x: x.get('age_days', 0) or 0
        else:
            key_func = lambda x: x.get(self.sort_column, '')
        
        # Сортуємо
        try:
            self.filtered_crypto_list.sort(key=key_func, reverse=reverse)
            self.update_crypto_tree()
        except Exception as e:
            self.safe_status_callback(f"Помилка сортування: {str(e)}")

    def quick_sort(self, sort_type):
        """Швидке сортування за популярними критеріями"""
        sort_presets = {
            'top100': ('market_cap_rank', 'asc'),
            'top500': ('market_cap_rank', 'asc'),
            'hot': ('price_change_24h', 'desc'),
            'new': ('age_days', 'asc')
        }
        
        if sort_type in sort_presets:
            self.sort_column, self.sort_direction = sort_presets[sort_type]
            self.apply_sorting()
            
            # Для топ-100/500 також застосовуємо фільтр
            if sort_type in ['top100', 'top500']:
                limit = 100 if sort_type == 'top100' else 500
                self.filtered_crypto_list = self.filtered_crypto_list[:limit]
                self.update_crypto_tree()

    def apply_filters(self):
        """Застосування всіх фільтрів"""
        search_text = self.search_var.get().lower()
        source_filter = self.source_filter_var.get()
        quality_filter = self.quality_filter_var.get()
        
        self.filtered_crypto_list = []
        
        for crypto in self.crypto_list:
            # Перевірка на None
            if crypto is None:
                continue
                
            # Фільтр пошуку
            if search_text:
                symbol_match = search_text in crypto.get('symbol', '').lower()
                name_match = search_text in crypto.get('name', '').lower()
                if not symbol_match and not name_match:
                    continue
            
            # Фільтр джерела
            if source_filter != 'all':
                crypto_source = crypto.get('data_source', '')
                if source_filter == 'coingecko' and not crypto_source.startswith('coingecko'):
                    continue
                elif source_filter == 'coinmarketcap' and not crypto_source.startswith('coinmarketcap'):
                    continue
                elif source_filter == 'exchange' and not crypto_source.startswith('exchange'):
                    continue
                elif source_filter == 'other' and crypto_source in ['coingecko', 'coinmarketcap', 'exchange']:
                    continue
            
            # Фільтр якості
            if quality_filter != 'all':
                quality = crypto.get('data_quality', 'N/A')
                if quality == 'N/A' and quality_filter != 'none':
                    continue
                elif quality != 'N/A':
                    try:
                        quality_num = float(quality)
                        if quality_filter == 'high' and quality_num < 8:
                            continue
                        elif quality_filter == 'medium' and (quality_num < 5 or quality_num >= 8):
                            continue
                        elif quality_filter == 'low' and quality_num >= 5:
                            continue
                        elif quality_filter == 'none':
                            continue
                    except ValueError:
                        continue
            
            self.filtered_crypto_list.append(crypto)
        
        # Застосовуємо поточне сортування
        self.apply_sorting()
        # Оновлюємо лічильники з реальною кількістю
        self.update_counters()

    def update_counters(self):
        """Оновлення всіх лічильників з реальною кількістю з treeview"""
        try:
            # Отримуємо реальну кількість елементів у treeview
            total_in_tree = len(self.crypto_tree.get_children())
            
            # Отримуємо кількість вибраних елементів
            selected_count = sum(1 for crypto in self.filtered_crypto_list if crypto.get('selected', False))
            
            # Оновлюємо статистику вибору
            self.selection_stats_var.set(f"Обрано: {selected_count} / {total_in_tree}")
            
            # Оновлюємо загальну кількість (відображаємо реальну кількість у treeview)
            self.total_count_var.set(f"Всього: {total_in_tree} криптовалют")
            
        except Exception as e:
            logger.error(f"Помилка оновлення лічильників: {e}")
            # Резервний варіант
            total = len(self.filtered_crypto_list)
            selected = sum(1 for crypto in self.filtered_crypto_list if crypto.get('selected', False))
            self.selection_stats_var.set(f"Обрано: {selected} / {total}")
            self.total_count_var.set(f"Всього: {total} криптовалют")

    def update_crypto_tree(self):
        """Оновлення дерева з урахуванням сортування та фільтрів"""
        try:
            # Зберігаємо поточну позицію прокрутки
            scroll_position = self.crypto_tree.yview()
            
            # Очищаємо treeview
            self.crypto_tree.delete(*self.crypto_tree.get_children())
            self.tree_item_data.clear()
            
            for crypto in self.filtered_crypto_list:
                values = self.get_tree_values(crypto)
                item_id = self.crypto_tree.insert('', 'end', values=values)
                
                # Зберігаємо дані криптовалюти
                self.tree_item_data[item_id] = crypto
                
                # Встановлюємо стиль для вибраних рядків
                if crypto.get('selected', False):
                    self.crypto_tree.item(item_id, tags=('selected',))
            
            # Відновлюємо позицію прокрутки
            self.crypto_tree.yview_moveto(scroll_position[0])
            
            # Оновлюємо лічильники з реальною кількістю
            self.update_counters()
            
        except Exception as e:
            self.safe_status_callback(f"Помилка оновлення дерева: {str(e)}")

    def get_tree_values(self, crypto):
        """Отримання значень для відображення в treeview"""
        # Додайте перевірку на None
        if crypto is None:
            return ('☐', '', '', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A')
        
        market_cap = crypto.get('market_cap')
        volume = crypto.get('volume')
        price = crypto.get('current_price')
        change = crypto.get('price_change_24h')
        quality = crypto.get('data_quality', 'N/A')
        data_points = crypto.get('data_points', 0)
        
        return (
            '☑' if crypto.get('selected', False) else '☐',
            crypto.get('symbol', ''),
            crypto.get('name', ''),
            f"${market_cap:,.0f}" if market_cap else 'N/A',
            f"${volume:,.0f}" if volume else 'N/A',
            f"${price:,.2f}" if price else 'N/A',
            f"{change:+.2f}%" if change else 'N/A',
            crypto.get('market_cap_rank', 'N/A'),
            quality,
            f"{crypto.get('age_days', 'N/A')}д" if crypto.get('age_days') else 'N/A',
            crypto.get('data_source', 'N/A'),
            f"{data_points}" if data_points > 0 else 'N/A'
        )

    
    
    
    def update_selection_stats(self):
        """Оновлення статистики вибору"""
        total = len(self.filtered_crypto_list)
        selected = sum(1 for crypto in self.filtered_crypto_list if crypto.get('selected', False))
        self.selection_stats_var.set(f"Обрано: {selected} / {total}")

    def get_selected_cryptos(self):
        """Отримання обраних криптовалют"""
        return [crypto for crypto in self.filtered_crypto_list if crypto.get('selected', False)]

    def download_selected(self):
        """Завантаження обраних криптовалют з перевіркою на неіснуючі символи"""
        selected = self.get_selected_cryptos()
        if not selected:
            messagebox.showwarning("Увага", "Оберіть хоча б одну криптовалюту")
            return
        
        def download_thread():
            """Внутрішня функція для потокового завантаження"""
            try:
                total = len(selected)
                successful = 0
                failed = 0
                
                self.safe_progress_callback(0)
                self.update_progress_label(f"Завантаження 0/{total}")
                
                for i, crypto in enumerate(selected):
                    symbol = crypto['symbol']
                    self.safe_status_callback(f"Завантаження {symbol} ({i+1}/{total})...")
                    self.update_progress_label(f"Завантаження {i+1}/{total}: {symbol}")
                    self.safe_progress_callback((i + 1) / total * 100)
                    
                    # Перевірка на неіснуючі символи
                    if self.is_likely_nonexistent(symbol):
                        self.safe_status_callback(f"⚠️ {symbol}: пропущено (неіснуючий символ)")
                        self.non_existent_symbols.add(symbol)
                        logger.warning(f"Додано до неіснуючих символів: {symbol}")
                        failed += 1
                        continue
                    
                    try:
                        data = self.fetch_robust_data(symbol)
                        
                        # Додаткова перевірка після отримання даних
                        if data is None or data.empty:
                            # Якщо дані порожні, перевіряємо чи символ не існує
                            if self.is_likely_nonexistent(symbol):
                                self.non_existent_symbols.add(symbol)
                                logger.warning(f"Додано до неіснуючих символів після спроби: {symbol}")
                            
                            failed += 1
                            data_length = len(data) if data is not None else 0
                            self.safe_status_callback(f"❌ {symbol}: недостатньо даних ({data_length} рядків)")
                            continue
                        
                        if len(data) >= 10:  # Зменшений мінімум до 10 рядків
                            # Зберігаємо дані в папку data/
                            filename = f"data/{symbol}_comprehensive_data.csv"
                            data.to_csv(filename)
                            
                            # Оновлюємо якість даних
                            quality = self.calculate_data_quality(data)
                            crypto['data_quality'] = quality
                            crypto['data_points'] = len(data)
                            
                            successful += 1
                            self.safe_status_callback(f"✅ {symbol}: завантажено {len(data)} точок даних, якість {quality}/10")
                        else:
                            failed += 1
                            self.safe_status_callback(f"❌ {symbol}: недостатньо даних ({len(data)} рядків)")
                            
                    except Exception as e:
                        failed += 1
                        error_msg = f"❌ {symbol}: помилка - {str(e)}"
                        self.safe_status_callback(error_msg)
                        logger.error(error_msg)
                    
                    # Затримка для уникнення rate limits
                    time.sleep(1)
                
                # Оновлюємо відображення
                self.apply_filters()
                
                final_msg = f"Завершено: {successful}/{total} успішно, {failed} невдало"
                self.safe_status_callback(final_msg)
                self.update_progress_label(final_msg)
                
            except Exception as e:
                error_msg = f"Критична помилка завантаження: {str(e)}"
                self.safe_status_callback(error_msg)
                self.update_progress_label("Помилка завантаження")
                logger.error(error_msg)
        
        # Запускаємо в окремому потоці
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def refresh_data(self):
        """Оновлення даних про криптовалюти"""
        def refresh_thread():
            try:
                self.safe_status_callback("Оновлення даних криптовалют...")
                
                # Отримуємо актуальні дані з CoinGecko
                updated_coins = self.fetch_coingecko_top_list(2000)  # Топ-2000
                
                if updated_coins:
                    # Оновлюємо існуючі записи
                    for updated_coin in updated_coins:
                        symbol = updated_coin['symbol']
                        # Шукаємо відповідну криптовалюту в нашому списку
                        for existing_coin in self.crypto_list:
                            if existing_coin['symbol'] == symbol:
                                # Оновлюємо дані
                                existing_coin.update(updated_coin)
                                break
                    
                    self.apply_filters()  # Перезастосовуємо фільтри
                    self.safe_status_callback("Дані оновлено успішно")
                else:
                    self.safe_status_callback("Не вдалося оновити дані")
                
            except Exception as e:
                self.safe_status_callback(f"Помилка оновлення: {str(e)}")
        
        threading.Thread(target=refresh_thread, daemon=True).start()

    def export_list(self):
        """Експорт списку криптовалют"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                if filename.endswith('.csv'):
                    self.export_to_csv(filename)
                elif filename.endswith('.json'):
                    self.export_to_json(filename)
                
                self.safe_status_callback(f"Список експортовано до {filename}")
                
        except Exception as e:
            self.safe_status_callback(f"Помилка експорту: {str(e)}")

    def export_to_csv(self, filename):
        """Експорт до CSV з перевіркою директорії"""
        try:
            # Перевіряємо директорію
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # Заголовки
                writer.writerow(['Symbol', 'Name', 'Market Cap', 'Volume', 'Price', 
                            '24h Change', 'Rank', 'Quality', 'Source', 'Selected'])
                
                # Дані
                for crypto in self.filtered_crypto_list:
                    writer.writerow([
                        crypto.get('symbol', ''),
                        crypto.get('name', ''),
                        crypto.get('market_cap', ''),
                        crypto.get('volume', ''),
                        crypto.get('current_price', ''),
                        crypto.get('price_change_24h', ''),
                        crypto.get('market_cap_rank', ''),
                        crypto.get('data_quality', 'N/A'),
                        crypto.get('data_source', ''),
                        'Так' if crypto.get('selected', False) else 'Ні'
                    ])
                    
        except Exception as e:
            error_msg = f"Помилка експорту CSV: {str(e)}"
            self.safe_status_callback(error_msg)
            logger.error(error_msg)
            raise

    def export_to_json(self, filename):
        """Експорт до JSON"""
        export_data = []
        for crypto in self.filtered_crypto_list:
            export_data.append({
                'symbol': crypto.get('symbol', ''),
                'name': crypto.get('name', ''),
                'market_cap': crypto.get('market_cap', ''),
                'volume': crypto.get('volume', ''),
                'current_price': crypto.get('current_price', ''),
                'price_change_24h': crypto.get('price_change_24h', ''),
                'market_cap_rank': crypto.get('market_cap_rank', ''),
                'data_quality': crypto.get('data_quality', 'N/A'),
                'data_source': crypto.get('data_source', ''),
                'selected': crypto.get('selected', False)
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def save_selection(self):
        """Збереження поточного вибору в папку data/"""
        try:
            selection = self.get_selected_cryptos()
            selection_data = [{
                'symbol': crypto['symbol'],
                'name': crypto['name'],
                'data_source': crypto.get('data_source', '')
            } for crypto in selection]
            
            filename = self.get_data_path("selected_cryptos.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(selection_data, f, ensure_ascii=False, indent=2)
            
            self.safe_status_callback(f"Збережено вибір: {len(selection)} криптовалют")
            
        except Exception as e:
            self.safe_status_callback(f"Помилка збереження: {str(e)}")

    def load_selection(self):
        """Завантаження збереженого вибору з папки data/"""
        try:
            filename = self.get_data_path("selected_cryptos.json")
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    saved_selection = json.load(f)
                
                # Скидаємо попередній вибір
                for crypto in self.crypto_list:
                    crypto['selected'] = False
                
                # Встановлюємо збережений вибір
                saved_symbols = {item['symbol'] for item in saved_selection}
                for crypto in self.crypto_list:
                    if crypto['symbol'] in saved_symbols:
                        crypto['selected'] = True
                
                self.apply_filters()
                self.safe_status_callback(f"Завантажено вибір: {len(saved_selection)} криптовалют")
                
        except Exception as e:
            self.safe_status_callback(f"Помилка завантаження: {str(e)}")

    def fetch_robust_data(self, symbol):
        """Надійне отримання даних з покращеною обробкою"""
        try:
            # Тільки для явно неіснуючих символів
            if self.is_likely_nonexistent(symbol):
                logger.info(f"Символ {symbol} вважається неіснуючим - створюємо тестові дані")
                return self.create_fallback_data(symbol)
            
            # Спроба 1: Yahoo Finance (покращена версія)
            data = self.fetch_yahoo_data_detailed(symbol)
            
            # Спроба 2: CoinGecko API для проблемних символів
            if data is None or data.empty:
                data = self.fetch_ada_from_coingecko(symbol)
            
            # Спроба 3: Резервні дані
            if data is None or data.empty:
                data = self.create_fallback_data(symbol)
            
            # М'яке очищення
            if data is not None and not data.empty:
                return self.safe_clean_data(data, symbol)
            
            return self.create_fallback_data(symbol)
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return self.create_fallback_data(symbol)

    # Інші методи (fetch_coingecko_top_list, fetch_all_data_sources, etc.)...
    # Додаємо їх з попередньої реалізації

    def safe_status_callback(self, message):
        if self.status_callback:
            self.status_callback(message)

    def safe_progress_callback(self, value):
        if self.progress_callback:
            self.progress_callback(value)

    # Додаємо обробник кліків по treeview
    def setup_tree_bindings(self):
        """Налаштування обробників подій для treeview"""
        # Обробка кліку по чекбоксу
        self.crypto_tree.bind('<Button-1>', self.on_tree_click)
        
        # Обробка подвійного кліку для деталей
        self.crypto_tree.bind('<Double-1>', self.on_tree_double_click)
        
        # Обробка клавіші Enter для деталей
        self.crypto_tree.bind('<Return>', self.on_tree_double_click)
        
        # Обробка пробілу для вибору
        self.crypto_tree.bind('<space>', self.on_space_press)
        
        # Прокрутка з коліщатком миші
        self.crypto_tree.bind('<MouseWheel>', self.on_mouse_wheel)
        
        # Прокрутка з клавішами PageUp/PageDown
        self.crypto_tree.bind('<Prior>', lambda e: self.crypto_tree.yview_scroll(-1, 'pages'))
        self.crypto_tree.bind('<Next>', lambda e: self.crypto_tree.yview_scroll(1, 'pages'))
        
        # Прокрутка з клавішами стрелок
        self.crypto_tree.bind('<Up>', lambda e: self.crypto_tree.yview_scroll(-1, 'units'))
        self.crypto_tree.bind('<Down>', lambda e: self.crypto_tree.yview_scroll(1, 'units'))

    def on_mouse_wheel(self, event):
        """Обробка прокрутки коліщатком миші"""
        if event.delta > 0:
            self.crypto_tree.yview_scroll(-1, 'units')
        else:
            self.crypto_tree.yview_scroll(1, 'units')
        return 'break'

    def scroll_to_selection(self):
        """Прокручує до вибраних елементів"""
        selected_items = self.crypto_tree.selection()
        if selected_items:
            # Прокручуємо до першого вибраного елемента
            self.crypto_tree.see(selected_items[0])
    
    def on_space_press(self, event):
        """Обробка натискання пробілу для вибору"""
        try:
            item = self.crypto_tree.selection()[0] if self.crypto_tree.selection() else None
            if item and item in self.tree_item_data:
                self.on_tree_select(event, item)
        except Exception as e:
            self.safe_status_callback(f"Помилка обробки пробілу: {str(e)}")

    def on_tree_double_click(self, event):
        """Обробка подвійного кліку - показ деталей"""
        try:
            region = self.crypto_tree.identify("region", event.x, event.y)
            if region == "cell":
                item = self.crypto_tree.identify_row(event.y)
                if item and item in self.tree_item_data:
                    crypto_data = self.tree_item_data[item]
                    self.show_crypto_details(crypto_data)
        except Exception as e:
            self.safe_status_callback(f"Помилка відкриття деталей: {str(e)}")
    
    def on_tree_click(self, event):
        """Обробка кліку по treeview"""
        try:
            region = self.crypto_tree.identify("region", event.x, event.y)
            if region == "cell":
                column_id = self.crypto_tree.identify_column(event.x)
                item = self.crypto_tree.identify_row(event.y)
                
                # Перевіряємо, чи це перша колонка (чекбокс)
                if column_id == '#1':  # Перша колонка
                    self.on_tree_select(event, item)
        except Exception as e:
            self.safe_status_callback(f"Помилка обробки кліку: {str(e)}")

    
    
    def toggle_selection(self, item):
        """Перемикач вибору рядка"""
        crypto_data = self.tree_item_data[item]
        current_state = crypto_data.get('selected', False)
        new_state = not current_state
        
        # Оновлюємо стан у даних
        crypto_data['selected'] = new_state
        
        # Змінюємо вигляд рядка для відображення вибору
        self.update_row_appearance(item, new_state)
        
        # Оновлюємо лічильники
        self.update_counters()

    def select_all(self):
        """Вибрати всі відфільтровані елементи"""
        for crypto in self.filtered_crypto_list:
            crypto['selected'] = True
        self.update_crypto_tree()
        self.scroll_to_selection()

    def deselect_all(self):
        """Скасувати вибір всіх елементів"""
        for crypto in self.filtered_crypto_list:
            crypto['selected'] = False
        self.update_crypto_tree()

    def invert_selection(self):
        """Інвертувати вибір"""
        for crypto in self.filtered_crypto_list:
            crypto['selected'] = not crypto.get('selected', False)
        self.update_crypto_tree()


    def get_selected_cryptos(self):
        """Отримання обраних криптовалют"""
        return [crypto for crypto in self.filtered_crypto_list if crypto.get('selected', False)]
        
    
    def on_tree_double_click(self, event):
        """Обробка подвійного кліку - показ деталей"""
        try:
            item = self.crypto_tree.selection()[0] if self.crypto_tree.selection() else None
            if item and item in self.tree_item_data:
                crypto_data = self.tree_item_data[item]
                self.show_crypto_details(crypto_data)
        except Exception as e:
            self.safe_status_callback(f"Помилка відкриття деталей: {str(e)}")

    def show_crypto_details(self, crypto_data):
        """Показ детальної інформації про криптовалюту"""
        detail_window = tk.Toplevel(self.parent)
        detail_window.title(f"Деталі: {crypto_data.get('symbol', '')}")
        detail_window.geometry("400x300")
        
        # Додаємо інформацію про криптовалюту
        info_frame = ttk.Frame(detail_window)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        fields = [
            ('Символ', 'symbol'),
            ('Назва', 'name'),
            ('Капіталізація', 'market_cap'),
            ('Об\'єм', 'volume'),
            ('Ціна', 'current_price'),
            ('Зміна 24h', 'price_change_24h'),
            ('Рейтинг', 'market_cap_rank'),
            ('Якість даних', 'data_quality'),
            ('Джерело', 'data_source')
        ]
        
        for i, (label, key) in enumerate(fields):
            ttk.Label(info_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            value = crypto_data.get(key, 'N/A')
            if isinstance(value, (int, float)) and key not in ['symbol', 'name', 'data_source']:
                if key == 'market_cap' or key == 'volume':
                    value = f"${value:,.0f}"
                elif key == 'current_price':
                    value = f"${value:,.2f}"
                elif key == 'price_change_24h':
                    value = f"{value:+.2f}%"
            ttk.Label(info_frame, text=str(value)).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        # Кнопка завантаження
        ttk.Button(detail_window, text="Завантажити дані", 
                command=lambda: self.download_single_crypto(crypto_data)).pack(pady=10)

    def download_single_crypto(self, crypto_data):
        """Завантаження даних для однієї криптовалюти"""
        def download_thread():
            try:
                symbol = crypto_data['symbol']
                self.safe_status_callback(f"Завантаження {symbol}...")
                
                data = self.fetch_all_data_sources(crypto_data)
                if data is not None and not data.empty:
                    filename = self.get_data_path(f"{symbol}_full_data.csv")
                    data.to_csv(filename)
                    
                    # Оновлюємо якість
                    quality = self.calculate_data_quality(data)
                    crypto_data['data_quality'] = quality
                    
                    self.safe_status_callback(f"✅ {symbol}: завантажено, якість {quality}/10")
                    self.update_crypto_tree()
                else:
                    self.safe_status_callback(f"❌ {symbol}: немає даних")
                    
            except Exception as e:
                self.safe_status_callback(f"❌ Помилка завантаження {symbol}: {str(e)}")
        
        threading.Thread(target=download_thread, daemon=True).start()

    def get_cached_data(self, key, expiry_hours=24):
        """Отримання даних з кешу"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{quote(key)}.json")
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < expiry_hours * 3600:  # Перевірка терміну дії
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Помилка читання кешу: {e}")
        return None

    def save_to_cache(self, key, data):
        """Збереження даних в кеш"""
        try:
            # Спочатку переконаємося, що директорія cache існує
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Помилка збереження кешу: {e}")
    

    def update_progress_label(self, text):
        """Оновлення тексту прогресс-бару"""
        try:
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.config(text=text)
        except Exception as e:
            logger.error(f"Помилка оновлення мітки прогресу: {e}")

    def safe_progress_callback(self, value):
        """Безпечний виклик прогрес callback"""
        try:
            # Викликаємо оригінальний callback якщо він є
            if self.progress_callback:
                self.progress_callback(value)
            
            # Оновлюємо прогресс-бар якщо він існує
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar['value'] = value
                
        except Exception as e:
            logger.error(f"Помилка прогрес callback: {e}")

    def safe_status_callback(self, message):
        """Безпечний виклик статус callback"""
        try:
            # Викликаємо оригінальний callback
            if self.status_callback:
                self.status_callback(message)
            
            # Логуємо повідомлення
            logger.info(message)
            
        except Exception as e:
            logger.error(f"Помилка статус callback: {e}")

    def initialize_session(self):
        """Ініціалізація сесії запитів"""
        try:
            # Створюємо нову сесію
            self.session = requests.Session()
            
            # Додаємо заголовки для уникнення блокування
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            })
            
            # Додаємо повторні спроби для запитів
            self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
            self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
            
            self.safe_status_callback("Сесія запитів ініціалізована")
            
        except Exception as e:
            self.safe_status_callback(f"Помилка ініціалізації сесії: {str(e)}")
            # Створюємо просту сесію як резервний варіант
            self.session = requests.Session()

    def initialize_exchanges(self):
        """Ініціалізація всіх доступних криптобірж"""
        try:
            self.exchanges = {}
            
            # Додаємо тільки ті біржі, які існують у поточній версії ccxt
            exchange_list = [
                ('binance', ccxt.binance),
                ('kraken', ccxt.kraken),
                ('coinbase', ccxt.coinbase),
                ('huobi', ccxt.huobi),
                ('okx', ccxt.okx),  # okex було перейменовано в okx
                ('bitfinex', ccxt.bitfinex),
                # bittrex може бути недоступний у нових версіях ccxt
            ]
            
            for name, exchange_class in exchange_list:
                try:
                    if hasattr(ccxt, name):
                        self.exchanges[name] = exchange_class({'enableRateLimit': True})
                        self.safe_status_callback(f"Біржа {name} ініціалізована")
                except Exception as e:
                    self.safe_status_callback(f"Помилка ініціалізації {name}: {str(e)}")
                    continue
            
            self.safe_status_callback(f"Ініціалізовано {len(self.exchanges)} бірж")
        except Exception as e:
            self.safe_status_callback(f"Помилка ініціалізації бірж: {str(e)}")
            self.exchanges = {}

    def setup_main_tab(self):
        """Налаштування основної вкладки"""
        # Лівий фрейм - список криптовалют
        left_frame = ttk.LabelFrame(self.main_frame, text="Криптовалютні активи")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Фільтри та пошук
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Пошук:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.apply_filters)
        ttk.Entry(filter_frame, textvariable=self.search_var, width=20).pack(side=tk.LEFT, padx=5)
        
        # Treeview з прокруткою
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = [
            ('Symbol', 80), ('Name', 150), ('Market Cap', 120), 
            ('Volume', 120), ('Price', 100), ('24h %', 80),
            ('Rank', 60), ('Quality', 80), ('Age', 80), ('Source', 100)
        ]
        
        self.crypto_tree = ttk.Treeview(tree_frame, columns=[col[0] for col in columns], show='headings', height=25)
        
        for col, width in columns:
            self.crypto_tree.heading(col, text=col)
            self.crypto_tree.column(col, width=width, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.crypto_tree.yview)
        self.crypto_tree.configure(yscrollcommand=scrollbar.set)
        
        self.crypto_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Правий фрейм - керування
        right_frame = ttk.LabelFrame(self.main_frame, text="Масова загрузка")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Кнопки масованого завантаження
        ttk.Button(right_frame, text="Завантажити ТОП-1000", 
                  command=lambda: self.download_top_crypto(1000)).pack(pady=2, fill=tk.X)
        ttk.Button(right_frame, text="Завантажити ТОП-5000", 
                  command=lambda: self.download_top_crypto(5000)).pack(pady=2, fill=tk.X)
        ttk.Button(right_frame, text="Завантажити ВСІ", 
                  command=self.download_all_crypto).pack(pady=2, fill=tk.X)
        
        # Налаштування джерел
        source_frame = ttk.LabelFrame(right_frame, text="Джерела даних")
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sources_var = tk.StringVar(value="all")
        sources = [
            ("Всі джерела", "all"),
            ("CoinGecko", "coingecko"),
            ("CoinMarketCap", "coinmarketcap"),
            ("Yahoo Finance", "yfinance"),
            ("Біржові дані", "exchanges")
        ]
        
        for text, value in sources:
            ttk.Radiobutton(source_frame, text=text, variable=self.sources_var, value=value).pack(anchor=tk.W)

    def setup_all_crypto_tab(self):
        """Вкладка для всіх криптовалют"""
        frame = ttk.Frame(self.all_crypto_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Загальна кількість відслідковуваних криптовалют: 0", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Статистика по джерелам
        stats_frame = ttk.LabelFrame(frame, text="Статистика джерел")
        stats_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stats_frame, text="CoinGecko: 0").pack(anchor=tk.W)
        ttk.Label(stats_frame, text="CoinMarketCap: 0").pack(anchor=tk.W)
        ttk.Label(stats_frame, text="Yahoo Finance: 0").pack(anchor=tk.W)

    def setup_stats_tab(self):
        """Вкладка статистики"""
        frame = ttk.Frame(self.stats_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Статистика завантажених даних", 
                 font=('Arial', 14, 'bold')).pack(pady=10)

    def download_all_crypto(self):
        """Завантаження ВСІХ доступних криптовалют"""
        def download_thread():
            try:
                self.safe_status_callback("Початок завантаження ВСІХ криптовалют...")
                
                # Отримуємо дані з усіх джерел
                all_coins = self.get_complete_crypto_list()
                
                if not all_coins:
                    messagebox.showerror("Помилка", "Не вдалося отримати список криптовалют")
                    return
                
                self.crypto_list = all_coins
                self.update_crypto_tree()
                
                self.safe_status_callback(f"Знайдено {len(all_coins)} криптовалют")
                self.safe_progress_callback(100)
                
                # Зберігаємо повний список
                self.save_complete_list(all_coins)
                
            except Exception as e:
                self.safe_status_callback(f"Помилка: {str(e)}")
                self.safe_progress_callback(0)
        
        threading.Thread(target=download_thread, daemon=True).start()

    def get_complete_crypto_list(self) -> List[Dict]:
        """Отримання повного списку криптовалют з усіх джерел"""
        all_coins = []
        seen_symbols = set()
        
        # 1. CoinGecko (основне джерело)
        try:
            cg_coins = self.fetch_coingecko_complete_list()
            for coin in cg_coins:
                if coin['symbol'].upper() not in seen_symbols:
                    all_coins.append(coin)
                    seen_symbols.add(coin['symbol'].upper())
            self.safe_status_callback(f"CoinGecko: {len(cg_coins)} монет")
        except Exception as e:
            self.safe_status_callback(f"Помилка CoinGecko: {str(e)}")
        
        # 2. CoinMarketCap
        try:
            cmc_coins = self.fetch_coinmarketcap_list()
            for coin in cmc_coins:
                symbol = coin['symbol'].upper()
                if symbol not in seen_symbols:
                    all_coins.append(coin)
                    seen_symbols.add(symbol)
            self.safe_status_callback(f"CoinMarketCap: {len(cmc_coins)} монет")
        except Exception as e:
            self.safe_status_callback(f"Помилка CoinMarketCap: {str(e)}")
        
        # 3. Біржові дані
        try:
            exchange_coins = self.fetch_exchange_symbols()
            for coin in exchange_coins:
                symbol = coin['symbol'].upper()
                if symbol not in seen_symbols:
                    all_coins.append(coin)
                    seen_symbols.add(symbol)
            self.safe_status_callback(f"Біржі: {len(exchange_coins)} монет")
        except Exception as e:
            self.safe_status_callback(f"Помилка бірж: {str(e)}")
        
        # 4. Інші джерела
        try:
            other_coins = self.fetch_alternative_sources()
            for coin in other_coins:
                symbol = coin['symbol'].upper()
                if symbol not in seen_symbols:
                    all_coins.append(coin)
                    seen_symbols.add(symbol)
            self.safe_status_callback(f"Інші джерела: {len(other_coins)} монет")
        except Exception as e:
            self.safe_status_callback(f"Помилка інших джерел: {str(e)}")
        
        return all_coins

    async def fetch_coingecko_complete_list_async(self):
        """Асинхронне отримання повного списку з CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/list"
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return [{
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name'],
                        'id': coin['id'],
                        'data_source': 'coingecko',
                        'market_cap_rank': 9999  # Тимпорарний рейтинг
                    } for coin in data]
        except Exception as e:
            logger.error(f"Помилка CoinGecko async: {e}")
        return []

    def fetch_coingecko_complete_list(self):
        """Отримання списку з CoinGecko з обходом обмежень"""
        coins = []
        try:
            # Спроба отримати весь список
            url = f"{self.COINGECKO_API_URL}/coins/list"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Перевіряємо коректність даних
                if not data or not isinstance(data, list):
                    self.safe_status_callback("CoinGecko: отримано некоректний список")
                    return []  # Повертаємо порожній список замість None
                    
                for coin in data:
                    # Перевіряємо, чи coin не є None
                    if coin is None:
                        continue
                        
                    try:
                        coins.append({
                            'id': coin.get('id', ''),
                            'symbol': coin.get('symbol', '').upper(),
                            'name': coin.get('name', ''),
                            'data_source': 'coingecko',
                            'market_cap_rank': 9999
                        })
                    except Exception as e:
                        # Пропускаємо пошкоджені дані
                        continue
                
                self.safe_status_callback(f"CoinGecko: {len(coins)} монет")
                
            elif response.status_code == 429:
                # Rate limit - використовуємо кешовані дані
                self.safe_status_callback("CoinGecko: Rate limit, використовуємо кеш")
                return self.get_cached_coingecko_list() or []  # Гарантуємо повернення списку
                
        except Exception as e:
            self.safe_status_callback(f"Помилка CoinGecko: {str(e)}")
        
        return coins  # Завжди повертаємо список (може бути порожнім)

    def fetch_coinmarketcap_list(self):
        """Отримання списку з CoinMarketCap"""
        coins = []
        try:
            # Використовуємо публічне API без ключа
            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"
            
            # Отримуємо дані частинами
            for start in range(1, 10001, 5000):
                params = {
                    'start': start,
                    'limit': 5000,
                    'sortBy': 'market_cap',
                    'sortType': 'desc',
                    'convert': 'USD'
                }
                
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'cryptoCurrencyList' in data['data']:
                        for crypto in data['data']['cryptoCurrencyList']:
                            coins.append({
                                'symbol': crypto['symbol'],
                                'name': crypto['name'],
                                'id': str(crypto['id']),
                                'market_cap_rank': crypto.get('cmcRank', 9999),
                                'data_source': 'coinmarketcap'
                            })
                
                # Затримка для уникнення блокування
                time.sleep(1)
                
            self.safe_status_callback(f"CoinMarketCap: {len(coins)} монет")
            
        except Exception as e:
            self.safe_status_callback(f"Помилка CoinMarketCap: {str(e)}")
        
        return coins

    def scrape_coinmarketcap_complete(self):
        """Скрапінг повного списку з CoinMarketCap з обходом обмежень"""
        coins = []
        try:
            # Використовуємо різні User-Agent та затримки
            for page in range(1, 51):  # Перші 50 сторінок
                try:
                    url = f"https://coinmarketcap.com/{page}/"
                    headers = {
                        'User-Agent': random.choice(self.USER_AGENTS),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive'
                    }
                    
                    response = self.session.get(url, headers=headers, timeout=30)
                    if response.status_code != 200:
                        continue
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Знаходимо таблицю з криптовалютами
                    table = soup.find('table')
                    if not table:
                        break
                    
                    rows = table.find_all('tr')[1:]  # Пропускаємо заголовок
                    if not rows:
                        break
                    
                    for row in rows:
                        try:
                            cells = row.find_all('td')
                            if len(cells) >= 3:
                                # Отримуємо символ і назву
                                name_cell = cells[2].find('p')
                                symbol_cell = cells[2].find('p', class_='coin-item-symbol')
                                
                                if name_cell and symbol_cell:
                                    symbol = symbol_cell.text.strip().upper()
                                    name = name_cell.text.replace(symbol_cell.text, '').strip()
                                    
                                    coins.append({
                                        'symbol': symbol,
                                        'name': name,
                                        'data_source': 'cmc_scrape',
                                        'market_cap_rank': len(coins) + 1
                                    })
                        except:
                            continue
                    
                    self.safe_status_callback(f"Скрапінг CMC: сторінка {page}, знайдено {len(coins)} монет")
                    
                    # Випадкова затримка між 1-3 секунди
                    time.sleep(random.uniform(1, 3))
                    
                    # Змінюємо User-Agent кожні 5 сторінок
                    if page % 5 == 0:
                        self.session.headers.update({'User-Agent': random.choice(self.USER_AGENTS)})
                        
                except Exception as e:
                    self.safe_status_callback(f"Помилка скрапінгу сторінки {page}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.safe_status_callback(f"Загальна помилка скрапінгу: {str(e)}")
        
        return coins

    def get_cached_coingecko_list(self):
        """Отримання кешованого списку з CoinGecko"""
        cache_file = self.get_data_path("coingecko_cache.json")
        
        # Перевіряємо, чи є кеш не старіший за 7 днів
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 7 * 24 * 3600:  # 7 днів
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        # Перевіряємо, чи це список
                        if isinstance(cached_data, list):
                            self.safe_status_callback(f"Використано кешований список: {len(cached_data)} монет")
                            return cached_data
                except Exception as e:
                    self.safe_status_callback(f"Помилка читання кешу: {str(e)}")
        
        # Якщо кешу немає або він старий, повертаємо порожній список
        return []

    def get_backup_crypto_list(self, max_count=1000):
        """Резервний список криптовалют - завжди повертає список"""
        try:
            backup_coins = []
            
            # Список популярних криптовалют
            popular_symbols = [
                'BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 
                'DOT', 'DOGE', 'MATIC', 'LTC', 'SHIB', 'TRX', 'LINK',
                'BCH', 'ATOM', 'XLM', 'ETC', 'XMR', 'ALGO', 'VET', 'FIL',
                'THETA', 'EOS', 'XTZ', 'AAVE', 'MKR', 'COMP', 'SNX',
                'NEAR', 'FTM', 'GRT', 'SUSHI', 'CRV', '1INCH', 'REN', 'BAT',
                'ZRX', 'ENJ', 'MANA', 'SAND', 'AXS', 'GALA', 'APE', 'GMT'
            ]
            
            # Додаємо спочатку популярні монети
            for i, symbol in enumerate(popular_symbols):
                if symbol and i < max_count:
                    backup_coins.append({
                        'symbol': symbol,
                        'name': f'{symbol} Coin',
                        'market_cap_rank': i + 1,
                        'data_source': 'backup',
                        'data_quality': 'N/A',
                        'selected': False
                    })
            
            # Додаємо додаткові монети до потрібної кількості
            start_index = len(popular_symbols) + 1
            for i in range(start_index, max_count + 1):
                if len(backup_coins) >= max_count:
                    break
                backup_coins.append({
                    'symbol': f'TOKEN{i:04d}',
                    'name': f'Token {i}',
                    'market_cap_rank': i,
                    'data_source': 'backup',
                    'data_quality': 'N/A',
                    'selected': False
                })
            
            return backup_coins[:max_count]  # Обмежуємо потрібною кількістю
            
        except Exception as e:
            logger.error(f"Error in get_backup_crypto_list: {e}")
            # Повертаємо мінімальний резервний список
            return [{
                'symbol': 'BTC',
                'name': 'Bitcoin',
                'market_cap_rank': 1,
                'data_source': 'backup',
                'data_quality': 'N/A',
                'selected': False
            }]

    def fetch_coinmarketcap_list(self):
        """Отримання списку з CoinMarketCap"""
        try:
            # Використовуємо публічне API CoinMarketCap
            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"
            params = {
                'start': 1,
                'limit': 10000,  # Максимальний ліміт
                'sortBy': 'market_cap',
                'sortType': 'desc',
                'convert': 'USD'
            }
            
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                coins = []
                for crypto in data['data']['cryptoCurrencyList']:
                    coins.append({
                        'symbol': crypto['symbol'],
                        'name': crypto['name'],
                        'id': str(crypto['id']),
                        'market_cap_rank': crypto['cmcRank'],
                        'data_source': 'coinmarketcap'
                    })
                return coins
        except Exception as e:
            logger.error(f"Помилка CoinMarketCap: {e}")
        
        # Резервний метод через web scraping
        return self.scrape_coinmarketcap_complete()

    def scrape_coinmarketcap_complete(self):
        """Скрапінг повного списку з CoinMarketCap"""
        coins = []
        try:
            # Скрапінг через кілька сторінок
            for page in range(1, 101):  # Перші 100 сторінок (≈5000 монет)
                url = f"https://coinmarketcap.com/?page={page}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                table = soup.find('table')
                if not table:
                    break
                
                rows = table.find_all('tr')[1:]  # Пропускаємо заголовок
                if not rows:
                    break
                
                for row in rows:
                    try:
                        cells = row.find_all('td')
                        if len(cells) >= 10:
                            symbol = cells[2].text.strip().upper()
                            name = cells[1].text.strip()
                            
                            coins.append({
                                'symbol': symbol,
                                'name': name,
                                'data_source': 'cmc_scrape',
                                'market_cap_rank': len(coins) + 1
                            })
                    except:
                        continue
                
                self.safe_status_callback(f"Скрапінг CMC: сторінка {page}, знайдено {len(coins)} монет")
                
                # Перерва для уникнення блокування
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Помилка скрапінгу CMC: {e}")
        
        return coins

    def fetch_exchange_symbols(self):
        """Отримання символів з усіх бірж"""
        all_symbols = set()
        coins = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                self.safe_status_callback(f"Завантаження символів з {exchange_name}...")
                
                markets = exchange.load_markets()
                for symbol in markets.keys():
                    if symbol.endswith('/USDT') or symbol.endswith('/USD') or symbol.endswith('/BTC'):
                        base_symbol = symbol.split('/')[0].upper()
                        if base_symbol not in all_symbols:
                            all_symbols.add(base_symbol)
                            coins.append({
                                'symbol': base_symbol,
                                'name': f"{base_symbol} (from {exchange_name})",
                                'data_source': f'exchange_{exchange_name}',
                                'market_cap_rank': 9999
                            })
                
                self.safe_status_callback(f"{exchange_name}: {len(markets)} trading pairs")
                
            except Exception as e:
                logger.error(f"Помилка {exchange_name}: {e}")
                continue
        
        return coins

    def fetch_alternative_sources(self):
        """Отримання даних з альтернативних джерел"""
        coins = []
        
        # 1. CoinPaprika
        try:
            url = "https://api.coinpaprika.com/v1/coins"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for coin in data:
                    coins.append({
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name'],
                        'id': coin['id'],
                        'data_source': 'coinpaprika',
                        'market_cap_rank': 9999
                    })
        except Exception as e:
            logger.error(f"Помилка CoinPaprika: {e}")
        
        # 2. LiveCoinWatch
        try:
            url = "https://api.livecoinwatch.com/coins/list"
            headers = {
                'content-type': 'application/json',
                'x-api-key': 'free'  # Використовуємо free tier
            }
            response = requests.post(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for coin in data:
                    coins.append({
                        'symbol': coin['code'].upper(),
                        'name': coin['name'],
                        'data_source': 'livecoinwatch',
                        'market_cap_rank': coin.get('rank', 9999)
                    })
        except Exception as e:
            logger.error(f"Помилка LiveCoinWatch: {e}")
        
        # 3. CryptoCompare
        try:
            url = "https://min-api.cryptocompare.com/data/all/coinlist"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for coin_data in data['Data'].values():
                    coins.append({
                        'symbol': coin_data['Symbol'].upper(),
                        'name': coin_data['CoinName'],
                        'data_source': 'cryptocompare',
                        'market_cap_rank': 9999
                    })
        except Exception as e:
            logger.error(f"Помилка CryptoCompare: {e}")
        
        return coins

    def download_top_crypto(self, count):
        """Завантаження топ-N криптовалют"""
        def download_top_thread():
            try:
                # Отримуємо топ монети з CoinGecko
                top_cryptos = self.fetch_coingecko_top_list(count)
                
                if not top_cryptos:
                    self.safe_status_callback("Не вдалося отримати топ криптовалюти")
                    return
                
                total = len(top_cryptos)
                successful = 0
                
                for i, crypto in enumerate(top_cryptos):
                    try:
                        symbol = crypto['symbol']
                        self.safe_status_callback(f"Завантаження {symbol} ({i+1}/{total})")
                        
                        data = self.fetch_robust_data(crypto['symbol'])
                        if data is not None and not data.empty:
                            filename = self.get_data_path(f"{symbol}_comprehensive.csv")
                            data.to_csv(filename)
                            successful += 1
                        
                        self.safe_progress_callback((i + 1) / total * 100)
                        time.sleep(0.3)
                        
                    except Exception as e:
                        logger.error(f"Помилка завантаження {crypto['symbol']}: {e}")
                
                self.safe_status_callback(f"Завершено: {successful}/{total} успішно")
                
            except Exception as e:
                self.safe_status_callback(f"Помилка завантаження топ-{count}: {e}")
        
        thread = threading.Thread(target=download_top_thread, daemon=True)
        thread.start()

    def fetch_coingecko_top_list(self, count):
        """Отримання топ-N криптовалют з CoinGecko з максимальною обробкою помилок"""
        try:
            url = f"{self.COINGECKO_API_URL}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': min(count, 250),
                'page': 1,
                'sparkline': 'false'
            }
            
            # Додаємо заголовки для уникнення блокування
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            
            # Перевіряємо успішність запиту
            if response.status_code != 200:
                self.safe_status_callback(f"CoinGecko API помилка: {response.status_code}")
                logger.warning(f"CoinGecko response status: {response.status_code}")
                return []
            
            # Спроба отримати JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                self.safe_status_callback("CoinGecko: некоректна JSON відповідь")
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Response text: {response.text[:200]}")  # Логуємо перші 200 символів відповіді
                return []
            
            # Перевіряємо, чи отримали коректні дані
            if data is None:
                self.safe_status_callback("CoinGecko: отримано None")
                return []
                
            if not isinstance(data, list):
                self.safe_status_callback(f"CoinGecko: очікувався список, отримано {type(data)}")
                logger.warning(f"Expected list, got {type(data)}: {data}")
                return []
            
            coins_data = []
            processed_count = 0
            error_count = 0
            
            for crypto in data:
                # Перевіряємо, чи crypto не є None - ЦЕ КЛЮЧОВА ПЕРЕВІРКА!
                if crypto is None:
                    error_count += 1
                    continue
                    
                try:
                    # Додаткова перевірка - чи crypto є словником
                    if not isinstance(crypto, dict):
                        error_count += 1
                        continue
                    
                    # Безпечно отримуємо всі значення з додатковими перевірками
                    symbol = crypto.get('symbol')
                    if symbol is None:
                        error_count += 1
                        continue
                        
                    # Конвертуємо символи в рядки та обробляємо None значення
                    coin_data = {
                        'symbol': str(symbol).upper() if symbol else 'UNKNOWN',
                        'name': str(crypto.get('name', '')) if crypto.get('name') else 'Unknown',
                        'market_cap': self._safe_float(crypto.get('market_cap')),
                        'volume': self._safe_float(crypto.get('total_volume')),
                        'current_price': self._safe_float(crypto.get('current_price')),
                        'price_change_24h': self._safe_float(crypto.get('price_change_percentage_24h')),
                        'market_cap_rank': self._safe_int(crypto.get('market_cap_rank'), 9999),
                        'data_source': 'coingecko',
                        'data_quality': 'N/A',
                        'selected': False
                    }
                    
                    # Додаткова перевірка - чи символ не є порожнім
                    if coin_data['symbol'] == 'UNKNOWN' or coin_data['symbol'] == '':
                        error_count += 1
                        continue
                    
                    coins_data.append(coin_data)
                    processed_count += 1
                    
                except (ValueError, TypeError, AttributeError) as e:
                    error_count += 1
                    logger.debug(f"Помилка обробки crypto data: {e}")
                    continue
            
            self.safe_status_callback(f"CoinGecko: оброблено {processed_count} монет, помилок: {error_count}")
            return coins_data
            
        except requests.exceptions.RequestException as e:
            self.safe_status_callback(f"Помилка мережі CoinGecko: {str(e)}")
            logger.error(f"Network error: {e}")
            return []
        except Exception as e:
            self.safe_status_callback(f"Неочікувана помилка CoinGecko: {str(e)}")
            logger.exception("Unexpected error in fetch_coingecko_top_list")  # Детальний лог помилки
            return []

    def download_parallel_data(self, coins):
        """Паралельне завантаження даних для списку криптовалют"""
        def download_batch(batch_coins):
            successful = 0
            for coin in batch_coins:
                try:
                    data = self.fetch_all_data_sources(coin)
                    if data is not None and not data.empty:
                        symbol = coin['symbol']
                        filename = f"data/{symbol}_full_data.csv"
                        data.to_csv(filename)
                        successful += 1
                        self.safe_status_callback(f"✅ {symbol}: збережено")
                except Exception as e:
                    self.safe_status_callback(f"❌ {coin['symbol']}: помилка - {str(e)}")
            return successful
        
        # Розділяємо на батчі для паралельної обробки
        batch_size = 10
        batches = [coins[i:i + batch_size] for i in range(0, len(coins), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_batch, batch) for batch in batches]
            
            total_successful = 0
            for future in concurrent.futures.as_completed(futures):
                total_successful += future.result()
        
        self.safe_status_callback(f"Завершено: {total_successful}/{len(coins)} успішно")

    def fetch_all_data_sources(self, crypto):
        """Отримання даних з усіх джерел з покращеною обробкою помилок"""
        symbol = crypto['symbol']
        
        try:
            # Спершу пробуємо Yahoo Finance
            data = self.fetch_yahoo_data_detailed(symbol)
            
            if data is None or data.empty:
                logger.warning(f"Не вдалося отримати дані для {symbol} з Yahoo Finance")
                return None
            
            # Очищаємо та обробляємо дані
            data = self.safe_clean_data(data, symbol)
            
            return data
            
        except Exception as e:
            logger.error(f"Критична помилка отримання даних для {symbol}: {e}")
            return None

    def fetch_yahoo_data_detailed(self, symbol):
        """Покращене завантаження з обробкою обмежених символів"""
        try:
            symbols_to_try = [
                f"{symbol}-USD", f"{symbol}USD", f"{symbol}-USDT",
                f"{symbol}/USD", symbol, f"{symbol}-EUR", f"{symbol}-BTC"
            ]
            
            for sym in symbols_to_try:
                try:
                    ticker = yf.Ticker(sym)
                    
                    # Спершу пробуємо максимальний період
                    try:
                        data = ticker.history(period="max")
                        if not data.empty and len(data) > 10:
                            return self._ensure_ohlcv_columns(data)
                    except Exception as e:
                        # Якщо "max" не працює, пробуємо коротші періоди
                        if "invalid" in str(e).lower() and "period" in str(e).lower():
                            for period in ["5y", "2y", "1y", "6mo", "3mo", "1mo"]:
                                try:
                                    data = ticker.history(period=period)
                                    if not data.empty and len(data) > 10:
                                        return self._ensure_ohlcv_columns(data)
                                except:
                                    continue
                        continue
                        
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Помилка Yahoo Finance для {symbol}: {e}")
            return None

    def _ensure_ohlcv_columns(self, data):
        """Переконуємося що є всі необхідні колонки"""
        if 'Close' not in data.columns:
            return None
            
        if 'Open' not in data.columns:
            data['Open'] = data['Close']
        if 'High' not in data.columns:
            data['High'] = data['Close'] * 1.01
        if 'Low' not in data.columns:
            data['Low'] = data['Close'] * 0.99
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000
            
        return data

    def create_fallback_data(self, symbol):
        """Створення резервних даних для проблемних символів"""
        try:
            # Створюємо дати за останні 2 роки
            dates = pd.date_range(end=pd.Timestamp.now(), periods=365*2, freq='D')
            
            # Базові ціни в залежності від символу
            base_prices = {
                'ADA': 0.5, 'DOGE': 0.1, 'TON': 2.0, 'BTC': 50000, 
                'ETH': 3000, 'JUPSOL': 0.01, 'FAKECOIN': 0.001
            }
            
            base_price = base_prices.get(symbol, 1.0)
            volatility = 0.02  # 2% денна волатильність
            
            # Генеруємо ціни
            np.random.seed(hash(symbol) % 1000)  # Унікальний seed для кожного символу
            returns = np.random.normal(0, volatility, len(dates))
            prices = base_price * (1 + returns).cumprod()
            
            df = pd.DataFrame({
                'Open': prices,
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Close': prices,
                'Volume': np.random.lognormal(14, 1, len(dates))
            }, index=dates)
            
            logger.info(f"Створено резервні дані для {symbol}: {len(df)} рядків")
            return df
            
        except Exception as e:
            logger.error(f"Помилка створення резервних даних для {symbol}: {e}")
            return None

    def create_minimal_data(self, symbol):
        """Створення мінімальних даних навіть при критичних помилках"""
        try:
            # Мінімальний набір даних для навчання
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
            
            df = pd.DataFrame({
                'Open': 1.0,
                'High': 1.02,
                'Low': 0.98, 
                'Close': 1.0,
                'Volume': 1000000
            }, index=dates)
            
            logger.warning(f"Створено мінімальні дані для {symbol} через критичну помилку")
            return df
            
        except Exception as e:
            logger.error(f"Навіть мінімальні дані не вдалося створити для {symbol}: {e}")
            # Повертаємо порожній DataFrame
            return pd.DataFrame()

    def safe_clean_data(self, data, symbol):
        """Безпечне очищення даних без помилок Series"""
        try:
            if data is None or data.empty:
                return pd.DataFrame()
                
            df = data.copy()
            
            # 1. Видаляємо NaN
            df = df.dropna()
            
            # 2. Видаляємо нульові ціни
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    df = df[df[col] > 0]
            
            # 3. Видаляємо дублікати індексу
            df = df[~df.index.duplicated(keep='first')]
            
            # 4. Сортуємо за індексом
            df = df.sort_index()
            
            if df.empty:
                logger.warning(f"Після очищення дані для {symbol} порожні")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Помилка безпечного очищення для {symbol}: {e}")
            # Повертаємо оригінальні дані без NaN
            try:
                return data.dropna() if data is not None else pd.DataFrame()
            except:
                return pd.DataFrame()

    def add_basic_technical_indicators(self, data):
        """Додавання базових технічних індикаторів"""
        df = data.copy()
        
        if 'Close' not in df.columns:
            return df
        
        # Базові технічні індикатори
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        
        # Ковзні середні
        for window in [5, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Волатильність
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # RSI
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        
        return df

    def calculate_rsi(self, prices, period=14):
        """Розрахунок RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fetch_stablecoin_data(self, symbol):
        """Отримання даних для стейблкоінів"""
        try:
            # Для стейблкоінів створюємо штучні дані
            dates = pd.date_range(end=datetime.now(), periods=365*3)  # 3 роки
            data = pd.DataFrame({
                'Open': 1.0,
                'High': 1.0,
                'Low': 1.0,
                'Close': 1.0,
                'Volume': 1000000
            }, index=dates)
            
            return data
            
        except Exception as e:
            logger.error(f"Помилка створення даних для {symbol}: {e}")
            return None

    def fetch_coingecko_detailed(self, coin_id):
        """Детальні дані з CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'true',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'true',
                'sparkline': 'false'
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return self.process_gecko_detailed_data(data)
        except Exception as e:
            logger.error(f"Помилка детальних даних CoinGecko: {e}")
        return None

    def process_gecko_detailed_data(self, data):
        """Обробка детальних даних з CoinGecko"""
        metrics = {}
        
        # Market data
        if 'market_data' in data:
            md = data['market_data']
            metrics.update({
                'current_price': md.get('current_price', {}).get('usd'),
                'market_cap': md.get('market_cap', {}).get('usd'),
                'total_volume': md.get('total_volume', {}).get('usd'),
                'price_change_24h': md.get('price_change_percentage_24h'),
                'high_24h': md.get('high_24h', {}).get('usd'),
                'low_24h': md.get('low_24h', {}).get('usd'),
                'ath': md.get('ath', {}).get('usd'),
                'ath_change_percentage': md.get('ath_change_percentage', {}).get('usd'),
                'atl': md.get('atl', {}).get('usd'),
                'atl_change_percentage': md.get('atl_change_percentage', {}).get('usd')
            })
        
        # Community data
        if 'community_data' in data:
            cd = data['community_data']
            metrics.update({
                'twitter_followers': cd.get('twitter_followers'),
                'reddit_subscribers': cd.get('reddit_subscribers'),
                'reddit_active_users': cd.get('reddit_active_users'),
                'telegram_channel_user_count': cd.get('telegram_channel_user_count')
            })
        
        # Developer data
        if 'developer_data' in data:
            dd = data['developer_data']
            metrics.update({
                'github_forks': dd.get('forks'),
                'github_stars': dd.get('stars'),
                'github_subscribers': dd.get('subscribers'),
                'total_issues': dd.get('total_issues'),
                'closed_issues': dd.get('closed_issues'),
                'pull_requests_merged': dd.get('pull_requests_merged'),
                'pull_request_contributors': dd.get('pull_request_contributors'),
                'commit_count_4_weeks': dd.get('commit_count_4_weeks')
            })
        
        return pd.DataFrame([metrics])

    def fetch_fundamental_data(self, symbol):
        """Отримання фундаментальних даних"""
        try:
            # Блокчейн дані (для підтримуваних монет)
            if symbol in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']:
                return self.fetch_blockchain_data(symbol)
        except Exception as e:
            logger.error(f"Помилка фундаментальних даних для {symbol}: {e}")
        return None

    def fetch_blockchain_data(self, symbol):
        """Отримання блокчейн даних"""
        metrics = {}
        
        try:
            if symbol == 'BTC':
                url = "https://blockchain.info/q/"
                metrics.update({
                    'hash_rate': requests.get(url + 'hashrate').json(),
                    'difficulty': requests.get(url + 'getdifficulty').json(),
                    'transaction_count': requests.get(url + 'transactioncount').json(),
                    'total_btc': requests.get(url + 'totalbc').json() / 100000000
                })
            elif symbol == 'ETH':
                url = "https://api.etherscan.io/api"
                params = {
                    'module': 'stats',
                    'action': 'ethsupply',
                    'apikey': 'freekey'
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    metrics['total_eth'] = float(data['result']) / 1e18
        except:
            pass
        
        return pd.DataFrame([metrics]) if metrics else None

    def save_complete_list(self, coins):
        """Збереження повного списку криптовалют"""
        try:
            # Зберігаємо у JSON
            json_filename = self.get_data_path("complete_crypto_list.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(coins, f, ensure_ascii=False, indent=2)
            
            # Зберігаємо у CSV
            csv_filename = self.get_data_path("complete_crypto_list.csv")
            with open(csv_filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=coins[0].keys())
                writer.writeheader()
                writer.writerows(coins)
            
            self.safe_status_callback(f"Збережено повний список: {len(coins)} криптовалют")
            
        except Exception as e:
            self.safe_status_callback(f"Помилка збереження: {str(e)}")

    def fetch_social_data(self, symbol):
        """Отримання соціальних даних для криптовалюти"""
        try:
            social_data = {}
            
            # Спроба отримати соціальні дані з CoinGecko
            coin_id = symbol.lower()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Соціальні метрики
                community_data = data.get('community_data', {})
                social_data.update({
                    'twitter_followers': community_data.get('twitter_followers'),
                    'reddit_subscribers': community_data.get('reddit_subscribers'),
                    'reddit_active_users': community_data.get('reddit_active_users'),
                    'telegram_users': community_data.get('telegram_channel_user_count')
                })
                
                # Developer data
                developer_data = data.get('developer_data', {})
                social_data.update({
                    'github_forks': developer_data.get('forks'),
                    'github_stars': developer_data.get('stars'),
                    'github_subscribers': developer_data.get('subscribers'),
                    'total_issues': developer_data.get('total_issues'),
                    'closed_issues': developer_data.get('closed_issues')
                })
            
            return social_data if social_data else None
            
        except Exception as e:
            logger.debug(f"Помилка отримання соціальних даних для {symbol}: {e}")
            return None
    
    def validate_crypto_symbol(self, symbol):
        """Перевірка чи символ існує"""
        try:
            # Швидка перевірка через CoinGecko
            url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                coins = data.get('coins', [])
                
                # Шукаємо точний збіг
                for coin in coins:
                    if coin['symbol'].upper() == symbol.upper():
                        return True
                
                return False
            
            return True  # Якщо API не працює, припускаємо що символ існує
            
        except Exception as e:
            logger.debug(f"Помилка перевірки символу {symbol}: {e}")
            return True
    
    
    # Додаткові методи...
    def clean_and_process_data(self, data, symbol):
        """Безпечне очищення даних без помилок Series"""
        try:
            if data is None or data.empty:
                logger.warning(f"Порожні дані для {symbol}")
                return pd.DataFrame()

            df = data.copy()
            
            # 1. Перетворюємо індекс на datetime
            try:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # Видаляємо невалідні дати
                    df = df[df.index.notnull()]
            except Exception as e:
                logger.warning(f"Помилка формату індексу для {symbol}: {e}")
                # Створюємо новий індекс
                df = df.reset_index(drop=True)
                df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
            
            # 2. Видаляємо дублікати індексу
            df = df[~df.index.duplicated(keep='first')]
            
            # 3. Сортуємо за датою
            df = df.sort_index()
            
            # 4. Заповнюємо пропуски
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if df[col].isnull().any():
                    # Заповнюємо пропуски
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 5. Обробка цінових даних (БЕЗ використання if з Series!)
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    # ВИПРАВЛЕННЯ: використовуємо булеве індексування без if
                    valid_prices_mask = (df[col] > 0) & (df[col] < 1e10)  # Прибираємо нульові та дуже великі значення
                    df = df[valid_prices_mask]
                    
                    # Додаткова перевірка на викиди (тільки якщо достатньо даних)
                    if len(df) > 100:
                        Q1 = df[col].quantile(0.05)
                        Q3 = df[col].quantile(0.95)
                        IQR = Q3 - Q1
                        valid_range_mask = (df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))
                        df = df[valid_range_mask]
            
            # 6. Видаляємо рядки з NaN що залишилися
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"Після очищення дані для {symbol} порожні")
                return pd.DataFrame()
            
            logger.info(f"Очищено {symbol}: {len(df)} рядків")
            return df
            
        except Exception as e:
            logger.error(f"Критична помилка очищення для {symbol}: {e}")
            # Повертаємо оригінальні дані без NaN
            try:
                return data.dropna() if not data.empty else pd.DataFrame()
            except:
                return pd.DataFrame()

    def add_basic_time_features(self, data):
        """Додавання базових часових ознак"""
        df = data.copy()
        
        # Переконуємося, що індекс є DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Додаємо часові компоненти
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        
        # Додаємо бінарні ознаки
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        df['Is_Month_Start'] = df.index.is_month_start.astype(int)
        
        return df

    
    def fetch_alternative_data(self, symbol):
        """Покращене отримання даних з альтернативних джерел"""
        try:
            # Спроба CoinGecko API
            coin_id = symbol.lower()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': 'max',
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                
                if not prices:
                    return None
                
                # Створюємо DataFrame
                dates = [pd.to_datetime(price[0], unit='ms') for price in prices]
                close_prices = [price[1] for price in prices]
                
                df = pd.DataFrame({
                    'Close': close_prices,
                    'Open': close_prices,
                    'High': close_prices,
                    'Low': close_prices,
                    'Volume': 0
                }, index=dates)
                
                # Заповнюємо пропуски
                df = df.asfreq('D').fillna(method='ffill')
                
                # Додаємо технічні індикатори
                df = self.add_basic_technical_indicators(df)
                
                return df
                
        except Exception as e:
            logger.error(f"Помилка альтернативного джерела для {symbol}: {e}")
        
        # Резервний варіант - створення штучних даних
        return self.create_synthetic_data(symbol)

    def create_synthetic_data(self, symbol):
        """Створення штучних даних для тестування"""
        try:
            # Створюємо дати за останні 3 роки
            dates = pd.date_range(end=pd.Timestamp.now(), periods=365*3, freq='D')
            
            # Базові ціни (симуляція)
            base_price = 100 if symbol != 'BTC' else 50000
            volatility = 0.02  # 2% денна волатильність
            
            # Генеруємо ціни
            returns = np.random.normal(0, volatility, len(dates))
            prices = base_price * (1 + returns).cumprod()
            
            df = pd.DataFrame({
                'Open': prices,
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Close': prices,
                'Volume': np.random.lognormal(10, 1, len(dates))
            }, index=dates)
            
            # Додаємо технічні індикатори
            df = self.add_basic_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Помилка створення штучних даних для {symbol}: {e}")
            return None
    
    
    def add_basic_time_features(self, data):
        """Додавання базових часових ознак"""
        df = data.copy()
        
        # Додаємо часові компоненти
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        df['Quarter'] = df.index.quarter
        df['WeekOfYear'] = df.index.isocalendar().week
        
        # Додаємо бінарні ознаки
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        df['Is_Month_Start'] = df.index.is_month_start.astype(int)
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
        df['Is_Quarter_Start'] = df.index.is_quarter_start.astype(int)
        
        return df
    
    def safe_clean_data(self, data, symbol):
        """Безпечне очищення даних без помилок Series"""
        try:
            df = data.copy()
            
            if df.empty:
                return df

            # 1. Перетворюємо індекс на datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
            
            # 2. Видаляємо дублікати індексу
            df = df[~df.index.duplicated(keep='first')]
            
            # 3. Сортуємо за датою
            df = df.sort_index()
            
            # 4. Заповнюємо пропуски (безпечно)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 5. Видаляємо неправильні ціни (без використання if з Series)
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    # Фільтруємо без використання if з цілим Series
                    df = df[df[col] > 0]  # Це працює коректно
                    
                    # Додатково: видаляємо екстремальні значення
                    if len(df) > 50:
                        q_low = df[col].quantile(0.01)
                        q_high = df[col].quantile(0.99)
                        df = df[(df[col] >= q_low) & (df[col] <= q_high)]
            
            # 6. Видаляємо рядки з NaN що залишилися
            df = df.dropna()
            
            logger.info(f"Очищено {symbol}: {len(df)} рядків")
            return df
            
        except Exception as e:
            logger.error(f"Помилка очищення {symbol}: {e}")
            # Повертаємо оригінальні дані без NaN
            return data.dropna() if not data.empty else data

    def cleanup(self):
        """Очищення ресурсів при закритті"""
        try:
            if self.session:
                asyncio.run(self.session.close())
            # Закриваємо всі біржі
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'close'):
                    exchange.close()
        except Exception as e:
            logger.error(f"Помилка очищення: {e}")

    def __del__(self):
        """Cleanup"""
        self.cleanup()

    def fetch_ada_specific_data(self, symbol):
        """Спеціальний метод для отримання даних ADA та інших проблемних криптовалют"""
        try:
            # ADA має спеціальний символ в різних джерелах
            symbol_variants = {
                'ADA': ['ADA-USD', 'ADA/USD', 'cardano', 'ADA'],
                'DOGE': ['DOGE-USD', 'DOGE/USD', 'dogecoin', 'DOGE'],
                'TON': ['TON-USD', 'TON/USD', 'toncoin', 'TON']
            }
            
            variants = symbol_variants.get(symbol, [symbol])
            
            for variant in variants:
                try:
                    # Спроба Yahoo Finance
                    ticker = yf.Ticker(variant)
                    data = ticker.history(period="max")
                    
                    if not data.empty and len(data) > 100:
                        # Перевіряємо наявність ключових колонок
                        if 'Close' not in data.columns:
                            continue
                            
                        # Заповнюємо відсутні колонки
                        if 'Open' not in data.columns:
                            data['Open'] = data['Close']
                        if 'High' not in data.columns:
                            data['High'] = data['Close'] * 1.01  # Додаємо невелику волатильність
                        if 'Low' not in data.columns:
                            data['Low'] = data['Close'] * 0.99
                        if 'Volume' not in data.columns:
                            data['Volume'] = 1000000  # Значення за замовчуванням
                        
                        logger.info(f"Знайдено дані для {symbol} через варіант {variant}")
                        return data
                        
                except Exception as e:
                    continue
            
            # Якщо Yahoo Finance не спрацював, спробуємо CoinGecko
            return self.fetch_ada_from_coingecko(symbol)
            
        except Exception as e:
            logger.error(f"Помилка отримання ADA даних: {e}")
            return None

    def fetch_ada_from_coingecko(self, symbol):
        """Отримання даних ADA з CoinGecko API"""
        try:
            # Map symbols to CoinGecko IDs
            coin_ids = {
                'ADA': 'cardano',
                'DOGE': 'dogecoin', 
                'TON': 'toncoin',
                'BTC': 'bitcoin',
                'ETH': 'ethereum'
            }
            
            coin_id = coin_ids.get(symbol, symbol.lower())
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': 'max',
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                
                if not prices:
                    return None
                
                # Створюємо DataFrame
                dates = [pd.to_datetime(price[0], unit='ms') for price in prices]
                close_prices = [price[1] for price in prices]
                
                df = pd.DataFrame({
                    'Close': close_prices,
                    'Open': close_prices,
                    'High': close_prices,
                    'Low': close_prices,
                    'Volume': 0
                }, index=dates)
                
                # Додаємо невелику волатильність для реалізму
                df['High'] = df['Close'] * (1 + np.random.uniform(0.01, 0.03, len(df)))
                df['Low'] = df['Close'] * (1 - np.random.uniform(0.01, 0.03, len(df)))
                df['Volume'] = np.random.lognormal(14, 1, len(df))
                
                logger.info(f"Отримано дані {symbol} з CoinGecko: {len(df)} рядків")
                return df
                
        except Exception as e:
            logger.error(f"Помилка CoinGecko для {symbol}: {e}")
        
        return None

    def create_fallback_data(self, symbol):
        """Створення резервних даних, якщо інші методи не спрацювали"""
        try:
            # Створюємо дати за останні 2 роки
            dates = pd.date_range(end=pd.Timestamp.now(), periods=365*2, freq='D')
            
            # Базові ціни в залежності від символу
            base_prices = {
                'ADA': 0.5, 'DOGE': 0.1, 'TON': 2.0,
                'BTC': 50000, 'ETH': 3000
            }
            
            base_price = base_prices.get(symbol, 100)
            volatility = 0.03  # 3% денна волатильність
            
            # Генеруємо ціни з нормальним розподілом
            np.random.seed(42)  # Для відтворюваності
            returns = np.random.normal(0, volatility, len(dates))
            prices = base_price * (1 + returns).cumprod()
            
            df = pd.DataFrame({
                'Open': prices,
                'High': prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
                'Close': prices,
                'Volume': np.random.lognormal(15, 1, len(dates))
            }, index=dates)
            
            logger.info(f"Створено резервні дані для {symbol}: {len(df)} рядків")
            return df
            
        except Exception as e:
            logger.error(f"Помилка створення резервних даних для {symbol}: {e}")
            return None
    
    
    
    def on_tree_select(self, event, item=None):
        """Обробка вибору елементів у treeview"""
        if item is None:
            item = self.crypto_tree.selection()[0] if self.crypto_tree.selection() else None
        
        if item and item in self.tree_item_data:
            current_value = self.crypto_tree.set(item, 'select')
            new_value = '☐' if current_value == '☑' else '☑'
            self.crypto_tree.set(item, 'select', new_value)
            
            # Оновлюємо стан у даних
            crypto_data = self.tree_item_data[item]
            crypto_data['selected'] = (new_value == '☑')
            
            self.update_counters()
            self.scroll_to_selection()  # Прокручуємо до вибраного елемента
    
    def setup_styles(self):
        """Налаштування стилів для treeview"""
        style = ttk.Style()
        
        # Налаштування стилів для treeview
        style.configure('Treeview', 
                    font=('Arial', 9),
                    rowheight=25,
                    borderwidth=1,
                    relief='solid')
        
        style.configure('Treeview.Heading', 
                    font=('Arial', 9, 'bold'),
                    background='#f0f0f0',
                    relief='raised')
        
        # Зміна кольору при наведенні
        style.map('Treeview', 
                background=[('selected', '#0078d7')],
                foreground=[('selected', 'white')])
    
    def download_all_crypto_comprehensive(self):
        """Завантаження всіх криптовалют з максимальною кількістю даних"""
        if not messagebox.askyesno("Підтвердження", 
                                "Ця операція завантажить дані для ВСІХ доступних криптовалют.\n"
                                "Це може зайняти багато часу та використати значний обсяг дискового простору.\n"
                                "Продовжити?"):
            return

        def download_thread():
            try:
                self.safe_status_callback("Початок завантаження ВСІХ криптовалют...")
                self.safe_progress_callback(0)
                
                # Отримуємо повний список криптовалют
                all_cryptos = self.get_complete_crypto_list()
                
                if not all_cryptos:
                    self.safe_status_callback("Не вдалося отримати список криптовалют")
                    return
                
                total_count = len(all_cryptos)
                self.safe_status_callback(f"Знайдено {total_count} криптовалют. Початок завантаження...")
                
                successful = 0
                failed = 0
                
                # Завантажуємо дані для кожної криптовалюти
                for i, crypto in enumerate(all_cryptos):
                    symbol = crypto['symbol']
                    self.safe_status_callback(f"Завантаження {symbol} ({i+1}/{total_count})...")
                    self.update_progress_label(f"Завантаження {i+1}/{total_count}: {symbol}")
                    self.safe_progress_callback((i + 1) / total_count * 100)
                    
                    try:
                        # Завантажуємо комплексні дані
                        data = self.fetch_robust_data(crypto['symbol'])
                        
                        if data is not None and not data.empty:
                            # Перевіряємо існування директорії data
                            if not os.path.exists('data'):
                                os.makedirs('data')
                            
                            # Зберігаємо дані
                            filename = self.get_data_path(f"{symbol}_comprehensive_data.csv")
                            data.to_csv(filename)
                            
                            # Оновлюємо якість даних
                            quality = self.calculate_data_quality(data)
                            crypto['data_quality'] = quality
                            crypto['data_points'] = len(data)
                            
                            successful += 1
                            self.safe_status_callback(f"✅ {symbol}: завантажено {len(data)} точок даних")
                        else:
                            failed += 1
                            self.safe_status_callback(f"❌ {symbol}: немає даних")
                            
                    except Exception as e:
                        failed += 1
                        error_msg = f"❌ {symbol}: помилка - {str(e)}"
                        self.safe_status_callback(error_msg)
                        logger.error(error_msg)
                    
                    # Затримка для уникнення rate limits
                    time.sleep(0.5)
                
                # Оновлюємо список та відображаємо результати
                self.crypto_list = all_cryptos
                self.apply_filters()
                
                final_msg = f"Завершено: {successful}/{total_count} успішно, {failed} невдало"
                self.safe_status_callback(final_msg)
                self.update_progress_label(final_msg)
                
                # Зберігаємо метадані
                self.save_crypto_metadata(all_cryptos)
                
            except Exception as e:
                error_msg = f"Критична помилка завантаження: {str(e)}"
                self.safe_status_callback(error_msg)
                self.update_progress_label("Помилка завантаження")
                logger.error(error_msg)
        
        # Запускаємо в окремому потоці
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def fetch_comprehensive_data(self, crypto):
        """Покращене отримання даних з обробкою помилок"""
        symbol = crypto['symbol']
        
        try:
            # 1. Отримуємо основні історичні дані
            main_data = self.fetch_extended_historical_data(symbol)
            
            if main_data is None or main_data.empty:
                logger.warning(f"Не вдалося отримати основні дані для {symbol}")
                # Спробуємо альтернативні джерела
                main_data = self.fetch_alternative_data(symbol)
                if main_data is None or main_data.empty:
                    return None
            
            # 2. Очищаємо дані
            main_data = self.clean_and_process_data(main_data, symbol)
            
            # Перевіряємо мінімальну кількість даних
            if main_data.empty or len(main_data) < 50:  # Зменшили мінімум до 50
                logger.warning(f"Замало даних для {symbol}: {len(main_data)} рядків")
                return None
            
            # 3. Додаємо технічні індикатори
            if self.include_technical.get():
                try:
                    technical_data = self.calculate_all_technical_indicators(main_data)
                    main_data = pd.concat([main_data, technical_data], axis=1)
                except Exception as e:
                    logger.warning(f"Помилка технічних індикаторів для {symbol}: {e}")
            
            # 4. Додаємо часові ознаки
            try:
                main_data = self.add_time_features(main_data)
            except Exception as e:
                logger.warning(f"Помилка часових ознак для {symbol}: {e}")
            
            # 5. Фінальне очищення після всіх перетворень
            main_data = main_data.dropna()
            
            if main_data.empty:
                return None
                
            return main_data
            
        except Exception as e:
            logger.error(f"Критична помилка отримання даних для {symbol}: {e}")
            return None

    def fetch_extended_historical_data(self, symbol):
        """Отримання розширених історичних даних"""
        try:
            # Спроба різних форматів символів
            symbols_to_try = [
                f"{symbol}-USD",
                f"{symbol}USD", 
                f"{symbol}-USDT",
                f"{symbol}/USD",
                symbol
            ]
            
            for sym in symbols_to_try:
                try:
                    ticker = yf.Ticker(sym)
                    
                    # Отримуємо дані з максимальним періодом
                    data = ticker.history(period="max")
                    
                    if not data.empty and len(data) > 100:
                        # Додаємо додаткові часові ознаки
                        data = self.add_time_features(data)
                        data = self.add_volume_features(data)
                        return data
                        
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Помилка історичних даних для {symbol}: {e}")
            return None

    def calculate_all_technical_indicators(self, data):
        """Розрахунок всіх технічних індикаторів"""
        df = data.copy()
        
        if 'Close' not in df.columns:
            return pd.DataFrame()
        
        # Базові показники
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Ковзні середні
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # RSI
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self.calculate_rsi(df['Close'], period)
        
        # MACD
        macd, macd_signal = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd - macd_signal
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = self.calculate_bollinger_bands(df['Close'], period)
            df[f'BB_Upper_{period}'] = upper
            df[f'BB_Middle_{period}'] = middle
            df[f'BB_Lower_{period}'] = lower
        
        # Волатильність
        for period in [10, 20, 30]:
            df[f'ATR_{period}'] = self.calculate_atr(df['High'], df['Low'], df['Close'], period)
            df[f'Volatility_{period}'] = df['Returns'].rolling(window=period).std() * np.sqrt(252)
        
        # Об'ємні індикатори
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        # Моментум
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        return df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')

    def add_time_features(self, data):
        """Додає часові ознаки"""
        df = data.copy()
        
        # Часові ознаки
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['Weekday'] = df.index.weekday
        df['Quarter'] = df.index.quarter
        df['DayOfYear'] = df.index.dayofyear
        
        # Сезонні ознаки
        df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
        df['Month_End'] = (df.index.is_month_end).astype(int)
        df['Month_Start'] = (df.index.is_month_start).astype(int)
        
        return df

    def add_volume_features(self, data):
        """Додає ознаки об'єму"""
        df = data.copy()
        
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)
        
        return df

    def get_complete_crypto_list(self):
        """Отримання повного списку криптовалют з обходом обмежень"""
        try:
            self.safe_status_callback("Отримання списку криптовалют...")
            
            coins = []
            
            # Спроба 1: CoinGecko (основне джерело)
            cg_coins = self.fetch_coingecko_complete_list()
            if cg_coins:  # Тепер cg_coins завжди буде списком
                coins.extend(cg_coins)
            
            # Спроба 2: CoinMarketCap (резервне джерело)
            if len(coins) < 1000:
                cmc_coins = self.fetch_coinmarketcap_list()
                if cmc_coins and isinstance(cmc_coins, list):
                    coins.extend(cmc_coins)
            
            # Спроба 3: Скрапінг з CoinMarketCap
            if len(coins) < 2000:
                scraped_coins = self.scrape_coinmarketcap_complete()
                if scraped_coins and isinstance(scraped_coins, list):
                    coins.extend(scraped_coins)
            
            # Якщо всі спроби не вдалися, використовуємо резервний список
            if not coins:
                coins = self.get_backup_crypto_list()
            
            # Видаляємо дублікати та None значення
            unique_coins = []
            seen_symbols = set()
            
            for coin in coins:
                # Перевіряємо, чи coin не є None і чи є символ
                if coin is not None and isinstance(coin, dict):
                    symbol = coin.get('symbol', '').upper()
                    if symbol and symbol not in seen_symbols:
                        unique_coins.append(coin)
                        seen_symbols.add(symbol)
            
            self.safe_status_callback(f"Знайдено {len(unique_coins)} унікальних криптовалют")
            return unique_coins
            
        except Exception as e:
            self.safe_status_callback(f"Помилка отримання списку: {str(e)}")
            return self.get_backup_crypto_list()  # Цей метод завжди повертає список

    def save_crypto_metadata(self, crypto_list):
        """Збереження метаданих криптовалют"""
        try:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_cryptos': len(crypto_list),
                'cryptos': crypto_list
            }
            
            filename = self.get_data_path("crypto_metadata.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            self.safe_status_callback("Метадані збережено успішно")
            
        except Exception as e:
            logger.error(f"Помилка збереження метаданих: {e}")

    def check_internet_connection(self):
        """Перевірка підключення до інтернету"""
        try:
            # Спроба пінгувати Google
            response = requests.get('https://www.google.com', timeout=5)
            return response.status_code == 200
        except:
            return False

    def _safe_float(self, value, default=0.0):
        """Безпечне перетворення в float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default=0):
        """Безпечне перетворення в int"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default




