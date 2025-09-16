import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
import os

# Імпорт модулів вкладок
from tabs.tab1_data_loader import UltimateCryptoDataLoader
from tabs.tab2_ml_models import MLModelsTab
from tabs.tab3_technical_analysis import TechnicalAnalysisTab
from tabs.tab4_neural_network import NeuralNetworkTab
#from tabs.tab5_multi_neural_network import MultiNeuralNetworkTab  # Додано нову вкладку
# Додайте ці імпорти
from tabs.tab5_training_neural_networks import TrainingNeuralNetworksTab
from tabs.tab6_trained_networks_graphs import TrainedNetworksGraphsTab
from tabs.tab7_real_trading_analysis import RealTradingAnalysisTab
from utils.live_data_manager import LiveDataManager
from utils.risk_manager import RiskManager
from utils.file_selector import FileSelector  # Додати цей рядок
from tabs.tab8_live_trading import LiveTradingTab

# Імпорт утиліт
from utils.logger import setup_logger

class CryptoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аналіз та прогнозування ринку криптовалют")
        
        # Встановлюємо розмір на весь екран
        self.root.state('zoomed')  # Розгортаємо на весь екран
        self.root.minsize(1024, 768)  # Мінімальний розмір для коректного відображення
        
        # Налаштування логування
        self.logger = setup_logger()
        self.logger.info("Запуск програми")
        
        # Спочатку створюємо статус бари
        self.create_status_bars()
        
        # Потім створюємо вкладки
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Створення вкладок
        self.create_tabs()
        
        # Підключення обробника зміни вкладок
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Ініціалізація нових менеджерів
        self.live_data_manager = LiveDataManager()
        self.risk_manager = RiskManager(initial_capital=10000)
        
        # Створення нової вкладки
        self.tab8 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab8, text="Live Trading")
        self.live_trading_tab = LiveTradingTab(self.tab8, self.update_status, self.update_progress)
        
        # Передаємо менеджери вкладці
        self.live_trading_tab.set_live_data_manager(self.live_data_manager)
        self.live_trading_tab.set_risk_manager(self.risk_manager)
        
        # Запуск live даних при старті
        self.setup_live_data()

        # Обробка зміни розміру вікна
        self.root.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        """Обробка зміни розміру вікна"""
        # Оновлюємо розміри елементів при зміні розміру вікна
        if hasattr(self, 'multi_neural_network'):
            try:
                # Оновлюємо розміри графіків
                self.multi_neural_network.update_plot_sizes()
            except Exception as e:
                self.logger.warning(f"Помилка оновлення розмірів: {str(e)}")

    def create_tabs(self):
        """Створення вкладок програми з адаптивним дизайном"""
        # Створення фреймів для всіх вкладок
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        self.tab5 = ttk.Frame(self.notebook)  # Нова вкладка 5
        self.tab6 = ttk.Frame(self.notebook)  # Нова вкладка 6
        self.tab7 = ttk.Frame(self.notebook)  # Нова вкладка 7
        
        # Додавання вкладок з правильними індексами
        self.notebook.add(self.tab1, text='Завантаження даних')
        self.notebook.add(self.tab2, text='ML Моделі')
        self.notebook.add(self.tab3, text='Технічний аналіз')
        self.notebook.add(self.tab4, text='Нейромережа')
        self.notebook.add(self.tab5, text='Навчання нейромереж')           # Вкладка 5
        self.notebook.add(self.tab6, text='Графіки для навчених нейромереж') # Вкладка 6
        self.notebook.add(self.tab7, text='Аналіз для реальної торгівлі')    # Вкладка 7
        
        # Ініціалізація вкладок з передачею callback функцій
        self.data_loader = UltimateCryptoDataLoader(self.tab1, self.update_status, self.update_progress)
        self.ml_models = MLModelsTab(self.tab2, self.update_status, self.update_progress)
        self.tech_analysis = TechnicalAnalysisTab(self.tab3, self.update_status, self.update_progress)
        self.neural_network = NeuralNetworkTab(self.tab4, self.update_status, self.update_progress)
        
        # Ініціалізація НОВИХ вкладок
        self.training_tab = TrainingNeuralNetworksTab(self.tab5, self.update_status, self.update_progress)
        self.graphs_tab = TrainedNetworksGraphsTab(self.tab6, self.update_status, self.update_progress)
        self.trading_tab = RealTradingAnalysisTab(self.tab7, self.update_status, self.update_progress)
        
        # Налаштування адаптивності для всіх вкладок
        self.configure_tab_adaptivity()

    def configure_tab_adaptivity(self):
        """Налаштування адаптивності для всіх вкладок"""
        # Додаємо обробники зміни розміру для кожної вкладки
        for tab in [self.tab1, self.tab2, self.tab3, self.tab4, self.tab5]:
            tab.bind('<Configure>', self.on_tab_resize)

    def on_tab_resize(self, event):
        """Обробка зміни розміру вкладки"""
        # Можна додати логіку для оновлення розмірів елементів на конкретній вкладці
        pass

    def create_status_bars(self):
        """Створення статус барів"""
        # Фрейм для статусних елементів
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Прогрес бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100, mode='determinate'
        )
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готово до роботи")
        self.status_bar = ttk.Label(
            status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))
        
        # Додатковий інформаційний бар
        self.info_var = tk.StringVar()
        self.info_var.set("Оберіть вкладку для роботи")
        self.info_bar = ttk.Label(
            status_frame, textvariable=self.info_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.info_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Оновлення статус бару"""
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
            self.logger.info(f"Статус: {message}")
            self.root.update_idletasks()
    
    def update_progress(self, value):
        """Оновлення прогрес бару"""
        if hasattr(self, 'progress_var'):
            self.progress_var.set(value)
            self.root.update_idletasks()
    
    def update_info(self, message):
        """Оновлення інформаційного бару"""
        if hasattr(self, 'info_var'):
            self.info_var.set(message)
            self.root.update_idletasks()
    
    def on_tab_changed(self, event):
        """Обробник зміни вкладок"""
        try:
            current_tab = self.notebook.index(self.notebook.select())
            total_tabs = self.notebook.index("end")  # Отримуємо загальну кількість вкладок
            
            if current_tab >= total_tabs:
                self.logger.error(f"Невірний індекс вкладки: {current_tab}, всього вкладок: {total_tabs}")
                return
                
            # Динамічне отримання назви
            tab_text = self.notebook.tab(current_tab, "text")
            self.update_info(f"Активна вкладка: {tab_text}")
            self.update_status("Готово до роботи")
            self.update_progress(0)
            
        except Exception as e:
            self.update_info("Активна вкладка: Помилка визначення")
            self.update_status("Готово до роботи")
            self.logger.error(f"Критична помилка зміни вкладки: {str(e)}")

    def setup_styles(self):
        """Налаштування стилів для адаптивного дизайну"""
        style = ttk.Style()
        
        # Компактніші стилі
        style.configure('TFrame', padding=2)
        style.configure('TLabelFrame', padding=4)
        style.configure('TLabel', font=('Arial', 9))
        style.configure('TButton', font=('Arial', 9), padding=4)
        style.configure('TEntry', font=('Arial', 9))
        style.configure('TCombobox', font=('Arial', 9))
        
        # Стилі для маленьких елементів
        style.configure('Small.TButton', font=('Arial', 8), padding=2)
        style.configure('Compact.Treeview', font=('Arial', 8))
        style.configure('Compact.Treeview.Heading', font=('Arial', 8, 'bold'))

    def setup_live_data(self):
        """Налаштування live даних"""
        # Отримати символи з наявних даних
        data_files = FileSelector.get_all_files()
        symbols = [f.replace('_data.csv', '') for f in data_files]
        
        if symbols:
            self.live_data_manager.start_live_feed(symbols[:3])  # Перші 3 символи
    
    def on_closing(self):
        """Обробник закриття програми"""
        self.live_data_manager.stop_feed()
        # Інший код закриття...




def main():
    root = tk.Tk()
    app = CryptoAnalysisApp(root)
    
    def on_closing():
        if messagebox.askokcancel("Вихід", "Ви впевнені, що хочете вийти?"):
            app.logger.info("Завершення роботи програми")
            root.quit()  # Зупиняємо головний цикл
            root.destroy()  # Знищуємо вікно
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
