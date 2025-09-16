import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime

class LiveTradingTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.is_updating = False
        self.setup_ui()
        
    def setup_ui(self):
        """Налаштування інтерфейсу реальної торгівлі"""
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Live data panel
        data_frame = ttk.LabelFrame(main_frame, text="Live Data")
        data_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.data_text = tk.Text(data_frame, height=8, width=80)
        self.data_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Trading signals panel
        signals_frame = ttk.LabelFrame(main_frame, text="Trading Signals")
        signals_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.signals_text = tk.Text(signals_frame, height=6, width=80)
        self.signals_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Risk management panel  
        risk_frame = ttk.LabelFrame(main_frame, text="Risk Management")
        risk_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.risk_text = tk.Text(risk_frame, height=4, width=80)
        self.risk_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Button(button_frame, text="Start Live Monitoring", 
                  command=self.start_monitoring).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Monitoring", 
                  command=self.stop_monitoring).pack(side=tk.LEFT, padx=5)
        
        # Initial message
        self.update_data_display("Live trading monitoring is ready. Click 'Start Live Monitoring' to begin.")
    
    def start_monitoring(self):
        """Запуск моніторингу"""
        if not self.is_updating:
            self.is_updating = True
            self.update_data_display("Starting live trading monitoring...")
            thread = threading.Thread(target=self._monitoring_loop)
            thread.daemon = True
            thread.start()
    
    def stop_monitoring(self):
        """Зупинка моніторингу"""
        self.is_updating = False
        self.update_data_display("Live trading monitoring stopped.")
    
    def _monitoring_loop(self):
        """Цикл моніторингу"""
        while self.is_updating:
            try:
                # Симулюємо отримання даних
                current_time = datetime.now().strftime("%H:%M:%S")
                message = f"[{current_time}] Monitoring live data...\n"
                
                # Оновлюємо UI
                self.update_data_display(message)
                
                time.sleep(5)  # Оновлення кожні 5 секунд
                
            except Exception as e:
                self.update_data_display(f"Error in monitoring: {str(e)}")
                time.sleep(10)
    
    def update_data_display(self, message):
        """Оновлення відображення даних"""
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, message)
        
        # Оновлюємо сигнали та ризики
        self.signals_text.delete(1.0, tk.END)
        self.signals_text.insert(tk.END, "No trading signals detected yet.")
        
        self.risk_text.delete(1.0, tk.END)
        self.risk_text.insert(tk.END, "Risk management: Normal")
    
    def set_live_data_manager(self, live_data_manager):
        """Встановлення менеджера live даних"""
        self.live_data_manager = live_data_manager
    
    def set_risk_manager(self, risk_manager):
        """Встановлення менеджера ризиків"""
        self.risk_manager = risk_manager