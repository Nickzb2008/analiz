import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
from utils.trading_engine import TradingEngine

class RealTradingAnalysisTab:
    def __init__(self, parent, status_callback, progress_callback):
        self.parent = parent
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.model_manager = ModelManager()
        self.trading_engine = TradingEngine()
        self.current_symbol = None
        self.analysis_results = {}
        self.multi_analysis_results = {}
        self.setup_ui()
        self.refresh_models()
    
    def setup_ui(self):
        """Налаштування інтерфейсу аналізу реальної торгівлі"""
        # Основні фрейми
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Лівий фрейм - вибір моделі та параметрів
        left_frame = ttk.LabelFrame(main_frame, text="Параметри торгівлі", width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        # Центральний фрейм - результати аналізу з прокруткою
        center_frame = ttk.LabelFrame(main_frame, text="Результати аналізу")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правий фрейм - керування
        right_frame = ttk.LabelFrame(main_frame, text="Керування", width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        # Вибір моделі
        ttk.Label(left_frame, text="Оберіть модель:").pack(pady=5)
        
        self.models_combobox = ttk.Combobox(left_frame, state='readonly')
        self.models_combobox.pack(pady=5, fill=tk.X, padx=5)
        self.models_combobox.bind('<<ComboboxSelected>>', self.on_model_select)
        
        # Параметри торгівлі
        ttk.Label(left_frame, text="Початковий капітал ($):").pack(pady=5)
        self.initial_capital_var = tk.DoubleVar(value=10000.0)
        ttk.Entry(left_frame, textvariable=self.initial_capital_var).pack(pady=2, fill=tk.X, padx=5)
        
        ttk.Label(left_frame, text="Ризик на угоду (%):").pack(pady=5)
        self.risk_per_trade_var = tk.DoubleVar(value=2.0)
        ttk.Entry(left_frame, textvariable=self.risk_per_trade_var).pack(pady=2, fill=tk.X, padx=5)
        
        ttk.Label(left_frame, text="Стратегія:").pack(pady=5)
        self.strategy_var = tk.StringVar(value="trend_following")
        strategies = [("Слідування за трендом", "trend_following"), 
                     ("Контртрендова", "counter_trend"),
                     ("Комбінована", "combined")]
        for text, value in strategies:
            ttk.Radiobutton(left_frame, text=text, variable=self.strategy_var, value=value).pack(anchor=tk.W)
        
        ttk.Label(left_frame, text="Горизонт прогнозу (днів):").pack(pady=5)
        self.forecast_horizon_var = tk.IntVar(value=10)
        ttk.Entry(left_frame, textvariable=self.forecast_horizon_var).pack(pady=2, fill=tk.X, padx=5)
        
        # Кнопки управління
        ttk.Button(left_frame, text="Оновити моделі", command=self.refresh_models).pack(pady=10, fill=tk.X, padx=5)
        ttk.Button(left_frame, text="Запустити аналіз", command=self.run_analysis).pack(pady=5, fill=tk.X, padx=5)
        ttk.Button(left_frame, text="Аналізувати моделі", command=self.analyze_multiple_models).pack(pady=5, fill=tk.X, padx=5)
        ttk.Button(left_frame, text="Очистити результати", command=self.clear_results).pack(pady=5, fill=tk.X, padx=5)
        
        # Центральна область з прокруткою для результатів
        self.canvas = tk.Canvas(center_frame)
        scrollbar = ttk.Scrollbar(center_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Додаємо прокрутку мишею
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", 
            lambda event: self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Права панель - кнопки експорту
        ttk.Button(right_frame, text="Експорт звіту", command=self.export_report).pack(pady=10, fill=tk.X, padx=5)
        ttk.Button(right_frame, text="Експорт всіх результатів", command=self.export_all_results).pack(pady=5, fill=tk.X, padx=5)
        ttk.Button(right_frame, text="Експорт графіків", command=self.export_all_charts).pack(pady=5, fill=tk.X, padx=5)
        
        # Статусна інформація
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(status_frame, text="Проаналізовано:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.models_count_var = tk.StringVar(value="0 моделей")
        ttk.Label(status_frame, textvariable=self.models_count_var).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Середня оцінка:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.avg_score_var = tk.StringVar(value="0/10")
        ttk.Label(status_frame, textvariable=self.avg_score_var).pack(anchor=tk.W)
        
        ttk.Label(status_frame, text="Найкраща модель:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.best_model_var = tk.StringVar(value="Немає")
        ttk.Label(status_frame, textvariable=self.best_model_var).pack(anchor=tk.W)
    
    def clear_results(self):
        """Очистити результати аналізу"""
        if messagebox.askyesno("Підтвердження", "Очистити всі результати аналізу?"):
            # Видаляємо всі віджети з прокручуваного фрейму
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            self.multi_analysis_results.clear()
            self.analysis_results.clear()
            self.current_symbol = None
            
            # Оновлюємо статусну інформацію
            self.models_count_var.set("0 моделей")
            self.avg_score_var.set("0/10")
            self.best_model_var.set("Немає")
            
            self.status_callback("Результати очищено")
    
    def display_single_analysis(self, symbol, results):
        """Відображення аналізу для однієї моделі"""
        # Очищаємо попередні результати
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Створюємо фрейм для цієї моделі
        model_frame = ttk.LabelFrame(self.scrollable_frame, text=f"Аналіз: {symbol}")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ліва частина - детальна інформація
        info_frame = ttk.Frame(model_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Права частина - графік
        chart_frame = ttk.Frame(model_frame)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Детальна інформація
        detail_text = tk.Text(info_frame, width=50, height=20, wrap=tk.WORD)
        detail_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=detail_text.yview)
        detail_text.configure(yscrollcommand=detail_scrollbar.set)
        
        detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Заповнюємо інформацію
        report = self.generate_detailed_report(symbol, results)
        detail_text.insert(tk.END, report)
        detail_text.config(state=tk.DISABLED)
        
        # Створюємо графік
        self.create_model_chart(chart_frame, symbol, results)
    
    def display_multi_analysis(self):
        """Відображення аналізу для декількох моделей"""
        # Очищаємо попередні результати
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Відображаємо кожну модель
        for symbol, results in self.multi_analysis_results.items():
            # Створюємо фрейм для кожної моделі
            model_frame = ttk.LabelFrame(self.scrollable_frame, text=f"Аналіз: {symbol}")
            model_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Ліва частина - детальна інформація
            info_frame = ttk.Frame(model_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Права частина - графік
            chart_frame = ttk.Frame(model_frame)
            chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
            
            # Детальна інформація
            detail_text = tk.Text(info_frame, width=50, height=15, wrap=tk.WORD)
            detail_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=detail_text.yview)
            detail_text.configure(yscrollcommand=detail_scrollbar.set)
            
            detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Заповнюємо інформацію
            report = self.generate_detailed_report(symbol, results)
            detail_text.insert(tk.END, report)
            detail_text.config(state=tk.DISABLED)
            
            # Створюємо графік
            self.create_model_chart(chart_frame, symbol, results)
        
        # Оновлюємо статусну інформацію
        self.update_status_info()
    
    def create_model_chart(self, parent, symbol, results):
        """Створення графіка для моделі"""
        try:
            # Завантаження даних
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                return
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Створюємо графік
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Останні 60 днів
            recent_data = data[-60:]
            ax.plot(recent_data.index, recent_data['Close'], 
                   label='Ціна закриття', color='blue', linewidth=2)
            
            # Додаємо середні ковзні
            if len(recent_data) > 20:
                recent_data['MA20'] = recent_data['Close'].rolling(window=20).mean()
                ax.plot(recent_data.index, recent_data['MA20'], label='MA20', linewidth=1, color='orange', alpha=0.7)
            
            # Додаємо критичні рівні, якщо є
            current_price = recent_data['Close'].iloc[-1]
            if 'stop_loss_price' in results:
                ax.axhline(y=results['stop_loss_price'], color='red', 
                          linestyle=':', label='Стоп-лосс', alpha=0.7)
            if 'take_profit_price' in results:
                ax.axhline(y=results['take_profit_price'], color='green',
                          linestyle=':', label='Тейк-профіт', alpha=0.7)
            
            ax.set_title(f'{symbol} - Торговий аналіз', fontsize=12, fontweight='bold')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Ціна (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматування дат
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=7))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            
            # Вставляємо графік у frame
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Помилка створення графіка {symbol}: {e}")
    
    def update_status_info(self):
        """Оновлення статусної інформації"""
        if not self.multi_analysis_results:
            return
        
        # Розраховуємо статистику
        total_models = len(self.multi_analysis_results)
        scores = [r.get('opportunity_score', 0) for r in self.multi_analysis_results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Знаходимо найкращу модель
        best_model = max(self.multi_analysis_results.items(), 
                        key=lambda x: x[1].get('opportunity_score', 0), 
                        default=(None, None))
        
        self.models_count_var.set(f"{total_models} моделей")
        self.avg_score_var.set(f"{avg_score:.1f}/10")
        if best_model[0]:
            best_score = best_model[1].get('opportunity_score', 0)
            self.best_model_var.set(f"{best_model[0]} ({best_score}/10)")
    
    def analyze_multiple_models(self):
        """Аналіз декількох моделей"""
        available_models = self.model_manager.get_available_models()
        if not available_models:
            messagebox.showwarning("Увага", "Немає навчених моделей")
            return
        
        # Використовуємо діалог для вибору моделей
        selected_models = FileSelector.ask_user_to_select_models_for_analysis(
            self.parent,
            available_models,
            title="Оберіть моделі для аналізу",
            prompt="Оберіть моделі для аналізу торгівлі:"
        )
        
        if selected_models:
            self.run_multi_analysis(selected_models)
    
    def run_multi_analysis(self, symbols):
        """Запуск аналізу для декількох моделей"""
        def analysis_thread():
            try:
                total_models = len(symbols)
                self.status_callback(f"Аналіз {total_models} моделей...")
                self.progress_callback(10)
                
                successful_analyses = 0
                
                for i, symbol in enumerate(symbols):
                    try:
                        self.status_callback(f"Аналіз {symbol} ({i+1}/{total_models})...")
                        self.progress_callback(10 + (i / total_models) * 80)
                        
                        # Виконуємо аналіз для кожної моделі
                        result = self.analyze_single_model(symbol)
                        if result:
                            self.multi_analysis_results[symbol] = result
                            successful_analyses += 1
                            
                    except Exception as e:
                        self.status_callback(f"Помилка аналізу {symbol}: {str(e)}")
                        continue
                
                # Відображаємо результати
                if successful_analyses > 0:
                    self.parent.after(0, self.display_multi_analysis)
                    self.status_callback(f"Аналіз завершено. Успішно: {successful_analyses}/{total_models}")
                else:
                    self.status_callback("Не вдалося проаналізувати жодну модель")
                
                self.progress_callback(100)
                
            except Exception as e:
                self.status_callback(f"Помилка аналізу: {str(e)}")
                self.progress_callback(0)
                messagebox.showerror("Помилка", f"Помилка аналізу: {str(e)}")
        
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Запуск аналізу для обраної моделі"""
        if not self.current_symbol:
            messagebox.showwarning("Увага", "Оберіть модель для аналізу")
            return
        
        def analysis_thread():
            try:
                self.status_callback(f"Аналіз {self.current_symbol}...")
                self.progress_callback(30)
                
                # Виконуємо аналіз
                result = self.analyze_single_model(self.current_symbol)
                if result:
                    self.analysis_results[self.current_symbol] = result
                    
                    # Відображаємо результати
                    self.parent.after(0, lambda: self.display_single_analysis(self.current_symbol, result))
                    
                    self.status_callback(f"Аналіз {self.current_symbol} завершено")
                    self.progress_callback(100)
                else:
                    self.status_callback(f"Помилка аналізу {self.current_symbol}")
                    self.progress_callback(0)
                    
            except Exception as e:
                self.status_callback(f"Помилка аналізу: {str(e)}")
                self.progress_callback(0)
                messagebox.showerror("Помилка", f"Помилка аналізу: {str(e)}")
        
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()
    
    def analyze_single_model(self, symbol):
        """Аналіз однієї моделі"""
        try:
            # Завантаження моделі
            model = self.model_manager.load_model_safe(symbol)
            if model is None:
                self.status_callback(f"Модель {symbol} не знайдена")
                return None
            
            # Завантаження даних
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                self.status_callback(f"Дані для {symbol} не знайдені")
                return None
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Параметри торгівлі
            initial_capital = self.initial_capital_var.get()
            risk_per_trade = self.risk_per_trade_var.get() / 100.0
            strategy = self.strategy_var.get()
            horizon = self.forecast_horizon_var.get()
            
            # Виконання аналізу
            results = self.trading_engine.analyze_trading_opportunity(
                data, model, initial_capital, risk_per_trade, 
                strategy, horizon
            )
            
            return results
            
        except Exception as e:
            self.status_callback(f"Помилка аналізу {symbol}: {str(e)}")
            return None
    
    def generate_detailed_report(self, symbol, results):
        """Генерація детального звіту для моделі"""
        report = f"""ДЕТАЛЬНИЙ АНАЛІЗ: {symbol}
=======================

ОСНОВНІ ПОКАЗНИКИ:
------------------
Початковий капітал: ${results.get('initial_capital', 0):.2f}
Очікуваний прибуток: ${results.get('expected_profit', 0):.2f}
Максимальний ризик: ${results.get('max_risk', 0):.2f}
Відношення прибутку до ризику: {results.get('profit_risk_ratio', 0):.2f}
Вірогідність успіху: {results.get('success_probability', 0):.1%}

ТОРГІВЕЛЬНІ СИГНАЛИ:
-------------------
Сигнал: {results.get('trade_signal', 'HOLD')}
Сила сигналу: {results.get('signal_strength', 0)}/10
Рекомендований розмір позиції: {results.get('position_size', 0):.2f}%

РИЗИКИ:
------
Рівень ризику: {results.get('risk_level', 'high')}
Основні ризики: {', '.join(results.get('key_risks', []))}
Волатильність: {results.get('volatility', 0):.2%}

РЕКОМЕНДОВАНІ ДІЇ:
-----------------
{results.get('recommended_actions', 'Немає рекомендацій')}

Попередження: {results.get('warnings', 'Немає попереджень')}

КОНТРОЛЬНІ ТОЧКИ:
-----------------
Стоп-лос: {results.get('stop_loss_pct', 0):.1%}
Тейк-профіт: {results.get('take_profit_pct', 0):.1%}
Максимальний ризик: {results.get('max_risk_pct', 0):.1%}

ПЕРСПЕКТИВИ:
-----------
Короткострокові: {results.get('short_term_outlook', 'Невідомо')}
Середньострокові: {results.get('medium_term_outlook', 'Невідомо')}

ОЦІНКА МОЖЛИВОСТІ: {results.get('opportunity_score', 0)}/10

СТАТУС: {'✅ ВИСОКА ЯКІСТЬ' if results.get('opportunity_score', 0) >= 7 else 
         '⚠️  СЕРЕДНЯ ЯКІСТЬ' if results.get('opportunity_score', 0) >= 5 else 
         '❌ НИЗЬКА ЯКІСТЬ'}
"""
        return report
    
    def export_all_charts(self):
        """Експорт всіх графіків"""
        if not self.multi_analysis_results:
            messagebox.showwarning("Увага", "Немає результатів для експорту")
            return
        
        try:
            # Створюємо папку для експорту
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = f'trading_charts_export_{timestamp}'
            os.makedirs(export_dir, exist_ok=True)
            
            # Експортуємо графіки для кожної моделі
            saved_count = 0
            for symbol in self.multi_analysis_results.keys():
                if self.save_model_chart(symbol, export_dir):
                    saved_count += 1
            
            self.status_callback(f"Збережено {saved_count} графіків у папку: {export_dir}")
            messagebox.showinfo("Успіх", f"Збережено {saved_count} графіків у папку:\n{export_dir}")
            
        except Exception as e:
            self.status_callback(f"Помилка експорту графіків: {str(e)}")
            messagebox.showerror("Помилка", f"Помилка експорту графіків: {str(e)}")

    def save_model_chart(self, symbol, export_dir):
        """Збереження графіка для моделі"""
        try:
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                return False
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            results = self.multi_analysis_results.get(symbol, {})
            
            # Створюємо графік
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Останні 60 днів
            recent_data = data[-60:]
            ax.plot(recent_data.index, recent_data['Close'], 
                label='Ціна закриття', color='blue', linewidth=2)
            
            # Додаємо середні ковзні
            if len(recent_data) > 20:
                recent_data['MA20'] = recent_data['Close'].rolling(window=20).mean()
                ax.plot(recent_data.index, recent_data['MA20'], label='MA20', linewidth=1, color='orange', alpha=0.7)
            
            # Додаємо критичні рівні
            current_price = recent_data['Close'].iloc[-1]
            if 'stop_loss_price' in results:
                ax.axhline(y=results['stop_loss_price'], color='red', 
                        linestyle=':', label='Стоп-лосс', alpha=0.7, linewidth=2)
            if 'take_profit_price' in results:
                ax.axhline(y=results['take_profit_price'], color='green',
                        linestyle=':', label='Тейк-профіт', alpha=0.7, linewidth=2)
            
            # Додаємо заголовок та легенду
            ax.set_title(f'{symbol} - Торговий аналіз', fontsize=16, fontweight='bold')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Ціна (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматування дат
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=7))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Додаємо сітку
            ax.grid(True, alpha=0.3)
            
            # Додаємо інформацію про сигнал
            signal = results.get('trade_signal', 'HOLD')
            score = results.get('opportunity_score', 0)
            signal_text = f"Сигнал: {signal}, Оцінка: {score}/10"
            ax.text(0.02, 0.98, signal_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
            
            # Зберігаємо графік
            chart_path = os.path.join(export_dir, f'{symbol}_trading_analysis.png')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            print(f"Помилка збереження графіка {symbol}: {e}")
            return False
    
    
    
    
    
    
    
    
    def on_analyzed_model_select(self, event):
        """Обробник вибору моделі зі списку проаналізованих"""
        selected_indices = self.analyzed_models_listbox.curselection()
        if not selected_indices:
            return
        
        selected_model = self.analyzed_models_listbox.get(selected_indices[0])
        results = self.multi_analysis_results.get(selected_model)
        
        if results:
            self.current_analysis_view = "multi"
            self.display_detailed_analysis(selected_model, results)
            
            # Оновлюємо графік
            file_path = f'data/{selected_model}_data.csv'
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                self.update_trading_chart(data, results)
    
    def display_detailed_analysis(self, symbol, results):
        """Відображення детального аналізу для обраної моделі"""
        detailed_text = self.generate_detailed_report(symbol, results)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, detailed_text)
        
        # Оновлюємо статусну інформацію
        self.risk_status_var.set(results.get('risk_level', 'Невідомо').upper())
        self.overall_score_var.set(f"{results.get('opportunity_score', 0)}/10")
    
    def show_comparison(self):
        """Показати порівняння моделей"""
        if not self.multi_analysis_results:
            messagebox.showwarning("Увага", "Немає даних для порівняння")
            return
        
        comparison_text = self.generate_comparison_report()
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(tk.END, comparison_text)
        
        # Переключаємося на вкладку порівняння
        self.results_notebook.select(1)
    
    
    
    
    def show_model_results(self, symbol):
        """Показати результати конкретної моделі"""
        results = self.multi_analysis_results.get(symbol)
        if results:
            # Вибираємо модель у списку
            for i in range(self.analyzed_models_listbox.size()):
                if self.analyzed_models_listbox.get(i) == symbol:
                    self.analyzed_models_listbox.selection_clear(0, tk.END)
                    self.analyzed_models_listbox.selection_set(i)
                    self.analyzed_models_listbox.activate(i)
                    break
            
            # Відображаємо детальний аналіз
            self.display_detailed_analysis(symbol, results)
            
            # Оновлюємо графік
            file_path = f'data/{symbol}_data.csv'
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                self.update_trading_chart(data, results)
    
    def display_multi_analysis_results(self):
        """Відображення результатів аналізу декількох моделей"""
        # Показуємо порівняння
        self.show_comparison()
        
        # Якщо є моделі, показуємо першу
        if self.multi_analysis_results:
            first_symbol = list(self.multi_analysis_results.keys())[0]
            self.show_model_results(first_symbol)
    
    
    def generate_comparison_report(self):
        """Генерація звіту порівняння моделей"""
        if not self.multi_analysis_results:
            return "Немає даних для порівняння"
        
        report = f"""ПОРІВНЯННЯ МОДЕЛЕЙ
=================

Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Проаналізовано моделей: {len(self.multi_analysis_results)}

ТОП-МОДЕЛІ:
----------
"""
        
        # Сортуємо моделі за оцінкою
        sorted_models = sorted(self.multi_analysis_results.items(), 
                              key=lambda x: x[1].get('opportunity_score', 0), 
                              reverse=True)
        
        # Додаємо топ-5 моделей
        for i, (symbol, results) in enumerate(sorted_models[:5], 1):
            score = results.get('opportunity_score', 0)
            signal = results.get('trade_signal', 'HOLD')
            risk = results.get('risk_level', 'high')
            profit = results.get('expected_profit', 0)
            
            report += f"{i}. {symbol}: {score}/10\n"
            report += f"   Сигнал: {signal}, Ризик: {risk.upper()}\n"
            report += f"   Очікуваний прибуток: ${profit:.2f}\n"
            report += f"   Відношення прибутку/ризику: {results.get('profit_risk_ratio', 0):.2f}\n"
            report += "-" * 40 + "\n"
        
        # Загальна статистика
        scores = [r.get('opportunity_score', 0) for r in self.multi_analysis_results.values()]
        signals = [r.get('trade_signal', 'HOLD') for r in self.multi_analysis_results.values()]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        hold_count = signals.count('HOLD')
        
        report += f"\nЗАГАЛЬНА СТАТИСТИКА:\n"
        report += f"Середня оцінка: {avg_score:.2f}/10\n"
        report += f"Купівельних сигналів: {buy_count}\n"
        report += f"Продажних сигналів: {sell_count}\n"
        report += f"Нейтральних сигналів: {hold_count}\n"
        
        # Рекомендації
        report += f"\nРЕКОМЕНДАЦІЇ:\n"
        if avg_score >= 7:
            report += "📈 Загальний ринок виглядає сприятливим для торгівлі\n"
            best_model = sorted_models[0][0] if sorted_models else "N/A"
            report += f"🎯 Найкраща модель: {best_model}\n"
        elif avg_score >= 5:
            report += "⚠️  Обережність рекомендована - вибирайте моделі з оцінкою ≥7/10\n"
        else:
            report += "📉 Утримання від торгівлі рекомендоване\n"
        
        return report
        
    def analyze_single_model(self, symbol):
        """Аналіз однієї моделі"""
        try:
            # Завантаження моделі
            if not self.model_manager.load_model(symbol):
                self.status_callback(f"Модель {symbol} не знайдена")
                return None
            
            model = self.model_manager.models[symbol]
            metrics = self.model_manager.get_model_metrics(symbol)
            
            # Завантаження даних
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                self.status_callback(f"Дані для {symbol} не знайдені")
                return None
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Параметри торгівлі
            initial_capital = self.initial_capital_var.get()
            risk_per_trade = self.risk_per_trade_var.get() / 100.0
            strategy = self.strategy_var.get()
            horizon = self.forecast_horizon_var.get()
            
            # Виконання аналізу
            results = self.trading_engine.analyze_trading_opportunity(
                data, model, initial_capital, risk_per_trade, 
                strategy, horizon
            )
            
            return results
            
        except Exception as e:
            self.status_callback(f"Помилка аналізу {symbol}: {str(e)}")
            return None
        
    def export_all_results(self):
        """Експорт всіх результатів аналізу"""
        if not self.multi_analysis_results:
            messagebox.showwarning("Увага", "Немає результатів для експорту")
            return
        
        try:
            # Створюємо папку для експорту
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = f'trading_analysis_export_{timestamp}'
            os.makedirs(export_dir, exist_ok=True)
            
            # Експортуємо дані для кожної моделі
            for symbol, results in self.multi_analysis_results.items():
                # Зберігаємо текстові дані
                report_content = self.generate_model_report(symbol, results)
                report_path = os.path.join(export_dir, f'{symbol}_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                # Зберігаємо графік
                self.save_model_chart(symbol, export_dir)
            
            # Зберігаємо зведений звіт
            summary_content = self.generate_summary_report()
            summary_path = os.path.join(export_dir, 'summary_report.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            self.status_callback(f"Всі результати збережено у папку: {export_dir}")
            messagebox.showinfo("Успіх", f"Всі результати збережено у папку:\n{export_dir}")
            
        except Exception as e:
            self.status_callback(f"Помилка експорту: {str(e)}")
            messagebox.showerror("Помилка", f"Помилка експорту: {str(e)}")
    
    def generate_model_report(self, symbol, results):
        """Генерація звіту для окремої моделі"""
        report = f"""ЗВІТ ТОРГІВЕЛЬНОГО АНАЛІЗУ: {symbol}
=======================

Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ОСНОВНІ ПОКАЗНИКИ:
------------------
Початковий капітал: ${results.get('initial_capital', 0):.2f}
Очікуваний прибуток: ${results.get('expected_profit', 0):.2f}
Максимальний ризик: ${results.get('max_risk', 0):.2f}
Відношення прибутку до ризику: {results.get('profit_risk_ratio', 0):.2f}
Вірогідність успіху: {results.get('success_probability', 0):.1%}

ТОРГІВЕЛЬНІ СИГНАЛИ:
-------------------
Сигнал: {results.get('trade_signal', 'HOLD')}
Сила сигналу: {results.get('signal_strength', 0)}/10
Рекомендований розмір позиції: {results.get('position_size', 0):.2f}%

РИЗИКИ:
------
Рівень ризику: {results.get('risk_level', 'high')}
Основні ризики: {', '.join(results.get('key_risks', []))}
Волатильність: {results.get('volatility', 0):.2%}

РЕКОМЕНДОВАНІ ДІЇ:
-----------------
{results.get('recommended_actions', 'Немає рекомендацій')}

Попередження: {results.get('warnings', 'Немає попереджень')}

КОНТРОЛЬНІ ТОЧКИ:
-----------------
Стоп-лос: {results.get('stop_loss_pct', 0):.1%}
Тейк-профіт: {results.get('take_profit_pct', 0):.1%}
Максимальний ризик: {results.get('max_risk_pct', 0):.1%}

ПЕРСПЕКТИВИ:
-----------
Короткострокові: {results.get('short_term_outlook', 'Невідомо')}
Середньострокові: {results.get('medium_term_outlook', 'Невідомо')}

ОЦІНКА МОЖЛИВОСТІ: {results.get('opportunity_score', 0)}/10
"""
        return report
    
    def generate_summary_report(self):
        """Генерація зведеного звіту"""
        if not self.multi_analysis_results:
            return "Немає даних для зведеного звіту"
        
        # Аналізуємо результати
        models = list(self.multi_analysis_results.keys())
        scores = [r.get('opportunity_score', 0) for r in self.multi_analysis_results.values()]
        signals = [r.get('trade_signal', 'HOLD') for r in self.multi_analysis_results.values()]
        
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        hold_signals = signals.count('HOLD')
        
        avg_score = sum(scores) / len(scores) if scores else 0
        best_model = max(self.multi_analysis_results.items(), 
                        key=lambda x: x[1].get('opportunity_score', 0), 
                        default=(None, None))
        
        report = f"""ЗВЕДЕНИЙ ЗВІТ ТОРГІВЕЛЬНОГО АНАЛІЗУ
===============================

Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Проаналізовано моделей: {len(models)}

ЗАГАЛЬНА СТАТИСТИКА:
-------------------
Середня оцінка: {avg_score:.2f}/10
Купівельних сигналів: {buy_signals}
Продажних сигналів: {sell_signals}
Нейтральних сигналів: {hold_signals}

НАЙКРАЩІ МОДЕЛІ:
---------------
"""
        
        # Додаємо топ-3 моделі
        sorted_models = sorted(self.multi_analysis_results.items(), 
                              key=lambda x: x[1].get('opportunity_score', 0), 
                              reverse=True)[:3]
        
        for i, (symbol, results) in enumerate(sorted_models, 1):
            score = results.get('opportunity_score', 0)
            signal = results.get('trade_signal', 'HOLD')
            report += f"{i}. {symbol}: {score}/10 ({signal})\n"
        
        if best_model[0]:
            report += f"\n🎯 НАЙКРАЩА МОДЕЛЬ: {best_model[0]}\n"
            report += f"   Оцінка: {best_model[1].get('opportunity_score', 0)}/10\n"
            report += f"   Сигнал: {best_model[1].get('trade_signal', 'HOLD')}\n"
            report += f"   Очікуваний прибуток: ${best_model[1].get('expected_profit', 0):.2f}\n"
        
        report += f"\nРЕКОМЕНДАЦІЇ:\n"
        if avg_score >= 7:
            report += "📈 Загальний ринок виглядає сприятливим для торгівлі\n"
        elif avg_score >= 5:
            report += "⚠️  Рисковий ринок - обережність рекомендована\n"
        else:
            report += "📉 Рисковий ринок - утримання від торгівлі рекомендоване\n"
        
        return report
    
    def save_model_chart(self, symbol, export_dir):
        """Збереження графіка для моделі"""
        try:
            file_path = f'data/{symbol}_data.csv'
            if not os.path.exists(file_path):
                return
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            results = self.multi_analysis_results.get(symbol, {})
            
            # Створюємо графік
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Останні 60 днів
            recent_data = data[-60:]
            ax.plot(recent_data.index, recent_data['Close'], 
                   label='Ціна закриття', color='blue', linewidth=2)
            
            # Додаємо заголовок та легенду
            ax.set_title(f'Торговий аналіз {symbol}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Ціна (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Форматування дат
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Зберігаємо графік
            chart_path = os.path.join(export_dir, f'{symbol}_chart.png')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Помилка збереження графіка {symbol}: {e}")
        
    def refresh_models(self):
        """Оновлення списку доступних моделей"""
        self.model_manager.load_all_models()
        available_models = self.model_manager.get_available_models()
        
        self.models_combobox['values'] = available_models
        if available_models:
            self.models_combobox.set(available_models[0])
            self.current_symbol = available_models[0]
    
    def on_model_select(self, event):
        """Обробник вибору моделі"""
        self.current_symbol = self.models_combobox.get()
        self.status_callback(f"Обрано модель: {self.current_symbol}")
    
    
    def update_results_display(self, results, metrics):
        """Оновлення відображення результатів"""
        text = f"""АНАЛІЗ ТОРГІВЕЛЬНИХ МОЖЛИВОСТЕЙ
================================

ОСНОВНІ ПОКАЗНИКИ:
------------------
Початковий капітал: ${results['initial_capital']:,.2f}
Очікуваний прибуток: ${results['expected_profit']:,.2f}
Максимальний ризик: ${results['max_risk']:,.2f}
Відношення прибутку до ризику: {results['profit_risk_ratio']:.2f}
Вірогідність успіху: {results['success_probability']:.1%}

ТЕХНІЧНІ ПОКАЗНИКИ:
-------------------
Точність моделі (MSE): {metrics.get('mse', 0):.6f}
Точність моделі (MAE): {metrics.get('mae', 0):.6f}
R² моделі: {metrics.get('r2', 0):.4f}

ТОРГІВЕЛЬНІ СИГНАЛИ:
-------------------
Сигнал: {results['trade_signal']}
Сила сигналу: {results['signal_strength']}/10
Рекомендований розмір позиції: {results['position_size']:.2f}%

РИЗИКИ:
------
Рівень ризику: {results['risk_level']}
Основні ризики: {', '.join(results['key_risks'])}
Волатильність: {results['volatility']:.2%}

РЕКОМЕНДОВАНІ ДІЇ:
-----------------
{results['recommended_actions']}
"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
    
    def update_recommendations(self, results):
        """Оновлення рекомендацій"""
        recommendations = f"""РЕКОМЕНДАЦІЇ ДЛЯ ТОРГІВЛІ
===========================

СТАТУС РИЗИКУ: {results['risk_level'].upper()}

{'⚠️ ВИСОКИЙ РИЗИК ⚠️' if results['risk_level'] == 'high' else 
 '⚠️ СЕРЕДНІЙ РИЗИК ⚠️' if results['risk_level'] == 'medium' else 
 '✅ НИЗЬКИЙ РИЗИК ✅'}

ОЦІНКА МОЖЛИВОСТІ: {results['opportunity_score']}/10

ОСНОВНІ РЕКОМЕНДАЦІЇ:
{results['recommended_actions']}

ПОПЕРЕДЖЕННЯ:
{results['warnings']}

КОНТРОЛЬНІ ТОЧКИ:
- Стоп-лос: {results['stop_loss_pct']:.1%}
- Тейк-профіт: {results['take_profit_pct']:.1%}
- Макс. ризик: {results['max_risk_pct']:.1%}

ЧАСОВІ РАМКИ:
- Короткостроковий: {results['short_term_outlook']}
- Середньостроковий: {results['medium_term_outlook']}
"""
        
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, recommendations)
        
        # Оновлення статусних індикаторів
        self.risk_status_var.set(results['risk_level'].upper())
        self.overall_score_var.set(f"{results['opportunity_score']}/10")
        
        # Зміна кольору залежно від ризику
        color = "red" if results['risk_level'] == 'high' else \
               "orange" if results['risk_level'] == 'medium' else "green"
        self.risk_status_var.set(color)
    
    def update_trading_chart(self, data, results):
        """Оновлення торгового графіка"""
        self.ax.clear()
        
        # Останні 60 днів
        recent_data = data[-60:]
        
        # Ціни
        self.ax.plot(recent_data.index, recent_data['Close'], 
                   label='Ціна закриття', color='blue', linewidth=2)
        
        # Прогнозовані точки
        if 'forecast_prices' in results:
            forecast_dates = pd.date_range(
                start=recent_data.index[-1] + timedelta(days=1),
                periods=len(results['forecast_prices']),
                freq='D'
            )
            self.ax.plot(forecast_dates, results['forecast_prices'],
                       label='Прогноз', color='red', linestyle='--', linewidth=2)
        
        # Критичні рівні
        current_price = recent_data['Close'].iloc[-1]
        if 'stop_loss_price' in results:
            self.ax.axhline(y=results['stop_loss_price'], color='red', 
                          linestyle=':', label='Стоп-лосс', alpha=0.7)
        if 'take_profit_price' in results:
            self.ax.axhline(y=results['take_profit_price'], color='green',
                          linestyle=':', label='Тейк-профіт', alpha=0.7)
        
        self.ax.set_title(f'Торговий аналіз {self.current_symbol}', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Дата')
        self.ax.set_ylabel('Ціна (USD)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Форматування дат
        self.ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator())
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
        
        self.canvas.draw()
    
    def export_report(self):
        """Експорт звіту аналізу"""
        if not self.current_symbol or self.current_symbol not in self.analysis_results:
            messagebox.showwarning("Увага", "Немає результатів для експорту")
            return
        
        try:
            results = self.analysis_results[self.current_symbol]
            
            # Створення звіту
            report = f"""ЗВІТ ТОРГІВЕЛЬНОГО АНАЛІЗУ
=======================

Криптовалюта: {self.current_symbol}
Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

РЕЗУЛЬТАТИ АНАЛІЗУ:
------------------
Початковий капітал: ${results['initial_capital']:,.2f}
Очікуваний прибуток: ${results['expected_profit']:,.2f}
Максимальний ризик: ${results['max_risk']:,.2f}
Відношення прибутку до ризику: {results['profit_risk_ratio']:.2f}
Вірогідність успіху: {results['success_probability']:.1%}

ТОРГІВЕЛЬНІ СИГНАЛИ:
-------------------
Сигнал: {results['trade_signal']}
Сила сигналу: {results['signal_strength']}/10
Рекомендований розмір позиції: {results['position_size']:.2f}%

РИЗИКИ ТА РЕКОМЕНДАЦІЇ:
----------------------
Рівень ризику: {results['risk_level']}
Основні ризики: {', '.join(results['key_risks'])}
Волатильність: {results['volatility']:.2%}

Рекомендовані дії: {results['recommended_actions']}

Попередження: {results['warnings']}

КОНТРОЛЬНІ ТОЧКИ:
-----------------
Стоп-лос: {results['stop_loss_pct']:.1%}
Тейк-профіт: {results['take_profit_pct']:.1%}
Максимальний ризик: {results['max_risk_pct']:.1%}

ПЕРСПЕКТИВИ:
-----------
Короткострокові: {results['short_term_outlook']}
Середньострокові: {results['medium_term_outlook']}
"""
            
            # Збереження
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trading_report_{self.current_symbol}_{timestamp}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.status_callback(f"Звіт збережено у {filename}")
            messagebox.showinfo("Успіх", f"Звіт збережено у {filename}")
            
        except Exception as e:
            self.status_callback(f"Помилка експорту: {str(e)}")
            messagebox.showerror("Помилка", f"Помилка експорту: {str(e)}")