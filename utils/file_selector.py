import tkinter as tk
from tkinter import ttk, messagebox
import os
from datetime import datetime
import json
from datetime import datetime

class FileSelector:
    @staticmethod
    def ask_user_to_select_file(parent, files, title="Оберіть файл", prompt="Оберіть файл для аналізу:"):
        """
        Діалог вибору файлу зі списку
        """
        if not files:
            return None
            
        if len(files) == 1:
            return files[0]
        
        selection_window = tk.Toplevel(parent)
        selection_window.title(title)
        selection_window.geometry("500x400")
        selection_window.transient(parent)
        selection_window.grab_set()
        
        selected_file = tk.StringVar()
        
        # Заголовок
        ttk.Label(selection_window, text=prompt, font=('Arial', 12)).pack(pady=10)
        
        # Фрейм з прокруткою
        frame = ttk.Frame(selection_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview для відображення файлів
        tree = ttk.Treeview(frame, columns=('File', 'Size', 'Modified'), show='headings', height=10)
        
        tree.heading('File', text='Файл')
        tree.heading('Size', text='Розмір')
        tree.heading('Modified', text='Змінено')
        
        tree.column('File', width=250)
        tree.column('Size', width=100)
        tree.column('Modified', width=150)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Заповнюємо список файлів
        for file in files:
            file_path = os.path.join('data', file)
            size = os.path.getsize(file_path)
            modified = os.path.getmtime(file_path)
            
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')
            
            tree.insert('', 'end', values=(file, size_str, modified_str))
        
        # Обробник вибору
        def on_select(event):
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                selected_file.set(item['values'][0])
        
        tree.bind('<<TreeviewSelect>>', on_select)
        
        # Кнопки
        def on_confirm():
            if selected_file.get():
                selection_window.result = selected_file.get()
                selection_window.destroy()
            else:
                messagebox.showwarning("Увага", "Оберіть файл зі списку")
        
        def on_cancel():
            selection_window.result = None
            selection_window.destroy()
        
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Обрати", command=on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Скасувати", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Центрування
        selection_window.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - selection_window.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - selection_window.winfo_height()) // 2
        selection_window.geometry(f"+{x}+{y}")
        
        # Очікуємо вибору
        selection_window.wait_window()
        
        return getattr(selection_window, 'result', None)

    @staticmethod
    def get_all_files(directory='data', pattern='_data.csv'):
        """
        Отримання всіх файлів без сортування
        """
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith(pattern)]

    @staticmethod
    def get_sorted_files(directory='data', pattern='_data.csv'):
        """
        Отримання відсортованого списку файлів (за замовчуванням)
        """
        files = FileSelector.get_all_files(directory, pattern)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
        return files
    

    @staticmethod
    def ask_user_to_select_multiple_files(parent, files, title="Оберіть файли", prompt="Оберіть файли для аналізу:"):
        """
        Діалог вибору кількох файлів
        """
        if not files:
            return []
        
        selection_window = tk.Toplevel(parent)
        selection_window.title(title)
        selection_window.geometry("600x500")
        selection_window.transient(parent)
        selection_window.grab_set()
        
        selected_files = []  # Список обраних файлів
        
        # Заголовок
        ttk.Label(selection_window, text=prompt, font=('Arial', 12)).pack(pady=10)
        
        # Фрейм з прокруткою
        frame = ttk.Frame(selection_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview з checkboxes
        tree = ttk.Treeview(frame, columns=('Select', 'File', 'Size', 'Modified'), show='tree headings', height=15)
        
        tree.heading('Select', text='Обрати')
        tree.heading('File', text='Файл')
        tree.heading('Size', text='Розмір')
        tree.heading('Modified', text='Змінено')
        
        tree.column('Select', width=60)
        tree.column('File', width=200)
        tree.column('Size', width=100)
        tree.column('Modified', width=150)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Заповнюємо список файлів з чекбоксами
        for file in files:
            file_path = os.path.join('data', file)
            size = os.path.getsize(file_path)
            modified = os.path.getmtime(file_path)
            
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')
            
            item = tree.insert('', 'end', values=('☐', file, size_str, modified_str))
            tree.set(item, 'Select', '☐')  # Початкове значення - не обрано
        
        # Обробник кліку по колонці Select
        def on_select_click(event):
            region = tree.identify("region", event.x, event.y)
            if region == "cell":
                column = tree.identify_column(event.x)
                if column == "#1":  # Колонка Select
                    item = tree.identify_row(event.y)
                    if item:
                        current_value = tree.set(item, 'Select')
                        new_value = '☑' if current_value == '☐' else '☐'
                        tree.set(item, 'Select', new_value)
        
        tree.bind('<Button-1>', on_select_click)
        
        # Кнопки
        def on_confirm():
            selected_files.clear()
            for item in tree.get_children():
                if tree.set(item, 'Select') == '☑':
                    selected_files.append(tree.set(item, 'File'))
            
            if selected_files:
                selection_window.result = selected_files
                selection_window.destroy()
            else:
                messagebox.showwarning("Увага", "Оберіть хоча б один файл")
        
        def select_all():
            for item in tree.get_children():
                tree.set(item, 'Select', '☑')
        
        def deselect_all():
            for item in tree.get_children():
                tree.set(item, 'Select', '☐')
        
        def on_cancel():
            selection_window.result = []
            selection_window.destroy()
        
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Обрати все", command=select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Скасувати все", command=deselect_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Обрати", command=on_confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Скасувати", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Центрування
        selection_window.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - selection_window.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - selection_window.winfo_height()) // 2
        selection_window.geometry(f"+{x}+{y}")
        
        # Очікуємо вибору
        selection_window.wait_window()
        
        return getattr(selection_window, 'result', [])


    @staticmethod
    def get_model_files(directory='models'):
        """
        Отримання списку файлів моделей
        """
        if not os.path.exists(directory):
            return []
        
        model_files = []
        for file in os.listdir(directory):
            if file.endswith('_model.h5'):
                symbol = file.replace('_model.h5', '')
                model_files.append(symbol)
        
        return sorted(model_files)


    @staticmethod
    def ask_user_to_select_models_for_analysis(parent, models, title="Оберіть моделі для аналізу", prompt="Оберіть моделі для аналізу торгівлі:"):
        """
        Спеціальний діалог для вибору моделей для аналізу
        """
        if not models:
            return []
        
        selection_window = tk.Toplevel(parent)
        selection_window.title(title)
        selection_window.geometry("700x500")
        selection_window.transient(parent)
        selection_window.grab_set()
        
        selected_models = []  # Список обраних моделей
        
        # Заголовок
        ttk.Label(selection_window, text=prompt, font=('Arial', 12)).pack(pady=10)
        
        # Фрейм з прокруткою
        frame = ttk.Frame(selection_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview з checkboxes
        tree = ttk.Treeview(frame, columns=('Select', 'Model', 'Trained', 'MSE', 'Type'), 
                        show='tree headings', height=15)
        
        tree.heading('Select', text='Обрати')
        tree.heading('Model', text='Модель')
        tree.heading('Trained', text='Навчено')
        tree.heading('MSE', text='MSE')
        tree.heading('Type', text='Тип')
        
        tree.column('Select', width=60, anchor='center')
        tree.column('Model', width=120)
        tree.column('Trained', width=120)
        tree.column('MSE', width=80)
        tree.column('Type', width=100)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Заповнюємо список моделей
        for model in models:
            # Отримуємо метрики моделі
            metrics_path = f'models/{model}_metrics.json'
            trained_date = "Невідомо"
            mse_value = "Невідомо"
            model_type = "Невідомо"
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                        # Дата навчання
                        timestamp = metrics.get('timestamp')
                        if timestamp:
                            if isinstance(timestamp, str):
                                try:
                                    date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    trained_date = date_obj.strftime('%Y-%m-%d')
                                except:
                                    trained_date = timestamp
                            else:
                                trained_date = "Невідомо"
                        
                        # MSE
                        mse = metrics.get('mse')
                        if mse is not None:
                            mse_value = f"{mse:.6f}"
                        
                        # Тип навчання
                        model_type = metrics.get('training_type', 'Невідомо')
                        
                except Exception as e:
                    print(f"Помилка читання метрик {model}: {e}")
            
            item = tree.insert('', 'end', values=('☐', model, trained_date, mse_value, model_type))
            tree.set(item, 'Select', '☐')  # Початкове значення - не обрано
        
        # Обробник кліку по колонці Select
        def on_select_click(event):
            region = tree.identify("region", event.x, event.y)
            if region == "cell":
                column = tree.identify_column(event.x)
                if column == "#1":  # Колонка Select
                    item = tree.identify_row(event.y)
                    if item:
                        current_value = tree.set(item, 'Select')
                        new_value = '☑' if current_value == '☐' else '☐'
                        tree.set(item, 'Select', new_value)
        
        tree.bind('<Button-1>', on_select_click)
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        # Кнопки
        def on_analyze():
            selected_models.clear()
            for item in tree.get_children():
                if tree.set(item, 'Select') == '☑':
                    model_name = tree.set(item, 'Model')
                    selected_models.append(model_name)
            
            if selected_models:
                selection_window.result = selected_models
                selection_window.destroy()
            else:
                messagebox.showwarning("Увага", "Оберіть хоча б одну модель для аналізу")
        
        def select_all():
            for item in tree.get_children():
                tree.set(item, 'Select', '☑')
        
        def deselect_all():
            for item in tree.get_children():
                tree.set(item, 'Select', '☐')
        
        def on_cancel():
            selection_window.result = []
            selection_window.destroy()
        
        # Розміщуємо кнопки в 2 ряди
        top_button_frame = ttk.Frame(button_frame)
        top_button_frame.pack(pady=5)
        
        bottom_button_frame = ttk.Frame(button_frame)
        bottom_button_frame.pack(pady=5)
        
        ttk.Button(top_button_frame, text="Обрати все", command=select_all, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_button_frame, text="Скасувати все", command=deselect_all, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_button_frame, text="Аналізувати", command=on_analyze, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_button_frame, text="Закрити", command=on_cancel, width=15).pack(side=tk.LEFT, padx=5)
        
        # Центрування
        selection_window.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - selection_window.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - selection_window.winfo_height()) // 2
        selection_window.geometry(f"+{x}+{y}")
        
        # Очікуємо вибору
        selection_window.wait_window()
        
        return getattr(selection_window, 'result', [])



