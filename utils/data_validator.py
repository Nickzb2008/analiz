import pandas as pd
import logging

class DataValidator:
    @staticmethod
    def check_data_requirements(data, status_callback=None):
        """
        Перевіряє, що дані містять всі необхідні колонки для аналізу
        
        Args:
            data: DataFrame з даними
            status_callback: функція для відображення статусу (опціонально)
        
        Returns:
            bool: True якщо дані відповідають вимогам
        """
        try:
            # Обов'язкові колонки
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                error_msg = f"Відсутні обов'язкові колонки: {missing_columns}"
                if status_callback:
                    status_callback(error_msg)
                raise ValueError(error_msg)
            
            # Додаткова інформація про доступність Volume
            if 'Volume' not in data.columns:
                if status_callback:
                    status_callback("Увага: Volume дані недоступні")
            
            # Перевірка наявності даних
            if data.empty:
                error_msg = "Отримано порожній DataFrame"
                if status_callback:
                    status_callback(error_msg)
                raise ValueError(error_msg)
            
            # Перевірка на NaN значення в ключових колонках
            for col in required_columns:
                if data[col].isnull().all():
                    error_msg = f"Колонка {col} містить тільки NaN значення"
                    if status_callback:
                        status_callback(error_msg)
                    raise ValueError(error_msg)
            
            if status_callback:
                status_callback("Дані відповідають вимогам для аналізу")
            
            return True
            
        except Exception as e:
            if status_callback:
                status_callback(f"Помилка перевірки даних: {str(e)}")
            raise

    @staticmethod
    def get_available_columns(data, status_callback=None):
        """
        Повертає інформацію про доступні колонки
        """
        available_columns = list(data.columns)
        if status_callback:
            status_callback(f"Доступні колонки: {', '.join(available_columns)}")
        return available_columns

    @staticmethod
    def validate_data_for_ml(data, status_callback=None):
        """
        Спеціальна перевірка для ML моделей
        """
        if status_callback:
            status_callback("Перевірка даних для ML моделей...")
        
        # Перевіряємо базові вимоги
        DataValidator.check_data_requirements(data, status_callback)
        
        # Додаткові вимоги для ML
        if len(data) < 100:
            warning_msg = "Мало даних для навчання ML моделі (мінімум 100 рядків)"
            if status_callback:
                status_callback(warning_msg)
        
        return True

    @staticmethod
    def validate_data_for_technical_analysis(data, status_callback=None):
        """
        Спеціальна перевірка для технічного аналізу
        """
        if status_callback:
            status_callback("Перевірка даних для технічного аналізу...")
        
        DataValidator.check_data_requirements(data, status_callback)
        
        # Для технічного аналізу бажано мати Volume
        if 'Volume' not in data.columns:
            if status_callback:
                status_callback("Увага: Технічний аналіз без Volume даних може бути обмеженим")
        
        return True