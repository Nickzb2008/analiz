import os
import json
import pickle
import tensorflow as tf
from datetime import datetime
import numpy as np  # Додайте цей імпорт

class ModelManager:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, symbol, model, scaler, metrics):
        """Збереження моделі та метрик з конвертацією NumPy типів"""
        try:
            # Конвертуємо метрики в JSON-сумісний формат
            serializable_metrics = self.convert_to_serializable(metrics)
            
            # Збереження моделі
            model_path = os.path.join(self.models_dir, f'{symbol}_model.h5')
            
            # Зберігаємо тільки ваги, щоб уникнути проблем з метриками
            model.save(model_path, save_format='h5', include_optimizer=True)
            
            # Збереження scaler (якщо є)
            if scaler:
                scaler_path = os.path.join(self.models_dir, f'{symbol}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Збереження метрик
            serializable_metrics['timestamp'] = datetime.now().isoformat()
            metrics_path = os.path.join(self.models_dir, f'{symbol}_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
            # Оновлюємо кеш
            self.models[symbol] = model
            self.metrics[symbol] = serializable_metrics
            
            return True
        except Exception as e:
            print(f"Помилка збереження моделі {symbol}: {e}")
            return False

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

    def load_model(self, symbol):
        """Завантаження моделі з обробкою помилок метрик"""
        try:
            model_path = os.path.join(self.models_dir, f'{symbol}_model.h5')
            if not os.path.exists(model_path):
                return False
            
            # Завантажуємо модель з ігноруванням помилок метрик
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.losses.MeanAbsoluteError(),
                },
                compile=False  # Не компілюємо автоматично
            )
            
            # Компілюємо модель вручну з правильними метриками
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            self.models[symbol] = model
            
            # Завантаження метрик
            metrics_path = os.path.join(self.models_dir, f'{symbol}_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.metrics[symbol] = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"Помилка завантаження моделі {symbol}: {e}")
            return False
    
    def load_model_safe(self, symbol):
        """Безпечне завантаження моделі для прогнозу (без метрик)"""
        try:
            model_path = os.path.join(self.models_dir, f'{symbol}_model.h5')
            if not os.path.exists(model_path):
                return None
            
            # Завантажуємо тільки архітектуру та ваги
            model = tf.keras.models.load_model(
                model_path,
                compile=False  # Не компілюємо - не потрібно для прогнозу
            )
            
            return model
            
        except Exception as e:
            print(f"Помилка безпечного завантаження моделі {symbol}: {e}")
            return None
    
    def load_all_models(self):
        """Завантаження всіх доступних моделей"""
        available_models = self.get_available_models()
        loaded_count = 0
        for symbol in available_models:
            if self.load_model(symbol):
                loaded_count += 1
        return loaded_count
    
    def get_available_models(self):
        """Отримання списку доступних моделей"""
        models = []
        if not os.path.exists(self.models_dir):
            return models
        
        try:
            # Шукаємо файли моделей .h5
            for file in os.listdir(self.models_dir):
                if file.endswith('_model.h5'):
                    symbol = file.replace('_model.h5', '')
                    models.append(symbol)
            return sorted(models)
        except Exception as e:
            print(f"Помилка читання папки models: {e}")
            return []

    def get_model_metrics(self, symbol):
        """Отримання метрик моделі"""
        if symbol in self.metrics:
            return self.metrics[symbol]
        
        # Спробуємо завантажити метрики з файлу
        try:
            metrics_path = os.path.join(self.models_dir, f'{symbol}_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    self.metrics[symbol] = metrics
                    return metrics
        except:
            pass
        
        return {}
    
    def delete_model(self, symbol):
        """Видалення моделі"""
        try:
            files_to_delete = [
                os.path.join(self.models_dir, f'{symbol}_model.h5'),
                os.path.join(self.models_dir, f'{symbol}_scaler.pkl'),
                os.path.join(self.models_dir, f'{symbol}_metrics.json')
            ]
            
            deleted_count = 0
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            
            # Видаляємо з кешу
            if symbol in self.models:
                del self.models[symbol]
            if symbol in self.metrics:
                del self.metrics[symbol]
            
            return deleted_count > 0
        except Exception as e:
            print(f"Помилка видалення моделі {symbol}: {e}")
            return False
    
    def get_models_info(self):
        """Отримання інформації про всі моделі"""
        info = []
        available_models = self.get_available_models()
        
        for symbol in available_models:
            metrics = self.get_model_metrics(symbol)
            info.append({
                'symbol': symbol,
                'mse': metrics.get('mse', 0),
                'mae': metrics.get('mae', 0),
                'r2': metrics.get('r2', 0),
                'timestamp': metrics.get('timestamp', 'Невідомо'),
                'training_type': metrics.get('training_type', 'Невідомо'),
                'feature_count': metrics.get('feature_count', 'Невідомо')
            })
        
        return info
    
    def cleanup_old_models(self, days=30):
        """Очищення старих моделей"""
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            deleted_count = 0
            
            for symbol in self.get_available_models():
                metrics_path = os.path.join(self.models_dir, f'{symbol}_metrics.json')
                if os.path.exists(metrics_path):
                    file_time = os.path.getmtime(metrics_path)
                    if file_time < cutoff_time:
                        if self.delete_model(symbol):
                            deleted_count += 1
            
            return deleted_count
        except Exception as e:
            print(f"Помилка очищення старих моделей: {e}")
            return 0