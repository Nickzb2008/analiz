import logging
import os
from datetime import datetime

def setup_logger():
    """Налаштування системи логування"""
    # Створення папки для логів
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Форматування часу для імені файлу
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/app_{timestamp}.log'
    
    # Налаштування логгера
    logger = logging.getLogger('CryptoAnalysisApp')
    logger.setLevel(logging.DEBUG)
    
    # Форматер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Додавання обробників
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_error(error_message, error_details=None):
    """Логування помилок з деталями"""
    logger = logging.getLogger('CryptoAnalysisApp')
    if error_details:
        logger.error(f"{error_message}\nДеталі: {error_details}")
    else:
        logger.error(error_message)

