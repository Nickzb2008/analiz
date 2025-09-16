import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from utils.risk_manager import RiskManager
from utils.trading_config import calculate_dynamic_risk_parameters, get_trading_recommendation

class TradingEngine:
    def __init__(self, risk_manager=None):
        self.risk_manager = risk_manager or RiskManager()

    def analyze_trading_opportunity(self, data, model, initial_capital=10000.0, 
                                  risk_per_trade=0.02, strategy='trend_following',
                                  forecast_horizon=10, live_data=None):
        """Аналіз з урахуванням live даних та ризику"""
        try:
            # Отримати live дані або використати останні історичні
            current_price = live_data.get('price', data['Close'].iloc[-1]) if live_data else data['Close'].iloc[-1]
            
            if current_price is None or current_price == 0:
                return self.get_default_results()
            
            # Прогноз моделі
            forecast_prices = self.generate_forecast(model, data, forecast_horizon)
            
            # Risk management
            risk_params = calculate_dynamic_risk_parameters(data)
            stop_loss_pct = risk_params['stop_loss_multiplier'] * data['Close'].pct_change().std()
            stop_loss_price = current_price * (1 - stop_loss_pct)
            
            # Position sizing
            position_size = self.risk_manager.calculate_position_size(
                current_price, stop_loss_price, risk_per_trade
            )
            
            # Trade validation
            is_valid, validation_message = self.risk_manager.validate_trade(
                "crypto", current_price, stop_loss_price, position_size
            )
            
            # Аналіз ризиків
            risk_analysis = self.analyze_risks(data, forecast_prices)
            
            # Генерація сигналів
            signals = self.generate_signals(data, forecast_prices, strategy)
            
            # Генерація рекомендації
            recommendation, recommendation_msg = get_trading_recommendation({
                'is_valid': is_valid,
                'validation_message': validation_message,
                'signals': signals,
                'risk_analysis': risk_analysis
            })
            
            # Результати
            results = {
                'initial_capital': initial_capital,
                'current_price': current_price,
                'forecast_prices': forecast_prices,
                'expected_profit': self.calculate_expected_profit(current_price, forecast_prices, position_size, initial_capital),
                'max_risk': initial_capital * risk_per_trade,
                'profit_risk_ratio': self.calculate_profit_risk_ratio(current_price, forecast_prices),
                'success_probability': self.calculate_success_probability(data, forecast_prices),
                'trade_signal': signals['signal'],
                'signal_strength': signals['strength'],
                'position_size': position_size,
                'risk_level': risk_analysis['risk_level'],
                'key_risks': risk_analysis['key_risks'],
                'volatility': risk_analysis['volatility'],
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': risk_analysis['take_profit_pct'],
                'max_risk_pct': risk_per_trade,
                'is_valid': is_valid,
                'validation_message': validation_message,
                'recommendation': recommendation,
                'recommendation_message': recommendation_msg,
                'live_data': live_data is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            print(f"Помилка аналізу торгової можливості: {e}")
            return self.get_default_results()
    
    def generate_forecast(self, model, input_data, horizon):
        """Генерація прогнозу з live даними"""
        predictions = []
        current_sequence = input_data.copy()
        
        for _ in range(horizon):
            next_pred = model.predict(current_sequence.reshape(1, 60, 1), verbose=0)
            predictions.append(next_pred[0, 0])
            
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        return np.array(predictions)    
    
    
    
    
    
    
    def analyze_risks(self, data, forecast_prices):
        """Аналіз ризиків"""
        # Розрахунок волатильності
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Річна волатильність
        
        # Визначення рівня ризику
        if volatility > 0.8:
            risk_level = 'high'
            stop_loss_pct = 0.05  # 5%
            take_profit_pct = 0.10  # 10%
        elif volatility > 0.4:
            risk_level = 'medium'
            stop_loss_pct = 0.03  # 3%
            take_profit_pct = 0.06  # 6%
        else:
            risk_level = 'low'
            stop_loss_pct = 0.02  # 2%
            take_profit_pct = 0.04  # 4%
        
        # Ключові ризики
        key_risks = []
        if volatility > 0.6:
            key_risks.append('Висока волатильність')
        if len(data) < 200:
            key_risks.append('Обмежена історія даних')
        if forecast_prices.std() / forecast_prices.mean() > 0.1:
            key_risks.append('Нестабільний прогноз')
        
        return {
            'risk_level': risk_level,
            'volatility': volatility,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'key_risks': key_risks
        }
    
    def generate_signals(self, data, forecast_prices, strategy):
        """Генерація торгових сигналів"""
        current_price = data['Close'].iloc[-1]
        forecast_change = (forecast_prices[-1] - current_price) / current_price
        
        if strategy == 'trend_following':
            # Слідування за трендом
            if forecast_change > 0.02:
                signal = 'BUY'
                strength = min(int(abs(forecast_change) * 100), 10)
            elif forecast_change < -0.02:
                signal = 'SELL'
                strength = min(int(abs(forecast_change) * 100), 10)
            else:
                signal = 'HOLD'
                strength = 0
                
        elif strategy == 'counter_trend':
            # Контртрендова стратегія
            if forecast_change > 0.05:  # Перекупленість
                signal = 'SELL'
                strength = min(int(abs(forecast_change) * 80), 10)
            elif forecast_change < -0.05:  # Перепроданість
                signal = 'BUY'
                strength = min(int(abs(forecast_change) * 80), 10)
            else:
                signal = 'HOLD'
                strength = 0
                
        else:  # combined
            # Комбінована стратегія
            ma_short = data['Close'].rolling(window=20).mean().iloc[-1]
            ma_long = data['Close'].rolling(window=50).mean().iloc[-1]
            
            if forecast_change > 0.03 and current_price > ma_short > ma_long:
                signal = 'BUY'
                strength = 8
            elif forecast_change < -0.03 and current_price < ma_short < ma_long:
                signal = 'SELL'
                strength = 8
            else:
                signal = 'HOLD'
                strength = 0
        
        return {'signal': signal, 'strength': strength}
    
    def calculate_position_size(self, capital, risk_per_trade, volatility):
        """Розрахунок розміру позиції"""
        # Базова формула: позиція = (капітал * ризик) / (волатильність * 2)
        base_size = (capital * risk_per_trade) / (volatility * 2)
        position_pct = min((base_size / capital) * 100, 20)  # Макс 20%
        return position_pct
    
    def calculate_expected_profit(self, current_price, forecast_prices, position_size, capital):
        """Розрахунок очікуваного прибутку"""
        expected_change = (forecast_prices[-1] - current_price) / current_price
        position_value = (position_size / 100) * capital
        return position_value * expected_change
    
    def calculate_profit_risk_ratio(self, current_price, forecast_prices):
        """Розрахунок співвідношення прибуток/ризик"""
        max_gain = (forecast_prices.max() - current_price) / current_price
        max_loss = (current_price - forecast_prices.min()) / current_price
        
        if max_loss == 0:
            return float('inf')
        return abs(max_gain / max_loss)
    
    def calculate_success_probability(self, data, forecast_prices):
        """Розрахунок ймовірності успіху"""
        # Спрощена оцінка на основі стабільності прогнозу
        forecast_std = forecast_prices.std() / forecast_prices.mean()
        historical_accuracy = 0.7  # Базова точність
        
        if forecast_std < 0.05:
            probability = 0.8
        elif forecast_std < 0.1:
            probability = 0.6
        else:
            probability = 0.4
        
        return probability * historical_accuracy
    
    def generate_recommendations(self, signals, risk_analysis):
        """Генерація рекомендацій"""
        recommendations = []
        
        if signals['signal'] == 'BUY':
            if risk_analysis['risk_level'] == 'low':
                recommendations.append("Рекомендується відкриття довгої позиції")
                recommendations.append(f"Стоп-лосс: -{risk_analysis['stop_loss_pct']*100:.1f}%")
                recommendations.append(f"Тейк-профіт: +{risk_analysis['take_profit_pct']*100:.1f}%")
            else:
                recommendations.append("Обережне відкриття довгої позиції")
                recommendations.append("Розглянути часткове входження")
                
        elif signals['signal'] == 'SELL':
            recommendations.append("Рекомендується закриття позиції або шорт")
            recommendations.append("Уважно моніторити ринок")
            
        else:
            recommendations.append("Залишитися в стороні")
            recommendations.append("Очікувати кращих умов для входження")
        
        if risk_analysis['risk_level'] == 'high':
            recommendations.append("❗ Обмежити розмір позиції через високий ризик")
        
        return ". ".join(recommendations)
    
    def generate_warnings(self, risk_analysis):
        """Генерація попереджень"""
        warnings = []
        
        if 'Висока волатильність' in risk_analysis['key_risks']:
            warnings.append("Висока волатильність може призвести до значних збитків")
        if 'Обмежена історія даних' in risk_analysis['key_risks']:
            warnings.append("Обмежена історія даних знижує точність прогнозу")
        if 'Нестабільний прогноз' in risk_analysis['key_risks']:
            warnings.append("Прогноз показує нестабільність - будьте обережні")
        
        return ". ".join(warnings) if warnings else "Немає критичних попереджень"
    
    def calculate_opportunity_score(self, signals, risk_analysis):
        """Розрахунок оцінки можливості"""
        score = 0
        
        # Сила сигналу
        score += signals['strength']
        
        # Рівень ризику
        if risk_analysis['risk_level'] == 'low':
            score += 3
        elif risk_analysis['risk_level'] == 'medium':
            score += 1
        else:
            score -= 2
        
        # Кількість ризиків
        risk_count = len(risk_analysis['key_risks'])
        score -= risk_count * 2
        
        return max(0, min(score, 10))
    
    def get_short_term_outlook(self, forecast_prices):
        """Короткострокові перспективи"""
        change_1d = (forecast_prices[0] - forecast_prices[1]) / forecast_prices[1] if len(forecast_prices) > 1 else 0
        
        if change_1d > 0.02:
            return "Позитивні"
        elif change_1d < -0.02:
            return "Негативні"
        else:
            return "Нейтральні"
    
    def get_medium_term_outlook(self, forecast_prices):
        """Середньострокові перспективи"""
        if len(forecast_prices) < 5:
            return "Невизначені"
        
        change_5d = (forecast_prices[4] - forecast_prices[0]) / forecast_prices[0]
        
        if change_5d > 0.05:
            return "Дуже позитивні"
        elif change_5d > 0.02:
            return "Позитивні"
        elif change_5d < -0.05:
            return "Дуже негативні"
        elif change_5d < -0.02:
            return "Негативні"
        else:
            return "Стабільні"
    
    def get_default_results(self):
        """Результати за замовчуванням при помилці"""
        return {
            'initial_capital': 10000.0,
            'current_price': 0,
            'expected_profit': 0,
            'max_risk': 200.0,
            'profit_risk_ratio': 0,
            'success_probability': 0,
            'trade_signal': 'HOLD',
            'signal_strength': 0,
            'position_size': 0,
            'risk_level': 'high',
            'key_risks': ['Помилка аналізу'],
            'volatility': 0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_risk_pct': 0.02,
            'recommended_actions': 'Очікувати покращення умов',
            'warnings': 'Помилка аналізу - будьте обережні',
            'opportunity_score': 0,
            'short_term_outlook': 'Невизначені',
            'medium_term_outlook': 'Невизначені'
        }
    

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


