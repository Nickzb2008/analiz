TRADING_CONFIG = {
    'risk_management': {
        'max_position_size_pct': 0.1,
        'max_daily_loss_pct': 0.05,
        'max_risk_per_trade_pct': 0.02,
        'stop_loss_method': 'volatility_based'
    },
    'trading_hours': {
        'start_hour': 9,
        'end_hour': 17,
        'timezone': 'UTC'
    },
    'data_sources': {
        'primary': 'binance',
        'fallback': 'coingecko',
        'update_interval_minutes': 5
    },
    'model_settings': {
        'min_confidence_threshold': 0.6,
        'retrain_interval_hours': 24,
        'use_ensemble': True
    }
}

def get_trading_recommendation(analysis_results, risk_tolerance='medium'):
    """Генерація торгових рекомендацій"""
    if not analysis_results.get('is_valid', True):
        return "NO_TRADE", analysis_results.get('validation_message', 'Invalid trade')
    
    signals = analysis_results.get('signals', {})
    
    if isinstance(signals, dict):
        signal_value = signals.get('signal', 'HOLD')
        strength = signals.get('strength', 0)
    else:
        signal_value = 'HOLD'
        strength = 0
    
    risk_analysis = analysis_results.get('risk_analysis', {})
    risk_level = risk_analysis.get('risk_level', 'medium')
    
    # Прийняття рішення
    if signal_value == 'BUY' and risk_level in ['low', 'medium']:
        return "BUY", f"Confidence: {strength}/10, Risk: {risk_level}"
    elif signal_value == 'SELL':
        return "SELL", f"Strong sell signal, Strength: {strength}/10"
    else:
        return "HOLD", "Wait for better opportunities"

def calculate_dynamic_risk_parameters(data):
    """Динамічний розрахунок параметрів ризику"""
    if data is None or len(data) == 0:
        return {
            'stop_loss_multiplier': 2.0,
            'position_size_multiplier': 1.0,
            'risk_level': 'medium'
        }
    
    returns = data['Close'].pct_change().dropna()
    if len(returns) == 0:
        volatility = 0.02
    else:
        volatility = returns.std()
    
    return {
        'stop_loss_multiplier': max(2.0, min(4.0, 3.0 * volatility * 100)),
        'position_size_multiplier': max(0.5, min(1.5, 1.0 / (volatility * 100))),
        'risk_level': 'high' if volatility > 0.03 else 'medium' if volatility > 0.015 else 'low'
    }