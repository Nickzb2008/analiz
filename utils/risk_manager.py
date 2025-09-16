class RiskManager:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.max_risk_per_trade = 0.02
        self.max_daily_loss = 0.05
        self.daily_pnl = 0
    
    def calculate_position_size(self, entry_price, stop_loss_price, risk_per_trade=None):
        """Розрахунок розміру позиції з ризик-менеджментом"""
        risk_per_trade = risk_per_trade or self.max_risk_per_trade
        
        if entry_price == 0 or stop_loss_price == 0:
            return 0
            
        risk_amount = self.capital * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
            
        position_size = risk_amount / risk_per_share
        return min(position_size, self.capital * 0.1)
    
    def validate_trade(self, symbol, entry_price, stop_loss, position_size):
        """Перевірка угоди на відповідність правилам ризику"""
        if entry_price == 0 or position_size == 0:
            return False, "Невірні параметри угоди"
        
        risk_amount = abs(entry_price - stop_loss) * position_size
        risk_percent = risk_amount / self.capital
        
        if risk_percent > self.max_risk_per_trade:
            return False, f"Ризик {risk_percent:.2%} > {self.max_risk_per_trade:.2%}"
        
        if self.daily_pnl < -self.capital * self.max_daily_loss:
            return False, "Денний ліміт втрат досягнуто"
        
        return True, "Угода відповідає правилам ризику"
    
    def update_after_trade(self, pnl):
        """Оновлення капіталу після угоди"""
        self.capital += pnl
        self.daily_pnl += pnl
    
    def reset_daily_pnl(self):
        """Скинути денний PNL"""
        self.daily_pnl = 0