class RiskAgent:
    def __init__(self, account_balance=10000, risk_per_trade_pct=1.0, stop_loss_pct=2.0, max_risk_pct=5.0):
        """
        account_balance: total paper-trading capital (USD)
        risk_per_trade_pct: % of account willing to risk on a single trade
        stop_loss_pct: default stop-loss distance from entry price (%)
        max_risk_pct: max allowed risk exposure before flagging a trade
        """
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_risk_pct = max_risk_pct

    def calculate_stop_loss(self, entry_price: float, direction: str = "BUY") -> float:
        """Calculate stop-loss price based on fixed percentage distance."""
        if direction == "BUY":
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
        else:  # SELL / short
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
        return round(stop_loss, 2)

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> dict:
        """
        Position size based on how much capital we're willing to risk
        and the distance between entry and stop-loss.
        """
        risk_amount = self.account_balance * (self.risk_per_trade_pct / 100)
        price_risk_per_unit = abs(entry_price - stop_loss_price)

        if price_risk_per_unit == 0:
            return {"error": "Entry price and stop-loss price cannot be equal."}

        position_size_units = risk_amount / price_risk_per_unit
        position_value = position_size_units * entry_price
        position_pct_of_account = (position_value / self.account_balance) * 100

        flagged = position_pct_of_account > self.max_risk_pct * 10  # sanity guard on leverage-like sizing

        return {
            "risk_amount_usd": round(risk_amount, 2),
            "position_size_units": round(position_size_units, 6),
            "position_value_usd": round(position_value, 2),
            "position_pct_of_account": round(position_pct_of_account, 2),
            "flagged_high_risk": flagged,
        }

    def evaluate(self, entry_price: float, direction: str = "BUY") -> dict:
        """Full risk evaluation for a given trade signal."""
        stop_loss_price = self.calculate_stop_loss(entry_price, direction)
        sizing = self.calculate_position_size(entry_price, stop_loss_price)

        return {
            "agent": "RiskAgent",
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            **sizing,
        }


if __name__ == "__main__":
    agent = RiskAgent(account_balance=10000, risk_per_trade_pct=1.0, stop_loss_pct=2.0)

    result = agent.evaluate(entry_price=65000, direction="BUY")

    print("Risk Assessment Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")