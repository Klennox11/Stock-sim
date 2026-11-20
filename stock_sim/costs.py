"""Trading costs. Defaults are roughly Interactive Brokers 'Fixed' pricing for US stocks."""
from dataclasses import dataclass


@dataclass
class Costs:
    per_share: float = 0.005     # $ commission per share
    min_commission: float = 1.00 # $ minimum per order
    max_pct: float = 0.01        # commission capped at 1% of trade value
    slippage_bps: float = 5.0    # we buy 0.05% higher / sell 0.05% lower than the open price

    def commission(self, shares: int, price: float) -> float:
        if shares <= 0:
            return 0.0
        fee = max(self.min_commission, shares * self.per_share)
        return min(fee, shares * price * self.max_pct)

    def fill_price(self, price: float, side: str) -> float:
        slip = price * self.slippage_bps / 10_000
        return price + slip if side == "BUY" else price - slip
