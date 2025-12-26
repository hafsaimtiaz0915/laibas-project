# Model Specification: ROI Calculator for Off-Plan Properties

## 1. Overview

**Model Type**: Financial Calculator (Deterministic)  
**Purpose**: Calculate projected Return on Investment (ROI) for off-plan property purchases, accounting for payment plan leverage, DLD fees, and appreciation forecasts.  
**Primary Use Case**: "If I buy this off-plan unit with a 60/40 payment plan, what ROI can I expect by handover?"

---

## 2. The Off-Plan Investment Mechanics

Off-plan properties in Dubai operate differently from ready properties:

1. **Payment Plans**: Buyers pay in installments (e.g., 10% down, 10% quarterly during construction, 30% on handover)
2. **Leverage Effect**: Investors only deploy a fraction of capital before the asset appreciates
3. **Exit Strategies**: Flip before handover (sell assignment) or hold through completion
4. **Risk Premium**: Off-plan trades at a discount to ready due to delivery risk

### 2.1 Why Cash-on-Cash ROI Matters

Simple price appreciation is misleading for off-plan. If a property appreciates 15% but the investor only paid 40% of the price, their actual return on deployed capital is much higher.

**Example:**
- Purchase Price: 2,000,000 AED
- Payment Made (40%): 800,000 AED + Fees
- Property Appreciates 15%: Now worth 2,300,000 AED
- Profit: 300,000 AED
- Cash-on-Cash ROI: 300,000 / 850,000 = **35.3%** (not 15%)

---

## 3. Core Formulas

### 3.1 Total Cost of Acquisition

$$C_{total} = P_{purchase} + F_{DLD} + F_{admin} + F_{oqood}$$

Where:
- $P_{purchase}$ = Purchase price (contract value)
- $F_{DLD}$ = Dubai Land Department fee = $P_{purchase} \times 0.04$ (4%)
- $F_{admin}$ = Admin/broker fees (typically 2%)
- $F_{oqood}$ = Oqood registration fee (off-plan specific)

```python
def calculate_total_cost(purchase_price: float, 
                         dld_rate: float = 0.04,
                         admin_rate: float = 0.02,
                         oqood_fee: float = 5000) -> dict:
    """
    Calculate total acquisition cost for off-plan purchase.
    """
    dld_fee = purchase_price * dld_rate
    admin_fee = purchase_price * admin_rate
    
    total_cost = purchase_price + dld_fee + admin_fee + oqood_fee
    
    return {
        "purchase_price": purchase_price,
        "dld_fee": dld_fee,
        "admin_fee": admin_fee,
        "oqood_fee": oqood_fee,
        "total_cost": total_cost,
        "fee_percentage": ((total_cost - purchase_price) / purchase_price) * 100
    }
```

### 3.2 Cash Invested Over Time

The key insight: cash is deployed gradually according to the payment plan.

$$\text{Cash Invested}(t) = \sum_{i=0}^{t} P_i + F_{upfront}$$

Where:
- $P_i$ = Payment at milestone $i$
- $F_{upfront}$ = Upfront fees (DLD, admin) paid at signing

```python
def calculate_cash_invested_schedule(
    purchase_price: float,
    payment_plan: list[dict],  # [{"milestone": "Booking", "pct": 10, "months": 0}, ...]
    fees: dict
) -> list[dict]:
    """
    Generate cash investment schedule over construction period.
    
    Args:
        purchase_price: Total purchase price
        payment_plan: List of payment milestones with percentage and timing
        fees: Dict with dld_fee, admin_fee, oqood_fee
    
    Returns:
        List of cumulative cash invested at each month
    """
    schedule = []
    cumulative = fees['dld_fee'] + fees['admin_fee'] + fees['oqood_fee']
    
    for milestone in payment_plan:
        payment = purchase_price * (milestone['pct'] / 100)
        cumulative += payment
        
        schedule.append({
            "month": milestone['months'],
            "milestone": milestone['milestone'],
            "payment": payment,
            "cumulative_invested": cumulative
        })
    
    return schedule
```

### 3.3 Cash-on-Cash ROI (Flip on Handover)

$$\text{CoC ROI} = \frac{P_{exit} - C_{total}}{\text{Cash Invested at Exit}} \times 100$$

For a flip before handover (selling the assignment):

$$\text{CoC ROI}_{flip} = \frac{P_{exit} - P_{purchase} - F_{selling}}{\text{Cash Invested}} \times 100$$

Where $F_{selling}$ = selling fees (typically 2% agent commission)

```python
def calculate_flip_roi(
    purchase_price: float,
    exit_price: float,
    cash_invested: float,
    selling_fee_rate: float = 0.02
) -> dict:
    """
    Calculate ROI for flipping off-plan before handover.
    """
    selling_fee = exit_price * selling_fee_rate
    
    gross_profit = exit_price - purchase_price
    net_profit = gross_profit - selling_fee
    
    # Note: DLD fee was already paid, it's a sunk cost
    # But the buyer pays their own DLD fee, so we include it in our cost basis
    
    coc_roi = (net_profit / cash_invested) * 100
    
    return {
        "purchase_price": purchase_price,
        "exit_price": exit_price,
        "gross_profit": gross_profit,
        "selling_fee": selling_fee,
        "net_profit": net_profit,
        "cash_invested": cash_invested,
        "coc_roi_pct": round(coc_roi, 2),
        "absolute_return": net_profit
    }
```

### 3.4 Annualized ROI (IRR)

For proper comparison across different investment horizons, calculate the Internal Rate of Return:

$$\sum_{t=0}^{n} \frac{CF_t}{(1+IRR)^t} = 0$$

Where:
- $CF_t$ = Cash flow at time $t$ (negative for payments, positive for exit)
- $n$ = Investment horizon in periods

```python
import numpy as np
from scipy.optimize import brentq

def calculate_irr(cash_flows: list[dict]) -> float:
    """
    Calculate Internal Rate of Return for off-plan investment.
    
    Args:
        cash_flows: List of {"month": int, "amount": float}
                    Negative = outflow (payment), Positive = inflow (sale)
    
    Returns:
        Annual IRR as percentage
    """
    # Convert to monthly periods
    max_month = max(cf['month'] for cf in cash_flows)
    monthly_flows = [0] * (max_month + 1)
    
    for cf in cash_flows:
        monthly_flows[cf['month']] += cf['amount']
    
    # NPV function
    def npv(rate):
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(monthly_flows))
    
    # Find monthly IRR
    try:
        monthly_irr = brentq(npv, -0.99, 10)
        # Annualize
        annual_irr = (1 + monthly_irr) ** 12 - 1
        return round(annual_irr * 100, 2)
    except:
        return None  # No valid IRR (e.g., all negative cash flows)
```

### 3.5 Break-Even Analysis

At what price appreciation does the investor break even?

$$P_{breakeven} = P_{purchase} + F_{total} + F_{selling}$$

$$\text{Breakeven Appreciation} = \frac{P_{breakeven} - P_{purchase}}{P_{purchase}} \times 100$$

```python
def calculate_breakeven(
    purchase_price: float,
    total_fees_paid: float,
    selling_fee_rate: float = 0.02
) -> dict:
    """
    Calculate the minimum exit price to break even.
    """
    # Breakeven: Exit Price - Selling Fee = Purchase + Fees Paid
    # Exit Price * (1 - selling_rate) = Purchase + Fees
    # Exit Price = (Purchase + Fees) / (1 - selling_rate)
    
    breakeven_price = (purchase_price + total_fees_paid) / (1 - selling_fee_rate)
    appreciation_needed = ((breakeven_price - purchase_price) / purchase_price) * 100
    
    return {
        "breakeven_price": round(breakeven_price, 0),
        "appreciation_needed_pct": round(appreciation_needed, 2),
        "fees_to_recover": total_fees_paid
    }
```

---

## 4. Payment Plan Structures

### 4.1 Common Dubai Off-Plan Payment Plans

| Plan Type | Structure | Typical Use |
|---|---|---|
| **60/40** | 60% during construction, 40% on handover | Standard |
| **80/20** | 80% during construction, 20% on handover | Premium developments |
| **Post-Handover** | 50% during, 50% over 3-5 years post-handover | Buyer-friendly |
| **1% Monthly** | 1% per month over 100 months | Emaar, extended plans |

### 4.2 Payment Plan Parser

```python
def parse_payment_plan(plan_type: str, 
                       construction_months: int,
                       purchase_price: float) -> list[dict]:
    """
    Generate payment schedule from plan type.
    """
    if plan_type == "60/40":
        return [
            {"milestone": "Booking", "pct": 10, "months": 0},
            {"milestone": "SPA Signing", "pct": 10, "months": 1},
            {"milestone": "Construction 20%", "pct": 10, "months": int(construction_months * 0.2)},
            {"milestone": "Construction 40%", "pct": 10, "months": int(construction_months * 0.4)},
            {"milestone": "Construction 60%", "pct": 10, "months": int(construction_months * 0.6)},
            {"milestone": "Construction 80%", "pct": 10, "months": int(construction_months * 0.8)},
            {"milestone": "Handover", "pct": 40, "months": construction_months}
        ]
    elif plan_type == "80/20":
        return [
            {"milestone": "Booking", "pct": 20, "months": 0},
            {"milestone": "SPA Signing", "pct": 20, "months": 1},
            {"milestone": "Construction 50%", "pct": 20, "months": int(construction_months * 0.5)},
            {"milestone": "Construction 100%", "pct": 20, "months": int(construction_months * 0.9)},
            {"milestone": "Handover", "pct": 20, "months": construction_months}
        ]
    # ... other plan types
```

---

## 5. Scenario Analysis Module

### 5.1 Multi-Scenario ROI Projection

```python
def scenario_analysis(
    purchase_price: float,
    payment_plan: list[dict],
    construction_months: int,
    appreciation_scenarios: dict  # {"bear": -5, "base": 10, "bull": 20}
) -> dict:
    """
    Generate ROI projections across market scenarios.
    """
    fees = calculate_total_cost(purchase_price)
    schedule = calculate_cash_invested_schedule(purchase_price, payment_plan, fees)
    
    # Cash invested at handover
    cash_at_handover = schedule[-1]['cumulative_invested']
    
    results = {}
    
    for scenario, appreciation_pct in appreciation_scenarios.items():
        exit_price = purchase_price * (1 + appreciation_pct / 100)
        
        roi = calculate_flip_roi(
            purchase_price=purchase_price,
            exit_price=exit_price,
            cash_invested=cash_at_handover
        )
        
        # Build cash flow for IRR
        cash_flows = [{"month": m['month'], "amount": -m['payment']} 
                      for m in schedule]
        cash_flows.append({
            "month": construction_months,
            "amount": exit_price - fees['total_cost'] + purchase_price  # Net proceeds
        })
        
        irr = calculate_irr(cash_flows)
        
        results[scenario] = {
            "appreciation_pct": appreciation_pct,
            "exit_price": exit_price,
            "coc_roi_pct": roi['coc_roi_pct'],
            "irr_pct": irr,
            "net_profit": roi['net_profit']
        }
    
    return results
```

### 5.2 Exit Timing Optimization

Should the investor flip at 50% construction or wait for handover?

```python
def optimize_exit_timing(
    purchase_price: float,
    payment_plan: list[dict],
    price_forecast: list[dict],  # From Time Series Model
    construction_months: int
) -> dict:
    """
    Determine optimal exit point based on price forecast.
    """
    exits = []
    
    for exit_month in range(6, construction_months + 1, 6):
        # Cash invested by this point
        payments_made = sum(
            p['payment'] for p in payment_plan 
            if p['months'] <= exit_month
        )
        cash_invested = payments_made + calculate_total_cost(purchase_price)['total_cost'] - purchase_price
        
        # Forecasted price at exit
        forecast = next((f for f in price_forecast if f['month'] == exit_month), None)
        if not forecast:
            continue
            
        exit_price = forecast['median']
        
        roi = calculate_flip_roi(purchase_price, exit_price, cash_invested)
        
        exits.append({
            "exit_month": exit_month,
            "construction_pct": (exit_month / construction_months) * 100,
            "cash_invested": cash_invested,
            "exit_price": exit_price,
            "coc_roi_pct": roi['coc_roi_pct'],
            "annualized_roi": roi['coc_roi_pct'] / (exit_month / 12)
        })
    
    # Find optimal
    optimal = max(exits, key=lambda x: x['annualized_roi'])
    
    return {
        "all_exits": exits,
        "optimal_exit": optimal,
        "recommendation": f"Exit at {optimal['exit_month']} months ({optimal['construction_pct']:.0f}% completion) for best annualized return of {optimal['annualized_roi']:.1f}%"
    }
```

---

## 6. Risk-Adjusted Returns

### 6.1 Project Risk Assessment

Off-plan carries risks that should discount the expected ROI:

| Risk Factor | Weight | Data Source |
|---|---|---|
| Developer Track Record | 30% | Projects.csv completion history |
| Construction Progress | 25% | `percent_completed` field |
| Escrow Bank Quality | 15% | `escrow_agent_name` field |
| Supply in Area | 20% | Supply Pressure Index |
| Payment Plan Risk | 10% | Post-handover exposure |

```python
def calculate_risk_score(
    developer_name: str,
    percent_completed: float,
    escrow_agent: str,
    supply_pressure: float,
    post_handover_pct: float,
    projects_df: pd.DataFrame
) -> dict:
    """
    Calculate risk score for off-plan investment.
    """
    scores = {}
    
    # 1. Developer track record (% of projects completed on time)
    dev_projects = projects_df[projects_df['developer_name'] == developer_name]
    completed = dev_projects[dev_projects['project_status'] == 'FINISHED']
    on_time = completed[completed['completion_date'] <= completed['project_end_date']]
    dev_score = len(on_time) / len(completed) * 100 if len(completed) > 0 else 50
    scores['developer'] = dev_score
    
    # 2. Construction progress (higher = lower risk)
    scores['construction'] = percent_completed
    
    # 3. Escrow bank (top-tier banks = 100, others = 70)
    top_banks = ['Emirates NBD', 'FAB', 'ADCB', 'Mashreq']
    scores['escrow'] = 100 if any(b in escrow_agent for b in top_banks) else 70
    
    # 4. Supply pressure (inverse: lower SPI = better)
    scores['supply'] = max(0, 100 - supply_pressure * 20)
    
    # 5. Payment plan (lower post-handover = lower risk)
    scores['payment_plan'] = 100 - post_handover_pct
    
    # Weighted average
    weights = {
        'developer': 0.30,
        'construction': 0.25,
        'escrow': 0.15,
        'supply': 0.20,
        'payment_plan': 0.10
    }
    
    overall = sum(scores[k] * weights[k] for k in weights)
    
    return {
        "scores": scores,
        "overall_risk_score": round(overall, 1),
        "risk_level": "Low" if overall > 75 else "Medium" if overall > 50 else "High"
    }
```

### 6.2 Risk-Adjusted ROI

$$\text{Risk-Adjusted ROI} = \text{Expected ROI} \times \frac{\text{Risk Score}}{100}$$

```python
def calculate_risk_adjusted_roi(expected_roi: float, risk_score: float) -> float:
    """
    Discount ROI by risk factor.
    """
    adjustment_factor = risk_score / 100
    return expected_roi * adjustment_factor
```

---

## 7. Output Schema

```json
{
    "property": {
        "project_name": "Creek Harbour Tower 3",
        "developer": "Emaar",
        "area": "Dubai Creek Harbour",
        "unit_type": "2 Bedroom",
        "purchase_price_aed": 2500000,
        "expected_handover": "Q4 2026",
        "percent_completed": 35
    },
    "costs": {
        "purchase_price": 2500000,
        "dld_fee": 100000,
        "admin_fee": 50000,
        "oqood_fee": 5000,
        "total_cost": 2655000
    },
    "payment_plan": {
        "type": "60/40",
        "schedule": [
            {"milestone": "Booking", "pct": 10, "amount": 250000, "month": 0, "cumulative": 405000},
            {"milestone": "SPA", "pct": 10, "amount": 250000, "month": 1, "cumulative": 655000},
            // ...
            {"milestone": "Handover", "pct": 40, "amount": 1000000, "month": 24, "cumulative": 2655000}
        ],
        "cash_at_50pct_construction": 905000,
        "cash_at_handover": 2655000
    },
    "roi_scenarios": {
        "bear": {
            "appreciation_pct": -5,
            "exit_price": 2375000,
            "coc_roi_pct": -14.2,
            "irr_pct": -8.1
        },
        "base": {
            "appreciation_pct": 12,
            "exit_price": 2800000,
            "coc_roi_pct": 18.5,
            "irr_pct": 10.2
        },
        "bull": {
            "appreciation_pct": 25,
            "exit_price": 3125000,
            "coc_roi_pct": 42.1,
            "irr_pct": 22.8
        }
    },
    "breakeven": {
        "breakeven_price": 2709000,
        "appreciation_needed_pct": 8.4
    },
    "risk_assessment": {
        "overall_risk_score": 72,
        "risk_level": "Medium",
        "factors": {
            "developer": 85,
            "construction": 35,
            "escrow": 100,
            "supply": 70,
            "payment_plan": 60
        }
    },
    "optimal_exit": {
        "recommended_exit_month": 18,
        "construction_at_exit_pct": 75,
        "projected_roi_pct": 28.5,
        "rationale": "Maximize annualized return before large handover payment due"
    },
    "recommendation": "This off-plan investment offers attractive leverage (60/40 plan) with a reputable developer. Base case ROI of 18.5% assumes 12% appreciation by handover. Key risk: 35% construction completion - consider entering at higher completion stages for lower risk."
}
```

---

## 8. Integration Points

### 8.1 Upstream Dependencies

| Dependency | Source | Usage |
|---|---|---|
| Time Series Forecast | Model 02 | Appreciation projections |
| Supply Pressure Index | Model 05 | Risk scoring |
| Projects.csv | Data Layer | Developer track record, completion % |

### 8.2 Downstream Consumers

| Consumer | Usage |
|---|---|
| Agentic Interface | Natural language ROI queries |
| Comparison Engine | Compare multiple off-plan opportunities |
| Alert System | Notify when project hits optimal exit point |

---

## 9. Validation & Testing

### 9.1 Test Cases

| Scenario | Expected Behavior |
|---|---|
| 60/40 plan, 15% appreciation | CoC ROI > 25% |
| 0% appreciation | Negative ROI (fees not recovered) |
| 100% completed project | Warn: "Ready property, use ready ROI model" |
| Unknown developer | Risk score penalized, warning issued |

### 9.2 Historical Validation

Backtest against actual off-plan transactions:
1. Select completed projects from 2020-2022
2. Get initial launch prices from historical data
3. Compare to actual resale prices
4. Validate model predictions against observed ROIs

---

## 10. Formula Reference Card

| Formula | Expression |
|---|---|
| **Total Cost** | $C = P + F_{DLD} + F_{admin} + F_{oqood}$ |
| **Cash Invested** | $\text{CI}(t) = \sum_{i=0}^{t} P_i + F_{upfront}$ |
| **Cash-on-Cash ROI** | $\text{CoC} = \frac{P_{exit} - C}{\text{CI}} \times 100$ |
| **IRR** | $\sum \frac{CF_t}{(1+r)^t} = 0$ |
| **Breakeven Price** | $P_{BE} = \frac{C}{1 - f_{sell}}$ |
| **Risk-Adjusted ROI** | $ROI_{adj} = ROI \times \frac{RS}{100}$ |

