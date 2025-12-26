## Output Contract v2 — Off‑Plan Investment Snapshot (Investor-first)

This contract is **formatting-only**. The assistant MUST only reformat numbers already present in the provided context.

CRITICAL RULES:
- DO NOT recalculate any values.
- DO NOT introduce new metrics that aren’t in the provided context.
- Use simple language. No technical feature names.
- If a value is missing, write **“Not available”** and keep the label.

---

### 1) Deal summary (header)
- Report title: Off‑Plan Investment Snapshot
- Date generated: <date>
- Area (display name): <area>
- Developer (brand): <developer>
- Unit type: <Apartment|Villa> + <Bedrooms>
- Stage: Off‑Plan
- Handover timing: <Handover in X months | handover date | Not available>

### 2) Required unit inputs (must be present)
- Unit size (sqft): <n>
- Purchase price (AED): <n>
- Purchase price per sqft (AED/sqft): <n>

### 3) Estimated value (AED total value; centerpiece)
- Estimated value at handover (AED):
  - Estimated: <AED X>
  - Range: <AED A – AED B>
- Estimated value 12 months after handover (AED):
  - Estimated: <AED X>
  - Range: <AED A – AED B>
- Projected uplift:
  - Uplift by handover: <+AED _> (<% vs purchase>)
  - Uplift by +12m post‑handover: <+AED _> (<% vs purchase>)

### 4) Rent & yield (post‑handover)
- Estimated annual rent after handover (AED/yr):
  - Estimated: <AED X>
  - Range: <AED A – AED B>
- Estimated gross yield after handover (%):
  - Estimated: <X%>
  - Range: <A% – B%>
- Optional benchmark (if present):
  - Benchmark rent (median) + contract count

### 5) Area market context (numbers only)
- Current area median price (AED/sqft): <n>
- 12‑month change (%): <n>
- 36‑month change (%): <n>
- Transactions (last 12m): <n>
- Supply pipeline / upcoming supply (units): <n>

### 6) Developer execution stats (numbers only)
- Projects completed / total: <n / n>
- Avg time to complete (months): <n>
- Avg delay (months): <n or Not available>
- Total units delivered (if available): <n or Not available>

### 7) What drives the forecast (sales script)
- Top 5 drivers (plain-English labels only; no technical feature names):
  - lifecycle timing to handover
  - supply pipeline / scheduled completions
  - area momentum
  - liquidity
  - rates (EIBOR)
- Optional sensitivities: include only if the text reads cleanly and deltas are meaningful.

### 8) Generate report prompt (CTA)
One line at the end:
**Press Generate Report to export the investor PDF.**