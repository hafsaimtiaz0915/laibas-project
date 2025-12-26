---
title: Landing Page Components Plan (Proprly)
last_updated: 2025-12-15
---

## Goals

- **Primary conversion**: Agents create an account → `/login` (Supabase sign-up already exists).
- **Secondary conversion**: View a credible sample report preview (screenshot section).
- **Conversion**: Help agents **convert more deals** by reducing buyer doubt with clear, consistent, data-backed reports.
- **Trust**: Confident but compliant language: “data-backed” and “validated” rather than guarantees.
- **Mobile-first**: Single-column first, full-width CTAs, fast load.

## Routing decision (required)

Right now `/` redirects to `/login` or `/chat` (`frontend/app/page.tsx`). For a real public landing page:

- **Make `/` the marketing landing page** (public).
- Keep the product at:
  - **`/login`** (sign-in / sign-up)
  - **`/chat`** (main app)
  - **`/settings`** (branding/settings)

Optional (recommended):
- Add **`/app`** that performs the current auth-check redirect to `/chat` or `/login`.

## Component plan (using the code you provided)

### 1) Hero (Section 1): Spline + overlay content + fixed navbar

**Use**: your `HeroSection` composition:
- `Navbar` (fixed, blurred background)
- `HeroSplineBackground` (lazy loaded Spline scene)
- `HeroContent` (left aligned overlay)

**Dependency**
- Install `@splinetool/react-spline` (not currently in `frontend/package.json`).

**Hero copy (Proprly) — agent-first (recommended)**
- **Badge**: “Proprly for Dubai real estate agents”
- **H1**: “Close faster with data-backed client reports.”
- **Subcopy**: “Turn investor questions into a clear answer: forecasts, yield signals, area momentum, and a branded PDF you can forward instantly.”
- **Primary CTA**: “Create account” → `/login`
- **Secondary CTA**: “View sample report” → `#sample-report`
- **Micro-trust line**: “Built for mobile. Consistent, buyer-ready output that removes doubt.”

**Hero proof points (1-line, optional under subcopy)**
- “Trained on large historical Dubai datasets (transactions + rent benchmarks) and designed for agent workflows.”

**Navbar items (keep minimal)**
- Features → `#features`
- Sample report → `#sample-report`
- FAQ → `#faq`
- Pricing (soon) → `#pricing`
- Sign in → `/login`
- CTA button: “Create account” → `/login`

**Performance**
- Keep Spline lazy-loaded with a `Suspense` fallback background to avoid layout jank.

**Mobile implementation notes (do these in code)**
- Use **`min-height: 100svh`** (not only `100vh`) to avoid iOS address-bar jump.
- Ensure overlay content uses `pointer-events: none` **except** CTA container (`pointer-events-auto`) so taps work reliably.
- Reduce scroll/parallax intensity on mobile (or disable under `prefers-reduced-motion`).
- Tap targets: keep buttons at least **44px** tall; avoid tiny text links in the primary row.

---

### 2) Sample report screenshot (Section 2): report preview card

**Use**: your `ScreenshotSection` pattern, but swap the image for our report screenshot.

**Asset**
- Place screenshot in: `frontend/public/sample-report.jpg` (or `.png`)
- Render via `next/image` for perf.

**Section copy (recommended)**
- **Eyebrow**: “Sample report”
- **H2**: “Remove doubt. Give buyers a clean, confident answer.”
- **Body**:
  - “Proprly generates a consistent, investor-friendly PDF: price outlook, rent & yield signals, area momentum, supply pipeline context, and developer delivery history—so you can handle objections with clarity.”

**CTAs under screenshot**
- “Create account” → `/login`
- “How it works” → `#how-it-works`

**Small disclaimer (recommended)**
- “Example only — informational, not financial advice.”

**Mobile implementation notes**
- Use `next/image` with a fixed aspect ratio container so the screenshot doesn’t cause layout shifts.
- Keep the screenshot card **full width** on mobile (`w-full`) with generous padding and one clear CTA.

---

### 3) Features (Section 3): concise feature grid

**Anchor**: `#features`

**Component**
- 21st.dev “feature grid” or simple card grid (4–6 items).

**Feature bullets (agent conversion framing)**
- **Answer investor objections quickly** (consistent, structured outputs)
- **Data-backed outlook** (forecast + supporting signals)
- **Rent & yield signals** (clear, buyer-friendly language)
- **Area momentum + liquidity** (transactions + change)
- **Supply pipeline context** (what’s landing soon)
- **Developer delivery history** (track record signals)
- **Branded PDF export** (forward to clients instantly)

---

### 4) How it works (Section 4): 3-step flow

**Anchor**: `#how-it-works`

**Component**
- 3 step cards / timeline:
  - Ask → Analyze → Export PDF

**Copy outline**
- Step 1: “Ask about a unit / project / area”
- Step 2: “Get forecasts + market signals”
- Step 3: “Download and forward a clean PDF report”

---

## Pricing section (use your slider UI, adapted to Proprly tiers)

You provided `LoopsPricingSlider`. We will keep the interaction pattern (slider + two cards) but change:

- “Subscribers” → **Plan tier / report volume / users**
- `$` → **AED**
- Copy → **reports, users, branding, support**
- CTA link → `/login` (or email for enterprise)

### Pricing (source of truth)

**Tier 1 — Starter (Boutique / Individual Brokers)**
- **AED 2,500 / month**
- **Up to 5 AI-validated reports / month**
- **1 user**
- Buyer-facing PDF output
- Standard report format (non-customisable)
- Basic area & unit forecasts
- Email support

**Tier 2 — Professional (Core Brokerage Tier)**
- **AED 7,500 / month**
- **Up to 40 reports / month**
- **4 users**
- Editable buyer-facing templates
- Broker branding (logo, footer)
- Internal “broker view” with deeper drivers
- Priority support

**Tier 3 — Advanced (High-Volume / Investment Desks)**
- **AED 20,000 / month**
- **100+ reports / month**
- **10 users**
- Fully branded reports

### Recommended slider “stops”

Instead of 0→1M subscriber steps, use discrete plan stops:
- Starter → Professional → Advanced → Enterprise (custom)

### Adapted pricing slider code (copy/paste)

```tsx
import React, { useMemo, useState } from "react"

type ProprlyPlan = {
  key: "starter" | "professional" | "advanced" | "enterprise"
  label: string
  priceAed: number | null // null => contact us
  reportsPerMonthLabel: string
  usersLabel: string
  includes: string[]
  idealFor: string[]
}

const PROPRLY_PLANS: ProprlyPlan[] = [
  {
    key: "starter",
    label: "Starter",
    priceAed: 2500,
    reportsPerMonthLabel: "Up to 5 reports / month",
    usersLabel: "1 user",
    includes: [
      "AI-validated buyer-facing PDF report",
      "Standard report format (non-customisable)",
      "Basic area & unit forecasts",
      "Email support",
    ],
    idealFor: ["Individual brokers", "Boutique teams"],
  },
  {
    key: "professional",
    label: "Professional",
    priceAed: 7500,
    reportsPerMonthLabel: "Up to 40 reports / month",
    usersLabel: "4 users",
    includes: [
      "Editable buyer-facing templates",
      "Broker branding (logo + footer)",
      "Internal broker view with deeper drivers",
      "Priority support",
    ],
    idealFor: ["Active off-plan brokerages", "Investment-focused agents"],
  },
  {
    key: "advanced",
    label: "Advanced",
    priceAed: 20000,
    reportsPerMonthLabel: "100+ reports / month",
    usersLabel: "10 users",
    includes: [
      "Fully branded reports",
      "High-volume usage",
      "Team workflows (multi-user)",
    ],
    idealFor: ["High-volume brokerages", "Investment desks", "Developer-aligned sales teams"],
  },
  {
    key: "enterprise",
    label: "Enterprise",
    priceAed: null,
    reportsPerMonthLabel: "Custom report volume",
    usersLabel: "Custom users",
    includes: [
      "Custom limits & rollout",
      "Custom templates & branding",
      "Dedicated onboarding (optional)",
    ],
    idealFor: ["Large broker networks", "Developer sales orgs"],
  },
]

export const ProprlyPricingSlider: React.FC = () => {
  const [sliderIndex, setSliderIndex] = useState(1) // default: Professional
  const plan = useMemo(() => PROPRLY_PLANS[sliderIndex], [sliderIndex])

  return (
    <section id="pricing" className="max-w-5xl mx-auto p-6">
      <div className="mb-8 text-center">
        <p className="text-xs uppercase tracking-widest text-neutral-500">Pricing</p>
        <h2 className="mt-2 text-3xl font-semibold text-black">Plans for brokerages</h2>
        <p className="mt-2 text-sm text-neutral-600 max-w-2xl mx-auto">
          Built to pay for itself: convert faster, reduce buyer doubt, and keep your advice consistent across the team.
        </p>
      </div>

      <div className="flex flex-col md:flex-row gap-6">
        {/* Left Card */}
        <div className="flex-1 rounded-xl border border-gray-200 p-8 relative bg-white">
          <h3 className="text-sm font-semibold text-neutral-800 mb-4">Choose a plan</h3>
          <div className="text-3xl font-bold text-black mb-2">{plan.label}</div>
          <div className="text-sm text-neutral-600 mb-8">
            {plan.reportsPerMonthLabel} • {plan.usersLabel}
          </div>

          <input
            type="range"
            min={0}
            max={PROPRLY_PLANS.length - 1}
            step={1}
            value={sliderIndex}
            onChange={(e) => setSliderIndex(Number(e.target.value))}
            className="w-full appearance-none h-3 rounded bg-gray-200 mb-12"
            style={{
              background: `linear-gradient(to right, #10B981 0%, #10B981 ${
                (sliderIndex / (PROPRLY_PLANS.length - 1)) * 100
              }%, #E5E7EB ${(sliderIndex / (PROPRLY_PLANS.length - 1)) * 100}%, #E5E7EB 100%)`,
            }}
            aria-label="Proprly plan selector"
          />

          <style>{`
            input[type='range']::-webkit-slider-thumb {
              -webkit-appearance: none;
              appearance: none;
              width: 28px;
              height: 28px;
              background: #ffffff;
              border: 2px solid #E5E7EB;
              border-radius: 50%;
              cursor: pointer;
              margin-top: -1px;
              box-shadow: 0 1px 5px rgba(0, 0, 0, 0.12);
              position: relative;
            }
            input[type='range']::-moz-range-thumb {
              width: 26px;
              height: 26px;
              background: #ffffff;
              border: 2px solid #E5E7EB;
              border-radius: 50%;
              cursor: pointer;
              box-shadow: 0 1px 5px rgba(0, 0, 0, 0.12);
            }
          `}</style>

          <div className="absolute bottom-8 left-8 right-8 flex items-center justify-between text-sm text-neutral-500">
            <span>Need a custom rollout?</span>
            <a
              href="mailto:hello@proprly.ai?subject=Proprly%20Enterprise%20Pricing"
              className="text-black font-medium flex items-center"
            >
              Contact sales <span className="ml-1">→</span>
            </a>
          </div>
        </div>

        {/* Right Card */}
        <div className="flex-1 rounded-xl border border-gray-200 p-8 bg-neutral-50">
          <h3 className="text-sm font-semibold text-neutral-800 mb-4">What’s included</h3>

          <div className="mb-3">
            <div className="text-3xl font-bold text-black">
              {plan.priceAed === null ? "Contact sales" : `AED ${plan.priceAed.toLocaleString()} / mo`}
            </div>
            <div className="mt-1 text-sm text-neutral-600">
              {plan.reportsPerMonthLabel} • {plan.usersLabel}
            </div>
          </div>

          <ul className="mt-6 space-y-2 text-sm text-neutral-700">
            {plan.includes.map((x) => (
              <li key={x} className="flex gap-2">
                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-emerald-500" />
                <span>{x}</span>
              </li>
            ))}
          </ul>

          <div className="mt-6">
            <p className="text-xs uppercase tracking-widest text-neutral-500">Ideal for</p>
            <p className="mt-2 text-sm text-neutral-600">{plan.idealFor.join(" • ")}</p>
          </div>

          <a
            href={
              plan.priceAed === null
                ? "mailto:hello@proprly.ai?subject=Proprly%20Enterprise%20Pricing"
                : "/login"
            }
            className="mt-8 inline-block bg-black text-white px-6 py-3 rounded-md font-semibold text-sm"
          >
            {plan.priceAed === null ? "Contact sales" : "Create account"}
          </a>

          <p className="mt-3 text-xs text-neutral-500">
            Payments aren’t enabled yet. This is our intended pricing; billing will launch in a later phase.
          </p>
        </div>
      </div>
    </section>
  )
}
```

**Mobile optimization notes for pricing**
- Ensure the section stacks as a single column on mobile (already: `flex-col md:flex-row`).
- Keep padding responsive: consider `p-5 sm:p-6 md:p-8` so mobile isn’t cramped.
- Ensure the slider thumb is easy to drag on mobile; keep thumb ≥ 28px (already) and add `touch-action: pan-y` if needed.

## FAQ (recommended)

**Anchor**: `#faq`  
**Component**: Accordion (21st.dev/shadcn style)

Suggested questions:
- “What markets does Proprly cover?”
- “What inputs do I need to generate a report?”
- “Can I brand the PDF report?”
- “Is this financial advice?”
- “Do you support teams?”

## Conversion-safe “hype” (copy guardrails)

We should sell the outcome agents want (conversion + trust) while avoiding risky guarantees:

- Use: “data-backed”, “trained on large historical datasets”, “confidence indicators”, “consistent outputs”, “client-ready formatting”
- Avoid: “guaranteed returns”, “always accurate”, “promise appreciation”

Suggested line you can use in multiple places:
- “Built on large historical Dubai datasets and designed to produce consistent, buyer-ready reports that help you answer objections with clarity.”

## Implementation checklist

- Add landing page route at `/` (stop redirecting).
- Ensure `@splinetool/react-spline` is installed.
- Confirm report screenshot exists at `frontend/public/sample-report.jpg`.
- Wire CTAs to `/login` and anchors.


