"use client"

import React, { useMemo, useState } from "react"

type ProprlyPlan = {
  key: "starter" | "professional" | "advanced" | "enterprise"
  label: string
  priceAed: number | null
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
    includes: ["Fully branded reports", "High-volume usage", "Team workflows (multi-user)"],
    idealFor: ["High-volume brokerages", "Investment desks", "Developer-aligned sales teams"],
  },
  {
    key: "enterprise",
    label: "Enterprise",
    priceAed: null,
    reportsPerMonthLabel: "Custom report volume",
    usersLabel: "Custom users",
    includes: ["Custom limits & rollout", "Custom templates & branding", "Dedicated onboarding (optional)"],
    idealFor: ["Large broker networks", "Developer sales orgs"],
  },
]

export function ProprlyPricingSlider() {
  const [sliderIndex, setSliderIndex] = useState(1)
  const plan = useMemo(() => PROPRLY_PLANS[sliderIndex], [sliderIndex])

  return (
    <section id="pricing" className="mx-auto max-w-6xl px-4 py-16 md:px-6 lg:px-8">
      <div className="mx-auto mb-10 max-w-2xl text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-white/60">Pricing</p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white md:text-4xl">
          Plans for brokerages
        </h2>
        <p className="mt-3 text-sm text-white/70 md:text-base">
          Pay for clarity, speed, and trust—so your team can answer buyers fast and keep deals moving.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Left Card */}
        <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur md:p-8">
          <h3 className="text-sm font-semibold text-white/90">Pick your volume</h3>
          <div className="mt-3 text-3xl font-semibold text-white">{plan.label}</div>
          <div className="mt-2 text-sm text-white/70">
            {plan.reportsPerMonthLabel} • {plan.usersLabel}
          </div>

          <div className="mt-8">
            <input
              type="range"
              min={0}
              max={PROPRLY_PLANS.length - 1}
              step={1}
              value={sliderIndex}
              onChange={(e) => setSliderIndex(Number(e.target.value))}
              className="h-3 w-full appearance-none rounded-full bg-white/10"
              style={{
                background: `linear-gradient(to right, rgba(16,185,129,1) 0%, rgba(16,185,129,1) ${
                  (sliderIndex / (PROPRLY_PLANS.length - 1)) * 100
                }%, rgba(255,255,255,0.12) ${(sliderIndex / (PROPRLY_PLANS.length - 1)) * 100}%, rgba(255,255,255,0.12) 100%)`,
              }}
              aria-label="Proprly plan selector"
            />

            <style>{`
              input[type='range']::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 28px;
                height: 28px;
                background: rgba(0,0,0,0.65);
                border: 2px solid rgba(255,255,255,0.22);
                border-radius: 999px;
                cursor: pointer;
                margin-top: -1px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.35);
              }
              input[type='range']::-moz-range-thumb {
                width: 26px;
                height: 26px;
                background: rgba(0,0,0,0.65);
                border: 2px solid rgba(255,255,255,0.22);
                border-radius: 999px;
                cursor: pointer;
                box-shadow: 0 10px 25px rgba(0,0,0,0.35);
              }
            `}</style>
          </div>

          <div className="mt-10 flex items-center justify-between text-sm text-white/60">
            <span>Need a custom plan?</span>
            <a
              href="mailto:hello@proprly.ai?subject=Proprly%20Enterprise%20Pricing"
              className="font-medium text-white hover:text-white/90"
            >
              Contact sales →
            </a>
          </div>
        </div>

        {/* Right Card */}
        <div className="rounded-2xl border border-white/10 bg-black/30 p-6 md:p-8">
          <h3 className="text-sm font-semibold text-white/90">What you get</h3>

          <div className="mt-3">
            <div className="text-3xl font-semibold text-white">
              {plan.priceAed === null ? "Contact sales" : `AED ${plan.priceAed.toLocaleString()} / mo`}
            </div>
            <div className="mt-2 text-sm text-white/70">
              {plan.reportsPerMonthLabel} • {plan.usersLabel}
            </div>
          </div>

          <ul className="mt-6 space-y-2 text-sm text-white/75">
            {plan.includes.map((x) => (
              <li key={x} className="flex gap-2">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                <span>{x}</span>
              </li>
            ))}
          </ul>

          <div className="mt-6">
            <p className="text-xs uppercase tracking-[0.2em] text-white/50">Ideal for</p>
            <p className="mt-2 text-sm text-white/70">{plan.idealFor.join(" • ")}</p>
          </div>

          <a
            href={
              plan.priceAed === null
                ? "mailto:hello@proprly.ai?subject=Proprly%20Enterprise%20Pricing"
                : "/login"
            }
            className="mt-8 inline-flex h-11 w-full items-center justify-center rounded-full border border-white/15 bg-white/10 px-6 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15"
          >
            {plan.priceAed === null ? "Contact sales" : "Create account"}
          </a>

          <p className="mt-3 text-xs text-white/50">
            Payments aren’t enabled yet. This is our intended pricing; billing will launch in a later phase.
          </p>
        </div>
      </div>
    </section>
  )
}


