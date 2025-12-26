"use client"

import React from "react"
import Link from "next/link"

type Plan = {
  key: string
  label: string
  priceAed: number
  reportsPerMonthLabel: string
  usersLabel: string
  includes: string[]
  idealFor: string[]
  featured?: boolean
}

const PLANS: Plan[] = [
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
    featured: true,
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
]

function PricingCard({ plan }: { plan: Plan }) {
  const isFeatured = plan.featured

  return (
    <div
      className={[
        "relative flex flex-col rounded-2xl border p-6 backdrop-blur transition-all duration-300",
        isFeatured
          ? "border-emerald-500/50 bg-emerald-950/20 shadow-lg shadow-emerald-500/10 scale-[1.02] z-10"
          : "border-white/10 bg-white/[0.03] hover:border-white/20 hover:bg-white/[0.05]",
      ].join(" ")}
    >
      {isFeatured && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2">
          <span className="inline-flex items-center rounded-full bg-emerald-500 px-3 py-1 text-xs font-semibold text-black">
            Most Popular
          </span>
        </div>
      )}

      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white">{plan.label}</h3>
        <p className="mt-1 text-xs text-white/50">
          {plan.reportsPerMonthLabel} • {plan.usersLabel}
        </p>
      </div>

      <div className="mb-6">
        <div className="flex items-baseline gap-1">
          <span className="text-sm text-white/60">AED</span>
          <span className="text-3xl font-bold tracking-tight text-white">{plan.priceAed.toLocaleString()}</span>
          <span className="text-sm text-white/60">/ mo</span>
        </div>
      </div>

      <ul className="mb-6 flex-1 space-y-2.5">
        {plan.includes.map((feature) => (
          <li key={feature} className="flex items-start gap-2.5 text-sm text-white/70">
            <svg
              className={[
                "mt-0.5 h-4 w-4 shrink-0",
                isFeatured ? "text-emerald-400" : "text-emerald-500/70",
              ].join(" ")}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2.5}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      <div className="mb-4 border-t border-white/10 pt-4">
        <p className="text-[10px] uppercase tracking-[0.15em] text-white/40">Ideal for</p>
        <p className="mt-1 text-xs text-white/60">{plan.idealFor.join(" • ")}</p>
      </div>

      <Link
        href="/login"
        className={[
          "inline-flex h-11 w-full items-center justify-center rounded-full text-sm font-semibold transition-all duration-200",
          isFeatured
            ? "bg-emerald-500 text-black hover:bg-emerald-400"
            : "border border-white/15 bg-white/10 text-white hover:bg-white/15",
        ].join(" ")}
      >
        Get started
      </Link>
    </div>
  )
}

export function Pricing() {
  return (
    <section id="pricing" className="mx-auto max-w-7xl px-4 py-16 md:px-6 lg:px-8">
      <div className="mx-auto mb-12 max-w-2xl text-center">
        <p className="text-xs uppercase tracking-[0.2em] text-white/60">Pricing</p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white md:text-4xl">
          Plans for brokerages
        </h2>
        <p className="mt-3 text-sm text-white/70 md:text-base">
          Pay for clarity, speed, and trust—so your team can answer buyers fast and keep deals moving.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {PLANS.map((plan) => (
          <PricingCard key={plan.key} plan={plan} />
        ))}
      </div>

      <div className="mt-8 text-center">
        <a
          href="mailto:hello@proprly.ai?subject=Proprly%20Enterprise%20Pricing"
          className="text-sm font-medium text-white/80 hover:text-white"
        >
          Need a custom rollout? Contact sales →
        </a>
      </div>

      <p className="mx-auto mt-8 max-w-xl text-center text-xs text-white/40">
        Payments aren't enabled yet. This is our intended pricing; billing will launch in a later phase.
      </p>
    </section>
  )
}

