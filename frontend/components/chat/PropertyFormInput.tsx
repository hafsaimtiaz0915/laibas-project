"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { Combobox, type ComboboxItem } from "@/components/ui/combobox"
import { ChevronUp, Send, Pencil, Sparkles } from "lucide-react"

type AreaItem = { value: string; label: string; dld_area?: string }

function onlyDigits(s: string): string {
  return s.replace(/[^\d]/g, "")
}

function buildDeterministicQuery(input: {
  area: string
  developer?: string
  bedroom?: string
  reg_type?: "OffPlan" | "Ready"
  property_type?: "Unit" | "Villa"
  price_aed?: string
  unit_sqft?: string
  handover_months?: string
}) {
  const parts: string[] = []
  parts.push(`Area: ${input.area}`)
  if (input.developer) parts.push(`Developer: ${input.developer}`)
  if (input.property_type) parts.push(`Property type: ${input.property_type}`)
  if (input.bedroom) parts.push(`Bedroom: ${input.bedroom}`)
  if (input.reg_type) parts.push(`Reg type: ${input.reg_type}`)
  if (input.price_aed) parts.push(`Price (AED): ${input.price_aed}`)
  if (input.unit_sqft) parts.push(`Size (sqft): ${input.unit_sqft}`)
  if (input.handover_months) parts.push(`Handover (months): ${input.handover_months}`)
  return parts.join(" | ")
}

// Animated segmented control with sliding pill and glow
function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  disabled,
  formatLabel,
}: {
  options: T[]
  value: T
  onChange: (v: T) => void
  disabled?: boolean
  formatLabel?: (v: T) => string
}) {
  const selectedIndex = options.indexOf(value)
  
  return (
    <div className="relative rounded-2xl p-1.5 bg-white/40 backdrop-blur-xl border border-white/60 shadow-[inset_0_1px_0_rgba(255,255,255,0.8),0_4px_16px_-4px_rgba(0,0,0,0.1)]">
      {/* Glow behind pill */}
      <motion.div
        className="absolute top-1.5 bottom-1.5 rounded-xl bg-gradient-to-r from-brand-400 to-cyan-400 blur-lg opacity-40"
        style={{ width: `calc(${100 / options.length}% - 8px)` }}
        animate={{ x: `calc(${selectedIndex * 100}% + ${selectedIndex * 6}px + 2px)` }}
        transition={{ type: "spring", stiffness: 500, damping: 35 }}
      />
      
      {/* Sliding pill indicator with gradient */}
      <motion.div
        className="absolute top-1.5 bottom-1.5 rounded-xl bg-gradient-to-br from-brand-500 via-brand-600 to-cyan-600 shadow-[0_4px_20px_-4px_rgba(14,165,233,0.5),inset_0_1px_0_rgba(255,255,255,0.2)]"
        style={{ width: `calc(${100 / options.length}% - 8px)` }}
        animate={{ x: `calc(${selectedIndex * 100}% + ${selectedIndex * 6}px + 2px)` }}
        transition={{ type: "spring", stiffness: 500, damping: 35 }}
      />
      
      <div className="relative z-10 grid" style={{ gridTemplateColumns: `repeat(${options.length}, 1fr)` }}>
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            disabled={disabled}
            onClick={() => onChange(opt)}
            className={cn(
              "h-9 rounded-xl text-xs font-bold transition-all duration-200",
              value === opt 
                ? "text-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]" 
                : "text-slate-600 hover:text-slate-900",
              disabled && "cursor-not-allowed opacity-60"
            )}
          >
            {formatLabel ? formatLabel(opt) : opt}
          </button>
        ))}
      </div>
    </div>
  )
}

export function PropertyFormInput({
  onSend,
  disabled = false,
  initialCollapsed = false,
}: {
  onSend: (query: string) => void
  disabled?: boolean
  initialCollapsed?: boolean
}) {
  const [collapsed, setCollapsed] = React.useState(initialCollapsed)
  const [areas, setAreas] = React.useState<AreaItem[]>([])
  const [developers, setDevelopers] = React.useState<ComboboxItem[]>([])
  const [loadingLookups, setLoadingLookups] = React.useState(true)

  // Sync collapsed state when initialCollapsed prop changes (e.g., after user submits)
  React.useEffect(() => {
    if (initialCollapsed) {
      setCollapsed(true)
    }
  }, [initialCollapsed])

  const [area, setArea] = React.useState<string>("")
  const [developer, setDeveloper] = React.useState<string>("")
  const [bedroom, setBedroom] = React.useState<string>("2BR")
  const [regType, setRegType] = React.useState<"OffPlan" | "Ready">("OffPlan")
  const [propertyType, setPropertyType] = React.useState<"Unit" | "Villa">("Unit")
  const [priceAed, setPriceAed] = React.useState<string>("")
  const [unitSqft, setUnitSqft] = React.useState<string>("")
  const [handoverMonths, setHandoverMonths] = React.useState<string>("")

  React.useEffect(() => {
    let cancelled = false
    const load = async () => {
      setLoadingLookups(true)
      try {
        const [aRes, dRes] = await Promise.all([
          fetch("/api/lookups/areas"),
          fetch("/api/lookups/developers"),
        ])
        const [aJson, dJson] = await Promise.all([aRes.json(), dRes.json()])
        if (cancelled) return
        setAreas(aJson.items ?? [])
        setDevelopers(dJson.items ?? [])
      } catch (e) {
        console.error("Failed to load lookups:", e)
      } finally {
        if (!cancelled) setLoadingLookups(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  const areaItems: ComboboxItem[] = React.useMemo(() => {
    return areas.map((a) => ({
      value: a.value,
      label: a.label,
      subtitle: a.dld_area ? `DLD: ${a.dld_area}` : undefined,
    }))
  }, [areas])

  const canSubmit = area.trim().length > 0 && !disabled && !loadingLookups

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!canSubmit) return
    const q = buildDeterministicQuery({
      area,
      developer: developer || undefined,
      bedroom,
      reg_type: regType,
      property_type: propertyType,
      price_aed: priceAed || undefined,
      unit_sqft: unitSqft || undefined,
      handover_months: regType === "OffPlan" ? handoverMonths || undefined : undefined,
    })
    onSend(q)
    setCollapsed(true)
  }

  return (
    <form onSubmit={submit}>
      <div className="relative group">
        {/* Outer glow ring - always visible */}
        <div className="pointer-events-none absolute -inset-1 rounded-[30px] bg-gradient-to-br from-brand-400/30 via-cyan-400/20 to-brand-500/30 blur-xl opacity-60 group-hover:opacity-100 transition-opacity duration-500" />
        
        {/* Gradient border - visible rainbow-ish */}
        <div className="pointer-events-none absolute -inset-[1.5px] rounded-[28px] bg-gradient-to-br from-brand-400/60 via-cyan-300/40 to-brand-300/50" />

        <div
          className={cn(
            "relative rounded-[26px] p-4",
            // Strong frosted glass effect
            "bg-white/70 backdrop-blur-2xl backdrop-saturate-[1.8]",
            "border border-white/80",
            // Multi-layer shadows for depth
            "shadow-[0_25px_60px_-15px_rgba(0,0,0,0.2),0_10px_30px_-10px_rgba(14,165,233,0.15),inset_0_2px_0_rgba(255,255,255,0.95)]",
            "transition-all duration-300"
          )}
        >
          {/* Inner highlight line - more visible */}
          <div className="pointer-events-none absolute inset-x-4 top-0 h-[2px] bg-gradient-to-r from-transparent via-white/90 to-transparent rounded-full" />
          
          <div className={cn("flex items-center justify-between gap-3 px-1", collapsed ? "pb-1" : "pb-4")}>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <div className="text-sm font-bold text-slate-800">Property details</div>
                <div className="flex items-center justify-center w-5 h-5 rounded-full bg-gradient-to-br from-brand-400 to-brand-600 shadow-[0_2px_8px_rgba(14,165,233,0.4)]">
                  <Sparkles className="h-3 w-3 text-white" />
                </div>
              </div>
              <div className="text-xs text-slate-500 mt-0.5">
                Structured inputs for consistent analysis
              </div>
            </div>
            <div className="text-xs font-medium">
              {loadingLookups ? (
                <span className="flex items-center gap-2 text-brand-600">
                  <span className="h-2 w-2 rounded-full bg-brand-500 animate-pulse shadow-[0_0_8px_rgba(14,165,233,0.6)]" />
                  Loading…
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-full bg-slate-900/5 text-slate-600 border border-slate-200/50">
                  {areas.length} areas
                </span>
              )}
            </div>
          </div>

          {collapsed ? (
            <div className="flex flex-wrap items-center gap-2 px-1 pt-2">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                {[
                  area || "Area",
                  developer,
                  bedroom,
                  regType === "OffPlan" ? "Off-plan" : "Ready",
                  priceAed ? `AED ${priceAed}` : null,
                  unitSqft ? `${unitSqft} sqft` : null,
                ].filter(Boolean).map((val, i) => (
                  <motion.span
                    key={val}
                    initial={{ opacity: 0, scale: 0.9, y: 5 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ delay: i * 0.04, type: "spring", stiffness: 400 }}
                    className="rounded-full px-3 py-1.5 font-semibold text-slate-700 bg-white/80 backdrop-blur-sm border border-slate-200/70 shadow-[0_2px_8px_rgba(0,0,0,0.06),inset_0_1px_0_rgba(255,255,255,1)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] hover:border-brand-200/50 transition-all duration-200 cursor-default"
                  >
                    {val}
                  </motion.span>
                ))}
              </div>

              <div className="ml-auto flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setCollapsed(false)}
                  className="inline-flex h-9 items-center justify-center gap-2 rounded-xl px-4 text-xs font-semibold text-brand-700 bg-gradient-to-b from-brand-50 to-brand-100/80 border border-brand-200/60 shadow-[0_2px_8px_rgba(14,165,233,0.15),inset_0_1px_0_rgba(255,255,255,0.8)] transition-all duration-200 hover:shadow-[0_4px_16px_rgba(14,165,233,0.25)] hover:from-brand-100 hover:to-brand-150/80 active:scale-[0.97]"
                >
                  <Pencil className="h-3.5 w-3.5" />
                  Edit
                </button>
                <button
                  type="button"
                  onClick={() => setCollapsed(false)}
                  className="inline-flex h-9 w-9 items-center justify-center rounded-xl text-slate-500 bg-white/80 backdrop-blur-sm border border-slate-200/70 shadow-[0_2px_8px_rgba(0,0,0,0.06),inset_0_1px_0_rgba(255,255,255,1)] transition-all duration-200 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] hover:text-slate-700 hover:bg-white active:scale-[0.95]"
                  aria-label="Expand"
                >
                  <ChevronUp className="h-4 w-4" />
                </button>
              </div>
            </div>
          ) : null}

          <AnimatePresence>
          {!collapsed && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="grid grid-cols-1 gap-3 md:grid-cols-2"
          >
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.02 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Area <span className="text-rose-400">*</span>
              </label>
              <Combobox
                items={areaItems}
                value={area || undefined}
                onChange={setArea}
                placeholder="Select area (search)…"
                disabled={disabled || loadingLookups}
                className="glass-input"
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.04 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Developer <span className="text-slate-400">(optional)</span>
              </label>
              <Combobox
                items={developers}
                value={developer || undefined}
                onChange={setDeveloper}
                placeholder="Select developer (search)…"
                disabled={disabled || loadingLookups}
                className="glass-input"
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.06 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Bedrooms
              </label>
              <SegmentedControl
                options={["Studio", "1BR", "2BR", "3BR", "4BR", "5BR"]}
                value={bedroom}
                onChange={setBedroom}
                disabled={disabled}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Off-plan / Ready
              </label>
              <SegmentedControl
                options={["OffPlan", "Ready"] as const}
                value={regType}
                onChange={setRegType}
                disabled={disabled}
                formatLabel={(v) => v === "OffPlan" ? "Off-plan" : "Ready"}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Price (AED)
              </label>
              <input
                value={priceAed}
                onChange={(e) => setPriceAed(onlyDigits(e.target.value))}
                inputMode="numeric"
                pattern="[0-9]*"
                placeholder="e.g. 2,200,000"
                disabled={disabled}
                className="glass-input h-11 w-full rounded-2xl px-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.12 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Size (sqft)
              </label>
              <input
                value={unitSqft}
                onChange={(e) => setUnitSqft(onlyDigits(e.target.value))}
                inputMode="numeric"
                pattern="[0-9]*"
                placeholder="e.g. 1,250"
                disabled={disabled}
                className="glass-input h-11 w-full rounded-2xl px-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
              />
              <div className="mt-1 text-[10px] text-slate-400">
                Numbers only — no units
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.14 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Property type
              </label>
              <SegmentedControl
                options={["Unit", "Villa"] as const}
                value={propertyType}
                onChange={setPropertyType}
                disabled={disabled}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.16 }}
            >
              <label className="mb-1.5 block text-xs font-medium text-slate-500">
                Handover (months){" "}
                <span className="text-slate-400">
                  {regType === "OffPlan" ? "" : "— off-plan only"}
                </span>
              </label>
              <input
                value={handoverMonths}
                onChange={(e) => setHandoverMonths(onlyDigits(e.target.value))}
                inputMode="numeric"
                pattern="[0-9]*"
                placeholder={regType === "OffPlan" ? "e.g. 18" : "—"}
                disabled={disabled || regType !== "OffPlan"}
                className="glass-input h-11 w-full rounded-2xl px-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
              />
            </motion.div>
          </motion.div>
          )}
          </AnimatePresence>

          {!collapsed && (
            <motion.div 
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.18 }}
              className="mt-4 flex items-center gap-2"
            >
              <button
                type="submit"
                disabled={!canSubmit}
                className={cn(
                  "group relative ml-auto inline-flex h-12 items-center justify-center gap-2.5 rounded-2xl px-6 text-sm font-bold transition-all duration-300 overflow-hidden",
                  "bg-gradient-to-r from-brand-600 via-brand-500 to-cyan-500 text-white",
                  "shadow-[0_8px_30px_-6px_rgba(14,165,233,0.5),inset_0_1px_0_rgba(255,255,255,0.2)]",
                  "hover:shadow-[0_12px_40px_-6px_rgba(14,165,233,0.6)] hover:scale-[1.03]",
                  "active:scale-[0.97]",
                  "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100 disabled:bg-slate-400 disabled:shadow-none"
                )}
              >
                {/* Shimmer effect */}
                <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent group-hover:animate-shimmer" />
                
                {/* Glow effect */}
                <span className="absolute inset-0 rounded-2xl bg-gradient-to-t from-black/10 to-transparent" />
                
                <Send className="relative h-4 w-4" />
                <span className="relative">Analyze</span>
              </button>
            </motion.div>
          )}
        </div>
      </div>
    </form>
  )
}

export default PropertyFormInput


