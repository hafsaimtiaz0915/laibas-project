"use client"

import React from "react"
import { cn } from "@/lib/utils"
import { SiriOrb } from "@/components/ui/siri-orb"

export function MobilePropertyDetailsDock({
  onOpen,
  className,
  disabled = false,
}: {
  onOpen: () => void
  className?: string
  disabled?: boolean
}) {
  return (
    <div
      className={cn(
        // Lift above Safari's bottom chrome + respect safe-area.
        "sm:hidden fixed left-0 right-0 z-50 px-4",
        className
      )}
      style={{ bottom: "calc(env(safe-area-inset-bottom) + 12px)" }}
    >
      <button
        type="button"
        onClick={onOpen}
        disabled={disabled}
        className={cn(
          "w-full rounded-3xl border border-slate-200/70 bg-white/85 backdrop-blur-xl",
          "shadow-[0_10px_40px_-20px_rgba(0,0,0,0.35)]",
          "px-4 py-3 flex items-center gap-3 transition",
          "active:scale-[0.99]",
          "disabled:opacity-60 disabled:active:scale-100"
        )}
        aria-label="Enter property details"
      >
        <SiriOrb size="26px" className="shrink-0" />
        <div className="flex-1 text-left min-w-0">
          <div className="text-sm font-semibold text-slate-900 truncate">Enter property details</div>
          <div className="text-xs text-slate-500 truncate">Area, bedrooms, price, size…</div>
        </div>
        <div className="text-xs font-medium text-slate-600">Open</div>
      </button>
    </div>
  )
}

export default MobilePropertyDetailsDock


