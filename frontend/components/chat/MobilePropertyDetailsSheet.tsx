"use client"

import React from "react"
import { AnimatePresence, motion } from "framer-motion"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"

export function MobilePropertyDetailsSheet({
  open,
  title = "Property details",
  subtitle = "Structured inputs for consistent analysis.",
  onClose,
  children,
}: {
  open: boolean
  title?: string
  subtitle?: string
  onClose: () => void
  children: React.ReactNode
}) {
  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.button
            type="button"
            className="sm:hidden fixed inset-0 z-50 bg-black/30 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            aria-label="Close property details"
          />

          {/* Sheet */}
          <motion.div
            className={cn(
              "sm:hidden fixed left-0 right-0 bottom-0 z-[60]",
              "rounded-t-[28px] border-t border-slate-200/30",
              "bg-gradient-to-b from-white/90 to-white/80 backdrop-blur-2xl",
              "shadow-[0_-25px_60px_-20px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.8)]"
            )}
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 420, damping: 40 }}
            role="dialog"
            aria-modal="true"
          >
            <div className="px-4 pt-3 pb-2">
              <div className="mx-auto h-1.5 w-12 rounded-full bg-gradient-to-r from-slate-300/50 via-slate-400/50 to-slate-300/50" />
              <div className="mt-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-base font-semibold text-slate-900">{title}</div>
                  <div className="text-xs text-slate-500">{subtitle}</div>
                </div>
                <button
                  type="button"
                  onClick={onClose}
                  className="h-10 w-10 rounded-full flex items-center justify-center transition-all duration-200 bg-gradient-to-b from-white to-slate-50/80 border border-slate-200/50 shadow-[0_1px_3px_rgba(0,0,0,0.04),inset_0_1px_0_rgba(255,255,255,1)] hover:shadow-[0_2px_8px_rgba(0,0,0,0.08)] hover:border-slate-300/60 active:scale-95"
                  aria-label="Close"
                >
                  <X className="h-5 w-5 text-slate-600" />
                </button>
              </div>
            </div>

            <div className="max-h-[78svh] overflow-y-auto px-4 pb-[max(18px,env(safe-area-inset-bottom))] custom-scrollbar">
              {children}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

export default MobilePropertyDetailsSheet





