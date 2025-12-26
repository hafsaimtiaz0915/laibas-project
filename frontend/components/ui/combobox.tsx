"use client"

import * as React from "react"
import * as Popover from "@radix-ui/react-popover"
import { cn } from "@/lib/utils"
import { ChevronDown, Check, Search } from "lucide-react"

export type ComboboxItem = {
  value: string
  label: string
  subtitle?: string
}

export function Combobox({
  items,
  value,
  onChange,
  placeholder = "Select…",
  disabled = false,
  className,
  contentClassName,
  searchPlaceholder = "Search…",
  emptyText = "No matches",
}: {
  items: ComboboxItem[]
  value?: string
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  className?: string
  contentClassName?: string
  searchPlaceholder?: string
  emptyText?: string
}) {
  const [open, setOpen] = React.useState(false)
  const [query, setQuery] = React.useState("")
  const [isMobile, setIsMobile] = React.useState(false)

  // Mobile UX: avoid auto-focusing inputs (keyboard takeover) and behave like a simple dropdown.
  React.useEffect(() => {
    const mql = window.matchMedia("(max-width: 640px)")
    const update = () => setIsMobile(Boolean(mql.matches))
    update()
    mql.addEventListener?.("change", update)
    return () => mql.removeEventListener?.("change", update)
  }, [])

  const selected = React.useMemo(
    () => items.find((i) => i.value === value),
    [items, value]
  )

  const filtered = React.useMemo(() => {
    // On mobile, we hide search to prevent keyboard takeover. Show the full list.
    if (isMobile) return items
    const q = query.trim().toLowerCase()
    if (!q) return items
    return items.filter((i) => {
      const hay = `${i.label} ${i.subtitle ?? ""}`.toLowerCase()
      return hay.includes(q)
    })
  }, [items, query, isMobile])

  const handleSelect = (v: string) => {
    onChange(v)
    setOpen(false)
    setQuery("")
  }

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "group inline-flex h-11 w-full items-center justify-between rounded-2xl px-4 text-left text-sm text-slate-900 transition-all duration-200",
            "bg-gradient-to-b from-white/80 to-slate-50/60 backdrop-blur-sm",
            "border border-slate-200/50",
            "shadow-[inset_0_1px_0_rgba(255,255,255,0.9),0_1px_3px_rgba(0,0,0,0.04)]",
            "hover:border-slate-300/60 hover:shadow-[inset_0_1px_0_rgba(255,255,255,1),0_2px_8px_rgba(0,0,0,0.06)]",
            "focus:outline-none focus-visible:border-brand-300 focus-visible:ring-[3px] focus-visible:ring-brand-100/50",
            "data-[state=open]:border-brand-300 data-[state=open]:ring-[3px] data-[state=open]:ring-brand-100/50",
            "disabled:cursor-not-allowed disabled:opacity-60",
            className
          )}
        >
          <span className={cn("truncate", !selected && "text-slate-400")}>
            {selected ? selected.label : placeholder}
          </span>
          <ChevronDown className="h-4 w-4 text-slate-400 transition-transform duration-200 group-data-[state=open]:rotate-180" />
        </button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          align="start"
          side="bottom"
          sideOffset={8}
          className={cn(
            // Must sit above mobile sheets/drawers.
            "z-[120] w-[--radix-popover-trigger-width] rounded-2xl p-2",
            "bg-white/95 backdrop-blur-xl",
            "border border-slate-200/60",
            "shadow-[0_10px_40px_-10px_rgba(0,0,0,0.2),0_4px_20px_-4px_rgba(0,0,0,0.1)]",
            "animate-in fade-in-0 zoom-in-95 slide-in-from-top-2",
            contentClassName
          )}
        >
          {!isMobile && (
            <div className="flex items-center gap-2 rounded-xl px-3 py-2.5 bg-gradient-to-b from-slate-100/80 to-slate-100/40 border border-slate-200/50 shadow-[inset_0_1px_0_rgba(255,255,255,0.8)]">
              <Search className="h-4 w-4 text-slate-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={searchPlaceholder}
                className="w-full bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
              />
            </div>
          )}

          <div className={cn("custom-scrollbar overflow-auto", isMobile ? "max-h-[50svh]" : "mt-2 max-h-64")}>
            {filtered.length === 0 ? (
              <div className="px-3 py-2 text-sm text-slate-500">
                {emptyText}
              </div>
            ) : (
              <div className="flex flex-col gap-0.5">
                {filtered.map((i) => {
                  const isSelected = i.value === value
                  return (
                    <button
                      key={i.value}
                      type="button"
                      onClick={() => handleSelect(i.value)}
                      className={cn(
                        "flex w-full items-start gap-2 rounded-xl px-3 py-2 text-left text-sm transition-all duration-150",
                        "hover:bg-gradient-to-b hover:from-slate-100/80 hover:to-slate-50/40",
                        "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-100",
                        isSelected && "bg-gradient-to-b from-brand-50/60 to-brand-50/30"
                      )}
                    >
                      <span className={cn(
                        "mt-0.5 h-4 w-4 shrink-0 transition-colors",
                        isSelected ? "text-brand-600" : "text-transparent"
                      )}>
                        <Check className="h-4 w-4" />
                      </span>
                      <span className="min-w-0">
                        <span className={cn(
                          "block truncate",
                          isSelected ? "text-brand-700 font-medium" : "text-slate-900"
                        )}>
                          {i.label}
                        </span>
                        {i.subtitle ? (
                          <span className="block truncate text-xs text-slate-500">
                            {i.subtitle}
                          </span>
                        ) : null}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}


