import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCurrency(value?: number | null, suffix: string = ''): string {
  if (!value) return 'N/A'
  if (value >= 1_000_000) {
    return `AED ${(value / 1_000_000).toFixed(2)}M${suffix}`
  } else if (value >= 1_000) {
    return `AED ${Math.round(value / 1_000)}K${suffix}`
  }
  return `AED ${Math.round(value)}${suffix}`
}

export function formatNumber(value?: number | null): string {
  if (!value) return 'N/A'
  return new Intl.NumberFormat('en-AE').format(value)
}
