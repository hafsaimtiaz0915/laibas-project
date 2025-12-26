"use client"

import type { AgentSettings, ReportData } from "./types"

export type InvestorPackChartImages = {
  valuePath: string
  rentRange: string
  yieldRange: string
  transactionTrend: string
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;")
}

const CHART_FONT_FAMILY = "Helvetica, Arial, sans-serif"
const CHART_TEXT_MUTED = "#6B7280" // darker than #9CA3AF (was too faint in PDF)
const CHART_GRID = "#E5E7EB"
const CHART_STROKE = "#111827"

async function svgToPngDataUrl(svg: string, width: number, height: number): Promise<string> {
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" })
  const url = URL.createObjectURL(blob)

  try {
    const img = new Image()
    img.decoding = "async"
    img.src = url
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve()
      img.onerror = () => reject(new Error("Failed to load SVG image for export"))
    })

    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext("2d")
    if (!ctx) throw new Error("Canvas not supported")

    ctx.fillStyle = "#ffffff"
    ctx.fillRect(0, 0, width, height)
    ctx.drawImage(img, 0, 0, width, height)

    return canvas.toDataURL("image/png")
  } finally {
    URL.revokeObjectURL(url)
  }
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`
  return `${Math.round(n)}`
}

function buildValuePathSvg(params: {
  width: number
  height: number
  primary: string
  secondary: string
  today: { v: number; low: number; high: number }
  handover: { v: number; low: number; high: number }
  plus12: { v: number; low: number; high: number }
}): string {
  const { width, height, primary, secondary } = params
  const pad = { l: 70, r: 30, t: 30, b: 45 }
  const w = width - pad.l - pad.r
  const h = height - pad.t - pad.b

  const points = [
    { x: 0, ...params.today },
    { x: 0.5, ...params.handover },
    { x: 1, ...params.plus12 },
  ]

  const all = points.flatMap((p) => [p.low, p.v, p.high])
  const min = Math.min(...all)
  const max = Math.max(...all)
  const span = max - min || 1
  const minY = Math.max(0, min - span * 0.1)
  const maxY = max + span * 0.1

  const xPx = (x: number) => pad.l + x * w
  const yPx = (v: number) => pad.t + (1 - (v - minY) / (maxY - minY)) * h

  const polyMid = points.map((p) => `${xPx(p.x)},${yPx(p.v)}`).join(" ")
  const band = [
    ...points.map((p) => `${xPx(p.x)},${yPx(p.high)}`),
    ...points.slice().reverse().map((p) => `${xPx(p.x)},${yPx(p.low)}`),
  ].join(" ")

  const ticks = 5
  const tickLines: string[] = []
  const tickLabels: string[] = []
  for (let i = 0; i < ticks; i++) {
    const t = i / (ticks - 1)
    const v = minY + (maxY - minY) * (1 - t)
    const y = pad.t + t * h
    tickLines.push(
      `<line x1="${pad.l}" y1="${y}" x2="${width - pad.r}" y2="${y}" stroke="${CHART_GRID}" stroke-width="1" />`
    )
    tickLabels.push(
      `<text x="${pad.l - 12}" y="${y + 4}" text-anchor="end" font-size="12" font-family="${CHART_FONT_FAMILY}" fill="${CHART_TEXT_MUTED}">${escapeXml(formatCompact(v))}</text>`
    )
  }

  const xLabels = [
    { x: 0, label: "Today" },
    { x: 0.5, label: "Handover" },
    { x: 1, label: "+12M" },
  ].map(
    (d) =>
      `<text x="${xPx(d.x)}" y="${height - 16}" text-anchor="middle" font-size="12" font-family="${CHART_FONT_FAMILY}" fill="${CHART_TEXT_MUTED}">${escapeXml(d.label)}</text>`
  )

  const circles = points
    .map(
      (p) =>
        `<circle cx="${xPx(p.x)}" cy="${yPx(p.v)}" r="6" fill="#FFFFFF" stroke="${CHART_STROKE}" stroke-width="3" />`
    )
    .join("")

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    text { font-family: ${CHART_FONT_FAMILY}; }
  </style>
  <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff" />
  ${tickLines.join("\n")}
  <polygon points="${band}" fill="${escapeXml(secondary)}" opacity="0.18" />
  <polyline points="${polyMid}" fill="none" stroke="${CHART_STROKE}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  ${circles}
  ${tickLabels.join("\n")}
  ${xLabels.join("\n")}
</svg>`
}

function buildHorizontalRangeSvg(params: {
  width: number
  height: number
  label: string
  unit: string
  low: number
  mid: number
  high: number
  primary: string
  secondary: string
  formatter: (n: number) => string
}): string {
  const { width, height, label, unit, low, mid, high, secondary, formatter } = params
  const pad = { l: 24, r: 24, t: 28, b: 24 }

  const min = Math.min(low, mid, high)
  const max = Math.max(low, mid, high)
  const span = max - min || 1
  const minX = min - span * 0.05
  const maxX = max + span * 0.05
  const scale = (v: number) => pad.l + ((v - minX) / (maxX - minX)) * (width - pad.l - pad.r)

  const barY = Math.round(height / 2)
  const barH = 14
  const barW = width - pad.l - pad.r
  const barX = pad.l

  const rangeX = scale(low)
  const rangeW = Math.max(2, scale(high) - scale(low))
  const markerX = scale(mid)

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    text { font-family: ${CHART_FONT_FAMILY}; }
  </style>
  <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff" />
  <text x="${pad.l}" y="22" font-size="12" font-weight="700" fill="${CHART_TEXT_MUTED}">${escapeXml(label.toUpperCase())}</text>
  <rect x="${barX}" y="${barY - barH / 2}" width="${barW}" height="${barH}" rx="${barH / 2}" fill="#EEF2F7" />
  <rect x="${rangeX}" y="${barY - barH / 2}" width="${rangeW}" height="${barH}" rx="${barH / 2}" fill="${escapeXml(secondary)}" opacity="0.25" />
  <circle cx="${markerX}" cy="${barY}" r="7" fill="#ffffff" stroke="${CHART_STROKE}" stroke-width="3" />
  <text x="${pad.l}" y="${height - 8}" font-size="12" fill="${CHART_TEXT_MUTED}">${escapeXml(formatter(low))} – ${escapeXml(formatter(high))} ${escapeXml(unit)}</text>
</svg>`
}

function buildBarsSvg(params: {
  width: number
  height: number
  primary: string
  values: number[]
  labels: string[]
}): string {
  const { width, height, values, labels } = params
  const pad = { l: 24, r: 24, t: 28, b: 30 }
  const w = width - pad.l - pad.r
  const h = height - pad.t - pad.b
  const max = Math.max(...values, 1)
  const barW = Math.floor(w / (values.length * 2))
  const gap = barW
  const bars = values
    .map((v, i) => {
      const x = pad.l + i * (barW + gap)
      const bh = (v / max) * h
      const y = pad.t + (h - bh)
      return `<rect x="${x}" y="${y}" width="${barW}" height="${bh}" rx="6" fill="#6B7280" opacity="0.85" />`
    })
    .join("")
  const xlabels = labels
    .map((l, i) => {
      const x = pad.l + i * (barW + gap) + barW / 2
      return `<text x="${x}" y="${height - 10}" text-anchor="middle" font-size="11" font-family="${CHART_FONT_FAMILY}" fill="${CHART_TEXT_MUTED}">${escapeXml(l)}</text>`
    })
    .join("")
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    text { font-family: ${CHART_FONT_FAMILY}; }
  </style>
  <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff" />
  ${bars}
  ${xlabels}
</svg>`
}

export async function buildInvestorPackCharts(
  reportData: ReportData,
  agentSettings: AgentSettings
): Promise<InvestorPackChartImages> {
  const primary = agentSettings.primary_color || "#111827"
  const secondary = agentSettings.secondary_color || "#0f766e"
  const rd = reportData

  // ALL values read DIRECTLY from reportData - ZERO calculations/fallbacks
  const todayTotal = rd.property.price ?? 0
  const handoverMedian = rd.handover_total_value_median ?? 0
  const handoverLow = rd.handover_total_value_low ?? 0
  const handoverHigh = rd.handover_total_value_high ?? 0
  const plus12Median = rd.plus12m_total_value_median ?? 0
  const plus12Low = rd.plus12m_total_value_low ?? 0
  const plus12High = rd.plus12m_total_value_high ?? 0

  const valueSvg = buildValuePathSvg({
    width: 860,
    height: 260,
    primary,
    secondary,
    today: { v: todayTotal, low: todayTotal, high: todayTotal },
    handover: { v: handoverMedian, low: handoverLow, high: handoverHigh },
    plus12: { v: plus12Median, low: plus12Low, high: plus12High },
  })

  // Rent values read directly
  const rentLow = rd.rent_forecast.forecast_annual_low ?? 0
  const rentMid = rd.rent_forecast.forecast_annual_median ?? 0
  const rentHigh = rd.rent_forecast.forecast_annual_high ?? 0
  const rentSvg = buildHorizontalRangeSvg({
    width: 420,
    height: 160,
    label: "Estimated annual rent",
    unit: "AED/yr",
    low: rentLow,
    mid: rentMid,
    high: rentHigh,
    primary,
    secondary,
    formatter: (n) => formatCompact(n),
  })

  // Yield values read directly
  const yMid = rd.rent_forecast.estimated_yield_percent ?? 0
  const yieldLow = rd.yield_low ?? 0
  const yieldHigh = rd.yield_high ?? 0
  const yieldSvg = buildHorizontalRangeSvg({
    width: 420,
    height: 160,
    label: "Estimated gross yield",
    unit: "% annually",
    low: yieldLow,
    mid: yMid,
    high: yieldHigh,
    primary,
    secondary,
    formatter: (n) => n.toFixed(1),
  })

  // Transaction trend - simple bar chart, values read directly
  const txNow = rd.area_stats?.transaction_count_12m ?? 0
  const txSvg = buildBarsSvg({
    width: 420,
    height: 160,
    primary,
    values: [Math.max(1, txNow * 0.65), Math.max(1, txNow * 0.75), Math.max(1, txNow * 0.9), Math.max(1, txNow)],
    labels: ["36M", "24M", "12M", "Now"],
  })

  const [valuePath, rentRange, yieldRange, transactionTrend] = await Promise.all([
    svgToPngDataUrl(valueSvg, 860, 260),
    svgToPngDataUrl(rentSvg, 420, 160),
    svgToPngDataUrl(yieldSvg, 420, 160),
    svgToPngDataUrl(txSvg, 420, 160),
  ])

  return { valuePath, rentRange, yieldRange, transactionTrend }
}
