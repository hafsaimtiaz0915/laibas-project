import React from "react"
import { Document, Page, Text, View, StyleSheet, Image } from "@react-pdf/renderer"
import type { AgentSettings, ReportData } from "@/lib/types"
import type { InvestorPackChartImages } from "@/lib/chart-export"

type Props = {
  reportData: ReportData
  agentSettings: AgentSettings
  chartImages: InvestorPackChartImages
}

// Formatters - display only, no calculations
function aed(n?: number | null): string {
  if (n === undefined || n === null) return "N/A"
  return `AED ${Math.round(n).toLocaleString("en-AE")}`
}

function aedCompact(n?: number | null): string {
  if (n === undefined || n === null) return "N/A"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`
  return `${Math.round(n)}`
}

function signedAedCompact(n?: number | null): string {
  if (n === undefined || n === null) return "N/A"
  const sign = n > 0 ? "+" : n < 0 ? "-" : ""
  return `${sign}${aedCompact(Math.abs(n))}`
}

function pct(n?: number | null, digits = 1): string {
  if (n === undefined || n === null) return "N/A"
  const prefix = n >= 0 ? "+" : ""
  return `${prefix}${n.toFixed(digits)}%`
}

const styles = StyleSheet.create({
  page: {
    flexDirection: "column",
    backgroundColor: "#ffffff",
    paddingTop: 36,
    paddingBottom: 36,
    paddingHorizontal: 40,
    fontFamily: "Helvetica",
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
  },
  brandRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  brandLogo: { width: 32, height: 32, borderRadius: 8, objectFit: "cover" },
  brandName: { fontSize: 11, color: "#111827", fontWeight: 700 },
  headerTitle: { textAlign: "right" },
  headerH1: { fontSize: 13, color: "#111827", fontWeight: 700 },
  headerDate: { fontSize: 9, color: "#9CA3AF", marginTop: 2 },

  card: {
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 12,
    padding: 14,
    backgroundColor: "#ffffff",
  },
  detailsGrid: { flexDirection: "row", flexWrap: "wrap", gap: 12 },
  detailItem: { width: "31%" },
  label: { fontSize: 8, color: "#6B7280", letterSpacing: 0.4 },
  valueBold: { fontSize: 11, color: "#111827", marginTop: 4, fontWeight: 700 },

  tilesGrid: { flexDirection: "row", flexWrap: "wrap", gap: 12, marginTop: 14 },
  tile: {
    width: "48%",
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 12,
    padding: 12,
  },
  tileTitle: { fontSize: 8, color: "#6B7280", letterSpacing: 0.3 },
  tileBig: { fontSize: 22, fontWeight: 700, color: "#111827", marginTop: 6 },
  tileUnit: { fontSize: 9, color: "#6B7280", marginLeft: 4 },
  tileRange: { fontSize: 9, color: "#9CA3AF", marginTop: 4 },

  upliftBar: {
    marginTop: 14,
    borderRadius: 12,
    padding: 14,
    flexDirection: "row",
    justifyContent: "space-between",
  },
  upliftCol: { width: "48%" },
  upliftTitle: { fontSize: 8, color: "#D1D5DB", letterSpacing: 0.3 },
  upliftValue: { fontSize: 20, fontWeight: 700, color: "#ffffff", marginTop: 4 },
  upliftPct: { fontSize: 10, color: "#D1D5DB", marginTop: 2 },

  sectionTitle: { fontSize: 9, color: "#6B7280", letterSpacing: 0.5, marginTop: 14, marginBottom: 8 },
  miniGrid: { flexDirection: "row", gap: 10 },
  miniTile: { flex: 1, borderWidth: 1, borderColor: "#E5E7EB", borderRadius: 10, padding: 10 },
  miniBig: { fontSize: 13, fontWeight: 700, color: "#111827", marginTop: 4 },
  miniSub: { fontSize: 8, color: "#9CA3AF", marginTop: 1 },

  chartCard: { marginTop: 12, borderWidth: 1, borderColor: "#E5E7EB", borderRadius: 12, padding: 14 },
  chartTitle: { fontSize: 8, color: "#6B7280", letterSpacing: 0.4 },
  chartCaption: { fontSize: 8, color: "#9CA3AF", textAlign: "center", marginTop: 6 },

  page2Header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 16 },
  page2Title: { fontSize: 12, fontWeight: 700, color: "#111827" },
  page2Sub: { fontSize: 8, color: "#9CA3AF", marginTop: 2 },

  twoCol: { flexDirection: "row", gap: 14 },
  half: { flex: 1 },
  smallChart: { borderWidth: 1, borderColor: "#E5E7EB", borderRadius: 12, padding: 12 },

  driversRow: { flexDirection: "row", gap: 10, marginTop: 12, flexWrap: "wrap" },
  driverTile: {
    width: "23%",
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 10,
    padding: 8,
  },
  driverText: { fontSize: 9, color: "#111827" },

  footer: { marginTop: "auto", paddingTop: 12, borderTopWidth: 1, borderTopColor: "#E5E7EB" },
  footerText: { fontSize: 8, color: "#9CA3AF" },
})

export const InvestorPack: React.FC<Props> = ({ reportData, agentSettings, chartImages }) => {
  const dateStr = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })

  // ALL values read directly from reportData - ZERO calculations
  const rd = reportData
  const brandName = agentSettings.company_name || agentSettings.name || "Proprly"
  // Keep the layout colors stable; use brand colors in charts/logo, not large background blocks
  const upliftBg = "#0B0F3A"

  // Get forecast drivers (just reading array)
  const rawDrivers: string[] = (rd.model_attribution?.top_drivers || [])
    .slice(0, 4)
    .map((d: any) => d?.feature ? String(d.feature) : "")
    .filter(Boolean)

  // Map internal feature names to investor-friendly labels (no calculations; pure string mapping)
  const driverLabel = (feature: string): string => {
    const f = feature.toLowerCase()
    if (f.includes("supply") && (f.includes("pipeline") || f.includes("units"))) return "Supply pipeline (upcoming inventory)"
    if (f.includes("complet") || f.includes("handover")) return "Scheduled completions (supply landing soon)"
    if (f.includes("eibor") || f.includes("rate") || f.includes("interest")) return "Rates (EIBOR)"
    if (f.includes("transact") || f.includes("liquid")) return "Liquidity (segment transactions)"
    if (f.includes("market") && f.includes("liquid")) return "Liquidity (wider market)"
    return feature.replace(/_/g, " ")
  }

  const drivers = rawDrivers.map(driverLabel)

  const ti = rd.trend_insights as any
  const metric3m = ti?.price_change_3m
  const metric12m = ti?.price_change_12m
  const metricLiq = ti?.area_transaction_volume_trend

  const signedPct = (n: any): string => {
    if (n === undefined || n === null || Number.isNaN(Number(n))) return "N/A"
    const v = Number(n)
    const sign = v >= 0 ? "+" : ""
    return `${sign}${v.toFixed(1)}%`
  }

  return (
    <Document>
      {/* Page 1: Investment Snapshot */}
      <Page size="A4" style={styles.page}>
        <View style={styles.headerRow}>
          <View style={styles.brandRow}>
            {agentSettings.logo_url ? (
              <Image src={agentSettings.logo_url} style={styles.brandLogo} />
            ) : (
              <View style={[styles.brandLogo, { backgroundColor: upliftBg }]} />
            )}
            <Text style={styles.brandName}>{brandName}</Text>
          </View>
          <View style={styles.headerTitle}>
            <Text style={styles.headerH1}>Off-Plan Investment Snapshot</Text>
            <Text style={styles.headerDate}>{dateStr}</Text>
          </View>
        </View>

        {/* Deal Details */}
        <View style={styles.card}>
          <View style={styles.detailsGrid}>
            <View style={styles.detailItem}>
              <Text style={styles.label}>AREA</Text>
              <Text style={styles.valueBold}>{rd.property.area_display || rd.property.area || "N/A"}</Text>
            </View>
            <View style={styles.detailItem}>
              <Text style={styles.label}>DEVELOPER</Text>
              <Text style={styles.valueBold}>{rd.property.developer || "N/A"}</Text>
            </View>
            <View style={styles.detailItem}>
              <Text style={styles.label}>BEDROOMS</Text>
              <Text style={styles.valueBold}>{rd.property.bedroom || "N/A"}</Text>
            </View>
            <View style={styles.detailItem}>
              <Text style={styles.label}>UNIT SIZE</Text>
              <Text style={styles.valueBold}>{rd.property.unit_sqft ? `${Math.round(rd.property.unit_sqft).toLocaleString()} sqft` : "N/A"}</Text>
            </View>
            <View style={styles.detailItem}>
              <Text style={styles.label}>PURCHASE PRICE</Text>
              <Text style={styles.valueBold}>{aed(rd.property.price)}</Text>
            </View>
            <View style={styles.detailItem}>
              <Text style={styles.label}>STAGE</Text>
              <Text style={styles.valueBold}>{rd.property.reg_type || "Off-Plan"}</Text>
            </View>
          </View>
        </View>

        {/* Value Tiles - ALL values read directly from rd */}
        <View style={styles.tilesGrid}>
          <View style={styles.tile}>
            <Text style={styles.tileTitle}>ESTIMATED MARKET VALUE AT HANDOVER (AI FORECAST)</Text>
            <View style={{ flexDirection: "row", alignItems: "baseline" }}>
              <Text style={styles.tileBig}>{aedCompact(rd.handover_total_value_median)}</Text>
              <Text style={styles.tileUnit}>AED</Text>
            </View>
            <Text style={styles.tileRange}>
              {rd.handover_total_value_low != null && rd.handover_total_value_high != null
                ? `Forecast Range: ${aedCompact(rd.handover_total_value_low)} – ${aedCompact(rd.handover_total_value_high)}`
                : "Forecast Range: Not available"}
            </Text>
          </View>

          <View style={styles.tile}>
            <Text style={styles.tileTitle}>ESTIMATED MARKET VALUE 12M POST‑HANDOVER (AI FORECAST)</Text>
            <View style={{ flexDirection: "row", alignItems: "baseline" }}>
              <Text style={styles.tileBig}>{aedCompact(rd.plus12m_total_value_median)}</Text>
              <Text style={styles.tileUnit}>AED</Text>
            </View>
            <Text style={styles.tileRange}>
              {rd.plus12m_total_value_low != null && rd.plus12m_total_value_high != null
                ? `Forecast Range: ${aedCompact(rd.plus12m_total_value_low)} – ${aedCompact(rd.plus12m_total_value_high)}`
                : "Forecast Range: Not available"}
            </Text>
          </View>

          <View style={styles.tile}>
            <Text style={styles.tileTitle}>RENTAL OUTLOOK (POST‑HANDOVER)</Text>
            <View style={{ flexDirection: "row", alignItems: "baseline" }}>
              <Text style={styles.tileBig}>{aedCompact(rd.rent_forecast.forecast_annual_median)}</Text>
              <Text style={styles.tileUnit}>AED/yr</Text>
            </View>
            <Text style={styles.tileRange}>
              {rd.rent_forecast.forecast_annual_low != null && rd.rent_forecast.forecast_annual_high != null
                ? `Range: ${aedCompact(rd.rent_forecast.forecast_annual_low)} – ${aedCompact(rd.rent_forecast.forecast_annual_high)}`
                : "Range: Not available"}
            </Text>
          </View>

          <View style={styles.tile}>
            <Text style={styles.tileTitle}>YIELD OUTLOOK (POST‑HANDOVER)</Text>
            <View style={{ flexDirection: "row", alignItems: "baseline" }}>
              <Text style={styles.tileBig}>{rd.rent_forecast.estimated_yield_percent != null ? `${rd.rent_forecast.estimated_yield_percent.toFixed(1)}%` : "N/A"}</Text>
              <Text style={styles.tileUnit}>annually</Text>
            </View>
            <Text style={styles.tileRange}>
              {rd.yield_low != null && rd.yield_high != null
                ? `Range: ${rd.yield_low.toFixed(1)}% – ${rd.yield_high.toFixed(1)}%`
                : "Range: Not available"}
            </Text>
          </View>
        </View>

        {/* Uplift Bar - values read directly from rd */}
        <View style={[styles.upliftBar, { backgroundColor: upliftBg }]}>
          <View style={styles.upliftCol}>
            <Text style={styles.upliftTitle}>PROJECTED CAPITAL UPLIFT BY HANDOVER</Text>
            <Text style={styles.upliftValue}>{signedAedCompact(rd.uplift_handover)}</Text>
            <Text style={styles.upliftPct}>{rd.uplift_handover_percent != null ? pct(rd.uplift_handover_percent) : ""}</Text>
          </View>
          <View style={styles.upliftCol}>
            <Text style={styles.upliftTitle}>PROJECTED CAPITAL UPLIFT +12M POST‑HANDOVER</Text>
            <Text style={styles.upliftValue}>{signedAedCompact(rd.uplift_plus12m)}</Text>
            <Text style={styles.upliftPct}>{rd.uplift_plus12m_percent != null ? pct(rd.uplift_plus12m_percent) : ""}</Text>
          </View>
        </View>

        {/* Market Context */}
        <Text style={styles.sectionTitle}>AREA MARKET CONTEXT | INDEPENDENT MARKET SIGNALS</Text>
        <View style={styles.miniGrid}>
          <View style={styles.miniTile}>
            <Text style={styles.label}>12M PRICE CHANGE</Text>
            <Text style={styles.miniBig}>
              {rd.area_stats?.price_change_12m != null
                ? pct(rd.area_stats.price_change_12m)
                : "N/A"}
            </Text>
          </View>
          <View style={styles.miniTile}>
            <Text style={styles.label}>36M PRICE CHANGE</Text>
            <Text style={styles.miniBig}>
              {rd.area_stats?.price_change_36m != null
                ? pct(rd.area_stats.price_change_36m)
                : "N/A"}
            </Text>
          </View>
          <View style={styles.miniTile}>
            <Text style={styles.label}>TRANSACTIONS (12M)</Text>
            <Text style={styles.miniBig}>
              {rd.area_stats?.transaction_count_12m != null
                ? rd.area_stats.transaction_count_12m.toLocaleString()
                : "N/A"}
            </Text>
          </View>
          <View style={styles.miniTile}>
            <Text style={styles.label}>SUPPLY PIPELINE</Text>
            <Text style={styles.miniBig}>
              {rd.area_stats?.supply_pipeline != null
                ? rd.area_stats.supply_pipeline.toLocaleString()
                : "N/A"}
            </Text>
            <Text style={styles.miniSub}>units</Text>
          </View>
        </View>

        {/* Value Path Chart */}
        <View style={styles.chartCard}>
          <Text style={styles.chartTitle}>PROJECTED VALUE PATH</Text>
          <View style={{ marginTop: 8 }}>
            <Image src={chartImages.valuePath} style={{ width: 460, height: 130, alignSelf: "center" }} />
          </View>
          <Text style={styles.chartCaption}>Shaded area represents forecast range</Text>
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>IMPORTANT NOTICE: This report is generated using an independent, data-driven forecasting model.</Text>
          <Text style={styles.footerText}>Informational purposes only — not investment advice. • {brandName} • {dateStr}</Text>
          {rd.match_info?.type && rd.match_info?.confidence != null ? (
            <Text style={styles.footerText}>
              Match: {String(rd.match_info.type)} • Confidence: {Number(rd.match_info.confidence).toFixed(0)}%
            </Text>
          ) : null}
        </View>
      </Page>

      {/* Page 2: Visuals & Developer Context */}
      <Page size="A4" style={styles.page}>
        <View style={styles.page2Header}>
          <View>
            <Text style={styles.page2Title}>Forecast Drivers & Credibility</Text>
            <Text style={styles.page2Sub}>Page 2</Text>
          </View>
          <Text style={styles.headerDate}>{dateStr}</Text>
        </View>

        {/* NOTE: Rent/yield and supply/liquidity visuals were removed from Page 2 because Page 1 already covers them.
            Page 2 is reserved for investor credibility: momentum, drivers, and training coverage. */}

        {/* Why the model landed here */}
        <View style={[styles.chartCard, { marginTop: 6 }]}>
          <Text style={styles.chartTitle}>WHY THE MODEL LANDED HERE</Text>
          <View style={{ marginTop: 10 }}>
            <Text style={{ fontSize: 10, color: "#111827" }}>
              Recent segment price (last ~3m): {signedPct(metric3m)}
            </Text>
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 4 }}>
              Recent segment price (last ~12m): {signedPct(metric12m)}
            </Text>
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 4 }}>
              Area liquidity (last 3 full months vs same period last year): {signedPct(metricLiq)}
            </Text>
          </View>
        </View>

        {/* What drives the forecast */}
        <View style={[styles.chartCard, { marginTop: 14 }]}>
          <Text style={styles.chartTitle}>WHY THE FORECAST LANDED HERE</Text>
          <View style={{ marginTop: 10 }}>
            <Text style={{ fontSize: 10, color: "#111827", marginBottom: 6 }}>
              The model’s forecast is primarily influenced by:
            </Text>
            {drivers.length ? (
              drivers.slice(0, 5).map((d, idx) => (
                <Text key={idx} style={{ fontSize: 10, color: "#111827", marginTop: idx === 0 ? 0 : 4 }}>
                  • {d}
                </Text>
              ))
            ) : (
              <Text style={{ fontSize: 10, color: "#9CA3AF" }}>Not available</Text>
            )}
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 10 }}>
              Recent pricing and liquidity trends have been factored into the forecast to moderate assumptions and reflect current market conditions.
            </Text>
          </View>
        </View>

        {/* Training data coverage (static, matches text) */}
        <View style={[styles.chartCard, { marginTop: 14 }]}>
          <Text style={styles.chartTitle}>DATA COVERAGE & MODEL CREDIBILITY</Text>
          <View style={{ marginTop: 10 }}>
            <Text style={{ fontSize: 10, color: "#111827" }}>• Sales transactions (raw): ~1.61M</Text>
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 4 }}>• Rental contracts (raw): ~9.52M</Text>
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 4 }}>
              • Projects/units/buildings/valuations used as supporting context: ~3,039 / ~2.34M / ~239K / ~87K
            </Text>
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 4 }}>
              • Model training table (monthly aggregates): 72,205 rows across 1,745 segment series (2003–2025)
            </Text>
            <Text style={{ fontSize: 10, color: "#111827", marginTop: 8 }}>
              The model evaluates each unit within its specific market segment, rather than relying on generic averages.
            </Text>
          </View>
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>Generated by {brandName} • {dateStr}</Text>
          <Text style={styles.footerText}>This report is for informational purposes only and does not constitute investment advice.</Text>
        </View>
      </Page>
    </Document>
  )
}

export default InvestorPack
