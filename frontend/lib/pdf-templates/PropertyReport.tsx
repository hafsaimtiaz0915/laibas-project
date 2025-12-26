import React from 'react'
import {
  Document,
  Page,
  Text,
  View,
  StyleSheet,
  Image,
} from '@react-pdf/renderer'
import type { ReportData, AgentSettings } from '@/lib/types'

// Styles
const styles = StyleSheet.create({
  page: {
    flexDirection: 'column',
    backgroundColor: '#ffffff',
    padding: 40,
    fontFamily: 'Helvetica',
    fontSize: 10,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 30,
    borderBottomWidth: 2,
    borderBottomColor: '#0f766e',
    paddingBottom: 20,
  },
  logo: {
    width: 80,
    height: 80,
    objectFit: 'contain',
  },
  headerText: {
    textAlign: 'right',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#0f766e',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 12,
    color: '#666666',
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#0f766e',
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e5e5',
    paddingBottom: 5,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  label: {
    color: '#666666',
    flex: 1,
  },
  value: {
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'right',
  },
  positiveValue: {
    color: '#059669',
    fontWeight: 'bold',
  },
  negativeValue: {
    color: '#dc2626',
    fontWeight: 'bold',
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  gridItem: {
    width: '50%',
    paddingRight: 10,
    marginBottom: 10,
  },
  highlight: {
    backgroundColor: '#f0fdf4',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  highlightTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#0f766e',
    marginBottom: 8,
  },
  highlightValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#059669',
  },
  highlightSubtext: {
    fontSize: 10,
    color: '#666666',
    marginTop: 4,
  },
  disclaimer: {
    marginTop: 30,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#e5e5e5',
  },
  disclaimerText: {
    fontSize: 8,
    color: '#999999',
    fontStyle: 'italic',
  },
  footer: {
    position: 'absolute',
    bottom: 40,
    left: 40,
    right: 40,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    borderTopWidth: 1,
    borderTopColor: '#e5e5e5',
    paddingTop: 10,
  },
  footerText: {
    fontSize: 8,
    color: '#666666',
  },
  confidenceBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 4,
    alignSelf: 'flex-start',
  },
  confidenceHigh: {
    backgroundColor: '#dcfce7',
  },
  confidenceMedium: {
    backgroundColor: '#fef3c7',
  },
  confidenceLow: {
    backgroundColor: '#fee2e2',
  },
  confidenceText: {
    fontSize: 10,
    fontWeight: 'bold',
  },
})

interface PropertyReportProps {
  reportData: ReportData
  agentSettings: AgentSettings
}

function formatCurrency(value?: number, suffix: string = ''): string {
  if (!value) return 'N/A'
  if (value >= 1_000_000) {
    return `AED ${(value / 1_000_000).toFixed(2)}M${suffix}`
  } else if (value >= 1_000) {
    return `AED ${Math.round(value / 1_000)}K${suffix}`
  }
  return `AED ${Math.round(value)}${suffix}`
}

function formatNumber(value?: number): string {
  if (!value) return 'N/A'
  return new Intl.NumberFormat('en-AE').format(value)
}

function formatAed(value?: number): string {
  if (value === undefined || value === null) return 'N/A'
  return `AED ${Math.round(value).toLocaleString('en-AE')}`
}

export const PropertyReport: React.FC<PropertyReportProps> = ({
  reportData,
  agentSettings,
}) => {
  const { property, price_forecast, rent_forecast, developer_stats, area_stats, match_info, model_attribution } = reportData

  const unitSqft = property.unit_sqft
  const purchasePrice = property.price
  const currentPricePerSqft =
    unitSqft && purchasePrice ? purchasePrice / unitSqft : undefined
  const forecastTotalLow =
    unitSqft && price_forecast.forecast_sqft_low ? price_forecast.forecast_sqft_low * unitSqft : undefined
  const forecastTotalMed =
    unitSqft && price_forecast.forecast_sqft_median ? price_forecast.forecast_sqft_median * unitSqft : undefined
  const forecastTotalHigh =
    unitSqft && price_forecast.forecast_sqft_high ? price_forecast.forecast_sqft_high * unitSqft : undefined
  
  const confidence = match_info.confidence
  const confidenceStyle = confidence >= 80
    ? styles.confidenceHigh
    : confidence >= 50
    ? styles.confidenceMedium
    : styles.confidenceLow

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {/* Header */}
        <View style={styles.header}>
          {agentSettings.logo_url && (
            <Image src={agentSettings.logo_url} style={styles.logo} />
          )}
          <View style={styles.headerText}>
            <Text style={[styles.title, { color: agentSettings.primary_color }]}>
              Property Analysis
            </Text>
            <Text style={styles.subtitle}>
              {new Date().toLocaleDateString('en-AE', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </Text>
          </View>
        </View>

        {/* Property Details */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: agentSettings.primary_color }]}>
            Property Details
          </Text>
          <View style={styles.grid}>
            <View style={styles.gridItem}>
              <View style={styles.row}>
                <Text style={styles.label}>Area</Text>
                <Text style={styles.value}>{property.area || 'N/A'}</Text>
              </View>
            </View>
            <View style={styles.gridItem}>
              <View style={styles.row}>
                <Text style={styles.label}>Developer</Text>
                <Text style={styles.value}>{property.developer || 'N/A'}</Text>
              </View>
            </View>
            <View style={styles.gridItem}>
              <View style={styles.row}>
                <Text style={styles.label}>Bedrooms</Text>
                <Text style={styles.value}>{property.bedroom || 'N/A'}</Text>
              </View>
            </View>
            <View style={styles.gridItem}>
              <View style={styles.row}>
                <Text style={styles.label}>Type</Text>
                <Text style={styles.value}>{property.reg_type || 'N/A'}</Text>
              </View>
            </View>
            {property.price && (
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Price</Text>
                  <Text style={styles.value}>{formatCurrency(property.price)}</Text>
                </View>
              </View>
            )}
            {property.unit_sqft && (
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Unit Size</Text>
                  <Text style={styles.value}>{formatNumber(property.unit_sqft)} sqft</Text>
                </View>
              </View>
            )}
            {currentPricePerSqft && (
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Current Price / sqft</Text>
                  <Text style={styles.value}>{formatAed(currentPricePerSqft)}/sqft</Text>
                </View>
              </View>
            )}
          </View>
        </View>

        {/* Price Forecast Highlight */}
        <View style={[styles.highlight, { backgroundColor: `${agentSettings.secondary_color}15` }]}>
          <Text style={[styles.highlightTitle, { color: agentSettings.primary_color }]}>
            6-Month Price Forecast (model)
          </Text>
          <Text style={styles.highlightValue}>
            AED {price_forecast.forecast_sqft_median?.toLocaleString() || 'N/A'}/sqft
          </Text>
          {unitSqft && forecastTotalMed !== undefined && (
            <Text style={styles.highlightSubtext}>
              Total value (est.): {formatAed(forecastTotalMed)}
              {forecastTotalLow !== undefined && forecastTotalHigh !== undefined
                ? ` (range ${formatAed(forecastTotalLow)} – ${formatAed(forecastTotalHigh)})`
                : ''}
            </Text>
          )}
          {price_forecast.appreciation_percent && (
            <Text style={styles.highlightSubtext}>
              6-month implied appreciation: {price_forecast.appreciation_percent >= 0 ? '+' : ''}
              {price_forecast.appreciation_percent.toFixed(1)}%
            </Text>
          )}
        </View>

        {/* Forecasts */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: agentSettings.primary_color }]}>
            Capital Appreciation (6/12/24 months)
          </Text>
          <View style={styles.grid}>
            <View style={styles.gridItem}>
              <View style={styles.row}>
                <Text style={styles.label}>Current Price</Text>
                <Text style={styles.value}>
                  AED {price_forecast.current_sqft?.toLocaleString() || 'N/A'}/sqft
                </Text>
              </View>
              <View style={styles.row}>
                <Text style={styles.label}>6M Low (P10)</Text>
                <Text style={styles.value}>
                  AED {price_forecast.forecast_sqft_low?.toLocaleString() || 'N/A'}/sqft
                </Text>
              </View>
              <View style={styles.row}>
                <Text style={styles.label}>6M High (P90)</Text>
                <Text style={styles.value}>
                  AED {price_forecast.forecast_sqft_high?.toLocaleString() || 'N/A'}/sqft
                </Text>
              </View>
              {price_forecast.forecast_sqft_median_12m && (
                <View style={styles.row}>
                  <Text style={styles.label}>12M (extrap.)</Text>
                  <Text style={styles.value}>
                    AED {price_forecast.forecast_sqft_median_12m.toLocaleString()}/sqft
                  </Text>
                </View>
              )}
              {price_forecast.forecast_sqft_median_24m && (
                <View style={styles.row}>
                  <Text style={styles.label}>24M (extrap.)</Text>
                  <Text style={styles.value}>
                    AED {price_forecast.forecast_sqft_median_24m.toLocaleString()}/sqft
                  </Text>
                </View>
              )}
            </View>
            <View style={styles.gridItem}>
              <View style={styles.row}>
                <Text style={styles.label}>Rent Forecast</Text>
                <Text style={styles.value}>
                  {formatCurrency(rent_forecast.forecast_annual_median, '/yr')}
                </Text>
              </View>
              <View style={styles.row}>
                <Text style={styles.label}>Rent Low</Text>
                <Text style={styles.value}>
                  {formatCurrency(rent_forecast.forecast_annual_low, '/yr')}
                </Text>
              </View>
              <View style={styles.row}>
                <Text style={styles.label}>Est. Yield</Text>
                <Text style={styles.value}>
                  {rent_forecast.estimated_yield_percent?.toFixed(1) || 'N/A'}%
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Model Explanation */}
        {model_attribution?.top_drivers?.length ? (
          <View style={styles.section}>
            <Text style={[styles.sectionTitle, { color: agentSettings.primary_color }]}>
              Why the model predicts this (non-causal)
            </Text>
            <View>
              <Text style={styles.label}>
                The 12/24 month numbers are extrapolations from the 6-month model-implied monthly growth. The items below are model drivers and sensitivities (not causal).
              </Text>
              <View style={{ marginTop: 8 }}>
                <Text style={{ fontWeight: 'bold', marginBottom: 4 }}>Top drivers (current + Δ6m)</Text>
                {model_attribution.top_drivers.slice(0, 5).map((d: any, idx: number) => (
                  <View key={idx} style={styles.row}>
                    <Text style={styles.label}>{d.feature}</Text>
                    <Text style={styles.value}>
                      {d.current_value !== null && d.current_value !== undefined ? `${Number(d.current_value).toFixed(2)}` : 'N/A'}
                      {d.change_6m !== null && d.change_6m !== undefined ? ` (Δ6m ${Number(d.change_6m) >= 0 ? '+' : ''}${Number(d.change_6m).toFixed(2)})` : ''}
                    </Text>
                  </View>
                ))}
              </View>
              {model_attribution.what_if_impacts?.length ? (
                <View style={{ marginTop: 8 }}>
                  <Text style={{ fontWeight: 'bold', marginBottom: 4 }}>What-if impacts (ceteris paribus)</Text>
                  {model_attribution.what_if_impacts.slice(0, 3).map((w: any, idx: number) => (
                    <View key={idx} style={styles.row}>
                      <Text style={styles.label}>
                        If {w.feature} is {w.assumption}
                      </Text>
                      <Text style={styles.value}>
                        {w.delta_percent_vs_baseline !== null && w.delta_percent_vs_baseline !== undefined
                          ? `${Number(w.delta_percent_vs_baseline) >= 0 ? '+' : ''}${Number(w.delta_percent_vs_baseline).toFixed(1)}%`
                          : 'N/A'}
                      </Text>
                    </View>
                  ))}
                </View>
              ) : null}
            </View>
          </View>
        ) : null}

        {/* Area Trends */}
        {area_stats && (
          <View style={styles.section}>
            <Text style={[styles.sectionTitle, { color: agentSettings.primary_color }]}>
              Area Trends - {area_stats.area_name}
            </Text>
            <View style={styles.grid}>
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>12M Price Change</Text>
                  <Text style={[
                    styles.value,
                    area_stats.price_change_12m && area_stats.price_change_12m >= 0
                      ? styles.positiveValue
                      : styles.negativeValue
                  ]}>
                    {area_stats.price_change_12m !== undefined
                      ? `${area_stats.price_change_12m >= 0 ? '+' : ''}${area_stats.price_change_12m.toFixed(1)}%`
                      : 'N/A'}
                  </Text>
                </View>
              </View>
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>36M Price Change</Text>
                  <Text style={[
                    styles.value,
                    area_stats.price_change_36m && area_stats.price_change_36m >= 0
                      ? styles.positiveValue
                      : styles.negativeValue
                  ]}>
                    {area_stats.price_change_36m !== undefined
                      ? `${area_stats.price_change_36m >= 0 ? '+' : ''}${area_stats.price_change_36m.toFixed(1)}%`
                      : 'N/A'}
                  </Text>
                </View>
              </View>
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Transactions (12M)</Text>
                  <Text style={styles.value}>{formatNumber(area_stats.transaction_count_12m)}</Text>
                </View>
              </View>
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Supply Pipeline</Text>
                  <Text style={styles.value}>{formatNumber(area_stats.supply_pipeline)} units</Text>
                </View>
              </View>
            </View>
          </View>
        )}

        {/* Developer Info */}
        {developer_stats && (
          <View style={styles.section}>
            <Text style={[styles.sectionTitle, { color: agentSettings.primary_color }]}>
              Developer Execution (historical)
            </Text>
            <View style={styles.grid}>
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Avg time to complete</Text>
                  <Text style={styles.value}>
                    {developer_stats.avg_duration_months ? `${Math.round(developer_stats.avg_duration_months)} months` : 'N/A'}
                  </Text>
                </View>
              </View>
              <View style={styles.gridItem}>
                <View style={styles.row}>
                  <Text style={styles.label}>Avg delay</Text>
                  <Text style={styles.value}>
                    {developer_stats.avg_delay_months !== undefined && developer_stats.avg_delay_months !== null
                      ? `${Math.round(developer_stats.avg_delay_months)} months`
                      : 'N/A'}
                  </Text>
                </View>
              </View>
            </View>
          </View>
        )}

        {/* Confidence Badge */}
        <View style={[styles.confidenceBadge, confidenceStyle]}>
          <Text style={styles.confidenceText}>
            Data Confidence: {Math.round(confidence)}% ({match_info.type})
          </Text>
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>
            DISCLAIMER: This report is for informational purposes only and does not constitute investment advice.
            Forecasts are based on historical data and market trends. Past performance does not guarantee future results.
            Please consult with a qualified professional before making any investment decisions.
          </Text>
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          {agentSettings.show_contact_info && (agentSettings.name || agentSettings.company_name) && (
            <View>
              {agentSettings.name && <Text style={styles.footerText}>{agentSettings.name}</Text>}
              {agentSettings.company_name && <Text style={styles.footerText}>{agentSettings.company_name}</Text>}
              {agentSettings.phone && <Text style={styles.footerText}>{agentSettings.phone}</Text>}
            </View>
          )}
          <View>
            {agentSettings.report_footer && (
              <Text style={styles.footerText}>{agentSettings.report_footer}</Text>
            )}
            <Text style={styles.footerText}>Generated by Proprly</Text>
          </View>
        </View>
      </Page>
    </Document>
  )
}

export default PropertyReport

