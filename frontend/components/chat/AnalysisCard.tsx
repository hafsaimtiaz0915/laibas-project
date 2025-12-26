"use client"

import { cn, formatCurrency, formatNumber } from "@/lib/utils"
import type { ReportData } from "@/lib/types"
import {
  TrendingUp,
  TrendingDown,
  Building2,
  MapPin,
  Home,
  Percent,
} from "lucide-react"

interface AnalysisCardProps {
  reportData: ReportData
  onGenerateReport?: (reportData: ReportData) => void
}

export const AnalysisCard: React.FC<AnalysisCardProps> = ({ reportData }) => {
  const { property, price_forecast, rent_forecast, area_stats, developer_stats } = reportData

  const priceChange = area_stats?.price_change_12m
  const isPositive = priceChange && priceChange >= 0

  return (
    <div className="w-full max-w-md rounded-2xl bg-white border border-slate-200 p-4 space-y-4 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <MapPin className="h-4 w-4 text-slate-500" />
            <span className="font-medium text-sm text-slate-900">{property.area || "Unknown Area"}</span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-500">
            {property.bedroom && (
              <span className="flex items-center gap-1">
                <Home className="h-3 w-3" />
                {property.bedroom}
              </span>
            )}
            {property.developer && (
              <span className="flex items-center gap-1">
                <Building2 className="h-3 w-3" />
                {property.developer}
              </span>
            )}
            {property.reg_type && (
              <span className={cn(
                "px-1.5 py-0.5 rounded text-[10px] font-medium",
                property.reg_type === "OffPlan"
                  ? "bg-amber-100 text-amber-700"
                  : "bg-green-100 text-green-700"
              )}>
                {property.reg_type}
              </span>
            )}
          </div>
        </div>

        {/* Match confidence */}
        <div className={cn(
          "px-2 py-1 rounded-full text-xs font-medium",
          reportData.match_info.confidence >= 80
            ? "bg-green-100 text-green-700"
            : reportData.match_info.confidence >= 50
            ? "bg-amber-100 text-amber-700"
            : "bg-red-100 text-red-700"
        )}>
          {Math.round(reportData.match_info.confidence)}% match
        </div>
      </div>

      {/* Price Forecast */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <span className="text-xs text-slate-500">Price Forecast ({price_forecast.forecast_horizon_months}m)</span>
          <div className="text-lg font-semibold text-slate-900">
            AED {price_forecast.forecast_sqft_median?.toLocaleString() || "N/A"}/sqft
          </div>
          {price_forecast.appreciation_percent && (
            <div className={cn(
              "flex items-center gap-1 text-xs",
              price_forecast.appreciation_percent >= 0 ? "text-green-600" : "text-red-600"
            )}>
              {price_forecast.appreciation_percent >= 0 ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              )}
              {price_forecast.appreciation_percent >= 0 ? "+" : ""}
              {price_forecast.appreciation_percent.toFixed(1)}%
            </div>
          )}
        </div>

        <div className="space-y-1">
          <span className="text-xs text-slate-500">Rent Forecast</span>
          <div className="text-lg font-semibold text-slate-900">
            {rent_forecast.forecast_annual_median
              ? formatCurrency(rent_forecast.forecast_annual_median, "/yr")
              : "N/A"}
          </div>
          {rent_forecast.estimated_yield_percent && (
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <Percent className="h-3 w-3" />
              {rent_forecast.estimated_yield_percent.toFixed(1)}% yield
            </div>
          )}
        </div>
      </div>

      {/* Area Stats */}
      {area_stats && (
        <div className="pt-2 border-t">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500">12M Price Change</span>
            <span className={cn(
              "font-medium",
              isPositive ? "text-green-600" : "text-red-600"
            )}>
              {isPositive ? "+" : ""}{priceChange?.toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center justify-between text-xs mt-1">
            <span className="text-slate-500">Transactions (12M)</span>
            <span className="font-medium text-slate-900">{formatNumber(area_stats.transaction_count_12m)}</span>
          </div>
          {area_stats.supply_pipeline > 0 && (
            <div className="flex items-center justify-between text-xs mt-1">
              <span className="text-slate-500">Upcoming Supply</span>
              <span className="font-medium text-slate-900">{formatNumber(area_stats.supply_pipeline)} units</span>
            </div>
          )}
        </div>
      )}

      {/* Developer Stats */}
      {developer_stats && (
        <div className="pt-2 border-t">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500">Developer Projects</span>
            <span className="font-medium text-slate-900">
              {developer_stats.projects_completed}/{developer_stats.projects_total} completed
            </span>
          </div>
          <div className="flex items-center justify-between text-xs mt-1">
            <span className="text-slate-500">Units Delivered</span>
            <span className="font-medium text-slate-900">{formatNumber(developer_stats.total_units)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default AnalysisCard

