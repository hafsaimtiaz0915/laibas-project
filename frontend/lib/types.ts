// API Types

export interface ChatRequest {
  query: string
  session_id?: string
}

export interface ChatResponse {
  session_id: string
  message_id: string
  content: string
  parsed_query?: ParsedQuery
  predictions?: Predictions
  report_data?: ReportData
  summary: string
}

export interface ParsedQuery {
  developer?: string
  area?: string
  bedroom?: string
  price?: number
  property_type?: string
  reg_type?: string
  handover_months?: number
  raw_query: string
  confidence: number
}

export interface Predictions {
  price_forecast: PriceForecast
  rent_forecast: RentForecast
  match_type: string
  confidence: number
}

export interface PriceForecast {
  current_sqft?: number
  forecast_sqft_low?: number
  forecast_sqft_median?: number
  forecast_sqft_high?: number
  forecast_horizon_months: number
  appreciation_percent?: number
  forecast_sqft_low_12m?: number
  forecast_sqft_median_12m?: number
  forecast_sqft_high_12m?: number
  appreciation_percent_12m?: number
  forecast_sqft_low_24m?: number
  forecast_sqft_median_24m?: number
  forecast_sqft_high_24m?: number
  appreciation_percent_24m?: number
  long_horizon_method?: string
  unit_sqft?: number
  current_total_value?: number
  forecast_total_value_low?: number
  forecast_total_value_median?: number
  forecast_total_value_high?: number
  // 12m horizon total values (for off-plan: handover + 12m)
  forecast_total_value_low_12m?: number
  forecast_total_value_median_12m?: number
  forecast_total_value_high_12m?: number
  // Handover-specific values (extrapolated to actual handover timeline)
  handover_total_value_median?: number
  handover_total_value_low?: number
  handover_total_value_high?: number
  post12_total_value_median?: number
  post12_total_value_low?: number
  post12_total_value_high?: number
  handover_months_target?: number
}

export interface RentForecast {
  current_annual?: number
  forecast_annual_low?: number
  forecast_annual_median?: number
  forecast_annual_high?: number
  has_actual_rent: boolean
  estimated_yield_percent?: number
}

export interface ReportData {
  property: {
    developer?: string
    developer_english?: string
    developer_arabic?: string
    is_building_developer?: boolean
    area?: string
    area_display?: string
    bedroom?: string
    property_type?: string
    reg_type?: string
    price?: number
    unit_sqft?: number
    area_confidence?: number
    area_resolution_method?: string
    developer_confidence?: number
    developer_resolution_method?: string
  }
  price_forecast: PriceForecast
  rent_forecast: RentForecast
  trend_insights?: any
  model_attribution?: any
  developer_stats?: DeveloperStats
  area_stats?: AreaStats
  rent_benchmark?: RentBenchmark
  lookup_audit?: any
  match_info: {
    type: string
    group_id?: string
    confidence: number
  }
  gating?: {
    area_confidence_threshold: number
    developer_confidence_threshold: number
    model_forecast_confidence_threshold: number
    model_forecast_match_types: string[]
  }
  caveats?: {
    developer_caveat?: string | null
  }
  // Pre-computed total values (calculated by backend, used directly by PDF - no recalculation)
  handover_total_value_median?: number
  handover_total_value_low?: number
  handover_total_value_high?: number
  plus12m_total_value_median?: number
  plus12m_total_value_low?: number
  plus12m_total_value_high?: number
  // Uplift values (also pre-computed by backend)
  uplift_handover?: number
  uplift_handover_percent?: number
  uplift_plus12m?: number
  uplift_plus12m_percent?: number
  // Yield range (pre-computed)
  yield_low?: number
  yield_high?: number

  // Debug payload (backend-provided) to explain how investor totals/uplift were computed
  investor_calc_debug?: {
    base_horizon_months?: number | null
    handover_months_target?: number | null
    handover_months_plus_12_target?: number | null
    unit_sqft?: number | null
    purchase_price_aed?: number | null
    current_sqft_anchor?: number | null
    current_sqft_source?: string
    forecast_sqft_median_base_horizon?: number | null
    forecast_sqft_low_base_horizon?: number | null
    forecast_sqft_high_base_horizon?: number | null
  }
}

export interface DeveloperStats {
  developer_name: string
  projects_total: number
  projects_completed: number
  projects_active: number
  total_units: number
  completion_rate: number
  avg_duration_months: number
  avg_delay_months: number
}

export interface AreaStats {
  area_name: string
  current_median_sqft?: number
  price_change_12m?: number
  price_change_36m?: number
  transaction_count_12m: number
  supply_pipeline: number
}

export interface RentBenchmark {
  area_name: string
  bedrooms: string
  median_annual_rent?: number
  rent_count: number
  median_rent_sqft?: number
}

// Session Types

export interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  parsed_query?: ParsedQuery
  predictions?: Predictions
  report_data?: ReportData
  report_id?: string
  pdf_url?: string
  pdf_storage_path?: string
  created_at: string
}

// Agent Types

export interface AgentSettings {
  name?: string
  company_name?: string
  phone?: string
  logo_url?: string
  primary_color: string
  secondary_color: string
  report_footer?: string
  show_contact_info: boolean
}

export interface AgentProfile extends AgentSettings {
  id: string
  email: string
}

