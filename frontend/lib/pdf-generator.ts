"use client"

import { pdf } from "@react-pdf/renderer"
import { InvestorPack } from "./pdf-templates/InvestorPack"
import type { ReportData, AgentSettings } from "./types"
import type { ReactElement } from "react"
import { buildInvestorPackCharts } from "./chart-export"

const defaultAgentSettings: AgentSettings = {
  primary_color: "#0f766e",
  secondary_color: "#10b981",
  show_contact_info: true,
}

export async function generatePDF(
  reportData: ReportData,
  agentSettings?: AgentSettings
): Promise<Blob> {
  const settings = agentSettings || defaultAgentSettings

  // Build chart images (browser-rendered) to ensure charts are always identical in the PDF
  const chartImages = await buildInvestorPackCharts(reportData, settings)

  const doc = InvestorPack({ reportData, agentSettings: settings, chartImages }) as ReactElement
  const blob = await pdf(doc).toBlob()
  
  return blob
}

export async function downloadPDF(
  reportData: ReportData,
  agentSettings?: AgentSettings,
  filename?: string
): Promise<void> {
  const blob = await generatePDF(reportData, agentSettings)
  
  // Create download link
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename || `property-analysis-${Date.now()}.pdf`
  
  // Trigger download
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  // Clean up
  // Delay revocation to avoid aborting downloads in some browsers
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

