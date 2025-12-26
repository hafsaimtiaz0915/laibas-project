"use client"

import { cn } from "@/lib/utils"
import type { Message, ReportData } from "@/lib/types"
import { User, Bot, FileText } from "lucide-react"
import { AnalysisCard } from "./AnalysisCard"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { useEffect, useMemo, useState } from "react"
import { useAgentStore } from "@/stores/agent-store"
import { generatePDF } from "@/lib/pdf-generator"
import { generateReport, uploadReportPdf, getReportDownloadUrl } from "@/lib/api"

interface ChatMessageProps {
  message: Message
  onGenerateReport?: (reportData: ReportData) => void
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onGenerateReport,
}) => {
  const isUser = message.role === "user"
  const hasReportData = message.report_data !== undefined

  const { settings } = useAgentStore()
  const [isGenerating, setIsGenerating] = useState(false)
  const [pdfStoragePath, setPdfStoragePath] = useState<string | undefined>(message.pdf_storage_path)

  useEffect(() => {
    setPdfStoragePath(message.pdf_storage_path)
  }, [message.pdf_storage_path])

  // Check if we have the required data for PDF generation (from original query)
  const canGeneratePdf = useMemo(() => {
    const rd = message.report_data
    if (!rd) return false
    const sqft = rd.property.unit_sqft || rd.price_forecast?.unit_sqft
    const price = rd.property.price
    return Boolean(sqft && sqft > 0 && price && price > 0)
  }, [message.report_data])

  const defaultFilename = useMemo(() => {
    const rd = message.report_data
    if (!rd) return `investor-pack-${Date.now()}.pdf`
    const area = rd.property.area || "property"
    const bedroom = rd.property.bedroom || ""
    return `${area}-${bedroom}-investor-pack-${Date.now()}.pdf`
  }, [message.report_data])

  // Generate PDF directly using data from original query - no modal needed
  const handleGeneratePdf = async () => {
    if (!message.report_data) return
    try {
      setIsGenerating(true)

      // Use the report data directly from the message
      const unitSqft = message.report_data.property.unit_sqft ?? message.report_data.price_forecast?.unit_sqft
      const purchasePriceAed = message.report_data.property.price
      const report = await generateReport(message.id, settings, unitSqft, purchasePriceAed)
      const reportData = report.report_data ?? message.report_data

      // Generate PDF blob (client-side)
      const blob = await generatePDF(reportData, report.agent_settings || settings)

      // Trigger download immediately
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = defaultFilename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      setTimeout(() => URL.revokeObjectURL(url), 1000)

      // Upload + persist URL for future downloads (non-blocking; download already happened)
      try {
        const uploaded = await uploadReportPdf(report.report_id, blob)
        if (uploaded?.pdf_storage_path) setPdfStoragePath(uploaded.pdf_storage_path)
      } catch (uploadErr) {
        console.warn("PDF downloaded but failed to upload for future downloads:", uploadErr)
        // Don't show a scary "failed to generate" message if the user already got the PDF.
        alert("PDF downloaded, but we couldn't save it to the chat for future downloads. Please try again.")
      }
    } catch (err) {
      console.error("Failed to generate PDF:", err)
      alert("Failed to generate the PDF. Please try again.")
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownloadSavedPdf = async () => {
    try {
      if (!message.report_id) return
      const { signed_url } = await getReportDownloadUrl(message.report_id)
      window.open(signed_url, "_blank", "noopener,noreferrer")
    } catch (err) {
      console.error("Failed to fetch signed download URL:", err)
      alert("Couldn't open the saved PDF. Please try generating it again.")
    }
  }

  return (
    <div
      className={cn(
        "flex gap-3 py-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full",
          isUser
            ? "bg-slate-900 text-white"
            : "bg-brand-100 text-brand-700"
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" />
        ) : (
          <Bot className="h-4 w-4" />
        )}
      </div>

      {/* Content */}
      <div
        className={cn(
          "flex flex-col gap-2 max-w-[80%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        {/* Message bubble */}
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5",
            isUser
              ? "bg-slate-900 text-white"
              : "bg-slate-50 border border-slate-200 text-slate-900"
          )}
        >
          {isUser ? (
            <div className="whitespace-pre-wrap text-sm">{message.content}</div>
          ) : (
            <div
              className="prose prose-sm max-w-none text-sm
              prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-2
              prose-h3:text-base prose-h4:text-sm
              prose-p:my-1.5 prose-p:leading-relaxed
              prose-ul:my-2 prose-ul:pl-4
              prose-li:my-0.5
              prose-strong:font-semibold prose-strong:text-slate-900
              [&>*:first-child]:mt-0"
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Analysis card for assistant messages */}
        {!isUser && hasReportData && message.report_data && (
          <AnalysisCard
            reportData={message.report_data}
            onGenerateReport={onGenerateReport}
          />
        )}

        {/* Report button for assistant messages with report data */}
        {!isUser && hasReportData && message.report_data && (
          <>
            {message.report_id && pdfStoragePath ? (
              <button
                onClick={handleDownloadSavedPdf}
                className="flex items-center gap-2 text-xs text-slate-600 hover:text-brand-700 transition-colors"
              >
                <FileText className="h-3.5 w-3.5" />
                Download PDF Report
              </button>
            ) : canGeneratePdf ? (
              <button
                onClick={handleGeneratePdf}
                disabled={isGenerating}
                className="flex items-center gap-2 text-xs text-slate-600 hover:text-brand-700 transition-colors disabled:opacity-50"
              >
                <FileText className="h-3.5 w-3.5" />
                {isGenerating ? "Generating..." : "Generate PDF Report"}
              </button>
            ) : (
              <span className="text-xs text-slate-500 italic">
                Include unit size &amp; price in your query to generate PDF
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default ChatMessage

