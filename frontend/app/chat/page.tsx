"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { sendMessage } from "@/lib/api"
import { useChatStore } from "@/stores/chat-store"
import { PropertyFormInput } from "@/components/chat/PropertyFormInput"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { LoadingState } from "@/components/chat/LoadingState"
import { downloadPDF } from "@/lib/pdf-generator"
import { SiriOrb } from "@/components/ui/siri-orb"
import { MobilePropertyDetailsDock } from "@/components/chat/MobilePropertyDetailsDock"
import { MobilePropertyDetailsSheet } from "@/components/chat/MobilePropertyDetailsSheet"
import type { Message, ReportData } from "@/lib/types"
export default function NewChatPage() {
  const router = useRouter()
  const { setMessages, addMessage, setCurrentSessionId, isLoading, setIsLoading, sessions, setSessions } = useChatStore()
  const [localMessages, setLocalMessages] = useState<Message[]>([])
  const [mobileDetailsOpen, setMobileDetailsOpen] = useState(false)

  const handleSend = async (query: string) => {
    // Add user message immediately
    const userMessage: Message = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: query,
      created_at: new Date().toISOString(),
    }
    setLocalMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await sendMessage({ query })
      
      // Update session ID and navigate
      setCurrentSessionId(response.session_id)
      
      // Add assistant message
      const assistantMessage: Message = {
        id: response.message_id,
        role: "assistant",
        content: response.content,
        parsed_query: response.parsed_query,
        predictions: response.predictions,
        report_data: response.report_data,
        created_at: new Date().toISOString(),
      }
      
      const allMessages = [
        { ...userMessage, id: `user-${Date.now()}` },
        assistantMessage,
      ]
      
      setMessages(allMessages)
      
      // Update sessions list
      setSessions([
        {
          id: response.session_id,
          title: response.summary || query.slice(0, 50),
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        ...sessions,
      ])
      
      // Navigate to the new session
      router.push(`/chat/${response.session_id}`)
    } catch (err) {
      console.error("Failed to send message:", err)
      // Add error message
      setLocalMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: "Sorry, I encountered an error processing your request. Please try again.",
          created_at: new Date().toISOString(),
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerateReport = async (reportData: ReportData) => {
    try {
      const area = reportData.property.area || "property"
      const bedroom = reportData.property.bedroom || ""
      const filename = `${area}-${bedroom}-analysis-${Date.now()}.pdf`
      await downloadPDF(reportData, undefined, filename)
    } catch (error) {
      console.error("Failed to generate PDF:", error)
      alert("Failed to generate PDF report. Please try again.")
    }
  }

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      {/* Messages */}
      <div className="flex flex-1 min-h-0 min-w-0 flex-col overflow-y-auto px-4 pb-2 sm:pb-0">
        {localMessages.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            {/* Mobile: orb loader */}
            <div className="sm:hidden flex flex-col items-center justify-center gap-4 px-6 text-center">
              <SiriOrb size="120px" />
              <div>
                <h2 className="text-2xl font-semibold text-slate-900">Property Analysis</h2>
                <p className="text-sm text-slate-600 mt-1 max-w-sm">
                  Select structured inputs below (area, developer, sqft, price) for consistent analysis.
                </p>
              </div>
            </div>

            {/* Desktop: keep existing loader */}
            <div className="hidden sm:flex flex-col items-center justify-center gap-4">
              <div className="loader-wrapper">
                <div className="loader"></div>
              </div>
              <div className="text-center">
                <h2 className="text-xl font-semibold text-slate-900">Property Analysis</h2>
                <p className="text-sm text-slate-600 mt-1 max-w-md">
                  Select structured inputs below (area, developer, sqft, price) for consistent analysis.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto py-4">
            {localMessages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onGenerateReport={handleGenerateReport}
              />
            ))}
            {isLoading && <LoadingState />}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="hidden sm:block border-t border-slate-200 bg-white p-4">
        <div className="max-w-3xl mx-auto">
          <PropertyFormInput 
            onSend={handleSend} 
            disabled={isLoading}
            initialCollapsed={localMessages.length > 0}
          />
        </div>
      </div>

      {/* Mobile: dock + full-screen details */}
      <MobilePropertyDetailsDock
        disabled={isLoading}
        onOpen={() => setMobileDetailsOpen(true)}
      />
      <MobilePropertyDetailsSheet
        open={mobileDetailsOpen}
        onClose={() => setMobileDetailsOpen(false)}
      >
        <PropertyFormInput
          onSend={(q) => {
            setMobileDetailsOpen(false)
            handleSend(q)
          }}
          disabled={isLoading}
          initialCollapsed={false}
        />
      </MobilePropertyDetailsSheet>
    </div>
  )
}

