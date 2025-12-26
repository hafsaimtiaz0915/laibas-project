"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { getSession as getSessionApi, sendMessage } from "@/lib/api"
import { useChatStore } from "@/stores/chat-store"
import { PropertyFormInput } from "@/components/chat/PropertyFormInput"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { LoadingState } from "@/components/chat/LoadingState"
import { downloadPDF } from "@/lib/pdf-generator"
import { SiriOrb } from "@/components/ui/siri-orb"
import { MobilePropertyDetailsDock } from "@/components/chat/MobilePropertyDetailsDock"
import { MobilePropertyDetailsSheet } from "@/components/chat/MobilePropertyDetailsSheet"
import type { Message, ReportData } from "@/lib/types"

export default function ChatSessionPage() {
  const params = useParams()
  const chatId = params.chatId as string
  const { messages, setMessages, addMessage, isLoading, setIsLoading } = useChatStore()
  const [isLoadingSession, setIsLoadingSession] = useState(true)
  const [mobileDetailsOpen, setMobileDetailsOpen] = useState(false)

  useEffect(() => {
    const loadSession = async () => {
      setIsLoadingSession(true)
      try {
        const session = await getSessionApi(chatId)
        setMessages(session.messages)
      } catch (err) {
        console.error("Failed to load session:", err)
      } finally {
        setIsLoadingSession(false)
      }
    }
    loadSession()
  }, [chatId, setMessages])

  const handleSend = async (query: string) => {
    // Add user message immediately
    const userMessage: Message = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: query,
      created_at: new Date().toISOString(),
    }
    addMessage(userMessage)
    setIsLoading(true)

    try {
      const response = await sendMessage({ query, session_id: chatId })
      
      // Replace temp message and add assistant response
      const assistantMessage: Message = {
        id: response.message_id,
        role: "assistant",
        content: response.content,
        parsed_query: response.parsed_query,
        predictions: response.predictions,
        report_data: response.report_data,
        created_at: new Date().toISOString(),
      }
      
      setMessages([
        ...messages.filter((m) => m.id !== userMessage.id),
        { ...userMessage, id: `user-${Date.now()}` },
        assistantMessage,
      ])
    } catch (err) {
      console.error("Failed to send message:", err)
      // Add error message
      addMessage({
        id: `error-${Date.now()}`,
        role: "assistant",
        content: "Sorry, I encountered an error processing your request. Please try again.",
        created_at: new Date().toISOString(),
      })
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

  if (isLoadingSession) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="sm:hidden flex flex-col items-center gap-4 px-6 text-center">
          <SiriOrb size="110px" />
          <div className="text-sm text-slate-600">Loading conversation…</div>
        </div>
        <div className="hidden sm:block animate-pulse text-slate-500">Loading conversation...</div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      {/* Messages */}
      <div className="flex flex-1 min-h-0 min-w-0 flex-col overflow-y-auto px-4 pb-2 sm:pb-0">
        <div className="max-w-3xl mx-auto py-4">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onGenerateReport={handleGenerateReport}
            />
          ))}
          {isLoading && <LoadingState />}
        </div>
      </div>

      {/* Input */}
      <div className="hidden sm:block border-t border-slate-200 bg-white p-4">
        <div className="max-w-3xl mx-auto">
          <PropertyFormInput 
            onSend={handleSend} 
            disabled={isLoading} 
            initialCollapsed={messages.length > 0}
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

