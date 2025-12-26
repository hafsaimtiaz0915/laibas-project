import { getAccessToken } from './supabase'
import type { 
  ChatRequest, 
  ChatResponse, 
  Session, 
  Message, 
  AgentProfile, 
  AgentSettings,
  ReportData
} from './types'

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === 'production' ? 'https://api.proprly.ae' : 'http://localhost:8000')

async function fetchWithAuth(url: string, options: RequestInit = {}) {
  const token = await getAccessToken()
  
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  }
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  const response = await fetch(`${API_URL}${url}`, {
    ...options,
    headers,
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `HTTP error ${response.status}`)
  }
  
  return response.json()
}

// Chat API
export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  return fetchWithAuth('/api/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

export async function testChat(query: string): Promise<any> {
  return fetchWithAuth('/api/chat/test', {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
}

// Sessions API
export async function getSessions(): Promise<Session[]> {
  return fetchWithAuth('/api/sessions')
}

export async function getSession(sessionId: string): Promise<{ id: string; title: string; messages: Message[] }> {
  return fetchWithAuth(`/api/sessions/${sessionId}`)
}

export async function deleteSession(sessionId: string): Promise<void> {
  return fetchWithAuth(`/api/sessions/${sessionId}`, {
    method: 'DELETE',
  })
}

export async function updateSessionTitle(sessionId: string, title: string): Promise<void> {
  return fetchWithAuth(`/api/sessions/${sessionId}/title?title=${encodeURIComponent(title)}`, {
    method: 'PUT',
  })
}

// Agent API
export async function getProfile(): Promise<AgentProfile> {
  return fetchWithAuth('/api/agent/profile')
}

export async function updateSettings(settings: Partial<AgentSettings>): Promise<AgentProfile> {
  return fetchWithAuth('/api/agent/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  })
}

export async function uploadLogo(file: File): Promise<{ logo_url: string }> {
  const token = await getAccessToken()
  
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await fetch(`${API_URL}/api/agent/logo`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
    },
    body: formData,
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || 'Upload failed')
  }
  
  return response.json()
}

export async function deleteLogo(): Promise<void> {
  return fetchWithAuth('/api/agent/logo', {
    method: 'DELETE',
  })
}

// Report API
export async function generateReport(
  messageId: string,
  agentSettings?: Partial<AgentSettings>,
  unitSqft?: number,
  purchasePriceAed?: number
): Promise<{
  report_id: string
  report_data: ReportData
  agent_settings: AgentSettings
}> {
  return fetchWithAuth('/api/report/generate', {
    method: 'POST',
    body: JSON.stringify({
      message_id: messageId,
      agent_settings: agentSettings,
      unit_sqft: unitSqft,
      purchase_price_aed: purchasePriceAed,
    }),
  })
}

export async function uploadReportPdf(reportId: string, pdf: Blob): Promise<{ pdf_storage_path: string }> {
  const token = await getAccessToken()
  const formData = new FormData()
  formData.append("file", pdf, "investor-pack.pdf")

  const response = await fetch(`${API_URL}/api/report/${reportId}/upload`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed" }))
    throw new Error(error.detail || "Upload failed")
  }

  return response.json()
}

export async function getReportDownloadUrl(reportId: string): Promise<{ signed_url: string }> {
  return fetchWithAuth(`/api/report/${reportId}/download-url`, {
    method: 'GET',
  })
}

