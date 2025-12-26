import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AgentSettings } from '@/lib/types'

interface AgentStore {
  settings: AgentSettings
  setSettings: (settings: Partial<AgentSettings>) => void
  resetSettings: () => void
}

const defaultSettings: AgentSettings = {
  name: undefined,
  company_name: undefined,
  phone: undefined,
  logo_url: undefined,
  primary_color: '#0f766e',
  secondary_color: '#10b981',
  report_footer: undefined,
  show_contact_info: true,
}

export const useAgentStore = create<AgentStore>()(
  persist(
    (set) => ({
      settings: defaultSettings,
      setSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),
      resetSettings: () => set({ settings: defaultSettings }),
    }),
    {
      name: 'agent-settings',
    }
  )
)

