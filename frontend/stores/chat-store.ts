import { create } from 'zustand'
import type { Session, Message } from '@/lib/types'

interface ChatStore {
  sessions: Session[]
  currentSessionId: string | null
  messages: Message[]
  isLoading: boolean
  setSessions: (sessions: Session[]) => void
  setCurrentSessionId: (id: string | null) => void
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  setIsLoading: (loading: boolean) => void
  reset: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isLoading: false,
  setSessions: (sessions) => set({ sessions }),
  setCurrentSessionId: (id) => set({ currentSessionId: id }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  setIsLoading: (loading) => set({ isLoading: loading }),
  reset: () =>
    set({
      sessions: [],
      currentSessionId: null,
      messages: [],
      isLoading: false,
    }),
}))

