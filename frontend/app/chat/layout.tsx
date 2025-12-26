"use client"

import { useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import Link from "next/link"
import { getSession, signOut } from "@/lib/supabase"
import { getSessions, deleteSession } from "@/lib/api"
import { useChatStore } from "@/stores/chat-store"
import { cn } from "@/lib/utils"
import { AnimatePresence, motion } from "framer-motion"
import { SiriOrb } from "@/components/ui/siri-orb"
import {
  MessageSquare,
  Plus,
  Settings,
  LogOut,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from "lucide-react"

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const { sessions, setSessions, currentSessionId, setCurrentSessionId } = useChatStore()

  useEffect(() => {
    const init = async () => {
      const session = await getSession()
      if (!session) {
        router.push("/login")
        return
      }

      try {
        const userSessions = await getSessions()
        setSessions(userSessions)
      } catch (err) {
        console.error("Failed to load sessions:", err)
      } finally {
        setIsLoading(false)
      }
    }
    init()
  }, [router, setSessions])

  // Update current session based on pathname
  useEffect(() => {
    const match = pathname.match(/\/chat\/([^/]+)/)
    if (match) {
      setCurrentSessionId(match[1])
    } else {
      setCurrentSessionId(null)
    }
  }, [pathname, setCurrentSessionId])

  const handleNewChat = () => {
    setCurrentSessionId(null)
    router.push("/chat")
  }

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (!confirm("Delete this chat?")) return
    
    try {
      await deleteSession(sessionId)
      setSessions(sessions.filter((s) => s.id !== sessionId))
      if (currentSessionId === sessionId) {
        router.push("/chat")
      }
    } catch (err) {
      console.error("Failed to delete session:", err)
    }
  }

  const handleSignOut = async () => {
    await signOut()
    router.push("/login")
  }

  if (isLoading) {
    return (
      <div className="flex min-h-[100svh] items-center justify-center bg-slate-50">
        <div className="sm:hidden flex flex-col items-center gap-4 px-6 text-center">
          <SiriOrb size="110px" />
          <div className="text-sm text-slate-600">Loading…</div>
        </div>
        <div className="hidden sm:block animate-pulse text-slate-500">Loading...</div>
      </div>
    )
  }

  return (
    <div className="flex min-h-[100svh] sm:h-screen flex-col sm:flex-row bg-slate-50 overflow-x-hidden">
      {/* Mobile top bar */}
      <div
        className="sm:hidden fixed top-0 left-0 right-0 z-50 border-b border-slate-200 bg-white/95 backdrop-blur"
        style={{ paddingTop: "env(safe-area-inset-top)" }}
      >
        <div className="flex h-14 w-full items-center justify-between px-5">
          <button
            type="button"
            onClick={() => setMobileNavOpen(true)}
            className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-700 shadow-sm"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
          <Link href="/" className="font-bold text-slate-900">
            Proprly.
          </Link>
          <div className="h-11 w-11" />
        </div>
      </div>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileNavOpen && (
          <>
            <motion.button
              type="button"
              className="sm:hidden fixed inset-0 z-50 bg-black/30 backdrop-blur-[2px]"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileNavOpen(false)}
              aria-label="Close menu"
            />
            <motion.aside
              className="sm:hidden fixed inset-y-0 left-0 z-[60] w-[86vw] max-w-[360px] bg-white border-r border-slate-200 shadow-2xl flex flex-col"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", stiffness: 420, damping: 40 }}
            >
              <div className="flex h-14 items-center justify-between border-b border-slate-200 px-4">
                <span className="font-bold text-slate-900">Proprly.</span>
                <button
                  type="button"
                  onClick={() => setMobileNavOpen(false)}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-700"
                  aria-label="Close"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="p-3">
                <button
                  onClick={() => {
                    setMobileNavOpen(false)
                    handleNewChat()
                  }}
                  className="flex w-full items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-900 hover:bg-slate-100 transition-colors"
                >
                  <Plus className="h-4 w-4" />
                  <span className="font-medium">New Chat</span>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1">
                {sessions.map((session) => (
                  <Link
                    key={session.id}
                    href={`/chat/${session.id}`}
                    onClick={() => setMobileNavOpen(false)}
                    className={cn(
                      "group flex items-center gap-2 rounded-2xl p-3 text-sm transition-colors",
                      currentSessionId === session.id
                        ? "bg-brand-50 text-brand-900 border border-brand-100"
                        : "hover:bg-slate-100 text-slate-700"
                    )}
                  >
                    <MessageSquare className="h-4 w-4 shrink-0" />
                    <span className="flex-1 truncate">{session.title}</span>
                    <button
                      onClick={(e) => handleDeleteSession(session.id, e)}
                      className="p-1 text-slate-400 hover:text-red-600"
                      aria-label="Delete chat"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </Link>
                ))}
              </div>

              <div className="border-t border-slate-200 p-3 space-y-1">
                <Link
                  href="/settings"
                  onClick={() => setMobileNavOpen(false)}
                  className="flex items-center gap-2 rounded-2xl p-3 text-sm hover:bg-slate-100 transition-colors text-slate-700"
                >
                  <Settings className="h-4 w-4" />
                  <span>Settings</span>
                </Link>
                <button
                  onClick={() => {
                    setMobileNavOpen(false)
                    handleSignOut()
                  }}
                  className="flex w-full items-center gap-2 rounded-2xl p-3 text-sm hover:bg-slate-100 transition-colors text-red-600"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Sign Out</span>
                </button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside
        className={cn(
          "hidden sm:flex flex-col border-r border-slate-200 bg-white transition-all duration-300",
          isCollapsed ? "w-16" : "w-64"
        )}
      >
        {/* Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-200 px-4">
          {!isCollapsed && (
            <Link href="/" className="flex items-center gap-2">
              <span className="font-bold text-slate-900">Proprly.</span>
            </Link>
          )}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="rounded-lg p-1.5 hover:bg-slate-100 text-slate-700"
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-2">
          <button
            onClick={handleNewChat}
            className={cn(
              "flex w-full items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-2.5 text-sm text-slate-900 hover:bg-slate-100 transition-colors",
              isCollapsed && "justify-center"
            )}
          >
            <Plus className="h-4 w-4" />
            {!isCollapsed && <span className="font-medium">New Chat</span>}
          </button>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.map((session) => (
            <Link
              key={session.id}
              href={`/chat/${session.id}`}
              className={cn(
                "group flex items-center gap-2 rounded-2xl p-2.5 text-sm transition-colors",
                currentSessionId === session.id
                  ? "bg-brand-50 text-brand-900 border border-brand-100"
                  : "hover:bg-slate-100 text-slate-700",
                isCollapsed && "justify-center"
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              {!isCollapsed && (
                <>
                  <span className="flex-1 truncate">{session.title}</span>
                  <button
                    onClick={(e) => handleDeleteSession(session.id, e)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-slate-400 hover:text-red-600"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </>
              )}
            </Link>
          ))}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-200 p-2 space-y-1">
          <Link
            href="/settings"
            className={cn(
              "flex items-center gap-2 rounded-2xl p-2.5 text-sm hover:bg-slate-100 transition-colors text-slate-700",
              isCollapsed && "justify-center"
            )}
          >
            <Settings className="h-4 w-4" />
            {!isCollapsed && <span>Settings</span>}
          </Link>
          <button
            onClick={handleSignOut}
            className={cn(
              "flex w-full items-center gap-2 rounded-2xl p-2.5 text-sm hover:bg-slate-100 transition-colors text-red-600",
              isCollapsed && "justify-center"
            )}
          >
            <LogOut className="h-4 w-4" />
            {!isCollapsed && <span>Sign Out</span>}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 min-w-0 min-h-0 overflow-hidden pt-[calc(56px+env(safe-area-inset-top))] pb-[calc(72px+env(safe-area-inset-bottom))] sm:pt-0 sm:pb-0">
        {children}
      </main>
    </div>
  )
}

