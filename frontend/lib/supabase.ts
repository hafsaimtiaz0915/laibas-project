import { createClient, type SupabaseClient } from "@supabase/supabase-js"

let _client: SupabaseClient | null = null

function requireSupabaseEnv() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!url || !anonKey) {
    // Important: do NOT throw at import-time. Next.js may load this module during
    // build/prerender even for client components.
    throw new Error(
      "Missing Supabase env vars. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY."
    )
  }

  return { url, anonKey }
}

// Lazily create the Supabase client to avoid build-time crashes.
export function getSupabaseClient(): SupabaseClient {
  if (_client) return _client
  const { url, anonKey } = requireSupabaseEnv()
  _client = createClient(url, anonKey)
  return _client
}

// Auth helpers
export async function getSession() {
  const supabase = getSupabaseClient()
  const {
    data: { session },
    error,
  } = await supabase.auth.getSession()
  if (error) {
    console.error("Error getting session:", error)
    return null
  }
  return session
}

export async function getUser() {
  const supabase = getSupabaseClient()
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser()
  if (error) {
    console.error("Error getting user:", error)
    return null
  }
  return user
}

export async function signIn(email: string, password: string) {
  const supabase = getSupabaseClient()
  return supabase.auth.signInWithPassword({ email, password })
}

export async function signUp(email: string, password: string) {
  const supabase = getSupabaseClient()
  return supabase.auth.signUp({ email, password })
}

export async function signOut() {
  const supabase = getSupabaseClient()
  return supabase.auth.signOut()
}

export async function getAccessToken(): Promise<string | null> {
  const supabase = getSupabaseClient()
  const {
    data: { session },
  } = await supabase.auth.getSession()
  return session?.access_token ?? null
}
