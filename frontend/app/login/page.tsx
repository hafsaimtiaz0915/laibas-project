"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { signIn, signUp } from "@/lib/supabase"
import Link from "next/link"
import { Building2, Loader2, ArrowRight } from "lucide-react"

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isSignUp, setIsSignUp] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      if (isSignUp) {
        const { data, error } = await signUp(email, password)
        if (error) throw error
        // If email confirmations are disabled in Supabase, `session` will be returned and the user is already signed in.
        if (data?.session) {
          router.push("/chat")
        } else {
          // If confirmations are enabled, Supabase creates the user but doesn't return a session until they confirm.
          setError("Check your email to confirm your account")
        }
      } else {
        const { error } = await signIn(email, password)
        if (error) throw error
        router.push("/chat")
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="container mx-auto px-6">
        <div className="min-h-screen flex flex-col items-center justify-center py-12">
          <div className="w-full max-w-md">
            <div className="mb-8 text-center">
              <Link href="/" className="inline-flex items-center gap-2">
                <span className="text-2xl font-bold tracking-tight text-slate-900">Proprly.</span>
              </Link>
              <h1 className="mt-6 text-3xl font-serif font-medium text-slate-900">
                {isSignUp ? "Create your account" : "Welcome back"}
              </h1>
              <p className="mt-2 text-sm text-slate-600">
                {isSignUp
                  ? "Start generating client-ready investment reports in minutes."
                  : "Sign in to continue to your workspace."}
              </p>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-8 shadow-lg">
              <form onSubmit={handleSubmit} className="space-y-5">
                <div className="space-y-4">
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-slate-700 mb-1.5">
                      Email
                    </label>
                    <input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-100"
                      placeholder="agent@example.com"
                    />
                  </div>

                  <div>
                    <label htmlFor="password" className="block text-sm font-medium text-slate-700 mb-1.5">
                      Password
                    </label>
                    <input
                      id="password"
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={6}
                      className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-100"
                      placeholder="••••••••"
                    />
                  </div>
                </div>

                {error && (
                  <p className={`text-sm ${error.includes("Check your email") ? "text-green-600" : "text-red-600"}`}>
                    {error}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full rounded-full bg-slate-900 py-3.5 font-semibold text-white hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-100 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {isSignUp ? "Creating Account..." : "Signing In..."}
                    </>
                  ) : (
                    <>
                      {isSignUp ? "Create Account" : "Sign In"} <ArrowRight className="h-4 w-4" />
                    </>
                  )}
                </button>

                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => setIsSignUp(!isSignUp)}
                    className="text-sm text-slate-600 hover:text-brand-700 transition-colors"
                  >
                    {isSignUp ? "Already have an account? Sign in" : "Need an account? Sign up"}
                  </button>
                </div>
              </form>
            </div>

            <div className="mt-6 text-center text-xs text-slate-500">
              By continuing you agree to our terms. Informational only — not financial advice.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

