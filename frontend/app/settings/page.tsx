"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { getProfile, updateSettings, uploadLogo, deleteLogo } from "@/lib/api"
import { useAgentStore } from "@/stores/agent-store"
import type { AgentProfile } from "@/lib/types"
import { useDropzone } from "react-dropzone"
import { ArrowLeft, Upload, Trash2, Loader2, Save } from "lucide-react"
import Image from "next/image"

export default function SettingsPage() {
  const router = useRouter()
  const { settings, setSettings } = useAgentStore()
  const [profile, setProfile] = useState<AgentProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [isUploadingLogo, setIsUploadingLogo] = useState(false)
  const [message, setMessage] = useState("")

  // Form state
  const [name, setName] = useState("")
  const [companyName, setCompanyName] = useState("")
  const [phone, setPhone] = useState("")
  const [primaryColor, setPrimaryColor] = useState("#0f766e")
  const [secondaryColor, setSecondaryColor] = useState("#10b981")
  const [reportFooter, setReportFooter] = useState("")
  const [showContactInfo, setShowContactInfo] = useState(true)

  useEffect(() => {
    const loadProfile = async () => {
      try {
        const data = await getProfile()
        setProfile(data)
        setName(data.name || "")
        setCompanyName(data.company_name || "")
        setPhone(data.phone || "")
        setPrimaryColor(data.primary_color || "#0f766e")
        setSecondaryColor(data.secondary_color || "#10b981")
        setReportFooter(data.report_footer || "")
        setShowContactInfo(data.show_contact_info ?? true)
        
        // Update store
        setSettings(data)
      } catch (err) {
        console.error("Failed to load profile:", err)
      } finally {
        setIsLoading(false)
      }
    }
    loadProfile()
  }, [setSettings])

  const handleSave = async () => {
    setIsSaving(true)
    setMessage("")
    
    try {
      const updated = await updateSettings({
        name: name || undefined,
        company_name: companyName || undefined,
        phone: phone || undefined,
        primary_color: primaryColor,
        secondary_color: secondaryColor,
        report_footer: reportFooter || undefined,
        show_contact_info: showContactInfo,
      })
      
      setProfile(updated)
      setSettings(updated)
      setMessage("Settings saved successfully!")
    } catch (err: any) {
      setMessage(err.message || "Failed to save settings")
    } finally {
      setIsSaving(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [".jpeg", ".jpg", ".png", ".webp"] },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024, // 5MB
    onDrop: async (acceptedFiles) => {
      if (acceptedFiles.length === 0) return
      
      setIsUploadingLogo(true)
      try {
        const result = await uploadLogo(acceptedFiles[0])
        setProfile((prev) => prev ? { ...prev, logo_url: result.logo_url } : prev)
        setSettings({ logo_url: result.logo_url })
      } catch (err: any) {
        setMessage(err.message || "Failed to upload logo")
      } finally {
        setIsUploadingLogo(false)
      }
    },
  })

  const handleDeleteLogo = async () => {
    if (!confirm("Delete your logo?")) return
    
    try {
      await deleteLogo()
      setProfile((prev) => prev ? { ...prev, logo_url: undefined } : prev)
      setSettings({ logo_url: undefined })
    } catch (err: any) {
      setMessage(err.message || "Failed to delete logo")
    }
  }

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading settings...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="max-w-3xl mx-auto px-4 py-4 flex items-center gap-4">
          <button
            onClick={() => router.push("/chat")}
            className="rounded-lg p-2 hover:bg-accent"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <h1 className="text-xl font-semibold">Settings</h1>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-3xl mx-auto px-4 py-8 space-y-8">
        {/* Logo Upload */}
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Logo</h2>
          <p className="text-sm text-muted-foreground">
            Your logo will appear on generated PDF reports.
          </p>
          
          <div className="flex items-start gap-4">
            {profile?.logo_url ? (
              <div className="relative">
                <Image
                  src={profile.logo_url}
                  alt="Logo"
                  width={120}
                  height={120}
                  className="rounded-lg object-contain border"
                />
                <button
                  onClick={handleDeleteLogo}
                  className="absolute -top-2 -right-2 rounded-full bg-destructive p-1.5 text-white hover:bg-destructive/90"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : (
              <div
                {...getRootProps()}
                className={`flex h-32 w-32 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors ${
                  isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/30 hover:border-muted-foreground/50"
                }`}
              >
                <input {...getInputProps()} />
                {isUploadingLogo ? (
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                ) : (
                  <>
                    <Upload className="h-6 w-6 text-muted-foreground mb-2" />
                    <span className="text-xs text-muted-foreground text-center">
                      Drop logo or click
                    </span>
                  </>
                )}
              </div>
            )}
          </div>
        </section>

        {/* Profile Info */}
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Profile</h2>
          
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium mb-1.5">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="John Doe"
                className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1.5">Company</label>
              <input
                type="text"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                placeholder="ABC Real Estate"
                className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1.5">Phone</label>
              <input
                type="tel"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                placeholder="+971 50 123 4567"
                className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          </div>
        </section>

        {/* Report Settings */}
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Report Settings</h2>
          
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium mb-1.5">Primary Color</label>
              <div className="flex gap-2">
                <input
                  type="color"
                  value={primaryColor}
                  onChange={(e) => setPrimaryColor(e.target.value)}
                  className="h-10 w-10 rounded cursor-pointer"
                />
                <input
                  type="text"
                  value={primaryColor}
                  onChange={(e) => setPrimaryColor(e.target.value)}
                  className="flex-1 rounded-lg border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1.5">Secondary Color</label>
              <div className="flex gap-2">
                <input
                  type="color"
                  value={secondaryColor}
                  onChange={(e) => setSecondaryColor(e.target.value)}
                  className="h-10 w-10 rounded cursor-pointer"
                />
                <input
                  type="text"
                  value={secondaryColor}
                  onChange={(e) => setSecondaryColor(e.target.value)}
                  className="flex-1 rounded-lg border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1.5">Report Footer</label>
            <textarea
              value={reportFooter}
              onChange={(e) => setReportFooter(e.target.value)}
              placeholder="Custom footer text for your reports..."
              rows={2}
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="showContactInfo"
              checked={showContactInfo}
              onChange={(e) => setShowContactInfo(e.target.checked)}
              className="rounded border-muted-foreground/30"
            />
            <label htmlFor="showContactInfo" className="text-sm">
              Show contact info on reports
            </label>
          </div>
        </section>

        {/* Save Button */}
        <div className="flex items-center gap-4">
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {isSaving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save Changes
          </button>
          
          {message && (
            <span className={`text-sm ${message.includes("success") ? "text-green-500" : "text-destructive"}`}>
              {message}
            </span>
          )}
        </div>
      </main>
    </div>
  )
}

