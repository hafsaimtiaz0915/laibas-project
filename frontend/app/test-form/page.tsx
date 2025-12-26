"use client"

import { PropertyFormInput } from "@/components/chat/PropertyFormInput"

export default function TestFormPage() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Gradient mesh background for glass effect to show */}
      <div className="absolute inset-0 bg-gradient-to-br from-brand-100 via-slate-100 to-cyan-50" />
      <div className="absolute top-20 -left-20 w-96 h-96 bg-brand-300/40 rounded-full blur-3xl" />
      <div className="absolute bottom-20 -right-20 w-80 h-80 bg-cyan-300/30 rounded-full blur-3xl" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-purple-200/20 rounded-full blur-3xl" />
      
      <div className="relative z-10 p-8">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold mb-6 text-slate-900">Property Form Test</h1>
          <PropertyFormInput 
            onSend={(q) => console.log("Submitted:", q)} 
            disabled={false}
            initialCollapsed={false}
          />
        </div>
      </div>
    </div>
  )
}

