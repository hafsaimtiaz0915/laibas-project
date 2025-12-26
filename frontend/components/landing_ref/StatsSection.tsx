import React from "react"

export const StatsSection: React.FC = () => {
  const stats = [
    { value: "30%", label: "Increase in close rate" },
    { value: "3,000+", label: "Projects covered" },
    { value: "7M+", label: "Data points analysed" },
    { value: "1.6M", label: "Verified DLD transactions" },
  ]

  return (
    <section className="py-12 md:py-16 border-y border-slate-100 bg-slate-50/50">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-slate-900 mb-2 font-serif">{stat.value}</div>
              <div className="text-xs md:text-sm font-medium text-slate-500 uppercase tracking-wide">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}


