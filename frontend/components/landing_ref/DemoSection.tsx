import React from "react"
import { BrainCircuit, FileDown, MousePointerClick } from "lucide-react"
import Image from "next/image"

export const DemoSection: React.FC = () => {
  const steps = [
    {
      icon: <MousePointerClick className="w-6 h-6" />,
      title: "1. Select",
      desc: "Choose the Developer, Project, and Unit Type from our structured database.",
      color: "bg-blue-100 text-blue-600",
    },
    {
      icon: <BrainCircuit className="w-6 h-6" />,
      title: "2. Analyze",
      desc: "Our model runs the parameters against 7 million data points to forecast value.",
      color: "bg-purple-100 text-purple-600",
    },
    {
      icon: <FileDown className="w-6 h-6" />,
      title: "3. Export",
      desc: "Generate a branded PDF investment report to send to your client instantly.",
      color: "bg-green-100 text-green-600",
    },
  ]

  return (
    <section id="demo" className="py-16 md:py-24 bg-slate-50 border-y border-slate-200">
      <div className="container mx-auto px-6">
        <div className="flex flex-col lg:flex-row gap-12 lg:gap-16 items-center">
          <div className="flex-1">
            <h2 className="text-3xl md:text-4xl font-serif font-medium text-slate-900 mb-6">
              From input to client-ready report in seconds.
            </h2>
            <p className="text-lg text-slate-600 mb-12">
              Streamline your investment advisory workflow. No more digging through spreadsheets or relying on developer
              marketing brochures.
            </p>

            <div className="space-y-8">
              {steps.map((step, idx) => (
                <div key={idx} className="flex gap-6 group">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${step.color}`}>
                    {step.icon}
                  </div>
                  <div>
                    <h4 className="text-xl font-semibold text-slate-900 mb-2">{step.title}</h4>
                    <p className="text-slate-600 leading-relaxed">{step.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex-1 w-full perspective-1000">
            {/* Abstract UI Representation of the Report */}
            <div className="relative bg-white w-full max-w-sm md:max-w-md mx-auto aspect-[1/1.41] shadow-2xl rounded-sm border border-slate-200 p-6 md:p-8 transform rotate-y-6 md:rotate-y-12 hover:rotate-y-0 transition-transform duration-700">
              <Image
                src="/sample-report.svg"
                alt="Sample Proprly report"
                fill
                className="object-contain"
                sizes="(max-width: 768px) 90vw, 420px"
                priority={false}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}


