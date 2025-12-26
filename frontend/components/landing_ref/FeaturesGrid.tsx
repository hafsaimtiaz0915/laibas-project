import React from "react"
import Image from "next/image"
import { Activity, DraftingCompass, Mail, Zap } from "lucide-react"

export const FeaturesGrid: React.FC = () => {
  return (
    <section id="features" className="py-16 md:py-32 bg-white">
      <div className="mx-auto max-w-xl md:max-w-6xl px-6">
        <div className="grid items-center gap-12 md:grid-cols-2 md:gap-12 lg:grid-cols-5 lg:gap-24">
          <div className="lg:col-span-2">
            <div className="md:pr-6 lg:pr-0">
              <h2 className="text-4xl font-serif font-medium text-slate-900 lg:text-5xl">
                Built for Scaling Teams
              </h2>
              <p className="mt-6 text-slate-600 leading-relaxed">
                From solo agents to enterprise brokerages, Proprly scales with your workflow. Generate consistent, data-backed reports that build trust and close deals faster.
              </p>
            </div>
            <ul className="mt-8 divide-y border-y border-slate-200 *:flex *:items-center *:gap-3 *:py-4 text-slate-700">
              <li>
                <Mail className="size-5 text-brand-600" />
                Email and web support
              </li>
              <li>
                <Zap className="size-5 text-brand-600" />
                Fast response time
              </li>
              <li>
                <Activity className="size-5 text-brand-600" />
                Monitoring and analytics
              </li>
              <li>
                <DraftingCompass className="size-5 text-brand-600" />
                Architectural review
              </li>
            </ul>
          </div>
          <div className="relative lg:col-span-3 flex items-center justify-center">
            <Image
              src="/Component2.png"
              alt="Proprly features illustration"
              width={1207}
              height={929}
              className="w-full h-auto max-w-4xl scale-110 lg:scale-125"
              quality={95}
            />
          </div>
        </div>
      </div>
    </section>
  )
}
