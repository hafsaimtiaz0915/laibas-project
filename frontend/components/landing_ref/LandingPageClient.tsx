"use client"

import React, { useEffect, useState } from "react"
import { ArrowRight } from "lucide-react"
import Link from "next/link"

import { Navbar } from "./Navbar"
import { Hero } from "./Hero"
import { StatsSection } from "./StatsSection"
import { FeaturesGrid } from "./FeaturesGrid"
import { DemoSection } from "./DemoSection"
import { HowItWorks } from "./HowItWorks"
import { Pricing } from "./Pricing"
import { Footer } from "./Footer"
import { Features } from "@/components/ui/features-7"

export function LandingPageClient() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <div className="min-h-screen bg-white selection:bg-brand-100 selection:text-brand-900">
      <Navbar scrolled={scrolled} />

      <main>
        <Hero />
        <Features />
        <StatsSection />
        <FeaturesGrid />
        <DemoSection />
        <Pricing />
        <HowItWorks />

        {/* Final CTA */}
        <section className="py-24 bg-brand-900 text-white overflow-hidden relative">
          <div className="absolute top-0 left-0 w-full h-full overflow-hidden opacity-10">
            <div className="absolute -top-[50%] -left-[20%] w-[1000px] h-[1000px] rounded-full bg-white blur-3xl"></div>
            <div className="absolute bottom-0 right-0 w-[800px] h-[800px] rounded-full bg-brand-500 blur-3xl"></div>
          </div>

          <div className="container mx-auto px-6 relative z-10 text-center">
            <h2 className="text-4xl md:text-5xl font-serif font-medium mb-6">Ready to close more deals?</h2>
            <p className="text-xl text-brand-100 mb-10 max-w-2xl mx-auto font-light leading-relaxed">
              Join top performing agents who use Proprly to validate investments and build trust with clients.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link
                href="/login"
                className="px-8 py-4 bg-white text-brand-900 rounded-full font-semibold text-lg hover:bg-brand-50 transition-all shadow-lg hover:shadow-xl flex items-center gap-2"
              >
                Start Free Trial <ArrowRight className="w-5 h-5" />
              </Link>
              <a
                href="#demo"
                className="px-8 py-4 bg-transparent border border-brand-200 text-white rounded-full font-semibold text-lg hover:bg-white/10 transition-all"
              >
                Book a Demo
              </a>
            </div>
            <p className="mt-6 text-sm text-brand-200">No credit card required for 7-day trial.</p>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  )
}


