"use client"

import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react"
import Image from "next/image"
import Link from "next/link"

const Spline = lazy(() => import("@splinetool/react-spline"))

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false)

  useEffect(() => {
    const mql = window.matchMedia("(prefers-reduced-motion: reduce)")
    const update = () => setReduced(Boolean(mql.matches))
    update()
    mql.addEventListener?.("change", update)
    return () => mql.removeEventListener?.("change", update)
  }, [])

  return reduced
}

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false)
  useEffect(() => {
    const update = () => setIsMobile(window.innerWidth < 640)
    update()
    window.addEventListener("resize", update)
    return () => window.removeEventListener("resize", update)
  }, [])
  return isMobile
}

function HeroSplineBackground() {
  return (
    <div className="relative h-[100svh] w-full overflow-hidden">
      <Suspense fallback={<div className="h-[100svh] w-full bg-black" />}>
        <Spline
          style={{
            width: "100%",
            height: "100svh",
            pointerEvents: "auto",
          }}
          scene="https://prod.spline.design/us3ALejTXl6usHZ7/scene.splinecode"
        />
      </Suspense>

      {/* readability overlays */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: `
            linear-gradient(to right, rgba(0, 0, 0, 0.82), transparent 30%, transparent 70%, rgba(0, 0, 0, 0.82)),
            linear-gradient(to bottom, transparent 45%, rgba(0, 0, 0, 0.92))
          `,
        }}
      />
    </div>
  )
}

function Navbar() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 1024) setOpen(false)
    }
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  const links = useMemo(
    () => [
      { href: "#features", label: "Features" },
      { href: "#sample-report", label: "Sample report" },
      { href: "#faq", label: "FAQ" },
      { href: "#pricing", label: "Pricing (soon)" },
    ],
    []
  )

  return (
    <nav
      className="fixed left-0 right-0 top-0 z-20"
      style={{
        backgroundColor: "rgba(13, 13, 24, 0.30)",
        backdropFilter: "blur(10px)",
        WebkitBackdropFilter: "blur(10px)",
        borderRadius: "0 0 15px 15px",
      }}
    >
      <div className="container mx-auto flex items-center justify-between px-4 py-4 md:px-6 lg:px-8">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-sm font-semibold tracking-tight text-white">
            Proprly
          </Link>

          <div className="hidden lg:flex items-center gap-6">
            {links.map((l) => (
              <a key={l.href} href={l.href} className="text-sm text-white/70 hover:text-white transition-colors">
                {l.label}
              </a>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Link href="/login" className="hidden sm:block text-sm text-white/70 hover:text-white transition-colors">
            Sign in
          </Link>
          <Link
            href="/login"
            className="inline-flex h-10 items-center justify-center rounded-full border border-white/15 bg-white/10 px-4 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15"
          >
            Create account
          </Link>

          <button
            type="button"
            className="lg:hidden inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-black/30 text-white"
            aria-label="Toggle mobile menu"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d={open ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"}
              />
            </svg>
          </button>
        </div>
      </div>

      <div
        className={[
          "lg:hidden absolute top-full left-0 right-0 z-30 overflow-hidden transition-all duration-300 ease-in-out",
          "bg-black/60 border-t border-white/10",
          open ? "max-h-screen opacity-100 pointer-events-auto" : "max-h-0 opacity-0 pointer-events-none",
        ].join(" ")}
        style={{ backdropFilter: "blur(10px)", WebkitBackdropFilter: "blur(10px)" }}
      >
        <div className="px-4 py-6 flex flex-col space-y-4">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-white/80 hover:text-white"
              onClick={() => setOpen(false)}
            >
              {l.label}
            </a>
          ))}
          <Link href="/login" className="text-sm text-white/80 hover:text-white" onClick={() => setOpen(false)}>
            Sign in / Create account
          </Link>
        </div>
      </div>
    </nav>
  )
}

function HeroContent() {
  return (
    <div className="max-w-3xl px-4 pt-16 text-left text-white sm:pt-24 md:pt-32">
      <p className="mb-4 text-xs uppercase tracking-[0.2em] text-white/60">
        Built for Dubai brokers
      </p>

      <h1 className="text-3xl font-semibold leading-[1.05] tracking-tight sm:text-5xl md:text-7xl">
        Win trust fast.
        <br className="hidden sm:block" /> Close with confidence.
      </h1>

      <p className="mt-5 max-w-xl text-base text-white/75 sm:text-lg md:text-xl">
        Turn buyer “maybe” into “send me the details” with a crisp, data-backed report: forecasts, yield signals, area
        momentum, supply context, and a branded PDF you can forward instantly.
      </p>

      <div className="pointer-events-auto mt-7 flex flex-col gap-3 sm:flex-row">
        <Link
          href="/login"
          className="inline-flex h-12 w-full items-center justify-center rounded-full border border-white/15 bg-white/10 px-6 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15 sm:w-auto"
        >
          Create account
        </Link>
        <a
          href="#sample-report"
          className="inline-flex h-12 w-full items-center justify-center rounded-full border border-white/15 bg-black/40 px-6 text-sm font-medium text-white/90 transition hover:bg-black/55 sm:w-auto"
        >
          View sample report
        </a>
      </div>

      <p className="mt-5 max-w-xl text-xs text-white/55">
        Trained on large historical Dubai datasets (transactions + rent benchmarks) and built for real conversations.
      </p>
    </div>
  )
}

function SampleReportSection({ screenshotRef }: { screenshotRef: React.RefObject<HTMLDivElement> }) {
  const [imgSrc, setImgSrc] = useState<string>("/sample-report.jpg")

  return (
    <section id="sample-report" className="relative z-10 mx-auto max-w-6xl px-4 pt-10 md:px-6 lg:px-8">
      <div className="mx-auto max-w-3xl">
        <p className="text-xs uppercase tracking-[0.2em] text-white/60">What you send to buyers</p>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight text-white md:text-3xl">
          A report that kills doubt on the spot.
        </h2>
        <p className="mt-3 text-sm text-white/70 md:text-base">
          Proprly generates a clean, investor-friendly PDF: price outlook, rent &amp; yield signals, area momentum, supply
          pipeline context, and developer delivery history—so you handle objections with clarity and speed.
        </p>

        <div
          ref={screenshotRef}
          className="mx-auto mt-8 w-full max-w-[520px] overflow-hidden rounded-2xl border border-white/10 bg-white/5 shadow-2xl"
        >
          {/* Mobile-first: render as a portrait page (A4-ish ratio). */}
          <div className="relative aspect-[7/10] w-full bg-black/40 p-2 sm:p-3">
            <Image
              src={imgSrc}
              alt="Proprly sample report preview"
              fill
              className="rounded-xl object-contain"
              sizes="(max-width: 640px) 92vw, 520px"
              priority
              onError={() => setImgSrc("/sample-report.svg")}
            />
          </div>
        </div>

        <p className="mt-3 text-xs text-white/50">Example only — informational, not financial advice.</p>

        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
          <Link
            href="/login"
            className="inline-flex h-12 w-full items-center justify-center rounded-full border border-white/15 bg-white/10 px-6 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15 sm:w-auto"
          >
            Create account
          </Link>
          <a
            href="#how-it-works"
            className="inline-flex h-12 w-full items-center justify-center rounded-full border border-white/15 bg-black/40 px-6 text-sm font-medium text-white/90 transition hover:bg-black/55 sm:w-auto"
          >
            How it works
          </a>
        </div>
      </div>
    </section>
  )
}

export function HeroSection() {
  const screenshotRef = useRef<HTMLDivElement>(null)
  const heroContentRef = useRef<HTMLDivElement>(null)

  const reducedMotion = usePrefersReducedMotion()
  const isMobile = useIsMobile()

  useEffect(() => {
    if (reducedMotion || isMobile) return

    const handleScroll = () => {
      if (!screenshotRef.current || !heroContentRef.current) return
      requestAnimationFrame(() => {
        const scrollY = window.scrollY || window.pageYOffset
        screenshotRef.current!.style.transform = `translateY(-${scrollY * 0.18}px)`

        const maxScroll = 420
        const opacity = 1 - Math.min(scrollY / maxScroll, 1)
        heroContentRef.current!.style.opacity = opacity.toString()
      })
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [reducedMotion, isMobile])

  return (
    <div className="relative bg-black">
      <Navbar />

      <div className="relative min-h-[100svh]">
        <div className="absolute inset-0 z-0">
          <HeroSplineBackground />
        </div>

        <div
          ref={heroContentRef}
          className="pointer-events-none absolute inset-0 z-10 flex items-center"
          style={{ height: "100svh" }}
        >
          <div className="container mx-auto">
            <HeroContent />
          </div>
        </div>
      </div>

      <div className="relative z-10 bg-black" style={{ marginTop: "-10svh" }}>
        <SampleReportSection screenshotRef={screenshotRef} />
      </div>
    </div>
  )
}


