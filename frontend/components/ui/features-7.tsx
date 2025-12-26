import Image from "next/image"
import { Cpu, Lock, Sparkles, Zap } from "lucide-react"

export function Features() {
  return (
    <section className="overflow-hidden bg-white py-16 md:py-24">
      <div className="mx-auto max-w-7xl space-y-10 px-6 md:space-y-14">
        {/* Heading */}
        <div className="relative z-10 max-w-3xl">
          <h2 className="text-4xl font-serif font-medium leading-[1.05] tracking-tight text-slate-900 md:text-5xl lg:text-6xl">
            Investors demand certainty.
            <br />
            <span className="text-brand-600">Marketing brochures don&apos;t provide it.</span>
          </h2>
          <p className="mt-6 text-lg text-slate-600">
            Replace guesswork with a consistent, data-backed analysis you can share confidently—value outlook, yield
            signals, momentum, and supply context in one place.
          </p>
        </div>

        {/* Desktop: cards left (1/3), image right (2/3) */}
        <div className="grid items-start gap-8 lg:grid-cols-12 lg:gap-10">
          <div className="space-y-4 lg:order-1 lg:col-span-3">
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2">
                  <Zap className="size-4 text-slate-900" />
                  <h3 className="text-sm font-medium text-slate-900">Fast</h3>
                </div>
                <p className="mt-2 text-sm text-slate-600">
                  Generate consistent analysis quickly, so you can respond while the lead is still warm.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2">
                  <Cpu className="size-4 text-slate-900" />
                  <h3 className="text-sm font-medium text-slate-900">Powerful</h3>
                </div>
                <p className="mt-2 text-sm text-slate-600">
                  Built on large historical datasets to support defensible decisions.
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2">
                  <Lock className="size-4 text-slate-900" />
                  <h3 className="text-sm font-medium text-slate-900">Secure</h3>
                </div>
                <p className="mt-2 text-sm text-slate-600">Keep your workflow and outputs clean, controlled, and shareable.</p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2">
                  <Sparkles className="size-4 text-slate-900" />
                  <h3 className="text-sm font-medium text-slate-900">AI Powered</h3>
                </div>
                <p className="mt-2 text-sm text-slate-600">
                  Turn structured inputs into a clear narrative buyers and investors can trust.
                </p>
              </div>
            </div>
          </div>

          <div className="relative rounded-3xl bg-slate-50 p-3 lg:order-2 lg:col-span-9">
            <div className="[perspective:800px] max-w-2xl mx-auto">
              <div className="[transform:skewY(-2deg)skewX(-2deg)rotateX(6deg)]">
                <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_45px_120px_-55px_rgba(15,23,42,0.55)]">
                  {/* Match the image ratio (1366x768 = 16:9) so it never crops */}
                  <div className="relative aspect-[16/9]">
                    <Image
                      src="/section2-image.png"
                      alt="Property analysis form"
                      fill
                      className="object-contain"
                      sizes="(max-width: 768px) 90vw, 600px"
                      quality={95}
                      priority={false}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}


