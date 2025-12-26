import React from 'react';
import { FileText, TrendingUp, ShieldCheck, Sliders } from 'lucide-react';

export const FeaturesGrid: React.FC = () => {
  return (
    <section id="features" className="py-16 md:py-24 bg-white">
      <div className="container mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-12 md:mb-16">
          <h2 className="text-3xl md:text-4xl font-serif font-medium text-slate-900 mb-4">
             Turn Skeptical Investors into <br/> <span className="text-brand-600 italic">Confident Buyers</span>
          </h2>
          <p className="text-lg text-slate-600">
            Stop losing deals to uncertainty. Arm your clients with data-backed intelligence that proves the ROI and sets you apart from every other agent in Dubai.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Feature 1 - Large */}
          <div className="md:col-span-2 bg-slate-50 rounded-3xl p-8 border border-slate-100 hover:border-brand-100 transition-colors group">
            <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-6 text-brand-600">
              <Sliders className="w-6 h-6" />
            </div>
            <h3 className="text-2xl font-semibold text-slate-900 mb-3">Structured Deal Qualification</h3>
            <p className="text-slate-600 mb-6 max-w-lg">
              No guesswork required. Simply select the Project, Area, and Unit Type from our comprehensive database. Our AI instantly benchmarks your selection against 7 million data points to generate a precise investment outlook.
            </p>
            <div className="bg-white rounded-xl p-4 border border-slate-100 shadow-sm max-w-md">
              <div className="flex flex-col sm:flex-row sm:items-center gap-3 text-sm text-slate-700 font-medium">
                <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-brand-500"></div> Project: Binghatti Grove</span>
                <span className="hidden sm:block text-slate-300">|</span>
                <span className="flex items-center gap-2">Area: JVC</span>
              </div>
            </div>
          </div>

          {/* Feature 2 */}
          <div className="bg-slate-50 rounded-3xl p-8 border border-slate-100 hover:border-brand-100 transition-colors">
            <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-6 text-indigo-600">
              <TrendingUp className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900 mb-3">Precision Forecasting</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Get estimated value at handover and +12 months post-handover. Powered by a Temporal Fusion Transformer model trained on 10 years of market patterns.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="bg-slate-50 rounded-3xl p-8 border border-slate-100 hover:border-brand-100 transition-colors">
            <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-6 text-rose-600">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900 mb-3">Developer Vetting</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Validate claims with hard data. Instantly access average delay times, completion rates, and total units delivered for every developer.
            </p>
          </div>

          {/* Feature 4 - Large - Reports */}
          <div className="md:col-span-2 bg-brand-600 rounded-3xl p-8 border border-brand-500 text-white relative overflow-hidden group">
            <div className="relative z-10">
              <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center mb-6 text-white">
                <FileText className="w-6 h-6" />
              </div>
              <h3 className="text-2xl font-semibold mb-3">Your Unfair Advantage: White-Labeled Reports</h3>
              <p className="text-brand-100 mb-6 max-w-xl leading-relaxed">
                Give your clients the institutional-grade clarity they can't get anywhere else. Generate professional, branded investment reports that crush objections and validate value. Position yourself as the trusted authority and give investors the confidence to sign the SPA on the spot.
              </p>
              <button className="px-6 py-3 bg-white text-brand-700 text-sm font-bold rounded-lg hover:bg-brand-50 transition-colors shadow-lg">
                View Sample Report
              </button>
            </div>
            
            {/* Decorative BG */}
            <div className="absolute right-0 bottom-0 w-64 h-64 bg-white/10 rounded-full blur-3xl transform translate-x-1/3 translate-y-1/3"></div>
          </div>

        </div>
      </div>
    </section>
  );
};