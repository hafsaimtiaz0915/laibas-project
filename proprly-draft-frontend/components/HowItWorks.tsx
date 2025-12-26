import React from 'react';
import { Database, LineChart, Layers } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  return (
    <section className="py-16 md:py-24 bg-white">
      <div className="container mx-auto px-6 text-center">
        <div className="inline-block px-3 py-1 bg-brand-50 text-brand-700 text-xs font-bold rounded-full uppercase tracking-wider mb-6">
          Our Methodology
        </div>
        <h2 className="text-3xl md:text-4xl font-serif font-medium text-slate-900 mb-12 md:mb-16 max-w-2xl mx-auto">
          We don't give advice. <br/>We give you the data to make it.
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12 max-w-5xl mx-auto relative">
           {/* Connecting Line (Desktop) */}
           <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-0.5 bg-gradient-to-r from-slate-200 via-brand-200 to-slate-200 -z-10"></div>

          <div className="bg-white p-6 rounded-2xl border border-transparent hover:border-slate-100 transition-colors">
            <div className="w-20 h-20 md:w-24 md:h-24 bg-white border border-slate-100 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg z-10 relative">
              <Database className="w-8 h-8 md:w-10 md:h-10 text-brand-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-3">1. Aggregate</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              We ingest 1.6M+ official DLD transactions, rent contracts, and project data points daily.
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-transparent hover:border-slate-100 transition-colors">
            <div className="w-20 h-20 md:w-24 md:h-24 bg-white border border-slate-100 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg z-10 relative">
              <Layers className="w-8 h-8 md:w-10 md:h-10 text-brand-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-3">2. Train</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              Our Temporal Fusion Transformer model learns pricing dynamics, lifecycle timing, and supply pressure.
            </p>
          </div>

          <div className="bg-white p-6 rounded-2xl border border-transparent hover:border-slate-100 transition-colors">
            <div className="w-20 h-20 md:w-24 md:h-24 bg-white border border-slate-100 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg z-10 relative">
              <LineChart className="w-8 h-8 md:w-10 md:h-10 text-brand-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-3">3. Forecast</h3>
            <p className="text-slate-600 text-sm leading-relaxed">
              We generate probabilistic forecasts with confidence intervals, not just single-point guesses.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};