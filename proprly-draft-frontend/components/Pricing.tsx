import React from 'react';
import { Check } from 'lucide-react';

export const Pricing: React.FC = () => {
  return (
    <section id="pricing" className="py-16 md:py-24 bg-slate-50 border-t border-slate-200">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12 md:mb-16">
          <h2 className="text-3xl md:text-4xl font-serif font-medium text-slate-900 mb-4">
            Simple, transparent pricing
          </h2>
          <p className="text-slate-600">
            Choose the plan that fits your deal flow.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          
          {/* Starter Plan */}
          <div className="bg-white rounded-2xl p-6 md:p-8 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-6">
              <h3 className="text-lg font-bold text-slate-900">Starter</h3>
              <p className="text-sm text-slate-500 mb-4">For individual brokers getting started.</p>
              <div className="flex items-baseline">
                <span className="text-3xl md:text-4xl font-bold text-slate-900">AED 2,500</span>
                <span className="text-slate-500 ml-2">/mo</span>
              </div>
            </div>
            <ul className="space-y-4 mb-8">
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> 5 PDF Reports / month
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> 1 User Seat
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> Standard Report Format
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> Basic Forecasts
              </li>
            </ul>
            <button className="w-full py-3 border-2 border-slate-900 text-slate-900 font-bold rounded-xl hover:bg-slate-50 transition-colors">
              Start Trial
            </button>
          </div>

          {/* Professional Plan */}
          <div className="bg-white rounded-2xl p-6 md:p-8 border-2 border-brand-600 shadow-xl relative transform md:-translate-y-4 order-first md:order-none">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-brand-600 text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wide">
              Most Popular
            </div>
            <div className="mb-6">
              <h3 className="text-lg font-bold text-slate-900">Professional</h3>
              <p className="text-sm text-slate-500 mb-4">For active agents closing deals.</p>
              <div className="flex items-baseline">
                <span className="text-3xl md:text-4xl font-bold text-slate-900">AED 7,500</span>
                <span className="text-slate-500 ml-2">/mo</span>
              </div>
            </div>
            <ul className="space-y-4 mb-8">
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-brand-600 shrink-0" /> 40 PDF Reports / month
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-brand-600 shrink-0" /> 4 User Seats
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-brand-600 shrink-0" /> <strong>Custom Branding (Logo)</strong>
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-brand-600 shrink-0" /> Detailed Forecast Drivers
              </li>
            </ul>
            <button className="w-full py-3 bg-brand-600 text-white font-bold rounded-xl hover:bg-brand-700 transition-colors shadow-lg shadow-brand-200">
              Get Started
            </button>
          </div>

          {/* Advanced Plan */}
          <div className="bg-white rounded-2xl p-6 md:p-8 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-6">
              <h3 className="text-lg font-bold text-slate-900">Advanced</h3>
              <p className="text-sm text-slate-500 mb-4">For investment teams & agencies.</p>
              <div className="flex items-baseline">
                <span className="text-3xl md:text-4xl font-bold text-slate-900">AED 20,000</span>
                <span className="text-slate-500 ml-2">/mo</span>
              </div>
            </div>
            <ul className="space-y-4 mb-8">
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> 100+ PDF Reports / month
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> 10 User Seats
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> Fully Branded Templates
              </li>
              <li className="flex items-start gap-3 text-sm text-slate-700">
                <Check className="w-5 h-5 text-green-500 shrink-0" /> Team Workflows
              </li>
            </ul>
            <button className="w-full py-3 border-2 border-slate-900 text-slate-900 font-bold rounded-xl hover:bg-slate-50 transition-colors">
              Contact Sales
            </button>
          </div>

        </div>
      </div>
    </section>
  );
};