import React from 'react';
import { ArrowRight, CheckCircle2, SlidersHorizontal } from 'lucide-react';

export const Hero: React.FC = () => {
  return (
    <section className="pt-28 pb-16 md:pt-48 md:pb-32 overflow-hidden">
      <div className="container mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
          
          {/* Content */}
          <div className="flex-1 max-w-2xl animate-fade-in-up">
            
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-serif font-medium text-slate-900 leading-[1.1] mb-6">
              Sell More Off-Plan Units <br/>
              <span className="text-brand-600 italic">With Confidence.</span>
            </h1>
            
            <p className="text-lg md:text-xl text-slate-600 mb-8 leading-relaxed max-w-lg">
              Instantly predict the future value of any off-plan property. Our AI model, trained on <span className="font-semibold text-slate-900">7 million data points</span>, gives you the data-backed answers to validate investments and close more deals.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 mb-10">
              <button className="px-8 py-4 bg-slate-900 text-white rounded-full font-semibold text-lg hover:bg-slate-800 transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2 active:scale-95">
                Sign Up <ArrowRight className="w-5 h-5" />
              </button>
            </div>

            <div className="flex flex-col gap-3">
              <p className="text-sm font-medium text-slate-400 uppercase tracking-wide">Trusted by</p>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold font-serif text-slate-700">100+</span>
                <span className="text-lg font-medium text-slate-500">Agents in Dubai</span>
              </div>
            </div>
          </div>

          {/* Visual */}
          <div className="flex-1 w-full relative mt-8 lg:mt-0">
            <div className="relative z-10 bg-white rounded-2xl shadow-2xl border border-slate-100 overflow-hidden transform rotate-1 hover:rotate-0 transition-transform duration-500">
              {/* Header of Mock App */}
              <div className="bg-slate-50 border-b border-slate-100 p-4 flex items-center gap-4">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-400"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                  <div className="w-3 h-3 rounded-full bg-green-400"></div>
                </div>
                <div className="flex-1 text-center text-xs font-medium text-slate-400">Proprly Analyst</div>
              </div>
              
              {/* Body of Mock App */}
              <div className="p-6 bg-slate-50/50 min-h-[350px] md:min-h-[400px] flex flex-col gap-6">
                
                {/* Input Simulation (Fixed Inputs) */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4">
                  <div className="flex items-center gap-2 mb-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    <SlidersHorizontal className="w-4 h-4" /> Property Parameters
                  </div>
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div className="bg-slate-50 p-2 rounded border border-slate-100 text-sm">
                      <span className="text-slate-400 text-xs block">Project</span>
                      <span className="font-medium text-slate-900">Binghatti Grove</span>
                    </div>
                    <div className="bg-slate-50 p-2 rounded border border-slate-100 text-sm">
                      <span className="text-slate-400 text-xs block">Area</span>
                      <span className="font-medium text-slate-900">JVC</span>
                    </div>
                    <div className="bg-slate-50 p-2 rounded border border-slate-100 text-sm">
                      <span className="text-slate-400 text-xs block">Unit</span>
                      <span className="font-medium text-slate-900">2 Bedroom</span>
                    </div>
                     <div className="bg-slate-50 p-2 rounded border border-slate-100 text-sm">
                      <span className="text-slate-400 text-xs block">Price</span>
                      <span className="font-medium text-slate-900">AED 2.2M</span>
                    </div>
                  </div>
                </div>

                {/* AI Response */}
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-brand-600 flex-shrink-0 flex items-center justify-center text-white font-bold text-xs hidden sm:flex">P</div>
                  <div className="bg-white border border-slate-100 px-4 md:px-6 py-5 rounded-2xl rounded-tl-sm w-full shadow-sm">
                    <div className="flex items-center justify-between mb-4 border-b border-slate-100 pb-3">
                      <div className="flex flex-col">
                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Projected Uplift</span>
                        <span className="text-xl md:text-2xl font-bold text-green-600">+AED 750,000</span>
                      </div>
                      <div className="text-right">
                         <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Gross Yield</span>
                         <div className="text-lg md:text-xl font-bold text-slate-900">5.9%</div>
                      </div>
                    </div>
                    
                    <div className="space-y-3 mb-4">
                      <div className="flex justify-between text-sm">
                         <span className="text-slate-500">Est. Value at Handover</span>
                         <span className="font-medium text-slate-900">AED 2.78M</span>
                      </div>
                       <div className="flex justify-between text-sm">
                         <span className="text-slate-500">Confidence Range</span>
                         <span className="font-medium text-slate-900">2.65M – 2.91M</span>
                      </div>
                    </div>

                    <div className="bg-brand-50 rounded-lg p-3 text-xs text-brand-800 leading-relaxed mb-4">
                      <strong>Forecast Driver:</strong> Strong area momentum in JVC (+18% YoY) and moderate supply pipeline.
                    </div>

                    <button className="w-full py-2 bg-white border border-slate-200 text-slate-700 text-sm font-semibold rounded-lg hover:bg-slate-50 flex items-center justify-center gap-2">
                       <CheckCircle2 className="w-4 h-4 text-brand-600" /> Generate Client PDF
                    </button>
                  </div>
                </div>

              </div>
            </div>

            {/* Decorative Elements */}
            <div className="absolute -top-10 -right-10 w-40 h-40 md:w-64 md:h-64 bg-brand-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
            <div className="absolute -bottom-10 -left-10 w-40 h-40 md:w-64 md:h-64 bg-purple-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
          </div>
        </div>
      </div>
    </section>
  );
};