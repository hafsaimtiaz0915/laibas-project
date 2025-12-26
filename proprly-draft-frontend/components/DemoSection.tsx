import React from 'react';
import { Search, BrainCircuit, FileDown, MousePointerClick } from 'lucide-react';

export const DemoSection: React.FC = () => {
  const steps = [
    {
      icon: <MousePointerClick className="w-6 h-6" />,
      title: "1. Select",
      desc: "Choose the Developer, Project, and Unit Type from our structured database.",
      color: "bg-blue-100 text-blue-600"
    },
    {
      icon: <BrainCircuit className="w-6 h-6" />,
      title: "2. Analyze",
      desc: "Our model runs the parameters against 7 million data points to forecast value.",
      color: "bg-purple-100 text-purple-600"
    },
    {
      icon: <FileDown className="w-6 h-6" />,
      title: "3. Export",
      desc: "Generate a branded PDF investment report to send to your client instantly.",
      color: "bg-green-100 text-green-600"
    }
  ];

  return (
    <section id="demo" className="py-16 md:py-24 bg-slate-50 border-y border-slate-200">
      <div className="container mx-auto px-6">
        <div className="flex flex-col lg:flex-row gap-12 lg:gap-16 items-center">
          
          <div className="flex-1">
            <h2 className="text-3xl md:text-4xl font-serif font-medium text-slate-900 mb-6">
              From input to client-ready report in seconds.
            </h2>
            <p className="text-lg text-slate-600 mb-12">
              Streamline your investment advisory workflow. No more digging through spreadsheets or relying on developer marketing brochures.
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
              
              {/* Header */}
              <div className="flex justify-between items-start mb-6 md:mb-8 border-b border-slate-100 pb-6">
                <div className="w-10 h-10 md:w-12 md:h-12 bg-slate-900 rounded-lg"></div>
                <div className="text-right">
                  <div className="h-3 md:h-4 w-24 md:w-32 bg-slate-200 rounded mb-2 ml-auto"></div>
                  <div className="h-2 md:h-3 w-16 md:w-24 bg-slate-100 rounded ml-auto"></div>
                </div>
              </div>

              {/* Title */}
              <div className="mb-6 md:mb-8">
                <div className="h-5 md:h-6 w-3/4 bg-slate-900 rounded mb-3"></div>
                <div className="h-3 md:h-4 w-1/2 bg-slate-400 rounded"></div>
              </div>

              {/* Grid */}
              <div className="grid grid-cols-2 gap-3 md:gap-4 mb-6 md:mb-8">
                <div className="p-3 md:p-4 bg-green-50 rounded border border-green-100">
                  <div className="h-2 md:h-3 w-16 md:w-20 bg-green-200 rounded mb-2"></div>
                  <div className="h-5 md:h-6 w-12 md:w-16 bg-green-600 rounded"></div>
                </div>
                <div className="p-3 md:p-4 bg-slate-50 rounded border border-slate-100">
                  <div className="h-2 md:h-3 w-16 md:w-20 bg-slate-200 rounded mb-2"></div>
                  <div className="h-5 md:h-6 w-12 md:w-16 bg-slate-400 rounded"></div>
                </div>
              </div>

              {/* Chart Placeholder */}
              <div className="h-24 md:h-32 bg-slate-50 rounded border border-slate-100 mb-6 md:mb-8 flex items-end justify-between px-3 md:px-4 pb-4">
                 <div className="w-4 md:w-6 h-10 md:h-12 bg-slate-200 rounded-t"></div>
                 <div className="w-4 md:w-6 h-12 md:h-16 bg-slate-200 rounded-t"></div>
                 <div className="w-4 md:w-6 h-16 md:h-20 bg-slate-200 rounded-t"></div>
                 <div className="w-4 md:w-6 h-20 md:h-24 bg-brand-200 rounded-t"></div>
                 <div className="w-4 md:w-6 h-24 md:h-28 bg-brand-500 rounded-t shadow-lg shadow-brand-200"></div>
              </div>

              {/* Text Lines */}
              <div className="space-y-2 md:space-y-3">
                <div className="h-2 md:h-3 w-full bg-slate-100 rounded"></div>
                <div className="h-2 md:h-3 w-full bg-slate-100 rounded"></div>
                <div className="h-2 md:h-3 w-2/3 bg-slate-100 rounded"></div>
              </div>

              {/* Badge */}
              <div className="absolute top-6 right-[-10px] bg-brand-600 text-white text-[10px] md:text-xs font-bold px-3 py-1 rounded shadow-lg transform rotate-3">
                AI GENERATED
              </div>

            </div>
          </div>

        </div>
      </div>
    </section>
  );
};