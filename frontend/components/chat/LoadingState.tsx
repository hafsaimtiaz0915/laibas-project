"use client"

import { Bot } from "lucide-react"

interface LoadingStateProps {
  message?: string
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = "thinking",
}) => {
  return (
    <div className="flex gap-3 py-4">
      {/* Avatar - same as assistant */}
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full bg-slate-100 text-slate-600">
        <Bot className="h-4 w-4" />
      </div>

      {/* Thinking bubble */}
      <div className="flex flex-col gap-1 items-start">
        {/* Orb loader - smaller version */}
        <div className="rounded-2xl px-4 py-3 bg-slate-50 border border-slate-200">
          <div className="flex items-center gap-3">
            <div className="loader-small"></div>
            <span className="text-sm text-slate-600 animate-pulse">{message}</span>
          </div>
        </div>
      </div>

      <style jsx>{`
        .loader-small {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          animation: loader-rotate 2s ease-in-out infinite;
          box-shadow: 0 4px 8px 0 #fff inset,
            0 8px 12px 0 #ad5fff inset,
            0 24px 24px 0 #471eec inset;
        }

        @keyframes loader-rotate {
          0% {
            transform: rotate(90deg);
            box-shadow: 0 4px 8px 0 #fff inset,
              0 8px 12px 0 #ad5fff inset,
              0 24px 24px 0 #471eec inset;
          }
          50% {
            transform: rotate(270deg);
            box-shadow: 0 4px 8px 0 #fff inset,
              0 8px 4px 0 #d60a47 inset,
              0 16px 24px 0 #311e80 inset;
          }
          100% {
            transform: rotate(450deg);
            box-shadow: 0 4px 8px 0 #fff inset,
              0 8px 12px 0 #ad5fff inset,
              0 24px 24px 0 #471eec inset;
          }
        }
      `}</style>
    </div>
  )
}

export default LoadingState

