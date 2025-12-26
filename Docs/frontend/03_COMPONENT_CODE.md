# Frontend Component Code

> **Document Version**: 1.0  
> **Last Updated**: 2025-12-11  
> **Source**: 21st.dev community components

---

## Overview

Specific components to use from 21st.dev, customized for our property analysis chat.

---

## 1. Sidebar Component

Collapsible sidebar with chat history. Expands on hover (desktop) or hamburger menu (mobile).

**File**: `components/ui/sidebar.tsx`

```tsx
"use client";

import { cn } from "@/lib/utils";
import Link, { LinkProps } from "next/link";
import React, { useState, createContext, useContext } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Menu, X } from "lucide-react";

interface Links {
  label: string;
  href: string;
  icon: React.JSX.Element | React.ReactNode;
}

interface SidebarContextProps {
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  animate: boolean;
}

const SidebarContext = createContext<SidebarContextProps | undefined>(
  undefined
);

export const useSidebar = () => {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider");
  }
  return context;
};

export const SidebarProvider = ({
  children,
  open: openProp,
  setOpen: setOpenProp,
  animate = true,
}: {
  children: React.ReactNode;
  open?: boolean;
  setOpen?: React.Dispatch<React.SetStateAction<boolean>>;
  animate?: boolean;
}) => {
  const [openState, setOpenState] = useState(false);

  const open = openProp !== undefined ? openProp : openState;
  const setOpen = setOpenProp !== undefined ? setOpenProp : setOpenState;

  return (
    <SidebarContext.Provider value={{ open, setOpen, animate }}>
      {children}
    </SidebarContext.Provider>
  );
};

export const Sidebar = ({
  children,
  open,
  setOpen,
  animate,
}: {
  children: React.ReactNode;
  open?: boolean;
  setOpen?: React.Dispatch<React.SetStateAction<boolean>>;
  animate?: boolean;
}) => {
  return (
    <SidebarProvider open={open} setOpen={setOpen} animate={animate}>
      {children}
    </SidebarProvider>
  );
};

export const SidebarBody = (props: React.ComponentProps<typeof motion.div>) => {
  return (
    <>
      <DesktopSidebar {...props} />
      <MobileSidebar {...(props as React.ComponentProps<"div">)} />
    </>
  );
};

export const DesktopSidebar = ({
  className,
  children,
  ...props
}: React.ComponentProps<typeof motion.div>) => {
  const { open, setOpen, animate } = useSidebar();
  return (
    <motion.div
      className={cn(
        "h-full px-4 py-4 hidden md:flex md:flex-col bg-neutral-100 dark:bg-neutral-800 w-[300px] flex-shrink-0",
        className
      )}
      animate={{
        width: animate ? (open ? "300px" : "60px") : "300px",
      }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export const MobileSidebar = ({
  className,
  children,
  ...props
}: React.ComponentProps<"div">) => {
  const { open, setOpen } = useSidebar();
  return (
    <>
      <div
        className={cn(
          "h-10 px-4 py-4 flex flex-row md:hidden items-center justify-between bg-neutral-100 dark:bg-neutral-800 w-full"
        )}
        {...props}
      >
        <div className="flex justify-end z-20 w-full">
          <Menu
            className="text-neutral-800 dark:text-neutral-200 cursor-pointer"
            onClick={() => setOpen(!open)}
          />
        </div>
        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ x: "-100%", opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: "-100%", opacity: 0 }}
              transition={{
                duration: 0.3,
                ease: "easeInOut",
              }}
              className={cn(
                "fixed h-full w-full inset-0 bg-white dark:bg-neutral-900 p-10 z-[100] flex flex-col justify-between",
                className
              )}
            >
              <div
                className="absolute right-10 top-10 z-50 text-neutral-800 dark:text-neutral-200 cursor-pointer"
                onClick={() => setOpen(!open)}
              >
                <X />
              </div>
              {children}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
};

export const SidebarLink = ({
  link,
  className,
  ...props
}: {
  link: Links;
  className?: string;
  props?: LinkProps;
}) => {
  const { open, animate } = useSidebar();
  return (
    <Link
      href={link.href}
      className={cn(
        "flex items-center justify-start gap-2 group/sidebar py-2",
        className
      )}
      {...props}
    >
      {link.icon}
      <motion.span
        animate={{
          display: animate ? (open ? "inline-block" : "none") : "inline-block",
          opacity: animate ? (open ? 1 : 0) : 1,
        }}
        className="text-neutral-700 dark:text-neutral-200 text-sm group-hover/sidebar:translate-x-1 transition duration-150 whitespace-pre inline-block !p-0 !m-0"
      >
        {link.label}
      </motion.span>
    </Link>
  );
};
```

---

## 2. Chat Input (PromptBox)

Rich chat input with send button. Simplified for property analysis (removed tools, image upload).

**File**: `components/chat/ChatInput.tsx`

```tsx
"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

// Send Icon
const SendIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <path d="M12 5.25L12 18.75" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M18.75 12L12 5.25L5.25 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

interface ChatInputProps {
  onSubmit: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

export const ChatInput = ({ onSubmit, isLoading, placeholder = "Ask about any property..." }: ChatInputProps) => {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const [value, setValue] = React.useState("");

  React.useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const newHeight = Math.min(textarea.scrollHeight, 200);
      textarea.style.height = `${newHeight}px`;
    }
  }, [value]);

  const handleSubmit = () => {
    if (value.trim() && !isLoading) {
      onSubmit(value.trim());
      setValue("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const hasValue = value.trim().length > 0;

  return (
    <div className={cn(
      "flex flex-col rounded-[28px] p-2 shadow-sm transition-colors",
      "bg-white border dark:bg-[#303030] dark:border-transparent"
    )}>
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isLoading}
        className={cn(
          "w-full resize-none border-0 bg-transparent p-3",
          "text-foreground dark:text-white",
          "placeholder:text-muted-foreground dark:placeholder:text-gray-400",
          "focus:ring-0 focus-visible:outline-none min-h-12"
        )}
      />

      <div className="mt-0.5 p-1 pt-0 flex justify-end">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!hasValue || isLoading}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-full",
            "text-sm font-medium transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            "disabled:pointer-events-none",
            "bg-black text-white hover:bg-black/80",
            "dark:bg-white dark:text-black dark:hover:bg-white/80",
            "disabled:bg-black/40 dark:disabled:bg-[#515151]"
          )}
        >
          <SendIcon className="h-5 w-5" />
          <span className="sr-only">Send message</span>
        </button>
      </div>
    </div>
  );
};
```

---

## 3. Loading Indicator (SiriOrb)

Animated orb shown when AI is processing.

**File**: `components/ui/siri-orb.tsx`

```tsx
"use client";

import { cn } from "@/lib/utils";
import React from "react";

interface SiriOrbProps {
  size?: string;
  className?: string;
  animationDuration?: number;
}

export const SiriOrb: React.FC<SiriOrbProps> = ({
  size = "48px",
  className,
  animationDuration = 20,
}) => {
  const colors = {
    bg: "transparent",
    c1: "oklch(75% 0.15 350)",
    c2: "oklch(80% 0.12 200)",
    c3: "oklch(78% 0.14 280)",
  };

  const sizeValue = parseInt(size.replace("px", ""), 10);
  const blurAmount = Math.max(sizeValue * 0.08, 8);
  const contrastAmount = Math.max(sizeValue * 0.003, 1.8);

  return (
    <div
      className={cn("siri-orb", className)}
      style={{
        width: size,
        height: size,
        "--bg": colors.bg,
        "--c1": colors.c1,
        "--c2": colors.c2,
        "--c3": colors.c3,
        "--animation-duration": `${animationDuration}s`,
        "--blur-amount": `${blurAmount}px`,
        "--contrast-amount": contrastAmount,
      } as React.CSSProperties}
    >
      <style jsx>{`
        @property --angle {
          syntax: "<angle>";
          inherits: false;
          initial-value: 0deg;
        }

        .siri-orb {
          display: grid;
          grid-template-areas: "stack";
          overflow: hidden;
          border-radius: 50%;
          position: relative;
          background: radial-gradient(
            circle,
            rgba(0, 0, 0, 0.08) 0%,
            rgba(0, 0, 0, 0.03) 30%,
            transparent 70%
          );
        }

        .dark .siri-orb {
          background: radial-gradient(
            circle,
            rgba(255, 255, 255, 0.08) 0%,
            rgba(255, 255, 255, 0.02) 30%,
            transparent 70%
          );
        }

        .siri-orb::before {
          content: "";
          display: block;
          grid-area: stack;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          background:
            conic-gradient(from calc(var(--angle) * 1.2) at 30% 65%, var(--c3) 0deg, transparent 45deg 315deg, var(--c3) 360deg),
            conic-gradient(from calc(var(--angle) * 0.8) at 70% 35%, var(--c2) 0deg, transparent 60deg 300deg, var(--c2) 360deg),
            conic-gradient(from calc(var(--angle) * -1.5) at 65% 75%, var(--c1) 0deg, transparent 90deg 270deg, var(--c1) 360deg),
            conic-gradient(from calc(var(--angle) * 2.1) at 25% 25%, var(--c2) 0deg, transparent 30deg 330deg, var(--c2) 360deg),
            conic-gradient(from calc(var(--angle) * -0.7) at 80% 80%, var(--c1) 0deg, transparent 45deg 315deg, var(--c1) 360deg),
            radial-gradient(ellipse 120% 80% at 40% 60%, var(--c3) 0%, transparent 50%);
          filter: blur(var(--blur-amount)) contrast(var(--contrast-amount)) saturate(1.2);
          animation: rotate var(--animation-duration) linear infinite;
          transform: translateZ(0);
          will-change: transform;
        }

        @keyframes rotate {
          from { --angle: 0deg; }
          to { --angle: 360deg; }
        }

        @media (prefers-reduced-motion: reduce) {
          .siri-orb::before { animation: none; }
        }
      `}</style>
    </div>
  );
};
```

---

## 4. Shining Text (Loading Text)

Animated gradient text shown during loading.

**File**: `components/ui/shining-text.tsx`

```tsx
"use client";

import * as React from "react";
import { motion } from "framer-motion";

interface ShiningTextProps {
  text: string;
  className?: string;
}

export const ShiningText: React.FC<ShiningTextProps> = ({ text, className }) => {
  return (
    <motion.span
      className={`bg-[linear-gradient(110deg,#404040,35%,#fff,50%,#404040,75%,#404040)] bg-[length:200%_100%] bg-clip-text text-base font-regular text-transparent ${className}`}
      initial={{ backgroundPosition: "200% 0" }}
      animate={{ backgroundPosition: "-200% 0" }}
      transition={{
        repeat: Infinity,
        duration: 2,
        ease: "linear",
      }}
    >
      {text}
    </motion.span>
  );
};
```

---

## 5. Loading State Component

Combines SiriOrb and ShiningText for loading state.

**File**: `components/chat/LoadingState.tsx`

```tsx
"use client";

import { SiriOrb } from "@/components/ui/siri-orb";
import { ShiningText } from "@/components/ui/shining-text";

export const LoadingState = () => {
  return (
    <div className="flex items-center gap-3 p-4">
      <SiriOrb size="32px" animationDuration={15} />
      <ShiningText text="Analyzing property data..." />
    </div>
  );
};
```

---

## 6. Chat Message Component

Individual message bubble (user or assistant).

**File**: `components/chat/ChatMessage.tsx`

```tsx
"use client";

import { cn } from "@/lib/utils";
import { AnalysisCard } from "./AnalysisCard";
import { PropertyAnalysis } from "@/lib/types";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  analysis?: PropertyAnalysis;
  onExportPDF?: () => void;
}

export const ChatMessage = ({ role, content, analysis, onExportPDF }: ChatMessageProps) => {
  const isUser = role === "user";

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div className={cn(
        "max-w-[80%] rounded-2xl px-4 py-3",
        isUser 
          ? "bg-black text-white dark:bg-white dark:text-black" 
          : "bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
      )}>
        <p className="whitespace-pre-wrap">{content}</p>
        
        {/* Analysis card for assistant messages */}
        {!isUser && analysis && (
          <div className="mt-4">
            <AnalysisCard analysis={analysis} onExportPDF={onExportPDF} />
          </div>
        )}
      </div>
    </div>
  );
};
```

---

## 7. Analysis Card Component

Inline property analysis with PDF export button.

**File**: `components/chat/AnalysisCard.tsx`

```tsx
"use client";

import { PropertyAnalysis } from "@/lib/types";
import { FileDown, TrendingUp, Building2 } from "lucide-react";

interface AnalysisCardProps {
  analysis: PropertyAnalysis;
  onExportPDF?: () => void;
}

export const AnalysisCard = ({ analysis, onExportPDF }: AnalysisCardProps) => {
  const { query, predictions, trends } = analysis;

  const formatAED = (value: number) => `AED ${value.toLocaleString()}`;
  const formatPercent = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;

  return (
    <div className="bg-white dark:bg-neutral-900 rounded-xl border border-neutral-200 dark:border-neutral-700 overflow-hidden">
      {/* Header */}
      <div className="bg-neutral-50 dark:bg-neutral-800 px-4 py-3 flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-neutral-900 dark:text-white">
            {query.developer} • {query.area} • {query.bedroom}
          </h3>
          <p className="text-neutral-500 dark:text-neutral-400 text-sm">
            Purchase: {formatAED(query.purchasePrice)}
          </p>
        </div>
        {onExportPDF && (
          <button
            onClick={onExportPDF}
            className="flex items-center gap-2 bg-black dark:bg-white text-white dark:text-black px-3 py-1.5 rounded-lg text-sm font-medium hover:opacity-80 transition-opacity"
          >
            <FileDown className="w-4 h-4" />
            PDF Report
          </button>
        )}
      </div>

      {/* Predictions */}
      <div className="p-4 grid grid-cols-3 gap-3">
        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-neutral-500 dark:text-neutral-400 text-xs mb-1">
            <TrendingUp className="w-3 h-3" />
            Handover Value
          </div>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
            {formatAED(predictions.handoverValue.median)}
          </p>
          <p className="text-xs text-neutral-400">
            {formatAED(predictions.handoverValue.low)} - {formatAED(predictions.handoverValue.high)}
          </p>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-neutral-500 dark:text-neutral-400 text-xs mb-1">
            <TrendingUp className="w-3 h-3" />
            Appreciation
          </div>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
            {formatPercent(predictions.appreciation.percentMedian)}
          </p>
          <p className="text-xs text-neutral-400">
            {predictions.timeHorizon} months
          </p>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center gap-1 text-neutral-500 dark:text-neutral-400 text-xs mb-1">
            <Building2 className="w-3 h-3" />
            Rental Yield
          </div>
          <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
            {predictions.rentalYield.grossYield.toFixed(1)}%
          </p>
          <p className="text-xs text-neutral-400">
            {formatAED(predictions.rentalYield.estimatedAnnualRent)}/yr
          </p>
        </div>
      </div>

      {/* Trend Summary */}
      <div className="border-t border-neutral-200 dark:border-neutral-700 px-4 py-3 grid grid-cols-2 gap-4 text-sm">
        <div>
          <h4 className="font-medium text-neutral-900 dark:text-white mb-2">Developer</h4>
          <div className="space-y-1 text-neutral-600 dark:text-neutral-300">
            <div className="flex justify-between">
              <span>Projects Completed</span>
              <span className="font-medium">{trends.developer.projectsCompleted}</span>
            </div>
            <div className="flex justify-between">
              <span>Avg Delay</span>
              <span className="font-medium">{trends.developer.avgDelayMonths}mo</span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="font-medium text-neutral-900 dark:text-white mb-2">Area Trends</h4>
          <div className="space-y-1 text-neutral-600 dark:text-neutral-300">
            <div className="flex justify-between">
              <span>12m Change</span>
              <span className={`font-medium ${trends.area.priceChange12Months >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                {formatPercent(trends.area.priceChange12Months)}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Supply Pipeline</span>
              <span className="font-medium">{trends.area.supplyPipeline.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
```

---

## 8. Full Chat Layout

Complete layout combining sidebar and chat area (ChatGPT style).

**File**: `app/chat/layout.tsx`

```tsx
"use client";

import { useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { MessageSquare, Plus, Settings } from "lucide-react";
import Link from "next/link";
import Image from "next/image";

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  // Example chat history - would come from database
  const chatHistory = [
    { id: "1", title: "Binghatti JVC 2BR", date: "Today" },
    { id: "2", title: "Emaar Creek Harbour", date: "Today" },
    { id: "3", title: "Sobha Hartland Villa", date: "Yesterday" },
  ];

  return (
    <div className="flex h-screen bg-white dark:bg-neutral-900">
      <Sidebar open={open} setOpen={setOpen}>
        <SidebarBody className="justify-between gap-10">
          <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
            {/* Logo */}
            <div className="flex items-center gap-2 py-2">
              <div className="h-8 w-8 rounded-full bg-black dark:bg-white flex items-center justify-center">
                <span className="text-white dark:text-black font-bold text-sm">P</span>
              </div>
              {open && (
                <span className="font-semibold text-neutral-900 dark:text-white">
                  Property AI
                </span>
              )}
            </div>

            {/* New Chat Button */}
            <Link
              href="/chat"
              className="flex items-center gap-2 mt-4 p-2 rounded-lg border border-neutral-200 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            >
              <Plus className="h-4 w-4 text-neutral-600 dark:text-neutral-300" />
              {open && (
                <span className="text-sm text-neutral-600 dark:text-neutral-300">
                  New Chat
                </span>
              )}
            </Link>

            {/* Chat History */}
            <div className="mt-6 flex-1">
              {open && (
                <>
                  <p className="text-xs text-neutral-500 px-2 mb-2">Today</p>
                  {chatHistory.filter(c => c.date === "Today").map(chat => (
                    <Link
                      key={chat.id}
                      href={`/chat/${chat.id}`}
                      className="flex items-center gap-2 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                    >
                      <MessageSquare className="h-4 w-4 text-neutral-400" />
                      <span className="text-sm text-neutral-700 dark:text-neutral-200 truncate">
                        {chat.title}
                      </span>
                    </Link>
                  ))}

                  <p className="text-xs text-neutral-500 px-2 mb-2 mt-4">Yesterday</p>
                  {chatHistory.filter(c => c.date === "Yesterday").map(chat => (
                    <Link
                      key={chat.id}
                      href={`/chat/${chat.id}`}
                      className="flex items-center gap-2 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                    >
                      <MessageSquare className="h-4 w-4 text-neutral-400" />
                      <span className="text-sm text-neutral-700 dark:text-neutral-200 truncate">
                        {chat.title}
                      </span>
                    </Link>
                  ))}
                </>
              )}
              {!open && (
                <div className="flex flex-col items-center gap-2">
                  {chatHistory.slice(0, 3).map(chat => (
                    <Link
                      key={chat.id}
                      href={`/chat/${chat.id}`}
                      className="p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
                    >
                      <MessageSquare className="h-4 w-4 text-neutral-400" />
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Settings at bottom */}
          <div className="border-t border-neutral-200 dark:border-neutral-700 pt-4">
            <Link
              href="/settings"
              className="flex items-center gap-2 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            >
              <Settings className="h-4 w-4 text-neutral-600 dark:text-neutral-300" />
              {open && (
                <span className="text-sm text-neutral-600 dark:text-neutral-300">
                  Settings
                </span>
              )}
            </Link>
          </div>
        </SidebarBody>
      </Sidebar>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {children}
      </main>
    </div>
  );
}
```

---

## 9. Chat Page (New Chat)

Empty state for new conversation.

**File**: `app/chat/page.tsx`

```tsx
"use client";

import { useState } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { LoadingState } from "@/components/chat/LoadingState";
import { useRouter } from "next/navigation";

export default function NewChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (message: string) => {
    // Add user message
    setMessages(prev => [...prev, { role: "user", content: message }]);
    setIsLoading(true);

    try {
      // Call API
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message }),
      });

      const data = await response.json();

      // Add assistant message
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          analysis: data.report,
        },
      ]);

      // Redirect to chat ID
      // router.push(`/chat/${data.chatId}`);
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6">
        {messages.length === 0 ? (
          // Empty State
          <div className="h-full flex flex-col items-center justify-center">
            <div className="h-16 w-16 rounded-full bg-black dark:bg-white flex items-center justify-center mb-4">
              <span className="text-white dark:text-black font-bold text-2xl">P</span>
            </div>
            <h1 className="text-2xl font-semibold text-neutral-900 dark:text-white mb-2">
              Property Analysis
            </h1>
            <p className="text-neutral-500 dark:text-neutral-400 text-center max-w-md">
              Ask about any off-plan property in Dubai. Get predictions, developer track records, and area trends.
            </p>
          </div>
        ) : (
          // Messages
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                role={msg.role}
                content={msg.content}
                analysis={msg.analysis}
                onExportPDF={msg.analysis ? () => handleExportPDF(msg.analysis) : undefined}
              />
            ))}
            {isLoading && <LoadingState />}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-neutral-200 dark:border-neutral-800">
        <div className="max-w-3xl mx-auto">
          <ChatInput
            onSubmit={handleSubmit}
            isLoading={isLoading}
            placeholder="Binghatti JVC 2BR at 2.2M - what's the outlook?"
          />
        </div>
      </div>
    </div>
  );
}

function handleExportPDF(analysis: any) {
  // Call PDF generation API
  console.log("Export PDF", analysis);
}
```

---

## File Structure Summary

```
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                    # Redirect to /chat
│   ├── login/
│   │   └── page.tsx
│   ├── chat/
│   │   ├── layout.tsx              # Sidebar + main area
│   │   ├── page.tsx                # New chat (empty state)
│   │   └── [chatId]/
│   │       └── page.tsx            # Existing chat
│   └── settings/
│       └── page.tsx
│
├── components/
│   ├── chat/
│   │   ├── ChatInput.tsx           # Message input
│   │   ├── ChatMessage.tsx         # Message bubble
│   │   ├── AnalysisCard.tsx        # Property analysis
│   │   └── LoadingState.tsx        # SiriOrb + ShiningText
│   ├── settings/
│   │   └── SettingsModal.tsx
│   └── ui/
│       ├── sidebar.tsx             # 21st.dev sidebar
│       ├── siri-orb.tsx            # Loading animation
│       └── shining-text.tsx        # Loading text
│
└── lib/
    ├── utils.ts                    # cn() helper
    └── types.ts                    # TypeScript types
```

---

## Dependencies

```json
{
  "dependencies": {
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.300.0",
    "@radix-ui/react-tooltip": "^1.0.0",
    "@radix-ui/react-popover": "^1.0.0",
    "@radix-ui/react-dialog": "^1.0.0"
  }
}
```

