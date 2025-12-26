# Frontend Architecture: Agent Dashboard & PDF Report Generator

> **Document Version**: 1.0  
> **Last Updated**: 2025-12-11  
> **Purpose**: Define the frontend architecture for the agent-facing dashboard and client PDF report generation.

---

## 1. Overview

The frontend provides real estate agents with:
1. **Chat Interface** - Natural language queries about off-plan investments
2. **Dashboard** - Market overview and portfolio tracking
3. **PDF Report Generator** - White-labeled client reports with agent branding

---

## 2. System Architecture

Simple ChatGPT/Claude-style interface with chat history.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEXT.JS FRONTEND                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         PAGES                                        │    │
│  │  /login         → Agent authentication                               │    │
│  │  /chat          → Main interface (chat + sidebar)                    │    │
│  │  /chat/[id]     → Specific conversation                              │    │
│  │  /settings      → Agent branding (modal or page)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      MAIN INTERFACE                                  │    │
│  │                                                                      │    │
│  │  ┌────────────┐  ┌──────────────────────────────────────────────┐  │    │
│  │  │ SIDEBAR    │  │              CHAT AREA                        │  │    │
│  │  │            │  │                                               │  │    │
│  │  │ + New Chat │  │  [User Message]                               │  │    │
│  │  │            │  │  "Binghatti JVC 2BR at 2.2M?"                 │  │    │
│  │  │ Today      │  │                                               │  │    │
│  │  │ • Chat 1   │  │  [Assistant Response]                         │  │    │
│  │  │ • Chat 2   │  │  Based on current trends...                   │  │    │
│  │  │            │  │                                               │  │    │
│  │  │ Yesterday  │  │  ┌────────────────────────────────────────┐  │  │    │
│  │  │ • Chat 3   │  │  │ Analysis Card + [Download PDF]         │  │  │    │
│  │  │            │  │  └────────────────────────────────────────┘  │  │    │
│  │  │            │  │                                               │  │    │
│  │  │ ────────   │  ├──────────────────────────────────────────────┤  │    │
│  │  │ ⚙ Settings │  │  [Message Input]                     [Send]  │  │    │
│  │  └────────────┘  └──────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ API Calls
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND API (FastAPI)                              │
│                                                                              │
│  POST /api/chat         → Process query, return analysis                     │
│  POST /api/report/pdf   → Generate branded PDF report                        │
│  POST /api/agent/logo   → Upload agent logo                                  │
│  GET  /api/agent/settings → Get agent branding settings                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| **Framework** | Next.js 14+ (App Router) | Server components, API routes |
| **Styling** | Tailwind CSS | Rapid UI development |
| **UI Components** | 21st.dev + shadcn/ui | Modern AI chat components |
| **State** | Zustand | Lightweight global state |
| **PDF Generation** | @react-pdf/renderer | React-native PDF creation |
| **File Upload** | react-dropzone | Logo uploads |
| **Auth** | NextAuth.js | JWT authentication |
| **Hosting** | Vercel | Automatic deployments |

---

## 4. Project Structure

Simple ChatGPT/Claude-style interface with chat history sidebar.

```
frontend/
├── app/
│   ├── layout.tsx                 # Root layout
│   ├── page.tsx                   # Redirect to /chat or /login
│   ├── login/
│   │   └── page.tsx               # Agent login
│   └── chat/
│       ├── layout.tsx             # Chat layout with sidebar
│       ├── page.tsx               # New chat (empty state)
│       └── [chatId]/
│           └── page.tsx           # Specific chat conversation
│
├── components/
│   ├── chat/
│   │   ├── ChatSidebar.tsx        # Left sidebar with chat list
│   │   ├── ChatList.tsx           # Chat history items
│   │   ├── ChatMessages.tsx       # Message display area
│   │   ├── ChatInput.tsx          # Query input
│   │   └── AnalysisCard.tsx       # Inline analysis with PDF export
│   ├── settings/
│   │   ├── SettingsModal.tsx      # Settings dialog
│   │   └── LogoUploader.tsx       # Logo upload
│   └── ui/                        # 21st.dev components
│
├── lib/
│   ├── api.ts                     # API client
│   ├── pdf-templates/
│   │   └── PropertyReport.tsx     # PDF template
│   └── types.ts                   # TypeScript types
│
├── stores/
│   └── agent-store.ts             # Agent settings state
│
└── public/
    └── default-logo.png           # Fallback logo
```

---

## 5. Agent Branding System

### 5.1 Agent Settings Schema

```typescript
// lib/types.ts

interface AgentSettings {
  // Identity
  agentId: string;
  name: string;
  company: string;
  email: string;
  phone: string;
  
  // Branding
  logo: string | null;           // URL to uploaded logo
  primaryColor: string;          // Hex color for headers
  secondaryColor: string;        // Hex color for accents
  
  // Report Customization
  reportHeader: string;          // Custom header text
  reportFooter: string;          // Legal disclaimer
  showContactInfo: boolean;      // Include contact in report
  
  // Metadata
  createdAt: string;
  updatedAt: string;
}
```

### 5.2 Logo Upload Component

```tsx
// components/settings/LogoUploader.tsx
'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';
import { useAgentStore } from '@/stores/agent-store';

export function LogoUploader() {
  const { settings, updateLogo } = useAgentStore();
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    
    setUploading(true);
    const file = files[0];
    
    // Create form data
    const formData = new FormData();
    formData.append('logo', file);
    
    // Upload to backend
    const response = await fetch('/api/agent/logo', {
      method: 'POST',
      body: formData,
    });
    
    const { logoUrl } = await response.json();
    updateLogo(logoUrl);
    setUploading(false);
  }, [updateLogo]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.svg'] },
    maxSize: 2 * 1024 * 1024, // 2MB max
    multiple: false,
  });

  return (
    <div className="space-y-4">
      <label className="text-sm font-medium text-slate-300">
        Company Logo
      </label>
      
      {/* Current Logo Preview */}
      {settings.logo && (
        <div className="flex items-center gap-4">
          <Image
            src={settings.logo}
            alt="Company logo"
            width={120}
            height={60}
            className="object-contain bg-white rounded-lg p-2"
          />
          <button
            onClick={() => updateLogo(null)}
            className="text-red-400 text-sm hover:underline"
          >
            Remove
          </button>
        </div>
      )}
      
      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors
          ${isDragActive 
            ? 'border-emerald-500 bg-emerald-500/10' 
            : 'border-slate-700 hover:border-slate-600'
          }
        `}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <p className="text-slate-400">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-emerald-400">Drop logo here</p>
        ) : (
          <div className="space-y-2">
            <p className="text-slate-400">
              Drag & drop your logo, or click to select
            </p>
            <p className="text-slate-500 text-sm">
              PNG, JPG, or SVG (max 2MB)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## 6. PDF Report Generation

### 6.1 Report Schema

```typescript
// lib/types.ts

interface PropertyAnalysisReport {
  // Report Metadata
  reportId: string;
  generatedAt: string;
  agentSettings: AgentSettings;
  
  // Query Details
  query: {
    developer: string;
    area: string;
    bedroom: string;
    purchasePrice: number;
    propertyType: 'Unit' | 'Villa';
  };
  
  // Predictions
  predictions: {
    handoverValue: {
      low: number;       // P10 (pessimistic)
      median: number;    // P50 (most likely)
      high: number;      // P90 (optimistic)
    };
    appreciation: {
      percentLow: number;
      percentMedian: number;
      percentHigh: number;
    };
    rentalYield: {
      grossYield: number;
      netYield: number;
      estimatedAnnualRent: number;
    };
    timeHorizon: number;  // Months to handover
  };
  
  // Trend Data (Simple, factual)
  trends: {
    developer: {
      name: string;
      projectsCompleted: number;
      avgDelayMonths: number;
      totalUnitsDelivered: number;
    };
    area: {
      name: string;
      currentMedianPriceSqft: number;
      priceChange12Months: number;
      priceChange36Months: number;
      supplyPipeline: number;
      currentMedianRent: number;
    };
    market: {
      eibor: number;
      dubaiTransactions12m: number;
    };
  };
}
```

### 6.2 PDF Template (Simple Trend Report)

```tsx
// lib/pdf-templates/PropertyReport.tsx

import {
  Document,
  Page,
  Text,
  View,
  Image,
  StyleSheet,
} from '@react-pdf/renderer';
import { PropertyAnalysisReport } from '@/lib/types';

const styles = StyleSheet.create({
  page: {
    padding: 40,
    fontFamily: 'Helvetica',
    fontSize: 10,
    color: '#1e293b',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 30,
    paddingBottom: 20,
    borderBottom: '2px solid #0f766e',
  },
  logo: {
    width: 140,
    height: 60,
    objectFit: 'contain',
  },
  reportTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#0f172a',
    textAlign: 'right',
  },
  reportSubtitle: {
    fontSize: 11,
    color: '#64748b',
    textAlign: 'right',
    marginTop: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#0f172a',
    marginBottom: 12,
    paddingBottom: 6,
    borderBottom: '1px solid #e2e8f0',
  },
  row: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  label: {
    width: 180,
    fontSize: 10,
    color: '#64748b',
  },
  value: {
    flex: 1,
    fontSize: 10,
    color: '#0f172a',
    fontWeight: 'bold',
  },
  highlight: {
    backgroundColor: '#f0fdf4',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
  },
  highlightTitle: {
    fontSize: 11,
    color: '#166534',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  highlightRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  highlightLabel: {
    fontSize: 10,
    color: '#166534',
  },
  highlightValue: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#166534',
  },
  trendPositive: {
    color: '#16a34a',
  },
  trendNegative: {
    color: '#dc2626',
  },
  disclaimer: {
    marginTop: 20,
    padding: 12,
    backgroundColor: '#fef9c3',
    borderRadius: 6,
    borderLeft: '3px solid #ca8a04',
  },
  disclaimerText: {
    fontSize: 8,
    color: '#713f12',
    lineHeight: 1.5,
  },
  footer: {
    position: 'absolute',
    bottom: 30,
    left: 40,
    right: 40,
    borderTop: '1px solid #e2e8f0',
    paddingTop: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  footerText: {
    fontSize: 8,
    color: '#94a3b8',
  },
});

const formatAED = (value: number) => `AED ${value.toLocaleString('en-AE')}`;
const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;

export function PropertyReportPDF({ report }: { report: PropertyAnalysisReport }) {
  const { agentSettings, query, predictions, trends } = report;
  
  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {/* Header with Agent Logo */}
        <View style={styles.header}>
          <View>
            {agentSettings.logo ? (
              <Image src={agentSettings.logo} style={styles.logo} />
            ) : (
              <Text style={{ fontSize: 18, fontWeight: 'bold' }}>
                {agentSettings.company}
              </Text>
            )}
          </View>
          <View>
            <Text style={styles.reportTitle}>Property Analysis Report</Text>
            <Text style={styles.reportSubtitle}>
              {query.developer} | {query.area} | {query.bedroom}
            </Text>
          </View>
        </View>

        {/* Property Details */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Property Details</Text>
          <View style={styles.row}>
            <Text style={styles.label}>Developer</Text>
            <Text style={styles.value}>{query.developer}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Area</Text>
            <Text style={styles.value}>{query.area}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Configuration</Text>
            <Text style={styles.value}>{query.bedroom} {query.propertyType}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Purchase Price</Text>
            <Text style={styles.value}>{formatAED(query.purchasePrice)}</Text>
          </View>
        </View>

        {/* Predicted Outcomes - Highlighted */}
        <View style={styles.highlight}>
          <Text style={styles.highlightTitle}>Predicted Outcomes</Text>
          <View style={styles.highlightRow}>
            <Text style={styles.highlightLabel}>Estimated Handover Value</Text>
            <Text style={styles.highlightValue}>{formatAED(predictions.handoverValue.median)}</Text>
          </View>
          <View style={styles.highlightRow}>
            <Text style={styles.highlightLabel}>Value Range</Text>
            <Text style={styles.highlightValue}>
              {formatAED(predictions.handoverValue.low)} - {formatAED(predictions.handoverValue.high)}
            </Text>
          </View>
          <View style={styles.highlightRow}>
            <Text style={styles.highlightLabel}>Expected Appreciation</Text>
            <Text style={styles.highlightValue}>{formatPercent(predictions.appreciation.percentMedian)}</Text>
          </View>
          <View style={styles.highlightRow}>
            <Text style={styles.highlightLabel}>Estimated Rental Yield</Text>
            <Text style={styles.highlightValue}>{predictions.rentalYield.grossYield.toFixed(1)}% gross</Text>
          </View>
          <View style={styles.highlightRow}>
            <Text style={styles.highlightLabel}>Estimated Annual Rent</Text>
            <Text style={styles.highlightValue}>{formatAED(predictions.rentalYield.estimatedAnnualRent)}</Text>
          </View>
        </View>

        {/* Developer Trends */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Developer: {trends.developer.name}</Text>
          <View style={styles.row}>
            <Text style={styles.label}>Projects Completed</Text>
            <Text style={styles.value}>{trends.developer.projectsCompleted}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Total Units Delivered</Text>
            <Text style={styles.value}>{trends.developer.totalUnitsDelivered.toLocaleString()}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Average Delivery Delay</Text>
            <Text style={styles.value}>{trends.developer.avgDelayMonths} months</Text>
          </View>
        </View>

        {/* Area Trends */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Area: {trends.area.name}</Text>
          <View style={styles.row}>
            <Text style={styles.label}>Current Median Price</Text>
            <Text style={styles.value}>{formatAED(trends.area.currentMedianPriceSqft)}/sqft</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Price Change (12 months)</Text>
            <Text style={[styles.value, trends.area.priceChange12Months >= 0 ? styles.trendPositive : styles.trendNegative]}>
              {formatPercent(trends.area.priceChange12Months)}
            </Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Price Change (36 months)</Text>
            <Text style={[styles.value, trends.area.priceChange36Months >= 0 ? styles.trendPositive : styles.trendNegative]}>
              {formatPercent(trends.area.priceChange36Months)}
            </Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Current Median Rent ({query.bedroom})</Text>
            <Text style={styles.value}>{formatAED(trends.area.currentMedianRent)}/year</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Supply Pipeline (24 months)</Text>
            <Text style={styles.value}>{trends.area.supplyPipeline.toLocaleString()} units</Text>
          </View>
        </View>

        {/* Market Overview */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Market Overview</Text>
          <View style={styles.row}>
            <Text style={styles.label}>EIBOR (12 month)</Text>
            <Text style={styles.value}>{trends.market.eibor}%</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.label}>Dubai Transactions (12 months)</Text>
            <Text style={styles.value}>{trends.market.dubaiTransactions12m.toLocaleString()}</Text>
          </View>
        </View>

        {/* Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>
            This analysis is for informational purposes only and does not constitute investment advice. 
            All projections are estimates based on historical data and market trends. Property values 
            can fluctuate. Consult qualified professionals before making investment decisions.
            {agentSettings.reportFooter ? `\n\n${agentSettings.reportFooter}` : ''}
          </Text>
        </View>

        {/* Footer */}
        <View style={styles.footer} fixed>
          <View>
            <Text style={styles.footerText}>
              Generated: {new Date(report.generatedAt).toLocaleDateString('en-AE', {
                year: 'numeric', month: 'long', day: 'numeric'
              })}
            </Text>
          </View>
          {agentSettings.showContactInfo && (
            <View style={{ textAlign: 'right' }}>
              <Text style={styles.footerText}>{agentSettings.name} | {agentSettings.phone}</Text>
              <Text style={styles.footerText}>{agentSettings.email}</Text>
            </View>
          )}
        </View>
      </Page>
    </Document>
  );
}
```

### 6.3 PDF Generation API Route

```typescript
// app/api/report/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { renderToBuffer } from '@react-pdf/renderer';
import { PropertyReportPDF } from '@/lib/pdf-templates/PropertyReport';

export async function POST(request: NextRequest) {
  try {
    const { reportData, agentSettings } = await request.json();
    
    // Combine report data with agent branding
    const fullReport = {
      ...reportData,
      agentSettings,
      reportId: `RPT-${Date.now()}`,
      generatedAt: new Date().toISOString(),
    };
    
    // Generate PDF
    const pdfBuffer = await renderToBuffer(
      <PropertyReportPDF report={fullReport} />
    );
    
    // Return PDF
    return new NextResponse(pdfBuffer, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `attachment; filename="Property-Analysis-${fullReport.query.area}-${fullReport.query.bedroom}.pdf"`,
      },
    });
  } catch (error) {
    console.error('PDF generation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate PDF' },
      { status: 500 }
    );
  }
}
```

---

## 7. Chat Interface with Export

### 7.1 Chat Interface Component

```tsx
// components/chat/ChatInterface.tsx
'use client';

import { useState } from 'react';
import { MessageList } from './MessageList';
import { AnalysisCard } from './AnalysisCard';
import { useAgentStore } from '@/stores/agent-store';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  analysis?: PropertyAnalysisReport;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const { settings } = useAgentStore();

  const handleSubmit = async () => {
    if (!input.trim()) return;
    
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });
      
      const data = await response.json();
      
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.summary,
        analysis: data.report,
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExportPDF = async (analysis: PropertyAnalysisReport) => {
    const response = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reportData: analysis,
        agentSettings: settings,
      }),
    });
    
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    
    // Download PDF
    const a = document.createElement('a');
    a.href = url;
    a.download = `Property-Analysis-${analysis.query.area}-${analysis.query.bedroom}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map(message => (
          <div key={message.id}>
            {message.role === 'user' ? (
              <div className="flex justify-end">
                <div className="bg-emerald-600 text-white px-4 py-3 rounded-2xl max-w-2xl">
                  {message.content}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="bg-slate-800 text-slate-200 px-4 py-3 rounded-2xl max-w-4xl">
                  {message.content}
                </div>
                {message.analysis && (
                  <AnalysisCard 
                    analysis={message.analysis}
                    onExportPDF={() => handleExportPDF(message.analysis!)}
                  />
                )}
              </div>
            )}
          </div>
        ))}
        
        {loading && (
          <div className="flex items-center gap-2 text-slate-400">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-100" />
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-200" />
            <span className="ml-2">Analyzing...</span>
          </div>
        )}
      </div>
      
      {/* Input */}
      <div className="border-t border-slate-800 p-4">
        <div className="flex gap-3 max-w-4xl mx-auto">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSubmit()}
            placeholder="Binghatti JVC 2BR at 2.2M - what's the outlook?"
            className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 
                       text-white placeholder-slate-500 focus:ring-2 focus:ring-emerald-500
                       focus:border-transparent outline-none"
          />
          <button
            onClick={handleSubmit}
            disabled={loading || !input.trim()}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 
                       text-white px-6 py-3 rounded-xl font-medium transition-colors"
          >
            Analyze
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 7.2 Analysis Card with Export Button

```tsx
// components/chat/AnalysisCard.tsx
'use client';

import { PropertyAnalysisReport } from '@/lib/types';
import { FileDown, TrendingUp, Building2, MapPin } from 'lucide-react';

interface AnalysisCardProps {
  analysis: PropertyAnalysisReport;
  onExportPDF: () => void;
}

export function AnalysisCard({ analysis, onExportPDF }: AnalysisCardProps) {
  const { query, predictions, trends } = analysis;
  
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden max-w-4xl">
      {/* Header */}
      <div className="bg-slate-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">
            {query.developer} • {query.area} • {query.bedroom}
          </h3>
          <p className="text-slate-400 text-sm">
            Purchase: AED {query.purchasePrice.toLocaleString()}
          </p>
        </div>
        <button
          onClick={onExportPDF}
          className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 
                     text-white px-4 py-2 rounded-lg font-medium transition-colors"
        >
          <FileDown className="w-4 h-4" />
          Export PDF Report
        </button>
      </div>
      
      {/* Predictions Grid */}
      <div className="p-6 grid grid-cols-3 gap-4">
        <div className="bg-slate-800/50 rounded-lg p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <TrendingUp className="w-4 h-4" />
            Handover Value
          </div>
          <p className="text-2xl font-bold text-emerald-400">
            AED {predictions.handoverValue.median.toLocaleString()}
          </p>
          <p className="text-slate-500 text-sm">
            {predictions.handoverValue.low.toLocaleString()} - {predictions.handoverValue.high.toLocaleString()}
          </p>
        </div>
        
        <div className="bg-slate-800/50 rounded-lg p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <TrendingUp className="w-4 h-4" />
            Appreciation
          </div>
          <p className="text-2xl font-bold text-emerald-400">
            +{predictions.appreciation.percentMedian}%
          </p>
          <p className="text-slate-500 text-sm">
            Over {predictions.timeHorizon} months
          </p>
        </div>
        
        <div className="bg-slate-800/50 rounded-lg p-4">
          <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
            <Building2 className="w-4 h-4" />
            Rental Yield
          </div>
          <p className="text-2xl font-bold text-emerald-400">
            {predictions.rentalYield.grossYield.toFixed(1)}%
          </p>
          <p className="text-slate-500 text-sm">
            {predictions.rentalYield.estimatedAnnualRent.toLocaleString()}/yr
          </p>
        </div>
      </div>
      
      {/* Trend Summary */}
      <div className="px-6 pb-6 grid grid-cols-2 gap-6">
        {/* Developer Info */}
        <div>
          <h4 className="text-white font-medium mb-3">Developer</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Projects Completed</span>
              <span className="text-slate-200">{trends.developer.projectsCompleted}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Units Delivered</span>
              <span className="text-slate-200">{trends.developer.totalUnitsDelivered.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Avg Delay</span>
              <span className="text-slate-200">{trends.developer.avgDelayMonths} months</span>
            </div>
          </div>
        </div>
        
        {/* Area Info */}
        <div>
          <h4 className="text-white font-medium mb-3">Area Trends</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">12m Price Change</span>
              <span className={trends.area.priceChange12Months >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                {trends.area.priceChange12Months >= 0 ? '+' : ''}{trends.area.priceChange12Months}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">36m Price Change</span>
              <span className={trends.area.priceChange36Months >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                {trends.area.priceChange36Months >= 0 ? '+' : ''}{trends.area.priceChange36Months}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Supply Pipeline</span>
              <span className="text-slate-200">{trends.area.supplyPipeline.toLocaleString()} units</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## 8. Agent Settings Page

```tsx
// app/(dashboard)/settings/page.tsx
'use client';

import { useState } from 'react';
import { LogoUploader } from '@/components/settings/LogoUploader';
import { useAgentStore } from '@/stores/agent-store';

export default function SettingsPage() {
  const { settings, updateSettings } = useAgentStore();
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    await fetch('/api/agent/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    setSaving(false);
  };

  return (
    <div className="p-8 max-w-3xl">
      <h1 className="text-2xl font-bold text-white mb-8">Agent Settings</h1>
      
      <div className="space-y-8">
        {/* Logo Upload */}
        <section className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <h2 className="text-lg font-semibold text-white mb-4">
            Company Branding
          </h2>
          <LogoUploader />
        </section>
        
        {/* Contact Information */}
        <section className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <h2 className="text-lg font-semibold text-white mb-4">
            Contact Information
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-slate-400 block mb-2">Name</label>
              <input
                value={settings.name}
                onChange={(e) => updateSettings({ name: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 
                           text-white focus:ring-2 focus:ring-emerald-500 outline-none"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 block mb-2">Company</label>
              <input
                value={settings.company}
                onChange={(e) => updateSettings({ company: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 
                           text-white focus:ring-2 focus:ring-emerald-500 outline-none"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 block mb-2">Email</label>
              <input
                value={settings.email}
                onChange={(e) => updateSettings({ email: e.target.value })}
                type="email"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 
                           text-white focus:ring-2 focus:ring-emerald-500 outline-none"
              />
            </div>
            <div>
              <label className="text-sm text-slate-400 block mb-2">Phone</label>
              <input
                value={settings.phone}
                onChange={(e) => updateSettings({ phone: e.target.value })}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 
                           text-white focus:ring-2 focus:ring-emerald-500 outline-none"
              />
            </div>
          </div>
        </section>
        
        {/* Report Customization */}
        <section className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <h2 className="text-lg font-semibold text-white mb-4">
            PDF Report Settings
          </h2>
          <div className="space-y-4">
            <div>
              <label className="text-sm text-slate-400 block mb-2">
                Custom Footer Text (Legal Disclaimer)
              </label>
              <textarea
                value={settings.reportFooter}
                onChange={(e) => updateSettings({ reportFooter: e.target.value })}
                rows={3}
                placeholder="Add your company's legal disclaimer or additional notes..."
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 
                           text-white focus:ring-2 focus:ring-emerald-500 outline-none resize-none"
              />
            </div>
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="showContact"
                checked={settings.showContactInfo}
                onChange={(e) => updateSettings({ showContactInfo: e.target.checked })}
                className="w-4 h-4 rounded border-slate-700 bg-slate-800 text-emerald-600 
                           focus:ring-emerald-500"
              />
              <label htmlFor="showContact" className="text-slate-300">
                Include contact information in PDF reports
              </label>
            </div>
          </div>
        </section>
        
        {/* Save Button */}
        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700
                     text-white py-3 rounded-xl font-medium transition-colors"
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
```

---

## 9. Implementation Roadmap

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Phase 3.1** | Next.js setup, shadcn/ui, auth | 3 days |
| **Phase 3.2** | Chat interface, API integration | 5 days |
| **Phase 3.3** | Agent settings, logo upload | 3 days |
| **Phase 3.4** | PDF report template | 4 days |
| **Phase 3.5** | Dashboard, market overview | 5 days |
| **Phase 3.6** | Testing, polish | 3 days |
| **Total** | | **~3 weeks** |

---

## 10. Dependencies

```json
{
  "dependencies": {
    "next": "^14.0.0",
    "@react-pdf/renderer": "^3.3.0",
    "react-dropzone": "^14.2.0",
    "zustand": "^4.4.0",
    "@tanstack/react-query": "^5.0.0",
    "tailwindcss": "^3.4.0",
    "lucide-react": "^0.300.0",
    "next-auth": "^4.24.0",
    "recharts": "^2.10.0"
  }
}
```

---

## 11. Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_SECRET=your-secret-key
NEXTAUTH_URL=http://localhost:3000

# Backend connection
BACKEND_API_KEY=your-api-key

# Storage (for logo uploads)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

