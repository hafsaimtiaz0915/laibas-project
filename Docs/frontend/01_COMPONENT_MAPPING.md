# Frontend Component Mapping (21st.dev)

> **Document Version**: 1.1  
> **Last Updated**: 2025-12-11  
> **Component Library**: [21st.dev/community/components](https://21st.dev/community/components)

---

## Overview

Simple AI chat interface design - similar to ChatGPT/Claude.

---

## App Structure

```
/login     → Agent login
/chat      → Main interface (chat + sidebar)
/settings  → Agent branding (accessed from chat)
```

**No separate dashboard or reports page.** Reports are generated inline within chat conversations.

---

## Main Interface Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌────────────────────┐  ┌──────────────────────────────────────────┐  │
│  │                    │  │                                          │  │
│  │  [Logo]            │  │                                          │  │
│  │                    │  │                                          │  │
│  │  ┌──────────────┐  │  │  [User Message]                         │  │
│  │  │ + New Chat   │  │  │  "Binghatti JVC 2BR at 2.2M outlook?"   │  │
│  │  └──────────────┘  │  │                                          │  │
│  │                    │  │                                          │  │
│  │  Today             │  │  [Assistant Response]                    │  │
│  │  ├─ Binghatti JVC  │  │  Based on current trends...              │  │
│  │  └─ Emaar Creek    │  │                                          │  │
│  │                    │  │  ┌────────────────────────────────────┐  │  │
│  │  Yesterday         │  │  │  Property Analysis                 │  │  │
│  │  ├─ Sobha Hartland │  │  │                                    │  │  │
│  │  └─ Damac Hills    │  │  │  Handover Value: AED 2.78M         │  │  │
│  │                    │  │  │  Appreciation: +26%                │  │  │
│  │  Last 7 Days       │  │  │  Rental Yield: 6.1%                │  │  │
│  │  └─ ...            │  │  │                                    │  │  │
│  │                    │  │  │  Developer Trends                  │  │  │
│  │                    │  │  │  • 12 projects completed           │  │  │
│  │                    │  │  │  • 4 months avg delay              │  │  │
│  │                    │  │  │                                    │  │  │
│  │                    │  │  │  Area Trends                       │  │  │
│  │                    │  │  │  • +18% (12m) | +42% (36m)         │  │  │
│  │                    │  │  │  • 12,000 units pipeline           │  │  │
│  │                    │  │  │                                    │  │  │
│  │                    │  │  │  [Download PDF Report]             │  │  │
│  │                    │  │  └────────────────────────────────────┘  │  │
│  │                    │  │                                          │  │
│  │                    │  │                                          │  │
│  │  ──────────────    │  │                                          │  │
│  │  [⚙️ Settings]     │  └──────────────────────────────────────────┘  │
│  │                    │                                               │
│  │                    │  ┌──────────────────────────────────────────┐  │
│  │                    │  │ Ask about any property...        [Send] │  │
│  │                    │  └──────────────────────────────────────────┘  │
│  └────────────────────┘                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pages

### 1. Login Page (`/login`)

Simple centered login form.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                                                             │
│                    ┌─────────────────┐                     │
│                    │                 │                     │
│                    │   [App Logo]    │                     │
│                    │                 │                     │
│                    │  Property AI    │                     │
│                    │                 │                     │
│                    │  ┌───────────┐  │                     │
│                    │  │ Email     │  │                     │
│                    │  └───────────┘  │                     │
│                    │                 │                     │
│                    │  ┌───────────┐  │                     │
│                    │  │ Password  │  │                     │
│                    │  └───────────┘  │                     │
│                    │                 │                     │
│                    │  [  Sign In  ]  │                     │
│                    │                 │                     │
│                    └─────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Component | 21st.dev Category | Notes |
|-----------|-------------------|-------|
| Form Container | **Sign Ins** (4) | Complete auth form |
| Email Input | **Inputs** (102) | Email field |
| Password Input | **Inputs** (102) | Password field |
| Submit Button | **Buttons** (130) | Sign in button |

---

### 2. Chat Page (`/chat`) - Main Interface

| Section | 21st.dev Category | Notes |
|---------|-------------------|-------|
| Left Sidebar | **Sidebars** (10) | Chat history list |
| Chat List Items | **Cards** (79) or custom | Chat titles |
| Chat Messages | **AI Chats** (30) ⭐ | Message bubbles |
| Analysis Card | **Cards** (79) | Inline report card |
| Message Input | **Text Areas** (22) | Query input |
| Send Button | **Buttons** (130) | Submit |
| New Chat Button | **Buttons** (130) | Start new conversation |
| Settings Link | **Links** (13) | Bottom of sidebar |

---

### 3. Settings Modal/Page (`/settings`)

Accessed from sidebar. Can be a modal or separate page.

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Settings                                    [X]      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Company Branding                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  [Logo Preview]     [Upload Zone]                   │   │
│  │                     Drop your logo here             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Contact Information                                        │
│  ┌────────────────────┐  ┌────────────────────┐           │
│  │ Name               │  │ Company            │           │
│  └────────────────────┘  └────────────────────┘           │
│  ┌────────────────────┐  ┌────────────────────┐           │
│  │ Email              │  │ Phone              │           │
│  └────────────────────┘  └────────────────────┘           │
│                                                             │
│  PDF Report Settings                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Custom disclaimer text...                           │   │
│  └─────────────────────────────────────────────────────┘   │
│  [✓] Include my contact info in PDF reports                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  [Save Settings]                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| Component | 21st.dev Category | Notes |
|-----------|-------------------|-------|
| Modal/Dialog | **Dialogs / Modals** (37) | Settings container |
| Logo Upload | **File Uploads** (7) | Drag & drop zone |
| Text Inputs | **Inputs** (102) | Name, email, phone |
| Text Area | **Text Areas** (22) | Custom disclaimer |
| Toggle | **Toggles** (12) | Show contact checkbox |
| Save Button | **Buttons** (130) | Save settings |

---

## 21st.dev Components Needed

### Priority 1 (Core)

| Component | Category | Purpose |
|-----------|----------|---------|
| AI Chat Interface | **AI Chats** (30) | Main chat UI |
| Sidebar | **Sidebars** (10) | Chat history |
| Cards | **Cards** (79) | Analysis results |
| Buttons | **Buttons** (130) | Actions |
| Inputs | **Inputs** (102) | Form fields |

### Priority 2 (Forms)

| Component | Category | Purpose |
|-----------|----------|---------|
| Sign In Form | **Sign Ins** (4) | Login |
| File Upload | **File Uploads** (7) | Logo upload |
| Text Area | **Text Areas** (22) | Query input, disclaimer |
| Toggle | **Toggles** (12) | Settings toggle |
| Modal | **Dialogs / Modals** (37) | Settings |

### Priority 3 (Polish)

| Component | Category | Purpose |
|-----------|----------|---------|
| Spinner | **Spinner Loaders** (21) | Loading state |
| Toast | **Toasts** (2) | Notifications |

---

## Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Message Input  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Chat History   │◄────│  New Message    │
│  (Left Sidebar) │     │  (Saved to DB)  │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Backend API    │
                        │  /api/chat      │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  TFT Model +    │
                        │  Claude LLM     │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Response +     │
                        │  Analysis Card  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  [Download PDF] │
                        │  Button inline  │
                        └─────────────────┘
```

---

## Chat History Structure

```typescript
// Database schema for chat history

interface Chat {
  id: string;
  agentId: string;
  title: string;           // Auto-generated from first query
  createdAt: Date;
  updatedAt: Date;
}

interface Message {
  id: string;
  chatId: string;
  role: 'user' | 'assistant';
  content: string;
  analysis?: PropertyAnalysis;  // Attached to assistant messages
  createdAt: Date;
}

interface PropertyAnalysis {
  query: {
    developer: string;
    area: string;
    bedroom: string;
    price: number;
  };
  predictions: { ... };
  trends: { ... };
}
```

---

## Mobile Responsive

```
Desktop (>1024px):
┌──────────┬────────────────────────┐
│ Sidebar  │     Chat Area          │
│          │                        │
└──────────┴────────────────────────┘

Mobile (<1024px):
┌────────────────────────────────────┐
│ [☰]  Property AI                   │  ← Hamburger menu
├────────────────────────────────────┤
│                                    │
│         Chat Area                  │
│                                    │
├────────────────────────────────────┤
│ [Message input...]          [Send] │
└────────────────────────────────────┘

Sidebar opens as overlay on mobile.
```

---

## File Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Redirect to /chat or /login
│   ├── login/
│   │   └── page.tsx            # Login page
│   └── chat/
│       ├── layout.tsx          # Chat layout with sidebar
│       ├── page.tsx            # New chat (empty)
│       └── [chatId]/
│           └── page.tsx        # Specific chat
│
├── components/
│   ├── chat/
│   │   ├── ChatSidebar.tsx     # Left sidebar with chat list
│   │   ├── ChatList.tsx        # List of chat items
│   │   ├── ChatMessages.tsx    # Message display area
│   │   ├── ChatInput.tsx       # Message input
│   │   └── AnalysisCard.tsx    # Inline analysis result
│   ├── settings/
│   │   ├── SettingsModal.tsx   # Settings dialog
│   │   └── LogoUploader.tsx    # Logo upload component
│   └── ui/                     # 21st.dev components
│
└── lib/
    ├── api.ts                  # API client
    └── types.ts                # TypeScript types
```
