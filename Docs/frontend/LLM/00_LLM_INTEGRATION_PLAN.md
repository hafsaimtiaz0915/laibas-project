# LLM Integration Plan

> **Document Version**: 2.0  
> **Last Updated**: 2025-12-11  
> **Purpose**: Plan for integrating OpenAI API with TFT model for natural language property analysis

---

## 1. Overview

The LLM acts as the **interface layer** between:
- **User** → Natural language queries
- **TFT Model** → Numeric predictions (price + rent)
- **Supabase** → Chat history, user sessions, agent settings
- **Response** → Natural language explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FULL ARCHITECTURE                               │
│                                                                              │
│  User Query                                                                  │
│      ↓                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   OpenAI    │    │   Entity    │    │    TFT      │    │   OpenAI    │  │
│  │   Parse     │ →  │  Validator  │ →  │  Inference  │ →  │  Generate   │  │
│  │   Query     │    │  (Arabic)   │    │  (CPU)      │    │  Response   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                            ↓                  ↓                             │
│                     ┌─────────────┐    ┌─────────────┐                     │
│                     │  Developer  │    │   Lookup    │                     │
│                     │  Mapping    │    │   Tables    │                     │
│                     │  (Ar↔En)    │    │   (Stats)   │                     │
│                     └─────────────┘    └─────────────┘                     │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          SUPABASE                                      │  │
│  │  • Chat sessions    • Message history    • Agent settings             │  │
│  │  • User auth        • Report storage     • Developer mapping          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Critical Requirements

### 2.1 TFT Model Input Format

The TFT model uses a specific `group_id` format:

```
{area}_{property_type}_{bedroom}_{reg_type}_{developer_arabic}
```

**Example**:
```
Business_Bay_Unit_2BR_OffPlan_اعمار_العقارية_ش__م__ع
```

### 2.2 Arabic Developer Names

⚠️ **CRITICAL**: The training data uses Arabic developer names.

| User Says | Must Resolve To |
|-----------|-----------------|
| "Binghatti" | `بن غاتي للتطوير العقاري` |
| "Emaar" | `اعمار العقارية ش. م. ع` |
| "Sobha" | `سوبها` |
| "Damac" | `داماك العقارية` |

### 2.3 Required Fields for TFT

| Field | Source | Required |
|-------|--------|----------|
| `area` | User query | ✅ Yes |
| `bedroom` | User query | ✅ Yes |
| `reg_type` | User query (infer if missing) | ✅ Yes |
| `developer_arabic` | Map from English name | ✅ Yes |
| `property_type` | User query (default: Unit) | ✅ Yes |
| `purchase_price` | User query | Optional |
| `forecast_months` | Infer from handover | Optional |

---

## 3. Implementation Phases

### Phase 0: Data Preparation (PREREQUISITE)

**Goal**: Create lookup tables and developer mapping before any code.

#### Task 0.1: Generate Lookup CSVs

**Script**: `scripts/generate_lookup_tables.py`

| Output File | Source | Content |
|-------------|--------|---------|
| `Data/lookups/developer_stats.csv` | Projects_Cleaned.csv | Completion rate, delays, units delivered |
| `Data/lookups/area_stats.csv` | Transactions_Cleaned.csv | 12m/36m price changes, supply pipeline |
| `Data/lookups/rent_benchmarks.csv` | Rent_Contracts_Cleaned.csv | Median rent by area/bedroom |

#### Task 0.2: Create Developer Name Mapping

**File**: `Data/lookups/developer_mapping.json`

Extract all 96 unique Arabic developers from `tft_training_data.csv` and map to English.

```json
{
  "mappings": [
    {
      "arabic": "اعمار العقارية ش. م. ع",
      "english": "Emaar Properties",
      "aliases": ["Emaar", "EMAAR", "Emaar Properties PJSC"]
    },
    {
      "arabic": "بن غاتي للتطوير العقاري",
      "english": "Binghatti Developers",
      "aliases": ["Binghatti", "Binghatti Properties"]
    }
  ]
}
```

⚠️ **MANUAL WORK REQUIRED**: ~2-3 hours to map all 96 developers.

#### Task 0.3: Create Area Abbreviation Map

**File**: `Data/lookups/area_mapping.json`

```json
{
  "abbreviations": {
    "JVC": "Jumeirah Village Circle",
    "JVT": "Jumeirah Village Triangle",
    "BB": "Business Bay",
    "DM": "Dubai Marina",
    "DIFC": "Dubai International Financial Centre"
  },
  "all_areas": ["Al Barsha First", "Business Bay", ...]
}
```

---

### Phase 1: Supabase Setup

**Goal**: Set up database for chat history, users, and agent settings.

#### Task 1.1: Supabase Schema

```sql
-- Users table (agents using the system)
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  company_name TEXT,
  logo_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat sessions
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  title TEXT,  -- Auto-generated from first message
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  -- Structured data for assistant messages
  parsed_query JSONB,
  predictions JSONB,
  report_data JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Generated reports (for PDF download)
CREATE TABLE reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES chat_messages(id) ON DELETE CASCADE,
  pdf_url TEXT,  -- Supabase Storage URL
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Developer mapping (alternative to JSON file)
CREATE TABLE developer_mapping (
  id SERIAL PRIMARY KEY,
  arabic_name TEXT UNIQUE NOT NULL,
  english_name TEXT NOT NULL,
  aliases TEXT[]  -- PostgreSQL array
);

-- Indexes
CREATE INDEX idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_messages_session ON chat_messages(session_id);
CREATE INDEX idx_messages_created ON chat_messages(created_at DESC);
```

#### Task 1.2: Row Level Security (RLS)

```sql
-- Enable RLS
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Users can only see their own sessions
CREATE POLICY "Users view own sessions" ON chat_sessions
  FOR ALL USING (auth.uid() = user_id);

-- Messages inherit session access
CREATE POLICY "Users view own messages" ON chat_messages
  FOR ALL USING (
    session_id IN (
      SELECT id FROM chat_sessions WHERE user_id = auth.uid()
    )
  );
```

---

### Phase 2: Query Parser

**Goal**: Extract structured data from natural language, including `reg_type`.

#### Task 2.1: Function Schema (OpenAI Tools Format)

```python
PARSE_QUERY_FUNCTION = {
    "name": "parse_property_query",
    "description": "Extract property details from Dubai real estate query",
    "parameters": {
        "type": "object",
        "properties": {
            "developer": {
                "type": "string",
                "description": "Developer name in ENGLISH. Examples: Emaar, Binghatti, Sobha, Damac"
            },
            "area": {
                "type": "string",
                "description": "Area/community. Expand: JVC=Jumeirah Village Circle, BB=Business Bay"
            },
            "bedroom": {
                "type": "string",
                "enum": ["Studio", "1BR", "2BR", "3BR", "4BR", "5BR", "6BR+", "Penthouse"],
                "description": "Bedrooms. Convert: '2 bed' -> '2BR', 'studio' -> 'Studio'"
            },
            "price": {
                "type": "number",
                "description": "Purchase price in AED. Convert: '2.2M' = 2200000"
            },
            "property_type": {
                "type": "string",
                "enum": ["Unit", "Villa"],
                "description": "Unit = apartment, Villa = villa/townhouse"
            },
            "reg_type": {
                "type": "string",
                "enum": ["OffPlan", "Ready"],
                "description": "OffPlan = under construction, Ready = completed/secondary"
            },
            "handover_months": {
                "type": "integer",
                "description": "Months until expected handover (for off-plan)"
            }
        },
        "required": []
    }
}
```

#### Task 2.2: Use OpenAI Tools API (Not Deprecated Functions)

```python
# ✅ CORRECT - Using tools API
response = await client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[...],
    tools=[{
        "type": "function",
        "function": PARSE_QUERY_FUNCTION
    }],
    tool_choice={
        "type": "function",
        "function": {"name": "parse_property_query"}
    }
)

# Extract result
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
```

#### Task 2.3: Infer reg_type if Missing

```python
def infer_reg_type(query: str, explicit_reg_type: str = None) -> str:
    """Infer OffPlan vs Ready from query context."""
    if explicit_reg_type:
        return explicit_reg_type
    
    query_lower = query.lower()
    
    offplan_keywords = ["off-plan", "offplan", "under construction", 
                        "handover", "launch", "payment plan", "phase"]
    ready_keywords = ["ready", "completed", "secondary", "resale", "move-in"]
    
    for kw in offplan_keywords:
        if kw in query_lower:
            return "OffPlan"
    
    for kw in ready_keywords:
        if kw in query_lower:
            return "Ready"
    
    return "OffPlan"  # Default for off-plan analysis tool
```

**Deliverable**: `services/query_parser.py`

---

### Phase 3: Entity Validator with Arabic Support

**Goal**: Resolve English developer names to Arabic, validate areas.

#### Task 3.1: Developer Name Resolution

```python
class EntityValidator:
    def __init__(self):
        self.mapping = self._load_developer_mapping()
        self.english_to_arabic = self._build_reverse_lookup()
    
    def _load_developer_mapping(self):
        # Load from Supabase or JSON file
        # Returns list of {arabic, english, aliases}
        ...
    
    def _build_reverse_lookup(self) -> dict:
        """Build English/alias -> Arabic lookup."""
        lookup = {}
        for m in self.mapping:
            lookup[m["english"].lower()] = m["arabic"]
            for alias in m.get("aliases", []):
                lookup[alias.lower()] = m["arabic"]
        return lookup
    
    def resolve_developer(self, english_name: str) -> tuple[str, str, float]:
        """
        Resolve English developer name to Arabic.
        
        Returns: (arabic_name, matched_english, confidence)
        """
        if not english_name:
            return None, None, 0.0
        
        name_lower = english_name.lower().strip()
        
        # Direct match
        if name_lower in self.english_to_arabic:
            return self.english_to_arabic[name_lower], english_name, 1.0
        
        # Fuzzy match
        all_names = list(self.english_to_arabic.keys())
        match = process.extractOne(name_lower, all_names, scorer=fuzz.token_sort_ratio)
        
        if match and match[1] >= 80:
            return self.english_to_arabic[match[0]], match[0].title(), match[1] / 100
        
        return "Unknown", english_name, 0.3
```

**Deliverable**: `services/entity_validator.py`

---

### Phase 4: TFT Inference

**Goal**: Run TFT model predictions with proper input construction.

#### Task 4.1: Group ID Construction

```python
def construct_group_id(
    area: str,
    property_type: str,
    bedroom: str,
    reg_type: str,
    developer_arabic: str
) -> str:
    """
    Construct group_id matching training data format.
    
    Format: {area}_{property_type}_{bedroom}_{reg_type}_{developer}
    """
    area_clean = area.replace(" ", "_")
    developer_clean = developer_arabic.replace(" ", "_")
    
    return f"{area_clean}_{property_type}_{bedroom}_{reg_type}_{developer_clean}"
```

#### Task 4.2: TFT Prediction Flow

```python
class TFTInference:
    def __init__(self, model_path: str, data_path: str):
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        self.model.eval()
        self.data = pd.read_csv(data_path)  # For encoder history
    
    def predict(
        self,
        area: str,
        property_type: str,
        bedroom: str,
        reg_type: str,
        developer_arabic: str,
        forecast_months: int = 12,
        purchase_price: float = None
    ) -> dict:
        # 1. Construct group_id
        group_id = self.construct_group_id(...)
        
        # 2. Find matching group (exact or fuzzy)
        matched = self.find_matching_group(group_id)
        
        # 3. Get historical encoder data (last 12-96 months)
        history = self.get_group_history(matched)
        
        # 4. Construct future known values (month_sin, month_cos, units_completing)
        future = self.construct_future_known(history, forecast_months)
        
        # 5. Run prediction
        predictions = self.model.predict(pd.concat([history, future]))
        
        # 6. Extract quantiles (p10, p50, p90)
        return self.extract_quantiles(predictions, purchase_price)
```

#### Task 4.3: Handle Unknown Groups

```python
def find_matching_group(self, group_id: str) -> str:
    """Find exact or closest matching group."""
    all_groups = self.data['group_id'].unique()
    
    # Exact match
    if group_id in all_groups:
        return group_id
    
    # Fuzzy match
    match = process.extractOne(group_id, all_groups, scorer=fuzz.token_sort_ratio)
    
    if match and match[1] >= 70:
        return match[0]
    
    # Fallback: Use developer overall or area overall
    return self._fallback_group(group_id)
```

**Deliverable**: `services/tft_inference.py`

---

### Phase 5: Trend Lookup

**Goal**: Fetch developer stats, area trends from Supabase or CSVs.

#### Task 5.1: Supabase Lookup

```python
from supabase import create_client

class TrendLookup:
    def __init__(self):
        self.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    def get_developer_stats(self, developer_arabic: str) -> dict:
        """Get developer stats from Supabase."""
        result = self.supabase.table('developer_stats') \
            .select('*') \
            .eq('developer_name', developer_arabic) \
            .single() \
            .execute()
        
        if result.data:
            return result.data
        return self._empty_developer_stats()
    
    def get_area_trends(self, area: str, bedroom: str = None) -> dict:
        """Get area price trends."""
        # Query area_stats table
        ...
```

**Deliverable**: `services/trend_lookup.py`

---

### Phase 6: Response Generator

**Goal**: Generate natural language from predictions.

(No changes from original - already correct)

**Deliverable**: `services/response_generator.py`

---

### Phase 7: Chat Endpoint with Supabase

**Goal**: Full endpoint with chat history persistence.

#### Task 7.1: Updated Chat Flow

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),  # From Supabase Auth
    supabase: Client = Depends(get_supabase)
):
    # 1. Get or create session
    session_id = request.session_id or await create_session(supabase, user.id)
    
    # 2. Save user message
    await save_message(supabase, session_id, "user", request.query)
    
    # 3. Parse query (with reg_type)
    parsed = await parser.parse(request.query)
    reg_type = validator.infer_reg_type(request.query, parsed.reg_type)
    
    # 4. Resolve developer (English -> Arabic)
    dev_arabic, dev_english, confidence = validator.resolve_developer(parsed.developer)
    
    # 5. Validate area
    area, area_confidence = validator.validate_area(parsed.area)
    
    # 6. TFT predictions
    predictions = tft.predict(
        area=area,
        property_type=parsed.property_type or "Unit",
        bedroom=parsed.bedroom,
        reg_type=reg_type,
        developer_arabic=dev_arabic,
        forecast_months=parsed.handover_months or 24,
        purchase_price=parsed.price
    )
    
    # 7. Lookup trends
    developer_stats = lookup.get_developer_stats(dev_arabic)
    area_trends = lookup.get_area_trends(area, parsed.bedroom)
    
    # 8. Generate response
    response_text = await generator.generate(
        query=request.query,
        predictions=predictions,
        developer_stats=developer_stats,
        area_trends=area_trends
    )
    
    # 9. Save assistant message with predictions
    await save_message(
        supabase, session_id, "assistant", response_text,
        parsed_query=parsed.dict(),
        predictions=predictions,
        report_data={...}
    )
    
    # 10. Return
    return ChatResponse(
        session_id=session_id,
        response=response_text,
        report={...}
    )
```

**Deliverable**: `routes/chat.py`

---

### Phase 8: PDF Reports

**Goal**: Generate branded PDF and store in Supabase Storage.

(Already defined in frontend architecture)

**Deliverable**: `routes/report.py`

---

### Phase 9: Testing

**Goal**: Validate all components work together.

| Test | Focus |
|------|-------|
| Query parsing with reg_type | `"off-plan 2BR JVC"` → `{reg_type: "OffPlan"}` |
| Arabic developer resolution | `"Binghatti"` → `بن غاتي للتطوير العقاري` |
| TFT group matching | Exact and fuzzy matching |
| Unknown group fallback | Developer doesn't exist in area |
| Supabase chat history | Messages persist correctly |

---

## 4. File Structure (Corrected)

```
backend/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py              # Settings, API keys
│   │   ├── supabase.py            # Supabase client
│   │   └── auth.py                # Supabase Auth helpers
│   ├── routes/
│   │   ├── chat.py                # POST /api/chat
│   │   ├── sessions.py            # GET /api/sessions
│   │   └── report.py              # POST /api/report/pdf
│   └── services/
│       ├── query_parser.py        # LLM: Parse query (with reg_type)
│       ├── entity_validator.py    # Arabic ↔ English resolution
│       ├── tft_inference.py       # TFT model prediction (full implementation)
│       ├── trend_lookup.py        # Supabase lookups
│       └── response_generator.py  # LLM: Generate response
│
├── models/
│   └── tft_final.ckpt             # Trained TFT model
│
└── requirements.txt

Data/
├── lookups/                       # Generated lookup tables
│   ├── developer_mapping.json     # Arabic ↔ English (MANUAL)
│   ├── area_mapping.json          # Abbreviations
│   ├── developer_stats.csv        # Generated from Projects
│   ├── area_stats.csv             # Generated from Transactions
│   └── rent_benchmarks.csv        # Generated from Rent Contracts
│
└── tft/
    └── tft_training_data.csv      # For TFT encoder history

scripts/
└── generate_lookup_tables.py      # Creates lookup CSVs
```

---

## 5. Environment Variables

```bash
# .env

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview

# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...  # For backend operations

# TFT Model
TFT_MODEL_PATH=models/tft_final.ckpt
TFT_DATA_PATH=Data/tft/tft_training_data.csv

# Lookups
LOOKUPS_PATH=Data/lookups
```

---

## 6. Supabase Tables Summary

| Table | Purpose |
|-------|---------|
| `users` | Agent accounts (from Supabase Auth) |
| `chat_sessions` | Chat threads per user |
| `chat_messages` | Messages with predictions |
| `reports` | Generated PDF storage |
| `developer_mapping` | Arabic ↔ English (optional, can use JSON) |
| `developer_stats` | Generated developer statistics |
| `area_stats` | Generated area statistics |

---

## 7. API Costs

| Operation | Model | Cost/Query |
|-----------|-------|------------|
| Query Parsing | GPT-4 Turbo | ~$0.006 |
| Response Generation | GPT-4 Turbo | ~$0.024 |
| **Total per query** | | **~$0.03** |

**Supabase**: Free tier (500MB, 2GB bandwidth)

---

## 8. Execution Checklist

| # | Task | Dependency | Est. Time |
|---|------|------------|-----------|
| 1 | Create `Data/lookups/` directory | None | 1 min |
| 2 | Run `generate_lookup_tables.py` | None | 30 min |
| 3 | **MANUAL**: Create developer_mapping.json | #2 | 2-3 hours |
| 4 | Set up Supabase project | None | 30 min |
| 5 | Run Supabase migrations (tables) | #4 | 15 min |
| 6 | Implement query_parser.py | #3 | 1 hour |
| 7 | Implement entity_validator.py | #3 | 1 hour |
| 8 | Implement tft_inference.py | TFT trained | 2-3 hours |
| 9 | Implement trend_lookup.py | #5 | 1 hour |
| 10 | Implement chat.py endpoint | #6-9 | 1 hour |
| 11 | Integration testing | #10 | 2 hours |

**Total: ~12-14 hours** (including manual mapping)

---

## 9. Key Risks

| Risk | Mitigation |
|------|------------|
| Arabic mapping errors | Manual review, test with common developers |
| TFT group not found | Fallback to developer overall or area overall |
| OpenAI rate limits | Implement retry with exponential backoff |
| Supabase free tier limits | Monitor usage, upgrade if needed |
