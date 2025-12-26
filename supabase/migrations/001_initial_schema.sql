-- Supabase Migration: Initial Schema for Property Analysis Chat
-- Run this in the Supabase SQL Editor

-- =============================================================================
-- 1. USERS TABLE (extends Supabase Auth)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  company_name TEXT,
  phone TEXT,
  logo_url TEXT,
  primary_color TEXT DEFAULT '#0f766e',
  secondary_color TEXT DEFAULT '#10b981',
  report_footer TEXT,
  show_contact_info BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 2. CHAT SESSIONS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT DEFAULT 'New Chat',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON public.chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON public.chat_sessions(updated_at DESC);

-- =============================================================================
-- 3. CHAT MESSAGES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  -- Structured data for assistant messages
  parsed_query JSONB,
  predictions JSONB,
  report_data JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON public.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON public.chat_messages(created_at DESC);

-- =============================================================================
-- 4. REPORTS TABLE (Generated PDF references)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES public.chat_messages(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  pdf_url TEXT,
  storage_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_user ON public.reports(user_id);

-- =============================================================================
-- 5. DEVELOPER MAPPING TABLE (Optional - can use JSON file instead)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.developer_mapping (
  id SERIAL PRIMARY KEY,
  arabic_name TEXT UNIQUE NOT NULL,
  english_name TEXT NOT NULL,
  aliases TEXT[] DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_dev_english ON public.developer_mapping(english_name);

-- =============================================================================
-- 6. DEVELOPER STATS TABLE (Lookup data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.developer_stats (
  id SERIAL PRIMARY KEY,
  developer_name TEXT UNIQUE NOT NULL,  -- Arabic name
  projects_total INTEGER DEFAULT 0,
  projects_completed INTEGER DEFAULT 0,
  projects_active INTEGER DEFAULT 0,
  total_units INTEGER DEFAULT 0,
  avg_completion_percent DECIMAL(5,2) DEFAULT 0,
  completion_rate DECIMAL(5,2) DEFAULT 0,
  avg_duration_months DECIMAL(5,1) DEFAULT 0,
  avg_delay_months DECIMAL(5,1) DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 7. AREA STATS TABLE (Lookup data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.area_stats (
  id SERIAL PRIMARY KEY,
  area_name TEXT UNIQUE NOT NULL,
  current_median_sqft DECIMAL(10,2),
  price_change_12m DECIMAL(5,1),
  price_change_36m DECIMAL(5,1),
  transaction_count_12m INTEGER DEFAULT 0,
  supply_pipeline INTEGER DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 8. RENT BENCHMARKS TABLE (Lookup data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.rent_benchmarks (
  id SERIAL PRIMARY KEY,
  area_name TEXT NOT NULL,
  bedrooms TEXT NOT NULL,
  median_annual_rent DECIMAL(10,0),
  rent_count INTEGER DEFAULT 0,
  median_rent_sqft DECIMAL(10,2),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(area_name, bedrooms)
);

CREATE INDEX IF NOT EXISTS idx_rent_area ON public.rent_benchmarks(area_name);

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
-- Lookup tables don't need RLS (public read)

-- Users can only see their own profile
CREATE POLICY "Users can view own profile" ON public.users
  FOR ALL USING (auth.uid() = id);

-- Users can only see their own sessions
CREATE POLICY "Users can view own sessions" ON public.chat_sessions
  FOR ALL USING (auth.uid() = user_id);

-- Messages inherit session access
CREATE POLICY "Users can view own messages" ON public.chat_messages
  FOR ALL USING (
    session_id IN (
      SELECT id FROM public.chat_sessions WHERE user_id = auth.uid()
    )
  );

-- Users can only see their own reports
CREATE POLICY "Users can view own reports" ON public.reports
  FOR ALL USING (auth.uid() = user_id);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for users table
CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON public.users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Trigger for chat_sessions table
CREATE TRIGGER update_sessions_updated_at
  BEFORE UPDATE ON public.chat_sessions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- STORAGE BUCKETS
-- =============================================================================
-- Run these separately in Storage settings or via Supabase CLI

-- Agent logos bucket (public)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('agent-logos', 'agent-logos', true);

-- Reports bucket (private)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('reports', 'reports', false);

-- =============================================================================
-- AUTO-CREATE USER PROFILE ON AUTH SIGNUP
-- =============================================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO public.users (id, email)
  VALUES (NEW.id, NEW.email);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger on auth.users
CREATE OR REPLACE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

