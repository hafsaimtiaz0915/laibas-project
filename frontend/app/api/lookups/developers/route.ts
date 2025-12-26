import { NextResponse } from "next/server"
import fs from "node:fs/promises"
import path from "node:path"
import { parseCsv } from "@/lib/csv"

export const runtime = "nodejs"
export const revalidate = 3600

function repoRoot(): string {
  return path.resolve(process.cwd(), "..")
}

type BuildingDevelopersJson = {
  building_developers_without_own_data?: Record<
    string,
    { aliases?: string[]; note?: string }
  >
}

export async function GET() {
  const root = repoRoot()

  const developerRefPath = path.join(root, "Data", "lookups", "developer_reference.csv")
  const buildingDevsPath = path.join(root, "Data", "lookups", "building_developers.json")

  const [devCsvText, buildingJsonText] = await Promise.all([
    fs.readFile(developerRefPath, "utf-8"),
    fs.readFile(buildingDevsPath, "utf-8"),
  ])

  const devRows = parseCsv(devCsvText)
  const building = JSON.parse(buildingJsonText) as BuildingDevelopersJson

  const normalizeKey = (name: string) =>
    name
      .toLowerCase()
      .replace(/[\.\(\)\[\],]/g, " ")
      .replace(/\b(properties|property|developers|developer|development|realty|pjsc|l\.l\.c|llc)\b/g, " ")
      .replace(/\s+/g, " ")
      .trim()

  // Keep the nicest display label per normalized key.
  // Prefer developer_reference.csv english_name over building_developers.json brand keys when both exist.
  const byKey = new Map<string, { label: string; sourceRank: number }>()

  // Preferred: english_name column for canonical, user-facing developer strings
  for (const r of devRows) {
    const n = (r["english_name"] ?? "").trim()
    if (!n) continue
    const key = normalizeKey(n)
    const existing = byKey.get(key)
    const sourceRank = 2
    if (!existing || sourceRank > existing.sourceRank || n.length > existing.label.length) {
      byKey.set(key, { label: n, sourceRank })
    }
  }

  // Also include curated "brand" developers that may not exist as english_name rows
  const buildingBrands = Object.keys(building.building_developers_without_own_data ?? {})
  for (const b of buildingBrands) {
    const n = (b ?? "").trim()
    if (!n) continue
    const key = normalizeKey(n)
    const existing = byKey.get(key)
    const sourceRank = 1
    if (!existing) {
      byKey.set(key, { label: n, sourceRank })
    }
  }

  const items = Array.from(byKey.values())
    .map((x) => x.label)
    .sort((a, b) => a.localeCompare(b))

  return NextResponse.json({ items: items.map((n) => ({ value: n, label: n })) })
}


