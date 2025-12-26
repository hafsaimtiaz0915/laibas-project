import { NextResponse } from "next/server"
import fs from "node:fs/promises"
import path from "node:path"
import { parseCsv } from "@/lib/csv"

export const runtime = "nodejs"
export const revalidate = 3600

function repoRoot(): string {
  // next dev/build runs with cwd at /frontend
  return path.resolve(process.cwd(), "..")
}

export async function GET() {
  const filePath = path.join(repoRoot(), "Data", "lookups", "area_reference.csv")
  const csvText = await fs.readFile(filePath, "utf-8")
  const rows = parseCsv(csvText)

  const items = rows
    .map((r) => ({
      value: r["master_project_en"]?.trim() ?? "",
      label: r["master_project_en"]?.trim() ?? "",
      dld_area: r["area_name_en"]?.trim() ?? "",
    }))
    .filter((x) => x.value)

  // Deduplicate by value
  const seen = new Set<string>()
  const deduped = items.filter((x) => {
    if (seen.has(x.value)) return false
    seen.add(x.value)
    return true
  })

  return NextResponse.json({
    items: deduped.sort((a, b) => a.label.localeCompare(b.label)),
  })
}






