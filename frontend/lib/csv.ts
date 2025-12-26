export function parseCsv(text: string): Array<Record<string, string>> {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let inQuotes = false

  const pushField = () => {
    row.push(field)
    field = ""
  }

  const pushRow = () => {
    // Drop trailing empty line
    if (row.length === 1 && row[0] === "" && rows.length === 0) return
    rows.push(row)
    row = []
  }

  for (let i = 0; i < text.length; i++) {
    const ch = text[i]

    if (inQuotes) {
      if (ch === '"') {
        const next = text[i + 1]
        if (next === '"') {
          field += '"'
          i++
        } else {
          inQuotes = false
        }
      } else {
        field += ch
      }
      continue
    }

    if (ch === '"') {
      inQuotes = true
      continue
    }

    if (ch === ",") {
      pushField()
      continue
    }

    if (ch === "\n") {
      pushField()
      pushRow()
      continue
    }

    if (ch === "\r") {
      // ignore; handled by \n
      continue
    }

    field += ch
  }

  // Flush last field/row
  pushField()
  if (row.length > 1 || row[0] !== "") {
    pushRow()
  }

  if (rows.length === 0) return []
  const header = rows[0].map((h) => h.trim())
  const out: Array<Record<string, string>> = []
  for (const r of rows.slice(1)) {
    const obj: Record<string, string> = {}
    for (let i = 0; i < header.length; i++) {
      obj[header[i]] = (r[i] ?? "").trim()
    }
    // Skip fully-empty rows
    if (Object.values(obj).every((v) => v === "")) continue
    out.push(obj)
  }
  return out
}






