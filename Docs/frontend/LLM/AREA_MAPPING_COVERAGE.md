## Area mapping coverage report
### Summary
- Total ambiguous names/aliases: **8**
- From `area_mapping.json:ambiguous_aliases`: **4**
- Inferred from `area_reference.csv` (multi-mapped): **4**

### What this means
- Any key listed here maps to **multiple DLD areas**. We intentionally resolve it with lower confidence (or require disambiguation) to avoid confidently wrong outputs.

### Needs manual disambiguation list
- JLT
- Arabian Ranches - 1
- Dubai Water Canal
- Jumeirah Lakes Towers

### Details (top 50)
| key | dominant_dld_area | candidates | source | reason |
|---|---|---|---|---|
| JLT | Al Thanyah Fifth | Al Thanyah Fifth | Al Thanyah Third | area_reference.csv | alias_maps_to_multiple_dld_areas |
| Arabian Ranches - 1 | Wadi Al Safa 6 | Al Hebiah Second | Wadi Al Safa 6 | area_mapping.json | maps_to_multiple_dld_areas |
| Dubai Water Canal | Al Wasl | Al Wasl | Jumeirah Second | area_mapping.json | maps_to_multiple_dld_areas |
| JLT | Al Thanyah Fifth | Al Thanyah Fifth | Al Thanyah Third | area_mapping.json | maps_to_multiple_dld_areas |
| Jumeirah Lakes Towers | Al Thanyah Fifth | Al Thanyah Fifth | Al Thanyah Third | area_mapping.json | maps_to_multiple_dld_areas |
| Arabian Ranches - 1 | Al Hebiah Second | Al Hebiah Second | Wadi Al Safa 6 | area_reference.csv | master_project_en_maps_to_multiple_dld_areas |
| Dubai Water Canal | Al Wasl | Al Wasl | Jumeirah Second | area_reference.csv | master_project_en_maps_to_multiple_dld_areas |
| Jumeirah Lakes Towers | Al Thanyah Fifth | Al Thanyah Fifth | Al Thanyah Third | area_reference.csv | master_project_en_maps_to_multiple_dld_areas |
