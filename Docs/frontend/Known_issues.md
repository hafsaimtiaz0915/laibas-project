## Known issues (frontend)

### PDF: downloads successfully but isn’t saved in chat (no persistent “Download PDF” link)
- **Symptom**: User clicks **Generate PDF**, the file downloads, but later there is no saved PDF link in the chat history (or they see an error after the download).
- **Root cause**: PDF generation is client-side and works, but the follow-up step that persists it fails:
  - Upload to backend endpoint `POST /api/report/{report_id}/upload` can fail (auth/token, storage overwrite behavior, CORS, backend errors, etc.).
  - When upload fails, we previously surfaced this as “PDF failed to generate” even though generation + download already happened.
- **Current behavior (mitigation)**:
  - Download happens immediately after PDF blob is created.
  - Upload is treated as **non-blocking**: if upload fails, we warn and show a softer message (“Downloaded, but couldn’t save to chat…”), rather than a generation failure.
- **Still to solve**:
  - Improve reliability of upload (retry/backoff, better error messaging, telemetry).
  - Ensure saved `pdf_url` is always reflected in chat history (session refresh / store update).

### PDF: report fields show as `N/A` in PDF while text has values
- **Symptom**: Rent/yield may render, but “Estimated value at handover”, “+12m”, and “uplift” show `N/A`.
- **Root cause**: The PDF template is a pure renderer and expects pre-computed investor fields on `report_data`. Older messages (or some report generation paths) may return stored `chat_messages.report_data` without the enriched fields.
- **Current behavior (mitigation)**:
  - `POST /api/report/generate` hydrates `report_data` with investor totals/uplift/yield ranges (server-side) before returning it, so PDF can render without doing calculations.
  - Hydrated `report_data` is written back to `chat_messages.report_data` so future PDF generations are consistent.
- **Still to solve**:
  - Backfill/migrate old messages if needed (so historical chats always have consistent `report_data`).

### PDF: misleading “failed to generate” toast/alert after file already downloaded
- **Symptom**: User sees “PDF failed to generate” despite having the PDF downloaded.
- **Root cause**: Upload step failure was caught by the same try/catch as PDF generation.
- **Current behavior (mitigation)**:
  - Only show “failed to generate” if blob generation fails.
  - Upload failures show a separate message indicating the PDF downloaded but wasn’t saved to chat.

### PDF: unexpected brand color changes / styling drift
- **Symptom**: Large blocks (e.g. uplift bar) change to agent primary color unexpectedly.
- **Root cause**: Template tied large background blocks to `agentSettings.primary_color`.
- **Current behavior (mitigation)**:
  - Keep large background blocks on a stable color and use agent colors primarily in charts/logo accents.
- **Still to solve**:
  - Align PDF styling strictly to the Figma template and lock it down (design tokens, snapshot testing).





