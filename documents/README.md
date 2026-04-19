## documents/

Place **source documents** here (uploads, transcripts, notes) that you want to convert into carousels.

Recommended patterns:
- Keep originals unchanged (treat as source-of-truth).
- Use clear filenames: `2026-04-15_sleep_podcast_transcript.vtt`
- Prefer plain text when possible (`.txt`, `.md`, `.vtt`, `.srt`).

Example run:

```bash
python -m carousel_agents --input "documents\\your_doc.txt" --out "data\\run.json"
```

