# Prompt baseline snapshot (simplification pass)

Static blocks in `prompts.py` (approximate, for regression awareness):

- `DOCMAP_CONTEXT_SUMMARY`: ~400 characters  
- `HOOK_BRIEF_SUMMARY`: ~280 characters (shortened vs earlier version)  
- `HOOK_LANGUAGE_POLICY`: ~420 characters  
- `DOCMAP_CTA_VOICE`: ~480 characters  

Re-measure after major copy changes: `python -c "from carousel_agents import prompts; ..."` or count in the editor.
