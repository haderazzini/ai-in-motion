# Instructions for agents — ai-in-motion (PUBLIC)

Public, English portfolio of **AI in Motion**. Everything here is visible to anyone.

## Rules
- Content arrives **only** from the private studio (`haderazzini/ai-in-motion-studio`) via `tools/export_public.py`
  → draft PR → Hader merges. Do not author episode content directly in this repo.
- Never commit: credentials, `.env`, internal production files (scripts drafts, storyboards, narration, review notes),
  absolute local paths, unreleased episodes, ALVA material.
- Allowed direct edits (still via PR): the root README index, licenses, typo fixes approved by Hader.
- Account: `haderazzini` only. `.githooks/pre-push` checks the active `gh` account, the remote owner and the commit
  email (`git config core.hooksPath .githooks`).
- Every episode README follows: question · answer · math · mechanism · application · run it · limitations ·
  references. Code must run with `uv run` from a clean clone.
