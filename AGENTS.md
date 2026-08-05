# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- The canonical research plan is `plans/project-plan.md`; keep durable
  experiment evidence, failed ideas, and next steps in the root `notebook.md`.
- Use `tools/cache_audit.py` for Stage-1 KV-cache artifact inspection. It is
  standalone on purpose and should not import the Triton-heavy package entrypoint.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
