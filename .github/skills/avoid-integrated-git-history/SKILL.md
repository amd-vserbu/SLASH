---
name: avoid-integrated-git-history
description: "Use when inspecting git history or debugging version changes. Do not use integrated git history features because they can hang this agent; use non-interactive git CLI commands instead."
---

# Avoid Integrated Git History

Integrated git history features are unreliable in this environment and may hang the agent.

## Rules

- Do not use integrated git history features.
- Use non-interactive terminal commands for history inspection.
- Use --no-pager for git history commands to avoid interactive paging.

## Preferred Commands

- git --no-pager log --oneline -n 50
- git --no-pager show <commit>
- git --no-pager diff <ref1> <ref2>
- git --no-pager blame <file>

## Avoid

- Any integrated history/timeline UI flow.
- Any interactive git UI that can block or hang.