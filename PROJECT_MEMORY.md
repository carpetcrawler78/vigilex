# Project Memory — job-search-system

This file records standing project instructions and decisions that should persist across ChatGPT sessions.

## Mandatory startup rule

At the start of every ChatGPT session that works on the `job-search-system` project, first read this file:

`PROJECT_MEMORY.md`

Then read the current handoff/status file:

`status-chatgpt.md`

Use both files before making plans, editing project files or discussing next steps.

## Source of truth

Google Drive folder:

`Projekte_Drive/job-search-system`

is the source of truth for the **job-search-system project**.

Important clarification:

- This Drive folder is the source of truth for the project structure, project-level working documents, final project exports and status handoff material.
- It is **not automatically** the source of truth for all CVs, Enhancv exports, historical application documents or certificates.
- Existing CVs and application materials in other Drive locations remain source material and may be used as input when explicitly selected.
- A CV or export becomes project-level source material only when it is deliberately copied/linked into the project workflow or documented in the repository.

## GitHub role

GitHub repository:

`carpetcrawler78/job-search-system`

is the versioned working repository for Markdown files, prompts, tracker templates, scoring rules, CV strategy drafts, cover-letter templates and project memory.

## Assistant role boundary

ChatGPT's role in this project is **discussion, planning, analysis, review and text/prompt drafting only**.

ChatGPT must **not** execute code, run local scripts, run terminal commands, perform implementation work, or act as the operative coding/execution agent for this project.

Operational execution belongs to Claude Code / Fable / Opus or another explicitly designated execution tool/agent.

Working model:

- ChatGPT = master planner / reviewer / strategist / prompt and document designer.
- Claude Code / Fable / Opus = execution-side master for code, filesystem implementation, scripts and operative changes.

If implementation is needed, ChatGPT should provide:

- exact plan
- acceptance criteria
- file-by-file instructions
- prompts for Claude Code / Fable / Opus
- review checklist

but should not itself execute code.

## Status handoff rule

At the end of every working session on the `job-search-system` project, append a new dated entry to:

`status-chatgpt.md`

Append-only rule: preserve all earlier status entries and add each new session update at the bottom of the file.

The appended update should summarize:

- current project state
- files changed or created
- strategic decisions made
- open tasks
- next recommended steps
- any important caveats or source-of-truth clarifications

## Strategic job-search rule

The project should avoid positioning Thomas Heger primarily as a classic operational Clinical Data Manager.

Clinical Data Management is used as domain proof, not as the target identity.

Target direction:

- Health AI / Clinical AI
- Healthcare Data Science
- Clinical Data Engineering
- Healthcare AI / Clinical Data Consulting

Avoid prioritizing:

- public-sector E9/E10 classic CDM roles
- medical documentation only
- query resolution only
- EDC maintenance only
- low-development operational CDM positions
