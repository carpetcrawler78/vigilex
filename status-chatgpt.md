# ChatGPT Status — job-search-system

Last updated: 2026-06-12

## Current state

The `job-search-system` project has been initialized in GitHub as a private repository:

`carpetcrawler78/job-search-system`

The repository contains the first working structure for a focused job-search system targeting healthcare AI, healthcare data science and clinical data engineering/consulting roles.

## Source-of-truth clarification

Google Drive folder:

`Projekte_Drive/job-search-system`

is the source of truth for the **job-search-system project**.

Clarification:

- The Drive folder is the source of truth for project-level files, final project exports, handoff/status documents and material deliberately moved into the project workflow.
- It is not automatically the source of truth for all historic CVs, Enhancv exports, application documents or certificates.
- Existing CVs in other Drive locations remain input/source material and can be used when explicitly selected.

## Assistant role boundary

Important standing rule added to `PROJECT_MEMORY.md`:

- ChatGPT must only handle discussion, planning, analysis, review, document strategy and prompt/text drafting.
- ChatGPT must not execute code, run terminal commands, run scripts, or perform operative implementation work for this project.
- Claude Code / Fable / Opus is the execution-side master for code, filesystem changes, scripts and operative implementation.
- Working model: ChatGPT = master planner/reviewer/strategist; Claude Code/Fable/Opus = execution master.
- At the start of every future ChatGPT session on this project, first read `PROJECT_MEMORY.md`, then read `status-chatgpt.md`.

## Files created in GitHub

- `README.md`
- `PROJECT_MEMORY.md`
- `00_master_profile/positioning.md`
- `01_cv_versions/cv_health_ai_engineer.md`
- `01_cv_versions/cv_healthcare_data_scientist.md`
- `01_cv_versions/cv_clinical_data_engineer_consultant.md`
- `02_cover_letters/template_health_ai.md`
- `02_cover_letters/template_healthcare_data_science.md`
- `02_cover_letters/template_clinical_data_engineer_consultant.md`
- `03_job_ads/.gitkeep`
- `04_applications_sent/.gitkeep`
- `05_tracker/scoring_model.md`
- `05_tracker/application_tracker_template.md`
- `06_prompts/enhancv_prompts.md`
- `07_evidence/evidence_matrix.md`
- `status-chatgpt.md`

## Strategic decisions

The project should avoid pushing Thomas Heger back into classic low-paid Clinical Data Manager positioning.

Core principle:

> Clinical Data Management is domain proof, not the target identity.

Target paths:

1. Health AI / Clinical AI
2. Healthcare Data Scientist / Clinical Data Scientist
3. Clinical Data Engineer / Healthcare Data Consultant

CDM/eCRF/EDC/GCP experience is used as the credibility bridge into these higher-value roles.

## Important evidence already identified

From previous Drive searches, existing material supports:

- mibeg Clinical Data Management I / Medical Information Management training, 2019–2020
- ICH-GCP / regulatory foundations in training
- Ergomed CDS internship in CRO context
- eCRF evaluation/adaptation
- EDC workflows including OPVerdi, OpenClinica, MARVIN, RAVE exposure
- query management
- data cleaning and plausibility checks
- SAS-based data handling
- DKFZ clinical data work
- AI Engineering / Data Science bootcamp
- SentinelAI / Vigilex healthcare AI project

## Open tasks

1. Identify the best existing Enhancv CV base version from Drive.
2. Build three concrete CV text versions from source material:
   - Health AI Engineer
   - Healthcare Data Scientist
   - Clinical Data Engineer / Healthcare Data Consultant
3. Add a reusable bullet library.
4. Add a LinkedIn profile draft.
5. Add a SentinelAI/Vigilex portfolio README summary for applications.
6. Decide whether final CV exports should be copied into the Drive project folder.
7. At the end of every future session, update this file.

## Next recommended step

Use the existing Enhancv AI/Data Scientist CV as the base and convert it first into the `Health AI Engineer` version. Then check for CDM-fallback wording and ATS readability.

For implementation work, prepare instructions/prompts for Claude Code/Fable/Opus rather than executing code directly in ChatGPT.

---

## Session update — 2026-06-12 — append-only status rule

### Change made

`PROJECT_MEMORY.md` was updated to clarify that `status-chatgpt.md` must be handled as an append-only handoff log.

### New rule

At the end of each ChatGPT working session on `job-search-system`, add a new dated update at the bottom of `status-chatgpt.md` and preserve all previous entries.

### Reason

The status file should function as a chronological project log, not as a single overwritten summary.

### Current open tasks remain

- Identify best existing Enhancv CV base version.
- Build three concrete CV text versions.
- Add bullet library.
- Add LinkedIn draft.
- Add SentinelAI/Vigilex portfolio summary.
- Keep ChatGPT role limited to planning/review/text/prompt work; Claude Code/Fable/Opus handles execution.
