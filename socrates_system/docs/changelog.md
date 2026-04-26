# Changelog

All commits to the repository grouped by semantic category. Commits whose intent could only be inferred from the message are marked **[inferred]**.

---

## Added

| Date | Commit | Message |
|---|---|---|
| 2025-07-19 | `Add Socrates Agent MVP` | Initial MVP implementation of the Socrates Agent |
| 2025-08-04 | `adding prompts templates` | Added prompt template files for LLM interactions |
| 2025-08-04 | `adding prompts templates` | Added prompt template files (duplicate push) |
| 2025-08-15 | `finsihed and tested claim extraction , categorisation, and external factuality checks` | Added and tested claim extraction, categorisation, and external factuality checking modules |
| 2025-08-16 | `clarification module` | Added clarification resolution module |
| 2025-08-17 | `Socrates question generator, KG` | Added Socratic question generator and Knowledge Graph manager |
| 2025-08-18 | `agla integration and MMHal-Bench evaluation` | Added AGLA remote API client and MMHal-Bench evaluator |
| 2025-08-18 | `KG, Evaluation , Full pipeline - cross-modal check` | Added full pipeline with KG, evaluation harness, and cross-modal verification |
| 2025-08-20 | `mitm, evaluation` | Added MitM middleware and evaluation harness integration |
| 2025-08-24 | `MMHal_Bench evaluation completed , and VLMEvalKit code for MITM completed` | Added complete MMHal-Bench evaluation and VLMEvalKit MitM integration |
| 2025-08-24 | `llava evaluation on MMHal-Bench` | Added LLaVA evaluation on MMHal-Bench |
| 2025-08-25 | `llava-evaluation` | Added LLaVA evaluation support **[inferred]** |
| 2025-08-26 | `Vendor VLMEvalKit into repo (replace submodule with tracked files)` | Vendored VLMEvalKit directly into the repository |
| 2025-08-26 | `Vendor VLMEvalKit into repo` | Vendored VLMEvalKit (initial) |

---

## Changed

| Date | Commit | Message |
|---|---|---|
| 2025-08-15 | `` `Updated Socrates System code with various changes and additions to claim extraction, categorization, and factuality checking modules.` `` | Updated claim extraction, categorization, and factuality checking modules |
| 2025-08-16 | `claim extraction,categorisation, external check` | Updated claim extraction, categorisation, and external check pipeline |
| 2025-08-27 | `claim extraction edit` | Updated claim extraction logic (6 iterations across the day) |
| 2025-08-27 | `SUT/pipeline` | Updated system-under-test and pipeline integration (3 iterations) |
| 2025-08-28 | `last version of ZANOBIA with edits ont he logic of verification with same modules` | Updated verification logic for ZANOBIA-based pipeline |
| 2025-08-29 | `Q0/Q1 edits` | Updated Q0/Q1 question generation or evaluation logic **[inferred]** |
| 2025-08-29 | `last edits` | Final round of edits before release **[inferred]** |
| 2025-08-29 | `final edits` | Final pre-submission edits **[inferred]** |
| 2025-08-29 | `hopfully final` | Further stabilization edits (3 iterations) **[inferred]** |
| 2025-09-01 | `editing llava"` | Updated LLaVA provider integration |
| 2025-08-26 | `edits` | Miscellaneous edits **[inferred]** |
| 2025-08-25 | `llava` | Updated LLaVA-related code (2 iterations) **[inferred]** |
| 2025-08-20 | `Update README.md` | Updated README |
| 2025-08-20 | `Merge branch 'main' of https://github.com/MohammedEsamaldin I edit in the readme file` | Merged remote main after README edits |

---

## Fixed

| Date | Commit | Message |
|---|---|---|
| 2025-08-26 | `Fix VLMEvalKit imports: make API and VLM imports optional with dynamic __all__` | Fixed VLMEvalKit import errors by making API/VLM imports optional |
| 2025-08-26 | `Fix VLMEvalKit config: add dummy classes for missing imports to prevent NameError` | Fixed NameError in VLMEvalKit config caused by missing import stubs |
| 2025-08-15 | `Add .gitignore and stop tracking secret files` | Removed secret/config files from tracking; added `.gitignore` |

---

## Removed

| Date | Commit | Message |
|---|---|---|
| 2025-07-21 | `Delete README.md` | Deleted README (replaced by updated version) |
| 2025-07-21 | `Delete socrates_agent_mvp directory` | Removed old MVP directory |
| 2025-08-26 | `deleted: .gitmodules` | Removed VLMEvalKit submodule reference in favour of vendored copy |

---

## Merges and repository structure

| Date | Commit | Message |
|---|---|---|
| 2025-07-19 | `Initial commit` | Initial repository commit |
| 2025-07-19 | `Merge pull request #1 from MohammedEsamaldin/codex/build-mvp-for-socrates-agent` | Merged MVP pull request |
| 2025-08-16 | `Merge origin/main resolving conflicts (prefer local)` | Merged remote main, keeping local changes |
| 2025-08-20 | `gitingore` | Committed `.gitignore` update **[inferred]** |
| 2025-08-26 | `Ignore local LLaVA directory` | Added local LLaVA directory to `.gitignore` **[inferred]** |
| 2025-08-29 | `I am dead` | Developer frustration commit during debugging **[inferred]** |
