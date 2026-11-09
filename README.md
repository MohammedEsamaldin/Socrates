# ZANOBIA: Socratic hallucination checks for multimodal LLMs

ZANOBIA wraps a multimodal LLM and checks claims **in both directions**. It checks the user's prompt before the model sees it, and the model's answer before the user sees it. Each claim is turned into a neutral, evidence-seeking "Socratic" question and answered by the right source: Wikipedia/Wikidata, a session knowledge graph, or the image itself. When a check fails, the system asks the source to clarify instead of guessing. It needs no retraining and works with any model.

MSc dissertation, Queen Mary University of London (2025, Distinction).
**[Paper](https://drive.google.com/file/d/1e7ZiQuJu0Q046NTe2LlNlZCAYu8Q-8Kg/view)** · **[Write-up on Substack](https://mohammedesamaldin.substack.com/p/its-not-there-zanobia-a-socratic)** · **[Technical docs](socrates_system/README.md)**

## Why both directions?

Most hallucination work treats the model as the only source of error. But many prompts carry their own errors: "What colour is the cat near the bus?" assumes there is a cat, and a helpful model will often invent one. Wu et al. (2024) report that about 47.8% of response hallucinations in their preliminary experiments were triggered by conflicts already in the question. ZANOBIA puts one agent with two gates between the user and the model:

```
 user prompt ──▶ [Gate 1: check prompt] ──▶ model ──▶ [Gate 2: check answer] ──▶ user
                        │                                      │
                        └──────────── same toolbox ────────────┘
     extract claims → label → write Socratic questions → check → pass to KG / clarify
```

| Check | Claim type | Evidence source |
|---|---|---|
| External fact | needs world knowledge | Wikipedia, Wikidata (+ optional fact-check APIs) |
| Self-consistency | must agree with earlier turns | session knowledge graph (SQLite + NetworkX) |
| Cross-alignment | must match the image | VQA with LLaVA-1.5-13B using AGLA attention grounding |

Rephrasing is the core move: "The Eiffel Tower is 500 m tall" becomes "What is the official height of the Eiffel Tower?". The checker looks up an answer instead of judging a sentence it was handed, so the framing of the prompt or the model's prior has less room to leak into the check.

## Results

**MMHal-Bench, LLaVA-1.5-7B** (96 questions, judged by GPT-4V):

| | Score ↑ | Hallucination rate ↓ |
|---|---|---|
| LLaVA-1.5-7B | 1.55 | 0.76 |
| LLaVA-1.5-7B + ZANOBIA | **1.85** | **0.64** |

The largest gain was on questions about objects **not in the image** (score 0.00 → 1.58), which is the case the input gate targets.

**MME results are mixed.** ZANOBIA helped object existence but hurt colour and position for LLaVA-1.5-13B, and lowered GPT-4o-mini's total.

**Limitations.** The benchmarks are small (MMHal-Bench has 12 questions per type). Every number comes from a single run with no confidence intervals. Parts of the pipeline run on OpenAI models while GPT-4V judges, so the checker and the judge are not fully independent. The GPT-4o-mini + ZANOBIA MMHal score (3.19) has no GPT-4o-mini-only baseline. The [write-up](https://mohammedesamaldin.substack.com/p/its-not-there-zanobia-a-socratic) discusses all of this.

## Quickstart

```bash
cd socrates_system
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# Check a text claim (uses a local Ollama model by default)
python -m socrates_system.pipeline --text "The Eiffel Tower is located in Berlin."

# Check text against an image with OpenAI
python -m socrates_system.pipeline --text "The image shows a red car." \
  --image path/to/image.jpg --llm-provider openai --llm-model gpt-4o-mini
```

Benchmark evaluators for MMHal-Bench, MME, POPE, AMBER, HallusionBench and SEED live in [`socrates_system/mllm_evaluation/`](socrates_system/mllm_evaluation). See the [technical README](socrates_system/README.md), [architecture](socrates_system/docs/architecture.md) and [setup guide](socrates_system/docs/setup.md) for configuration, the Flask API and module details.

## Repository layout

| Path | Contents |
|---|---|
| `socrates_system/` | The ZANOBIA pipeline, middleware, evaluators and tests |
| `mllm_evaluation/runs/` | Logged benchmark runs |
| `VLMEvalKit/` | Vendored copy of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) used for evaluation |
| `ragtest/` | GraphRAG experiments |

## Why the name

ZANOBIA is my mother's name. When I can't find something at home and tell her "it's not there", she walks over and picks it up from the exact place she named. A confident claim isn't trusted because it sounds sure: it is checked against what is actually there.

## Citation

```
Esamaldin, M. (2025). ZANOBIA: A Socratic Framework for Bidirectional Detection and
Mitigation of Hallucinations in Multimodal LLMs. MSc dissertation, Queen Mary University of London.
```
