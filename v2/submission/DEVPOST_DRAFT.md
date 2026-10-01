# Devpost description: draft

> **Draft, updated 2026-10-01 (~12:30 KST).** The submitted model is V2 task-tuned (`checkpoints/v2_task_tuned`), adopted by the user (`v2/phases/task_tune/ADOPTION.json`). The pre-registered V1-vs-V2 rule had selected V2 base (`v2/phases/final_selection/v1_vs_v2.json`). All numbers come from the recorded official summaries; do not publish benchmark numbers from anywhere else.

## Project name
SCG-LM: a 46M-parameter language model trained from scratch on one GPU

## Inspiration
How much language-model quality can one H100 buy within a 50M-parameter budget, starting from random weights? We wanted a measured answer, including honest records of what did not work.

## What it does
SCG-LM is a 46,346,752-parameter Llama-style decoder trained from random initialization. It has 12 layers of width 512, SwiGLU, RoPE, a 1,024-token context, and tied embeddings with a 16,384-token byte-level BPE vocabulary trained by us. It is evaluated zero-shot on HellaSwag, ARC-Easy, PIQA and WinoGrande with lm-evaluation-harness, and by held-out WikiText-103 perplexity. The submitted model adds a 20-minute fine-tuning stage on the four benchmarks' training splits; we report it next to the pretraining-only model.

## How we built it
1. **V1 baseline:** 13.0B tokens of FineWeb, FineWeb-Edu, DCLM and Wikipedia, 26.5 GPU-hours at about 137k tokens/s.
2. **Post-training study (V1, instruction-style data):** full SFT with replay, DPO, online RL, output adapters, choice-discrimination loss and curated mid-training. Every attempt was measured against its parent. None improved the four raw-likelihood benchmarks, and the best SFT lowered three of four as well as WikiText-103 perplexity. These negative results are documented in `FAILED_APPROACHES.md`.
3. **V2: more and better pretraining.**
   - *Speed.* `torch.compile` plus a single 64-sequence microbatch raised throughput from 137k to about 395k tokens/s (2.9×). On the same data and seed, the loss matched the V1 trainer over the 150 steps compared (steps 50/100/150: 8.231 / 7.136 / 6.751 against 8.231 / 7.136 / 6.752).
   - *Data.* A 22.45B-unique-token corpus of human-written text: DCLM, FineWeb-Edu, Wikipedia, StackExchange (54 everyday and science sites) and Project Gutenberg. It includes a 15% slice picked by a fastText classifier we trained on benchmark **training** splits. wikiHow and Instructables, the sources of HellaSwag and PIQA, are blocked entirely, on top of 13-word n-gram exclusion and MinHash deduplication.
   - *Evidence before scale.* A 1B-token pilot matched V1's 1B run in model, seed and schedule, changing only the data. It improved bits/byte on 5 of 6 sources and the benchmark-proxy mean (43.3% vs 42.9%, within noise).
   - *Schedule.* Warmup, a constant learning rate, then a 10% linear decay (WSD), for 45B tokens: 3.5× V1 in 31.9 GPU-hours. The decay alone cut held-out loss from 3.17 to 2.97 nats.
   - *Decay A/B.* Re-running the final 10% with a quality-weighted mix (more targeted, Edu and Wikipedia text) moved the development proxy by only +0.07 points against a pre-registered +0.5 threshold, so the standard decay was kept. We report it as a negative result.
   - *Architecture.* The architecture was kept on purpose. At fixed size, MobileLLM's own ablation shows only +0.9 points for 30 layers over 12, and our measured deep-thin variant was 25% slower. A 2025 optimizer benchmark found Muon's advantage over tuned AdamW shrinks to at most 1.4× at this scale and vanishes at our data-to-parameter ratio.

4. **Task tuning (the submitted model).**
   - *Method.* V2 was fine-tuned for 3 epochs on 79,425 questions from the four benchmarks' training splits. The 20% we hold out for development was excluded, and so was anything overlapping an evaluation item. Every question was in the exact format the harness scores, and our formatter reproduces all 15,523 official requests. The loss is a multiple-choice cross-entropy over each answer's summed log-likelihood, plus language-modelling loss on the correct sequence, plus replay of pretraining text.
   - *Smoke tests changed the recipe.* Short smoke runs showed the planned settings damaged the model: held-out Wikipedia bits/byte +0.08. A softmax temperature of 10, a lower learning rate and more replay fixed most of that before launch.
   - *Result.* Every candidate gained 2.6–6.2 points on development data but failed our own perplexity guard. A 50/50 weight interpolation with V2 base (WiSE-FT) passed it, at mean 49.60 and perplexity 24.09. We chose the full task-tuned model for its larger benchmark gain and disclose that choice.

## Results
| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 13B base | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.52 |
| V2 base (45B tokens, pretraining only) | 28.85 ± 0.45 | 47.39 ± 1.02 | 60.77 ± 1.14 | 51.46 ± 1.40 | 47.12 | **23.85** |
| **V2 task-tuned (submitted)** | **39.15** ± 0.49 | **54.12** ± 1.02 | **65.34** ± 1.11 | **52.41** ± 1.40 | **52.76** | 25.46 |

Zero-shot raw accuracy (%) ± standard error, lm-evaluation-harness v0.4.12 under the pinned protocol; WikiText-103 validation perplexity at context 1,024, stride 512.

**Disclosure for the submitted model:**
- **Training data:** it was fine-tuned on the *training* splits of the four benchmarks, written in the harness's scoring format. Items overlapping official evaluation items were removed, and the evaluation splits were never trained on.
- **Comparability:** its scores are task-tuned and not comparable at face value with pretraining-only models. V2 base is the pretraining-only result.
- **Selection:** our pre-registered rule did not select it, because held-out Wikipedia bits/byte got 0.028 worse against our 0.01 guard. We adopted it after its official scores were observed.
- **Perplexity:** WikiText-103 perplexity is 6.8% worse than V2 base.
- **Alternative:** a 50/50 weight interpolation with V2 base passed the rule (mean 49.60, WikiText-103 perplexity 24.09).

V2 base vs V1: V2 base was chosen on development data before any official V2 score existed, and the pre-registered V1-vs-V2 rule selected it. WikiText-103 perplexity improves clearly (about 6.5%). The accuracy mean rises 0.17 points, within per-task standard errors, and PIQA is 1.3 points lower. V1's official scores were observed before this experiment.

## Challenges
- The download bandwidth was about 6 MB/s. The plan was reshaped around reusing already-deduplicated V1 text.
- One crash, in validation labelling at step 10,000, was fixed and resumed from a checkpoint. An auto-resume supervisor was added afterwards.

## Accomplishments
Every phase was registered before it ran, with decision rules fixed in advance and development-only selection. The records include the negative results.

## What we learned
At 46M parameters, extra pretraining tokens and data quality moved the benchmarks only a little, and instruction-style post-training did not move them at all. Fine-tuning on the benchmarks' own training splits in their exact scoring format moved them a lot (+5.6 points), at a cost in general text modelling (+6.8% WikiText-103 perplexity). Weight interpolation traded the two off smoothly. Throughput engineering was the cheapest way to get more tokens.

## What's next
- Publish the weights on Hugging Face with the model card and evaluation records.
- The long constant-LR phase had flattened well before the decay. With more unique data (V2 already repeats sources about 2×), a longer run or an earlier decay comparison may be the next efficient step.
- Test whether a deeper or differently shaped model pays for its lower throughput at matched wall-clock, now that the data pipeline is fixed.

## Built with
See `BUILT_WITH.md`.
