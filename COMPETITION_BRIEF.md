# GIBC V2 — Competition Brief and Project Compass

> **Current (2026-10-01):** V2 is the selected model and the only active pipeline; see [README.md](README.md), [CURRENT_PIPELINE.md](docs/history/V1_CURRENT_PIPELINE.md) and [BEST_CHECKPOINT_TRAINING.md](docs/history/V1_BEST_CHECKPOINT_TRAINING.md). The rest of this document is a dated record; V1-era material is in [docs/history/](docs/history/).

**Completed DPO follow-up, 2026-09-28 08:00 KST:** preference generation produced 1,359 verified pairs; DPO completed all 85 updates; development and competition-protocol evaluations finished. Instruction macro accuracy changed from 49.10% to 49.05%, and WikiText-103 perplexity from 27.6288 to 27.6009. Changes were small and mixed; the pipeline retained the 13B base. RL remained deferred and confirmation stayed unopened. No training is active. **173 CPU tests passed; the completed artifact and training audit passed.** See [the separate DPO results](reports/posttraining/2026-09-28_dpo_followup.md) and [final audit](artifacts/posttraining_dpo_13b_v1/FINAL_AUDIT.json). The earlier active-run notices below are historical.

**Completed outcome, 2026-09-28 01:43 KST:** both 131-update SFT runs and their development/competition-protocol evaluations completed successfully. Human-only SFT used 1,077,789 supervised targets; mixed SFT used 1,077,749. Mixed SFT reached 49.10% on the narrow procedural instruction development tests, but both candidates increased raw-source NLL by approximately 0.06 nats, exceeding the fixed 0.02 limit. The registered decision retained the 13B base. DPO was skipped because neither SFT candidate qualified; RL remained deferred. Confirmation was not opened. No training is active. See [the completed post-training report](artifacts/posttraining_13b_v1/REPORT.md) and [final selection](artifacts/posttraining_13b_v1/selection/selection.json). All completed-stage artifact checksums were verified on the status check. Earlier launch notices below are historical.

**September 28 active update:** own-checkpoint post-training is now authorized by the user, who reports that the organizer permits it. Follow [POSTTRAINING_13B.md](FAILED_APPROACHES.md) and [the new live workflow](artifacts/posttraining_13b_v1/state.json). The persistent `gibc-post13` run verifies the completed 13B base and prepared data, then executes bounded SFT comparisons, conditional DPO, selection and reporting. **166 CPU tests passed.** Earlier skip decisions and packages are historical and remain unchanged; starting the workflow is not a claim that SFT or DPO has completed.

**Current checkpoint, 2026-09-27:** Track 01's scratch lineage reached **13,000,048,640 targets**. The 13B held-out gates and full automatic evaluation passed/completed; the evaluated base endpoint has a validated local [submission package](artifacts/release_13b/submission_package/README.md). Follow [RELEASE_13B.md](artifacts/release_13b/submission_package/README.md) for publication/video/team fields. The [latest rules/date check](reports/releases/2026-09-27_rules_recheck.md) retains the earlier-cutoff planning policy. No SFT/DPO/RL is part of this entry. Earlier implementation and launch descriptions below are dated project history.

**Last full competition review:** 2026-09-22; Rules, Schedule, and Updates rechecked 2026-09-23; date conflicts persist.  
**Project status updated:** 2026-09-23 (implementation authorized; scratch trainer implemented)  
**Planning timezone:** Asia/Seoul (KST, UTC+9)  
**Current phase:** 1B scratch baseline complete; user authorized the 12B extension. Corpus preparation and automatic audited pilot/full-run supervisor are active; see [execution runbook](docs/history/CONTINUATION_RUNBOOK.md). See [CORPUS_PLAN_10B_PLUS.md](docs/history/CORPUS_PLAN_10B_PLUS.md) and [RUN_STATUS.md](RUN_STATUS.md).  
**Selected track:** Track 01 — TECH (Foundational LLM Development), confirmed by the user on 2026-09-23.  
**Hardware:** one H100 confirmed by the user; guaranteed access hours and any spending allowance remain open.  
**Implementation plan:** [Track 01 research plan](artifacts/pipeline_v1/submission_package/TRACK_01_RESEARCH_PLAN.md), including architecture, experiment design, source evidence, and milestones.  
**Post-training plan:** [SFT, optional DPO, and gated RLVR](artifacts/pipeline_v1/submission_package/POST_TRAINING_PLAN.md); proposed, with own-model post-training eligibility unresolved.  
**Team, eligibility, and submission owner:** not yet confirmed.

This is our working reference for choosing, building, evaluating, and submitting a project. Read it at the start of future work sessions. Official requirements below are linked to their sources; our planning choices are explicitly labeled. This file is a dated summary, not a replacement for the full rules.

## 1. Our goal and scope

**Working goal — Track 01:** train a language model from scratch within the 50,000,000-trainable-parameter limit, including embeddings and the output head, and demonstrate its quality, reasoning performance, training efficiency, and technical contribution through reproducible experiments. The detailed constraints and required evaluations are in Section 3.

**Proposed research direction:** train a 46,346,752-parameter Llama-style model and compare bounded, evidence-guided data-mixture changes with fixed mixtures. Apply the user's SCG-ECC framework through lightweight training contracts and provenance. The [research plan](artifacts/pipeline_v1/submission_package/TRACK_01_RESEARCH_PLAN.md) distinguishes published evidence, our design choices, and unproven benefits.

Success has three parts:

1. **Eligibility:** the entry satisfies every applicable requirement.
2. **Execution:** training produces a usable model, evaluation is reproducible, and our claims have evidence.
3. **Communication:** a judge can understand the problem, contribution, results, and limitations quickly.

An award is an aspiration, not something we can guarantee. Our controllable objective is the strongest complete entry we can finish within the available time and resources.

**Current scope:** the 46,346,752-parameter scratch baseline completed 999,948,288 target tokens. The user now prioritizes 10B+ pretraining, beginning with corpus-source research. The [corpus plan](docs/history/CORPUS_PLAN_10B_PLUS.md) proposes a 12B cumulative continuation and records access/sample checks. The user authorized execution: extension corpus preparation and the automatic audited pilot/full-run supervisor are active. See [CONTINUATION_RUNBOOK.md](docs/history/CONTINUATION_RUNBOOK.md) for live state; GPU updates wait for the complete corpus. The adaptive candidate and post-training remain unlaunched. Preserve the 1B baseline and report changed data/compute separately.

## 2. Dates: unresolved official contradictions

**Internal planning decision:** work toward the earlier current Devpost cutoff, with an additional one-day submission buffer. Do not spend an advertised extension until the organizers resolve the discrepancy and the submission platform agrees.

| Event or source | Published date/time | Interpretation |
|---|---|---|
| Devpost submission window | July 11, 2026, 12:00 → October 1, 2026, 23:45, Taipei/UTC+8 | Current schedule cutoff. [Schedule](https://gibc-v2.devpost.com/details/dates) |
| Organizer extension announcement | October 2, 2026, 23:59, UTC+8 | Conflicts with the current cutoff. [Extension notice](https://gibc-v2.devpost.com/updates/46335-deadline-extended-you-now-have-until-oct-2-to-submit-your-project) |
| Devpost schedule: judging | October 2, 00:00 → October 16, 23:45, UTC+8 | Conflicts with Rules: October 1–7. [Schedule](https://gibc-v2.devpost.com/details/dates), [Rules](https://gibc-v2.devpost.com/rules) |
| Results/ceremony | Overview: October 8; schedule: October 17, 11:45, UTC+8 | Unresolved. [Overview](https://gibc-v2.devpost.com/), [Schedule](https://gibc-v2.devpost.com/details/dates) |

**Timezone conversions, calculated from the published UTC+8 timestamps:**

- Earlier cutoff: **2026-10-02 00:45 KST** = 2026-10-01 15:45 UTC.
- Announced extension: **2026-10-03 00:59 KST** = 2026-10-02 15:59 UTC.
- **Our proposed submission target: 2026-09-30 20:00 KST.** This leaves 28 hours 45 minutes before the earlier cutoff.

Devpost displays “CST”; its Rules specify Taipei/UTC+8. Use explicit offsets in our calendar. [Rules](https://gibc-v2.devpost.com/rules)

The separate [event landing page](https://gibc-9eb4d.web.app/competition-landing.html) still displays September 21, and the [conduct page](https://gibc-9eb4d.web.app/code-of-conduct.html) mentions September 28. These additional inconsistencies make rechecking essential; they do not establish an earlier active Devpost submission deadline.

## 3. Official participation and submission snapshot

### Participation and tracks

Students aged 13+; teams of 1–6; one project/track. Country exceptions apply; companies/professional organizations cannot enter as entities. Devpost registration required. [Overview](https://gibc-v2.devpost.com/)

| Track | Entry |
|---|---|
| **01 — TECH — SELECTED** | Train a language model from scratch. |
| 02 — Applied | Healthcare/finance: predictive ML, diagnostic pipeline, or automated financial system. |
| 03 — Open | Novel working software, hardware, or generative invention; no architectural/resource cap. |

Source: [Overview](https://gibc-v2.devpost.com/). Track-specific constraints follow below.

**Applicability:** use the common requirements and TECH-specific constraints for our entry. Applied and Open information is retained for reference only.

### Six required submission components

All six must be complete before the cutoff. [Overview](https://gibc-v2.devpost.com/)

| ID | Required component |
|---|---|
| S1 | Devpost description: problem, solution, technical approach. |
| S2 | Public GitHub/GitLab/Bitbucket/Hugging Face repository; README prerequisites, setup, usage. |
| S3 | Running demo/methodology video, 2–5 minutes; YouTube/Vimeo/Youku; public/unlisted; English audio/subtitles. |
| S4 | Built With: technologies, frameworks, libraries, APIs, datasets, AI tools, hardware. |
| S5 | Every teammate's real full name, Devpost account, and submission membership; required for certificates. |
| S6 | At least three high-quality images of interface/pipeline/training/build. |

Source: [Overview — submission requirements](https://gibc-v2.devpost.com/). No additional post-submission presentation is required there.

### Detailed constraints

The following compact checklist summarizes the [official Rules](https://gibc-v2.devpost.com/rules). Read the complete page before confirming eligibility or submitting.

- **Membership:** one submitting team/person; organizers/judges/immediate families excluded. Under-18s should obtain guardian permission for entry/name publication.
- **Originality:** build July 11–October 1; pre-existing/commercial/substantially repeated entries disqualified. Credit reused components in Built With/README.
- **AI:** assistants unrestricted; disclose tools in Built With and assisted portions in README; explain code.
- **Build:** functioning submission mandatory across tracks; papers/design documents/slides alone insufficient.
- **Access:** English throughout; unrestricted public repository; document paid/special dependencies, enabling evaluation without them. Inaccessibility: one contact, 48-hour response or no review. Deadline repository judged; later commits may be disregarded.
- **TECH:** ≤50,000,000 trainable parameters, including embeddings/output head. Provide count evidence (script/output)/configuration. Standard libraries/public datasets allowed. No pretrained initialization, fine-tuning, distillation, or hosted-API substitute. README: hardware/time/approximate compute; HellaSwag, ARC-Easy, PIQA, WinoGrande via lm-evaluation-harness; held-out WikiText-103 perplexity. Include evaluation script/results.
- **Applied:** empirical data; authorized public OR fully de-identified datasets; README sources/licenses. Identifiable patient data prohibited. No real patients/clinical decisions/money; trading simulated/historical. README: research prototype, not medical devices/diagnostic tools/financial advice.
- **Placement:** incorrect track risks reassignment/disqualification.
- **Awards:** Gold/Silver/Bronze per track. Ties: first criterion, then chair. Appeals within 72 hours of announcement.

Also review the linked Rules' ownership and promotion provisions before submitting.

**Post-training rule review, checked September 26:** the [official Resources page](https://gibc-v2.devpost.com/resources) explicitly excludes fine-tuning from Track 01, supplementing the Rules' prohibition on fine-tuning an existing model. No own-checkpoint exception was found in the public pages or discussion threads reviewed. Treat our proposed SFT/DPO as disallowed for this entry unless an applicable organizer exception is documented. The base-only pipeline has completed. See [dated evidence and interpretation](POST_TRAINING_ELIGIBILITY.md). Public synthetic datasets are not categorically banned; the Rules name TinyStories while separately prohibiting larger-model distillation.

### Conduct

Respect participants and privacy. Harassment, plagiarism, cheating, misrepresentation, and publishing others' private information without consent violate the conduct policy; sanctions can include expulsion or disqualification. [Code of Conduct](https://gibc-9eb4d.web.app/code-of-conduct.html)

## 4. Judging: evidence we should prepare

**Current Devpost criteria:** TECH includes Documentation & Demo; Applied uses Rigor & Validation. [Overview](https://gibc-v2.devpost.com/)

The tables below distinguish published website criteria/weights from **our proposed evidence**. The separate organizer website publishes percentages, but their applicability to the current Devpost competition is **unconfirmed**. Do not optimize a numerical score using them yet.

### Track 01 — TECH — our judging criteria

| Criterion | Website percentage, unconfirmed | Our evidence plan |
|---|---:|---|
| Perplexity & Accuracy | 40% | Versioned evaluation output, a comparable baseline, and a short explanation of the result. |
| Reasoning Performance | 25% | Per-task results plus examples of correct and incorrect behavior. |
| Training Efficiency | 20% | Experiment ledger, throughput, peak memory, and comparisons under matched conditions. |
| Innovation | 15% | A concrete hypothesis and an ablation showing what the design change contributes. |
| Documentation & Demo | Absent from website weighting | Clean-environment run instructions, artifact provenance, and an understandable demonstration. |

Percentages: [organizer LLM page](https://gibc-9eb4d.web.app/topic-llm.html). Its four percentages already total 100%, so adding the fifth Devpost criterion requires clarification.

### Track 02 — Applied — reference only

| Criterion | Website percentage, unconfirmed | Our evidence plan |
|---|---:|---|
| Innovation & Impact | 35% | A clearly defined user problem and a measured improvement over a simple baseline. |
| Technical Feasibility | 25% | A complete path from input data to an interpretable output, including failure cases. |
| Rigor & Validation | 25% for the website's differently named “Scientific Rigor” | A suitable split strategy, leakage checks, error analysis, and honest limitations. |
| Presentation | 15% | A concise demonstration linking system output to the problem being addressed. |

Percentages and the alternate criterion name: [organizer Applied page](https://gibc-9eb4d.web.app/topic-creation.html).

### Track 03 — Open — reference only

| Criterion | Website percentage, unconfirmed | Our evidence plan |
|---|---:|---|
| Creativity | 35% | Explain what is distinctive and demonstrate it against an existing approach. |
| Execution | 30% | A stable central workflow, usable interaction, and sensible failure handling. |
| Impact | 20% | Evidence of usefulness: a task comparison, observed use, or another relevant measurement. |
| Presentation | 15% | A clear narrative and a demonstration that makes the contribution immediately visible. |

Percentages: [organizer Open page](https://gibc-9eb4d.web.app/topic-open.html).

**Source conflict:** these website pages also describe live presentations and a July 10 registration cutoff; the LLM page allows unlimited team size. This conflicts with Devpost. Our working approach is to use Devpost's entry requirements and treat the website percentages as unresolved supporting information, rather than combining incompatible instructions.

## 5. Project definition — fill this before implementation

**Planning state:** track and one H100 are confirmed. Technical defaults below are proposals detailed in the linked research plan; unknown fields remain explicit.

| Decision | Current answer / required detail |
|---|---|
| Track | **Confirmed: 01 — TECH.** Our entry will be a language model trained from scratch. |
| Team | TBD: members, eligibility confirmation, responsibilities, availability. |
| Capability focus / intended use | Proposed: general English language modeling; required reasoning benchmarks as measured outcomes. |
| Research question | Proposed: can bounded, evidence-gated data-mixture updates improve held-out quality relative to fixed mixtures at this scale? |
| Baseline | Proposed: same architecture/recipe with fixed 80/20 web/Wikipedia token mixture; additional static-average-mixture ablation. |
| Technical contribution | Proposed: small, audited mixture interventions within data/resource contracts; benefits must be demonstrated. |
| Architecture and tokenizer | Proposed: 12 layers, width 512, 8 MHA heads, SwiGLU width 1,376, RMSNorm/RoPE, tied 16,384-token embeddings, context 1,024; analytic count 46,346,752. |
| Training data | Proposed: pinned FineWeb and English Wikipedia subsets; document-group splits and benchmark exclusions; exact snapshots/subsets to freeze. |
| Core demonstration | Model configuration → training evidence → saved model → inference and evaluation. Exact examples TBD. |
| Success measurements | Report the required evaluations from Section 3; set numerical targets after a baseline. Select supporting efficiency measurements. |
| Supporting evidence | TBD: ablations, learning curves, error analysis, and run-to-run variability where feasible. |
| Minimum scope | Proposed: one complete training run, a usable model artifact, reproducible evaluation, and a defensible comparison. |
| Explicit exclusions | TBD: features that do not fit the available time. |
| Resources | Confirmed: one H100. Read-only local query shows H100 NVL hardware with 95,830 MiB/device. Assigned device, access duration, CPU/disk allocation, and spending limit to record before runs. |
| Biggest uncertainty | Whether the candidate improves enough to justify probe overhead; throughput and a bounded pilot determine feasible experiment size. |
| Submission owner | TBD: person responsible for verifying and submitting the complete entry. |

**Project statement template:**

> We will train [model design] from scratch to test whether [specific change] improves [measured capability or efficiency] relative to [baseline] under [matched conditions]. We will use [data plan] and [compute budget], report the required evaluations, and demonstrate [model behavior]. Our numerical target is [baseline-informed target].

**Idea selection procedure — proposed internal practice:**

1. Reject candidates with unresolved eligibility or clear track incompatibility.
2. Compare at most three research directions on benchmark relevance, novelty, experiment cost, evidence quality, and available resources.
3. Identify the highest-risk assumption for each.
4. Choose the idea whose value and feasibility we can test earliest.
5. Record the decision and tradeoffs here before expanding scope.

Do not choose an idea solely because a sponsor provides credits, or because a large feature list sounds impressive.

### Track 01 experiment controls — proposed internal practice

- Confirm hardware access and the spending ceiling before sizing the training run.
- Measure throughput and memory on a small pilot before committing to a full run; reserve compute for evaluation and a recovery run.
- Keep evaluation examples out of training and record dataset versions and preprocessing decisions.
- Record configuration, code revision, seed, training progress, elapsed time, and hardware for each run.
- Compare one primary change at a time; report differences in data or compute that make comparisons imperfect.
- Save the tokenizer, configuration, and usable model checkpoint with clear provenance. This artifact bundle is our reproducibility plan, not a claim that the organizers specify a particular hosting format.
- Keep the demonstration focused on the model and its measured behavior. Add interface work only if it improves the explanation without displacing training or evaluation.

## 6. Delivery plan — proposed, not an organizer schedule

All dates below are **2026, KST**. The detailed milestones and fallback policy are in the [research plan](artifacts/pipeline_v1/submission_package/TRACK_01_RESEARCH_PLAN.md); preserve time for validation and submission.

| Target | Milestone | Evidence of completion |
|---|---|---|
| Sep 23 | Record confirmed track/H100; propose hypothesis and architecture; resolve remaining eligibility/access details | Research plan and explicit open decisions. |
| Sep 24 | Complete a training/evaluation pilot | Verified parameter count, measured throughput/memory, saved checkpoint, evaluation smoke check, and feasible run estimate. |
| Sep 26 | Complete a baseline training run and evaluation | Reproducible model artifact, training record, and baseline results. |
| Sep 28 | Complete the selected experiment, comparison, and error analysis | Required evaluation results, supporting artifacts, and limitations consistent with the evidence. |
| Sep 29 | Freeze experiments; finish documentation and media drafts | S1–S6 ready for review; unresolved issues limited to corrections. |
| Sep 30, 20:00 | Submit and verify | Confirmed Devpost submission, independently accessible artifacts, recorded revision. |
| Oct 1 | Contingency and final access check | Any necessary correction completed before the earlier cutoff. |

If a milestone slips, reduce optional scope first. Do not compensate by dropping evaluation, hiding limitations, or postponing the submission package to the last hour.

## 7. Definition of done and session discipline

**Internal completion checklist — proposed quality bar:**

- [ ] Project definition is complete; every major feature supports the core demonstration.
- [ ] Eligibility and applicable official constraints have been checked against the latest pages.
- [ ] Track 01 parameter-count evidence, training reports, required benchmark results, and evaluation script are present.
- [ ] The submitted model artifact corresponds to the documented run and reported results.
- [ ] The main claim is supported by saved evidence, with a baseline where appropriate.
- [ ] Someone following the instructions can reproduce the central result without undocumented help.
- [ ] Failures and limitations are visible and accurately described.
- [ ] S1–S6 are complete and mutually consistent.
- [ ] Repository and video access have been checked from a signed-out session.
- [ ] Final artifact URLs, submission confirmation, and repository revision are recorded.
- [ ] A team member is responsible for monitoring organizer messages during judging.

**At the start of each work session:** read this file, identify the next milestone, and state the specific evidence the session will produce.

**Before adding a feature:** identify the user outcome, judging evidence, implementation cost, and what must be removed or delayed. If its value is unclear, place it in a future-work list.

**At the end of each session:** update the decision log, evidence links, completed items, current blocker, and next action. Keep the research snapshot date separate from ordinary project-status edits.

### Evidence and artifact register

| Artifact | Link/path | State |
|---|---|---|
| Devpost entry | TBD | Not created/verified |
| Source repository and final revision | TBD | Not created/verified |
| Data/component provenance and AI-assistance record | [README](README.md), `data/processed/main/manifest.json`, run snapshots | Research, implementation, and documentation assistance disclosed; source identities recorded |
| Model configuration, tokenizer, and parameter count | [Model configuration](configs/model.json), [launch evidence](RUN_STATUS.md) | Scratch tokenizer prepared; exact count audited; endpoint export pending completion |
| Training records and model checkpoints | [Run record](RUN_STATUS.md), `runs/baseline_1b/` | Baseline complete; final checkpoint and HF export present |
| 10B+ corpus sources | [Corpus plan](docs/history/CORPUS_PLAN_10B_PLUS.md), `artifacts/corpus_research/` | Seven repositories pinned; bounded samples inspected from five; full preparation pending |
| Evaluation results and reproduction instructions | [Frozen protocol](EVALUATION_PROTOCOL.md), [README](README.md) | Protocol/runner prepared; official model scores still sealed |
| Post-training method and data plan | [POST_TRAINING_PLAN.md](artifacts/pipeline_v1/submission_package/POST_TRAINING_PLAN.md) | Formulated; execution and own-model eligibility remain pending |
| Video and image assets | TBD | Not created |
| Submission confirmation | TBD | Not submitted |

### Decision log

| Date | Decision | Basis | Status |
|---|---|---|---|
| 2026-09-22 | Establish this reference before implementation | User request | Completed |
| 2026-09-22 | Keep track and project undecided | No selection supplied at that time | Track superseded by Sep 23 decision; model concept still open |
| 2026-09-22 | Plan against earlier Devpost cutoff | Conflicting official dates | Conservative working assumption |
| 2026-09-22 | Use Sep 30, 20:00 KST as submission target | Buffer before cutoff | Proposed internal target |
| 2026-09-23 | Select Track 01 — train an LLM from scratch | Explicit user selection | Confirmed |
| 2026-09-23 | Adapt goals, experiment controls, and milestones to Track 01 | Selected track | Internal plan; hardware and model concept pending |
| 2026-09-23 | Record one H100 as available | Explicit user confirmation | Confirmed; supersedes hardware-pending status |
| 2026-09-23 | Propose Llama-style 46M model with bounded mixture experiments and SCG-ECC contracts | User request plus cited research | Plan documented; model not implemented |
| 2026-09-23 | Recheck Rules, Schedule, and Updates | Current official pages | Deadline contradiction still unresolved |
| 2026-09-23 | Implement scratch trainer and launch baseline | User authorization; correctness, deterministic recovery, and real-data pilot evidence | Running; supersedes earlier unimplemented status |
| 2026-09-23 | Formulate SFT → optional DPO and gated RLVR | User request; primary open-model reports; one-H100 budget | Documented separately; preserve base; resolve own-model post-training rule ambiguity before execution |
| 2026-09-23 | Increase pretraining scope to 10B+ and research sources first | Explicit user direction | Source plan complete; proposed 12B cumulative endpoint; original 1B baseline preserved |

## 8. Open questions and external dependencies

### Confirm with organizers if relevant

These are research follow-ups, not prerequisites for finishing this document. No messages have been sent.

1. Which submission timestamp controls: the platform cutoff or the extension announcement?
2. Which judging and results dates are current?
3. Which rubric weights apply, especially to TECH's fifth criterion?
4. For our TECH entry: which benchmark versions, splits, shot counts, and perplexity settings will judges use, and how should evaluation artifacts be provided?
5. May we apply SFT, DPO, or RL after pretraining our own model from scratch, and which post-training data provenance is acceptable? See the precise [eligibility gate](artifacts/pipeline_v1/submission_package/POST_TRAINING_PLAN.md#2-eligibility-gate-unresolved-before-execution).

Use [official Discord](https://discord.gg/4y6RghaSt5) or [organizer email](mailto:gibc.official.team@gmail.com). Save the reply's date and link here before treating it as a resolved instruction.

### Sponsor availability and prizes

- The latest access update says Featherless access ends **September 22** and reports unresolved Adaption Labs access problems. It supplies no precise expiry time. Do not assume these services will support the remaining build period. [Access update](https://gibc-v2.devpost.com/updates/46460-update-on-adaption-labs-and-featherless-ai)
- An offer advertises **$700 credits per registered participant**, claimable **before September 29, 2026**, using matching Devpost name/email. Access still needs verification. [Credit offer](https://gibc-v2.devpost.com/updates/46396-claim-700-in-free-adaption-labs-credits)
- Featherless redemption creates a subscription; review renewal/cancellation terms before relying on it. [Resources](https://gibc-v2.devpost.com/resources)
- Advertised prize dollar values refer to services/items, not cash payouts. [Prize clarification](https://gibc-v2.devpost.com/updates/46428-update-prize-descriptions-clarified-no-prizes-changed)

**Our dependency policy:** confirm access, duration, capacity, and cost before designing around any external service. Record a fallback if its loss would prevent demonstration or judging.

## 9. Source maintenance

Recheck the [Rules](https://gibc-v2.devpost.com/rules), [Overview](https://gibc-v2.devpost.com/), [Schedule](https://gibc-v2.devpost.com/details/dates), and [Updates](https://gibc-v2.devpost.com/updates) when selecting the project and again before submission.

When a source changes:

1. Record the source URL, verification date, and changed requirement.
2. Preserve the previous interpretation in the decision log.
3. Update the affected milestone, constraint, or artifact.
4. Keep unresolved contradictions visible until there is evidence resolving them.

**Research limitation:** public official pages were reviewed; the discussion forum could not be retrieved. Private Discord messages, the logged-in submission form, participant eligibility, and actual sponsor account access were not verified.
