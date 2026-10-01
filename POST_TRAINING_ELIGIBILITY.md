# Track 01 post-training eligibility review

> **Update (2026-10-01, user confirmation):** the user confirmed that Track 01 allows fine-tuning our own scratch-trained models. The ban covers fine-tuning a *previously existing* model (pretrained initialization). This supersedes the 2026-09-26 operational conclusion below, which treated all SFT/DPO as disallowed. The confirmation is the user's; no organizer document is recorded here. The first phase under it is [v2_task_tune_20261001](v2/phases/task_tune/REPORT.md).

> **Current (2026-10-01):** V2 is the selected model and the only active pipeline; see [README.md](README.md), [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md) and [BEST_CHECKPOINT_TRAINING.md](BEST_CHECKPOINT_TRAINING.md). The rest of this document is a dated record; V1-era material is in [docs/history/](docs/history/).

**Checked:** 2026-09-26.  
**Decision for this entry:** use the scratch-pretrained base endpoint. Treat SFT and DPO as disallowed by the published fine-tuning restriction unless the organizers explicitly provide an applicable exception.

## Published evidence

1. The [official Rules, Track 01](https://gibc-v2.devpost.com/rules) require random-origin training and prohibit pretrained initialization, adapting an existing model through fine-tuning, and larger-model distillation.
2. The [official Resources page, final Track 01 note](https://gibc-v2.devpost.com/resources) separately states that the track excludes both pretrained models and fine-tuning. Its general list of tools, including Unsloth, therefore does not grant permission to fine-tune a Track 01 submission.
3. Neither page states an exception for post-training the entrant's own scratch-trained checkpoint. Neither specifically names SFT, DPO, or RL. Applying the fine-tuning ban to our planned SFT/DPO is our interpretation of the published restriction, not a claim that the organizers answered our exact checkpoint scenario privately.

SFT adjusts the existing base weights using instruction/response examples; DPO adjusts them using preferred/rejected responses. Both planned stages are fine-tuning. Owning the checkpoint does not establish an exemption in the published text. Ordinary continuation of the same scratch pretraining experiment, checkpoint recovery, evaluation, export, and demos are separate from the proposed instruction/preference fine-tuning stages.

## Additional public sources checked

| Source | Result relevant to this question |
|---|---|
| [Devpost Overview](https://gibc-v2.devpost.com/) | Refers Track 01 entrants to the Rules; no own-checkpoint post-training exception found. |
| [Devpost Updates](https://gibc-v2.devpost.com/updates) | Visible notices concern deadlines and sponsor/prize matters; no post-training permission found. |
| [Public Discussions](https://gibc-v2.devpost.com/forum_topics) | Two visible topics: [track selection](https://gibc-v2.devpost.com/forum_topics/45283-track-selection) and [rewards](https://gibc-v2.devpost.com/forum_topics/45238-rewards). Neither supplies a post-training exception. |
| [Organizer's LLM website](https://gibc-9eb4d.web.app/topic-llm.html) | Describes scratch pretraining and flexible training design, but gives no specific SFT/DPO authorization. It also differs from Devpost on team size and presentation format; its general language is not evidence overriding the explicit Devpost restriction. |

Private Discord conversations and nonpublic organizer messages were not inspected. No organizer was contacted by this review. An actual later exception can be evaluated on its wording and scope; none was found in the public material reviewed.

## Correction to the earlier assessment

The earlier notes emphasized ambiguity around an entrant's own checkpoint and left SFT/DPO conditional. The explicit Track 01 note on the Resources page strengthens the case for excluding fine-tuning from the entry. The current operational conclusion is **base-model submission; no SFT/DPO without a documented exception**, rather than presuming that own-model post-training is allowed.

## Project state checked during this review

The existing pipeline completed on **2026-09-24 at 12:17:29 KST**. It recorded **12,000,034,816 cumulative pretraining targets**, selected the base endpoint, completed official evaluation, and created the local submission package. SFT and DPO were skipped because no applicable organizer clarification was recorded; RL was deferred. Those historical decisions and frozen artifacts remain intact.

- [Pipeline state](artifacts/pipeline_v1/state.json)
- [Frozen post-training decision](artifacts/pipeline_v1/posttraining_decision.json)
- [Frozen selection](artifacts/pipeline_v1/selection/selection.json)
- [Local package](artifacts/pipeline_v1/submission_package/)

This review adds a dated source assessment. It does not reopen model selection, rerun official benchmarks, or represent a submitted Devpost entry.
