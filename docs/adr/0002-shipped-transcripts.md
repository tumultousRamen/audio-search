# ADR 0002 — Use shipped transcripts as primary text signal

**Status:** Accepted · 2026-05-18

## Decision

Source transcripts from the datasets' shipped labels (CommonVoice `.tsv` `sentence` column, LibriSpeech `.trans.txt`). Do **not** run Whisper in the prototype's hot path. Keep a `Transcriber` interface so Whisper can swap in later.

## Context

Primary datasets ship hand-validated transcripts. Running Whisper on the same corpus burns ~1–2 hr of GPU/CPU time and adds ≤ 2 % WER on these clean splits — retrieval results are unchanged.

## Alternatives

| Option | Why rejected |
|---|---|
| Run Whisper-large-v3 on every clip | 1–2 hr inference, no measurable retrieval win on clean speech |
| Whisper only when shipped transcript missing | AudioCaps (stretch) treats captions ≠ transcripts; handled in a separate field, not by running ASR |

## Consequences

- Eval gold = shipped transcript (real ground truth, not Whisper's guess)
- Ingestion pipeline still includes a `Transcriber` stage; default impl returns shipped text
- Future: swap default impl for `WhisperTranscriber` when ingesting un-transcribed corpora
