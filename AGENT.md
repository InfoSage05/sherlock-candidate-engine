# AGENT.md — Sherlock Real-Time Candidate Identification & Authenticity Engine

## 1. Purpose

You are the reasoning core of **Sherlock's Candidate Identification & Authenticity Engine**, a system that
watches a live interview (Google Meet / Zoom / Teams) and continuously answers two linked questions:

1. **WHO** in this call is the actual interview candidate?
2. **HOW AUTHENTIC** is what that candidate is currently saying, typing, and doing?

Both questions are answered the same way: as a **running belief updated by many weak, independently
unreliable signals**, never as a single hard rule. You do not output a single final verdict and stop — you
maintain a live, explainable, continuously-revised confidence state for the entire duration of the call.

You are not the sole decision-maker for anything. LLM-based signals you produce are **one voice among many**
inside a fusion engine — never the sole basis for identifying or flagging a person.

---

## 2. Core Operating Principles

- **Multi-hypothesis, not single-shot classification.** Track a belief distribution over *every* participant
  in the call simultaneously. Never collapse to one answer prematurely.
- **Every signal is a log-likelihood update, not a verdict.** No individual signal is allowed to say
  "this is the candidate" or "this is fraud" — it can only shift a probability up or down, with a stated
  magnitude and a stated reason.
- **Explainability is a first-class output, not an afterthought.** Every confidence number must be traceable
  to the ordered list of evidence that produced it.
- **Silence is a valid answer.** If the top two hypotheses are close, report "ambiguous" explicitly rather
  than guessing. Do not force a decision to appear more capable than the evidence supports.
- **Fairness constraint on behavioral signals.** Signals like pause length, accent, or speaking rate must
  never be scored in isolation — they are only meaningful in combination with a second, corroborating
  signal (see §4.4). Flag the pattern, never the trait.
- **Cheap, real-time signals run continuously. Expensive signals (LLM calls, embedding lookups) run on
  triggers, not every frame.**

---

## 3. System Architecture

```
 ┌────────────────────┐
 │   Ingestion Layer    │  Meet/Zoom/Teams events, calendar API, per-participant audio/video streams,
 │                      │  speaker-attributed transcript, screen-share & IDE telemetry (if available)
 └─────────┬────────────┘
           │
 ┌─────────▼────────────┐
 │  Signal Extractors    │  Independent, parallel micro-services. Each emits small evidence packets:
 │  (Section 4)          │  { source, target_participant_id, delta_log_odds, confidence, rationale, ts }
 └─────────┬────────────┘
           │
 ┌─────────▼────────────┐
 │   Fusion Engine       │  Bayesian belief tracker, one instance per meeting, one distribution for
 │                      │  "who is candidate" and one authenticity score-stream per active speaker
 └─────────┬────────────┘
           │
 ┌─────────▼────────────┐
 │  Explanation Layer    │  On-demand: turns the evidence ledger into a plain-English rationale
 └─────────┬────────────┘
           │
 ┌─────────▼────────────┐
 │   Feedback Loop        │  Interviewer confirms/corrects → recalibrates per-signal weights over time
 └────────────────────┘
```

---

## 4. Signal Catalogue

### 4.1 Identity Priors (pre-call and structural)
- **Calendar match**: candidate name/email from invite vs. participant join identity (fuzzy embedding match,
  not exact string match — must survive nicknames, typos, "MacBook Pro" device names).
- **Interviewer negative prior**: known interviewer names/emails get an explicit down-weight, not just an
  absence of up-weight.
- **Email domain check**: external domain vs. company domain as a weak positive signal.

### 4.2 Behavioral / Conversational Signals
- **Turn-taking graph**: directed graph of who-addresses-whom across the transcript. Candidates show high
  in-degree (questions directed at them) and low out-degree.
- **Speaking ratio & burst pattern**: total duration + reactive (post-question) vs. initiating speech.
- **Screen-share correlation**: presenter during technical segments is a strong, not certain, signal.

### 4.3 Semantic Signal (LLM as a sensor, not a judge)
- Feed speaker-attributed transcript segments to an LLM role-classifier. Output per segment:
  `{ role_guess: interviewer|candidate, confidence, one-line rationale }`.
  This is fused as *one* weak vote, logged with its rationale — never used directly as the final answer.

### 4.4 Authenticity Signals (new — candidate performance integrity)

These extend the same belief-tracking pattern to a second axis: **is this participant's current answer
likely to be authentic, unassisted, in-the-moment work?**

**a) Disfluency-rate anomaly (spoken-answer authenticity)**
- Natural spontaneous speech contains filler words, false starts, self-corrections, variable sentence
  length. Segments that are unnaturally clean, well-structured, and uniform relative to the candidate's own
  baseline are a weak positive signal for "reading a generated/prepared answer."
- Always baseline against the *candidate's own* disfluency rate earlier in the call — never against a
  population average. A naturally articulate person is not penalized; a *shift* in their own pattern is
  what's meaningful.

**b) Pause-placement + fluency-recovery pattern (must be paired, never scored alone)**
- Pause length alone is explicitly excluded as a standalone signal — it unfairly penalizes ESL speakers,
  slower thinkers, and naturally deliberate communicators.
- The paired pattern that *is* meaningful: an unusually long pause immediately after a hard question,
  followed by an answer with abnormally low self-correction and high structural fluency (see 4.4a). It is
  the **combination** of "long think time" + "suspiciously polished output" that generates a signal, not
  either alone.

**c) Coding-session behavioral telemetry (not code-text analysis)**
- Do not attempt to classify finished code as "AI-generated" from the text alone — this is unreliable and
  easy to dispute.
- Instead capture process telemetry, the same approach used by Codeforces/HackerRank anti-cheat systems:
  paste events, large code blocks appearing in a single burst vs. incremental typing, keystroke cadence,
  tab/window-focus loss immediately preceding a burst of correct code.
- Emit a weak signal per burst event, timestamped and tied to the specific code segment — this becomes part
  of the evidence ledger, reviewable by a human, not an automatic accusation.

**d) Gaze / off-screen reading detection (computer vision)**
- Use face-mesh + gaze-vector estimation (e.g., MediaPipe Face Mesh, or a lightweight gaze model such as
  L2CS-Net) on the candidate's own webcam stream only.
- The signal is not "looked away once" — it is a **periodic, rhythmic horizontal saccade pattern to a fixed
  off-camera point**, which is the distinctive signature of reading, as opposed to the upward/unfocused gaze
  typical of recall or thinking, or the on-camera gaze typical of normal conversation.
- Emit signal strength as a function of saccade periodicity and consistency, not raw "eyes off camera" time.

All four are fused into the same log-odds ledger as the identity signals, but write to a **separate
authenticity score-stream** per active speaker rather than the "who is candidate" distribution — these are
different questions and must not silently contaminate each other's confidence.

---

## 5. Fusion Mechanism

- Maintain running **log-odds** per participant per question (`identity`, `authenticity`), updated on every
  incoming evidence packet.
- Each evidence packet is `{ source, target_participant_id, delta_log_odds, confidence, rationale, ts }`
  and is appended to an immutable per-meeting evidence ledger — this ledger *is* the explanation, not
  something reconstructed after the fact.
- Reported confidence = normalized posterior of the top hypothesis.
- If the gap between the top two identity hypotheses is below threshold → report `"status": "ambiguous"`
  explicitly, with the leading candidates and their evidence, instead of forcing a pick.
- Authenticity signals never silently flip identity, and identity uncertainty never silently suppresses an
  authenticity flag — the two streams are reported independently, then optionally cross-referenced for a
  human reviewer (e.g., "authenticity flag raised on a participant currently at only 54% identity
  confidence — verify identity before acting on the flag").

---

## 6. Feedback Loop

When an interviewer confirms or corrects the system's identification, or flags/dismisses an authenticity
alert, log this as labeled data. Periodically (not per-call) recalibrate per-signal weights — this is
lightweight weight recalibration, not full model retraining, and is what should be described as "continuous
learning" in the writeup.

---

## 7. Required Behaviors (do not violate)

- Never output a single hard verdict without an accompanying confidence and evidence trail.
- Never treat an LLM role-classification or an AI-text-detector output as ground truth — always fuse it as
  one weak signal.
- Never score raw pause length, speaking rate, or accent as a standalone fraud/authenticity signal.
- Always baseline behavioral signals against the individual's own earlier-call behavior where possible,
  not a population norm.
- Always prefer reporting "ambiguous" over a low-confidence forced answer.
- Always keep the "who is the candidate" belief and the "is this authentic" belief in separate,
  independently-inspectable streams.

---

## 8. Evaluation Checklist (for the submission)

Test against, at minimum:
- Candidate joins as a device name ("MacBook Pro")
- Candidate joins under a nickname
- Interviewer enters the wrong candidate name
- Multiple interviewers present
- Candidate changes display name mid-call
- Silent observers present
- Candidate swapped mid-interview (voice/embedding mismatch against an earlier session if available)
- Candidate never speaks (whiteboard-only round)
- Two legitimate candidates present (loop interview)
- Report not just top-1 accuracy but **calibration**: does a reported 70% confidence correspond to being
  correct ~70% of the time across the test suite?

---

## 9. Product Enhancement Roadmap — Frontend v2 & Operator Workflow

> This section is a planned redesign of the Streamlit operator console. Goal: move from a static
> three-column demo to a **purpose-driven, live-monitoring dashboard** that gives an interviewer or
> proctor enough detail to judge the candidate accurately throughout the call, with clear flags and
> a persistent confidence scoreboard.

---

### 9.1 Current UI Gaps

The existing `app.py` layout has three rigid columns and a bottom feedback bar. It is sufficient for
a quick demo but lacks the structure required for real operator workflow:

1. **No persistent live scoreboard.** Participant confidence is buried inside the center panel;
   the operator cannot glance at the whole call state at once.
2. **No flag / alert system.** Authenticity signals exist, but they are not surfaced as actionable
   alerts; there is no severity, no correlation with identity confidence, and no banner.
3. **Sections are stacked, not purpose-driven.** Evidence ledger, timeline, explain buttons, and
   export all live in the same column without clear hierarchy.
4. **Missing per-participant detail.** No candidate profile card, no behavioral summary, no
   per-person identity/authenticity timelines.
5. **Transcript and evidence are not time-linked.** The operator cannot click an evidence event
   and see the corresponding transcript, or vice versa.
6. **No "what changed" summary.** After each step the operator has to mentally diff the previous
   state from the current one.
7. **No operator notes.** Corrections are binary; there is no place to record *why* a correction
   was made.

---

### 9.2 Design Principles for v2

- **Purpose-driven sections, not data-driven columns.** Every panel answers a single operational
  question (e.g., "Who is the candidate?", "Are they being authentic right now?").
- **Scoreboard first.** The operator's primary question is "Who should I pay attention to?" — the
  scoreboard answers it at a glance.
- **Flags are first-class citizens.** A flag must be visible, severity-coded, and accompanied by a
  recommendation.
- **Progressive disclosure.** Summary cards are always visible; deep-dive tables and raw evidence
  are one click away.
- **Time-aware everything.** Every chart, card, and ledger entry must respect the meeting timeline
  and be scrubbable.
- **Responsive, not cramped.** The layout must remain usable with 2–8 participants and on a
  standard laptop screen.

---

### 9.3 Proposed Section Architecture

The dashboard is reorganized into a top command bar, a persistent left scoreboard, and tabbed or
stacked main sections.

#### A. Live Command Bar (always visible)

A single top row containing:

| Element | Purpose |
|---------|---------|
| Scenario selector + Load | Pick and load a fixture / live session. |
| Playback controls (Run / Pause / Reset / Speed) | Control replay. |
| Global status pill | `IDENTIFIED` / `AMBIGUOUS` / `UNDER REVIEW` with color. |
| Active alert banner | The highest-severity currently open flag, if any. |
| Meeting progress + timestamp | Scrubber + elapsed / total time. |

#### B. Live Scoreboard (left sidebar, sticky)

A participant list that stays visible while scrolling. Each card shows:

- Avatar placeholder + display name.
- **Identity probability bar** (0–100%, green/yellow/red gradient).
- **Authenticity gauge** when this participant is the current speaker; otherwise last-known score.
- Status badge: `CANDIDATE`, `INTERVIEWER`, `UNCLEAR`, or `FLAGGED`.
- Mini **confidence sparkline** over the last N steps.
- Clicking the card opens the **Candidate Intelligence** panel for that person.

#### C. Mission Control (main, top)

The "what is happening right now" area:

- **Top candidate highlight card**: name, confidence, gap to second place, and the single best
  piece of evidence.
- **Confidence-over-time chart**: one line per participant; current time marker; clickable to scrub.
- **Ambiguity gap indicator**: a progress-style bar showing how far above/below the threshold the
  current gap is.
- **Current speaker tile** with speaking duration and turn context.
- **Latest decisive evidence**: the 3 most recent evidence packets that moved the top probability
  by more than X%.

#### D. Live Event Feed (console-style)

A scrollable, time-ordered feed that acts like a mission log:

```
10:04:17  [Identity]      Bob Smith      INTERVIEWER_NEGATIVE  Δ=-2.50
10:04:22  [Behavioral]    Alice Johnson  TURN_TAKING           Δ=+2.00
10:04:45  [Authenticity]  Alice Johnson  CODING_TELEMETRY      ⚠ WARNING  Paste event
```

Each row has:
- Timestamp.
- Signal category and source.
- Target participant.
- Delta log-odds with sign.
- Severity icon if the event raised a flag.
- Click to jump the timeline to that moment and open the rationale.

#### E. Authenticity & Flags Monitor

Dedicated section for integrity monitoring:

- **Authenticity score timeline** per participant (line chart).
- **Active flags table**:
  - Severity: `INFO`, `WARNING`, `CRITICAL`.
  - Participant, signal source, timestamp, rationale.
  - Recommended action (e.g., "Verify identity before acting" if flag is on a non-top candidate).
- **Flag correlation note**: auto-generated text like:
  > "Authenticity WARNING on `MacBook Pro` while identity confidence is only 54% — confirm
  > who this is before treating the flag as candidate behavior."
- **Candidate-of-interest authenticity gauge**: the authenticity score of the current top candidate.

#### F. Candidate Intelligence Panel

A deep-dive panel opened by clicking a scoreboard card:

- **Profile card**: calendar name, email, device name, join time, display-name history.
- **Identity evidence breakdown by category** (bar chart or table).
- **Behavioral summary**:
  - Speaking ratio.
  - Reactive speech ratio (answers to questions).
  - Turn-taking in-degree / out-degree.
  - Average response latency.
- **Comparison table**: this participant vs others on key metrics.
- **All evidence for this participant** tab.

#### G. Evidence Room

A full-screen-capable evidence explorer:

- Filters: category, axis, source, participant, time range, severity.
- Search by rationale text.
- Sortable table with columns: time, source, target, Δ log-odds, confidence, rationale.
- Click row to show metadata JSON.
- Export JSON / CSV.

#### H. Interview Timeline

A unified timeline view:

- Horizontal scrubber with tick marks for evidence events and transcript segments.
- Transcript synchronized to the scrubber position.
- Question/answer pairs visually grouped.
- Click a transcript segment to jump to that time and filter evidence around it.

#### I. Operator Actions

A consistent action panel:

- **Confirm candidate** (top candidate).
- **Correct candidate** (dropdown + note).
- **Flag / dismiss authenticity alert**.
- **Add operator note** (free text, timestamped, attached to the evidence ledger).
- **Export full report** (PDF-style summary of identity decision + authenticity flags + evidence).

---

### 9.4 New Backend Data Requirements

The UI changes above require richer backend support:

1. **Snapshot history.** The replay engine should store every intermediate `IdentificationResult`
   so the scoreboard can draw sparklines and confidence-over-time charts.
2. **Per-participant identity timeline.** A time series of `identity_probability` per participant.
3. **Per-participant authenticity timeline.** A time series of `authenticity_probability` per
   active speaker.
4. **Behavioral summary metrics.** Pre-compute and store speaking ratio, turn in-degree/out-degree,
   response latency, and reactive ratio per participant per snapshot.
5. **Flag metadata on evidence packets.** Optional fields:
   - `severity`: `info` | `warning` | `critical` | `none`
   - `flag_type`: e.g., `authenticity_concern`, `identity_uncertainty`, `device_name`
   - `recommendation`: human-readable guidance
6. **Operator notes.** A new model `OperatorNote` with timestamp, author, text, linked evidence IDs.
7. **Candidate profile metadata.** Display-name history, device name, email, join offset, etc.
8. **Current-speaker detection.** The snapshot must reliably determine who is currently speaking,
   not just the last transcript segment.

---

### 9.5 Flag / Alert Semantics

A flag is **not** a verdict. It is a strong signal that the operator should look closer.

| Severity | Trigger | UI Treatment |
|----------|---------|--------------|
| `INFO` | Notable but expected (e.g., external email domain). | Subtle dot; no banner. |
| `WARNING` | Pattern of concern (e.g., disfluency drop + paste event). | Yellow badge; appears in flags table. |
| `CRITICAL` | Strong evidence of assisted work or swapped identity. | Red badge + top alert banner + sound/visual cue. |

All flags are tied to a participant, a signal source, and an evidence packet. The system must never
suppress an authenticity flag just because identity confidence is low; instead it shows the
"verify identity first" correlation message.

---

### 9.6 Implementation Phases

**Phase 1 — Foundation**
- Add snapshot history to `FusionEngine` / `SessionReplay`.
- Add flag severity and operator-note fields to models.
- Pre-compute behavioral summary metrics.

**Phase 2 — Scoreboard & Mission Control**
- Build the live scoreboard sidebar.
- Add confidence-over-time chart and ambiguity-gap indicator.
- Add top-candidate highlight card.

**Phase 3 — Live Feed & Flags**
- Build the event feed.
- Add flag severity badges and active-flags table.
- Add alert banner for critical flags.

**Phase 4 — Deep Dives**
- Candidate Intelligence panel.
- Evidence Room with filters/search.
- Interview Timeline with synchronized transcript.

**Phase 5 — Operator Workflow**
- Operator notes.
- Export full report.
- Feedback-loop calibration summary visible in UI.

---

### 9.7 Open Questions for Review

1. Should the scoreboard remain sticky while scrolling the main content, or scroll naturally?
2. Should critical flags produce an audible or pop-up notification, or only a visual banner?
3. Should the system support **multi-meeting candidate tracking** (e.g., compare voice/embeddings
   against prior interviews)?
4. What is the maximum number of participants the UI must support gracefully?
5. Should the operator be able to **manually flag a participant** (not just confirm/correct identity)?
6. Should the export report be a JSON dump, a formatted PDF, or both?
7. Do we want a **dark-only** theme, or a toggle for light/dark mode?

---

## 10. Production-Ready Audio/Video Fraud-Detection Roadmap

> **Purpose of this section:** The current Sherlock implementation solves the *identity-resolution* problem well, but it does **not** yet isolate the candidate's live audio/video streams and run real deepfake, voice-cloning, or behavioral-computer-vision checks. This section contains the implementation plan and coding prompts for closing that gap.

### 10.1 Goal

When the system is complete, it must:

1. Ingest live audio/video from Meet/Zoom/Teams **per participant**.
2. Use the existing identity engine to lock onto the single candidate.
3. Route **only the candidate's A/V frames** to the fraud-detection pipelines.
4. Run deepfake, voice-cloning, and behavioral-analysis models in real time.
5. Emit evidence packets into the existing `FusionEngine` so identity and authenticity scores stay in one explainable ledger.
6. Meet a target latency of **< 500 ms** from frame arrival to flag surfacing.

### 10.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Meeting Platform (Meet / Zoom / Teams / WebRTC)              │
│  → Per-participant audio stream                               │
│  → Per-participant video stream                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Ingestion Adapter Layer     │  Real-time demux + timestamp sync
        │   (sherlock/ingestion/)       │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Identity Lock-in            │  Existing FusionEngine decides
        │   (sherlock/fusion.py)        │  "this participant_id is the candidate"
        └──────────────┬──────────────┘
                       │ candidate_id
        ┌──────────────▼──────────────┐
        │   Stream Gate                 │  Only candidate frames pass through
        │   (sherlock/gate.py)          │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Real-Time Model Pipelines   │  Parallel, frame/batch-level inference
        │   (sherlock/pipelines/)       │
        └──────────────┬──────────────┘
                       │ EvidencePacket(s)
        ┌──────────────▼──────────────┐
        │   Existing Fusion Engine      │  Same ledger, same explainability
        │   + Operator Dashboard        │
        └─────────────────────────────┘
```

### 10.3 Implementation Prompts

Treat each subsection below as a self-contained coding task. Each prompt contains: goal, interface contract, suggested approach, files to create/modify, and acceptance criteria.

---

#### Prompt 10.3.1 — Ingestion Adapter Layer

**Goal:** Provide a unified interface that receives live per-participant audio and video from the meeting platform.

**Input contract:**
- `audio_chunk: bytes` — raw PCM or Opus-decoded 16-bit mono audio at 16 kHz.
- `video_frame: numpy.ndarray` — BGR or RGB image, shape `(H, W, 3)`.
- `participant_id: str` — platform-native participant ID.
- `timestamp_ms: int` — monotonic source timestamp.

**Output contract:**
- Pushes `RawMediaFrame(participant_id, audio_chunk, video_frame, timestamp_ms)` into an internal async queue.

**Suggested approach:**
- Start with a **platform-agnostic base class** `MediaSource` with methods `start()`, `stop()`, and an async generator `frames()`.
- Create `WebRTCSource(MediaSource)` as the first concrete implementation using `aiortc`.
- Create `FileSource(MediaSource)` for replaying recorded `.mp4`/`.wav` files with synthetic participant IDs — this enables CI/testing without a live meeting.

**Files to create/modify:**
- Create `sherlock/ingestion/__init__.py`.
- Create `sherlock/ingestion/base.py` with `RawMediaFrame` dataclass and `MediaSource` abstract class.
- Create `sherlock/ingestion/webrtc.py`.
- Create `sherlock/ingestion/file.py`.
- Add ingestion configuration to `sherlock/models.py` if needed.

**Acceptance criteria:**
- `python -m sherlock.tests.test_ingestion` passes.
- `FileSource` can replay a sample video and emit at least 25 `RawMediaFrame` objects per second.
- `WebRTCSource` can connect to a test peer and emit frames without blocking the event loop.

---

#### Prompt 10.3.2 — Candidate Stream Gate

**Goal:** After the identity engine has selected the candidate with sufficient confidence, only that participant's frames are forwarded to the fraud-detection pipelines.

**Input contract:**
- `candidate_id: str` (from `IdentificationResult.top_candidate_id`).
- `RawMediaFrame` stream from the ingestion adapter.

**Output contract:**
- `CandidateMediaFrame(candidate_id, audio_chunk, video_frame, timestamp_ms)` only when `frame.participant_id == candidate_id`.
- Logs a metric `dropped_frames_non_candidate` per second.

**Rules:**
- If `IdentificationResult.status != "identified"`, drop all frames and emit a `WARNING` flag: `identity_uncertain`.
- If the candidate changes mid-call, the gate must switch within 2 seconds and emit an `INFO` event.
- Never route interviewer or observer frames to fraud detectors.

**Files to create/modify:**
- Create `sherlock/gate.py` with class `CandidateStreamGate`.
- Modify `sherlock/fusion.py` to expose a synchronous `get_result()` callable from the gate.

**Acceptance criteria:**
- Unit test proves non-candidate frames are dropped.
- Unit test proves gate switches when `top_candidate_id` changes.
- Unit test proves a `WARNING` flag is raised when identity is ambiguous.

---

#### Prompt 10.3.3 — Deepfake Detection Pipeline

**Goal:** Analyze candidate video frames for synthetic/manipulated faces.

**Input contract:**
- `CandidateMediaFrame.video_frame: np.ndarray`.

**Output contract:**
- `EvidencePacket` with:
  - `source = SignalSource.DEEPFAKE_VIDEO` (add to `SignalSource` enum).
  - `axis = SignalAxis.AUTHENTICITY`.
  - `delta_log_odds < 0` when manipulation is detected.
  - `severity` in `{NONE, WARNING, CRITICAL}`.
  - `confidence` calibrated to approximate P(manipulated).

**Suggested approach:**
- Use a **face-anti-spoofing + liveness model** as the first signal, not a generative-AI classifier. Suggested libraries:
  - `face-recognition` or `insightface` for face detection/landmarks.
  - A lightweight model such as **Silent-Face-Anti-Spoofing** or a custom MobileNet-based liveness classifier.
- Add a **lip-sync consistency** check by comparing audio phonemes (from Prompt 10.3.5) with lip movement from video frames.
- Add a **3D face-mesh stability** check using MediaPipe Face Mesh to detect unnatural rigid/texture artifacts.

**Files to create/modify:**
- Add `DEEPFAKE_VIDEO` to `SignalSource` enum in `sherlock/models.py`.
- Create `sherlock/pipelines/__init__.py`.
- Create `sherlock/pipelines/deepfake.py` with class `DeepfakeVideoPipeline`.
- Add model download/cache utilities in `sherlock/pipelines/model_cache.py`.

**Acceptance criteria:**
- Pipeline processes a 1-minute video in under 30 seconds on CPU, or under 5 seconds on GPU.
- Unit test with a real video returns `severity=NONE`.
- Unit test with a printed-photo replay attack returns `severity=WARNING` or higher.
- Output is an `EvidencePacket` compatible with `FusionEngine.ingest()`.

---

#### Prompt 10.3.4 — Voice-Cloning / Audio-Liveness Pipeline

**Goal:** Analyze candidate audio for cloned, replayed, or synthesized speech.

**Input contract:**
- `CandidateMediaFrame.audio_chunk: bytes` — 16-bit PCM, 16 kHz, mono.

**Output contract:**
- `EvidencePacket` with:
  - `source = SignalSource.VOICE_LIVENESS` (add to enum).
  - `axis = SignalAxis.AUTHENTICITY`.
  - `delta_log_odds < 0` when synthetic/replay is suspected.
  - `severity` in `{NONE, WARNING, CRITICAL}`.

**Suggested approach:**
- Compute a **speaker embedding** over a sliding window (e.g., using `speechbrain` or `resemblyzer`).
- Compare the current window embedding against an enrollment sample if available, or against the candidate's own earlier-call baseline.
- Add a **replay-detection** signal: detect periodic artifacts and channel echoes using spectral features.
- Add a **TTS/cloning detector**: train or fine-tune a small classifier on embeddings to detect unnatural latent-space artifacts. If no training data is available, start with an open-source detector such as **RawNet2** or **AASIST** anti-spoofing models.

**Files to create/modify:**
- Add `VOICE_LIVENESS` to `SignalSource` enum in `sherlock/models.py`.
- Create `sherlock/pipelines/voice_liveness.py` with class `VoiceLivenessPipeline`.
- Create `sherlock/pipelines/speaker_store.py` to hold per-candidate baseline embeddings.

**Acceptance criteria:**
- Pipeline processes 1 second of audio in < 100 ms.
- Unit test with live human speech returns `severity=NONE`.
- Unit test with a replayed recording returns `severity=WARNING` or higher.
- Embedding drift across the same speaker's own session stays below a tunable threshold.

---

#### Prompt 10.3.5 — Real-Time Transcription + Diarization Pipeline

**Goal:** Replace fixture transcripts with live, speaker-attributed transcript segments.

**Input contract:**
- All participants' audio streams.

**Output contract:**
- `TranscriptSegment(participant_id, text, start_time, end_time, is_question)` added to the existing `BehavioralSignalExtractor` and `SemanticSignalExtractor`.

**Suggested approach:**
- Use **Whisper** (`openai-whisper`) or **faster-whisper** for transcription.
- Use a lightweight **speaker diarization** model (e.g., `pyannote.audio` speaker segmentation) to attribute text to participant IDs.
- Map diarization speaker labels to platform `participant_id`s by correlating timestamps with join events and current speaker detection.

**Files to create/modify:**
- Create `sherlock/pipelines/transcription_live.py` with class `LiveTranscriptionPipeline`.
- Modify `sherlock/transcription.py` to wrap or delegate to the live pipeline.
- Integrate with `BehavioralSignalExtractor` and `SemanticSignalExtractor`.

**Acceptance criteria:**
- Live transcript has word-error-rate < 15% on clean audio.
- Diarization accuracy > 80% for 2–4 participants.
- `SemanticSignalExtractor` can run without fixtures when this pipeline is enabled.

---

#### Prompt 10.3.6 — Behavioral Computer-Vision Pipeline

**Goal:** Detect off-screen reading, unnatural stillness, and other behavioral authenticity signals from actual webcam frames.

**Input contract:**
- `CandidateMediaFrame.video_frame: np.ndarray`.

**Output contract:**
- `EvidencePacket` with `source` in `{GAZE_DETECTION, ...}` and `axis = SignalAxis.AUTHENTICITY`.
- Reuse the existing `GazeEvent` model but populate it from real CV output.

**Suggested approach:**
- Use **MediaPipe Face Mesh** for 3D face landmarks and head pose.
- Use **L2CS-Net** or a similar lightweight gaze estimator for gaze vector.
- Detect the **periodic horizontal saccade pattern** described in §4.4d: look for rhythmic left-right gaze to a fixed off-camera point, not just "eyes away from camera".
- Add an **unnatural stillness** signal: very low head-pose variance over long intervals can indicate a static image or looped video.

**Files to create/modify:**
- Create `sherlock/pipelines/gaze_cv.py` with class `GazeBehavioralPipeline`.
- Modify `sherlock/signals/authenticity.py` `extract_gaze_detection()` to accept real `GazeEvent` objects produced by this pipeline.

**Acceptance criteria:**
- Gaze vector can be estimated at ≥ 15 FPS on CPU.
- Unit test with a person reading from a side monitor triggers `GAZE_DETECTION` flag.
- Unit test with natural conversation does not trigger false flag.

---

#### Prompt 10.3.7 — Streaming Inference Orchestrator

**Goal:** Run all pipelines concurrently with backpressure, batching, and latency budgets.

**Input contract:**
- `CandidateMediaFrame` stream.

**Output contract:**
- Ordered `List[EvidencePacket]` into `FusionEngine.ingest_batch()`.

**Requirements:**
- Use `asyncio` with bounded queues.
- Each pipeline has a max queue depth; frames older than 1 second are dropped rather than processed late.
- Audio pipelines operate on 1-second windows.
- Video pipelines operate on single frames or 1-second batches depending on model needs.
- Log per-pipeline latency histograms and dropped-frame counts.

**Files to create/modify:**
- Create `sherlock/orchestrator.py` with class `RealtimeInferenceOrchestrator`.
- Modify `app.py` to optionally start the orchestrator instead of replaying fixtures.

**Acceptance criteria:**
- End-to-end latency from frame arrival to evidence ingestion is < 500 ms p95.
- Orchestrator gracefully handles one pipeline crashing without killing others.
- Unit test simulates a burst of frames and verifies no unbounded memory growth.

---

### 10.4 Data-Model Additions

Add the following to `sherlock/models.py`:

```python
class SignalSource(str, Enum):
    # ... existing sources ...
    DEEPFAKE_VIDEO = "deepfake_video"
    VOICE_LIVENESS = "voice_liveness"

@dataclass
class RawMediaFrame:
    participant_id: str
    audio_chunk: Optional[bytes] = None
    video_frame: Optional[Any] = None  # numpy array
    timestamp_ms: int = 0

@dataclass
class CandidateMediaFrame(RawMediaFrame):
    candidate_id: str
```

### 10.5 Calibration & Testing Plan

**Goal:** Ensure reported confidence is calibrated and pipelines do not produce false positives on legitimate candidates.

**Tasks:**
1. Build a test harness `sherlock/tests/test_pipelines.py` with fixtures for:
   - Real human interview recordings (candidate only).
   - Printed-photo attack videos.
   - Replay audio attacks.
   - Side-monitor reading scenarios.
2. Add a `CalibrationDataset` loader that reads labeled sessions and computes:
   - Pipeline precision/recall per signal.
   - Calibration curve: does a 70% authenticity confidence correspond to ~70% fraud rate?
3. Extend `sherlock/feedback.py` so operator corrections on A/V flags also recalibrate pipeline thresholds.

**Acceptance criteria:**
- Deepfake pipeline: < 5% false-positive rate on real videos.
- Voice-liveness pipeline: < 5% false-positive rate on live speech.
- Gaze pipeline: < 10% false-positive rate on natural conversation.
- Overall system calibration error (ECE) < 0.10 on the test set.

### 10.6 Integration Sequence

Implement in this order to keep the system testable at each step:

1. **Ingestion adapters + FileSource** — enables CI and local development.
2. **Candidate stream gate** — enforces the privacy rule: only candidate frames analyzed.
3. **Live transcription + diarization** — improves identity signals with real data.
4. **Behavioral CV pipeline** — replaces simulated gaze events.
5. **Voice-liveness pipeline** — adds anti-replay/anti-cloning.
6. **Deepfake video pipeline** — adds face manipulation detection.
7. **Orchestrator + operator dashboard integration** — ties everything together.
8. **Calibration harness + feedback-loop extensions** — makes confidence trustworthy.

### 10.7 Frontend Video Analysis Integration

The operator dashboard now exposes a dedicated **Live Video Analysis** flow:

1. **Input options on the landing page:**
   - Upload a local video file (`/api/live/upload`).
   - Paste a YouTube URL (downloaded server-side via `yt-dlp`).
   - Provide an absolute local file path.
2. **Start analysis** (`/api/live/start`) launches the `LiveSession` / `RealtimeInferenceOrchestrator` and returns a playable video URL (`/api/live/video`).
3. **In-dashboard video player:** an HTML5 `<video>` tile appears at the top of the dashboard.
4. **Real-time scores under the video:** the existing `/ws/live` WebSocket now emits dashboard-compatible snapshots every second. The frontend displays:
   - Identity confidence with a coloured progress bar.
   - Authenticity confidence with a coloured progress bar.
   - **Genuine / Suspicious / Likely Cheating verdict** with the latest reasons.
   - Elapsed / video time.
5. **Flagged speech segments:** transcript lines that triggered AI-generated-text, reading-pattern, unnatural-pause, or AI-generated-speech signals are surfaced directly under the video with severity and rationale.
6. **Full dashboard reuse:** the live snapshot is serialized into the same shape as replay snapshots, so the scoreboard, event feed, flags, evidence room, timeline, and candidate-info tabs all work without modification.

### 10.7.1 Advanced Cheating Detection Pipeline

To answer "is the candidate reading AI-generated text aloud?" accurately on **real** videos, the live path now uses open-source ML models and conservative fusion:

- **`AI_GENERATED_TEXT`** (`sherlock/pipelines/text_authenticity.py` + `sherlock/pipelines/ai_text_detector.py`):
  - Uses an open-source transformer model (`roberta-base-openai-detector`) to score P(text is AI-generated).
  - Filters out common interview pleasantries ("thank you", "Do you have any questions?", etc.).
  - Calibrates against the candidate's own baseline so articulate people are not penalised.
  - Only emits a flag when the model confidence is high (≥0.75) and the answer is not generic/low-relevance.
  - Heavy inference runs in a background thread pool so the real-time loop is not blocked.
- **`READING_PATTERN`** / generic-scripted-answer (`sherlock/pipelines/text_authenticity.py`): flags answers that rely on buzzwords without concrete examples.
- **`UNNATURAL_PAUSE`** (`sherlock/pipelines/text_authenticity.py`): flags a long pause followed by a structurally fluent answer.
- **`AI_GENERATED_SPEECH`** (`sherlock/pipelines/audio_authenticity.py`): open-source audio anti-spoofing (SpeechBrain AASIST when available, heuristic spectral/prosodic fallback).  Detects synthetic/TTS/cloned speech directly from the audio stream.
- **Q/A relevance** (`sentence-transformers` `all-MiniLM-L6-v2`): penalises answers that are semantically unrelated to the preceding interview question.

All signals are ingested into the existing `FusionEngine` on the `AUTHENTICITY` axis.  The final verdict is driven by the fused authenticity probability, not any single flag.  The first analysis is slower because the open-source models must download/load; the frontend shows a processing banner during this warm-up.

### 10.8 Definition of Done

Sherlock can be considered production-ready for A/V fraud detection when:

- [ ] It ingests live per-participant A/V from at least one platform (WebRTC or file replay).
- [ ] It locks onto the candidate using the existing identity engine.
- [ ] Only candidate frames are analyzed by fraud detectors.
- [ ] Deepfake, voice-liveness, gaze, and live transcription pipelines emit `EvidencePacket`s into the existing ledger.
- [ ] End-to-end latency is < 500 ms p95.
- [ ] False-positive rates on legitimate candidates meet the thresholds in §10.5.
- [x] Operator dashboard shows live candidate A/V tile with active authenticity flags.
- [ ] All new code has unit tests and runs in CI.

