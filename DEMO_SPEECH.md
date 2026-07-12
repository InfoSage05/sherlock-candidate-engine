# Sherlock — Demo Speech Script (8–10 minutes)

---

## PART 1: INTRODUCTION & ARCHITECTURE (90 seconds)

"Hi, I'm going to walk you through **Sherlock**, a real-time candidate identification and authenticity engine built for live interviews on platforms like Google Meet, Zoom, and Microsoft Teams.

The core problem Sherlock solves is: **in a live interview, who is the actual candidate?** It sounds simple, but in practice candidates join with device names like 'MacBook Pro,' use nicknames, change their display name mid-call, or there are multiple interviewers and silent observers in the room. Traditional systems based on a single rule — 'the person whose name matches the calendar is the candidate' — break on all of these.

Sherlock takes a fundamentally different approach: **Bayesian multi-signal fusion.** Instead of one rule, we use 22 weak, independently unreliable signals. Each signal emits a small evidence packet — 'I think this person is the candidate because they're answering questions' — with a delta in log-odds space and a confidence score. The fusion engine continuously aggregates these into a running probability distribution over every participant. No single signal can dominate. The system is explainable: every confidence number traces back to the ordered list of evidence that produced it."

*(Switch to the architecture diagram in the README.)*

"Let me walk through the architecture quickly. On the left, the ingestion layer takes video files, YouTube URLs, or live WebRTC streams. A candidate stream gate routes only the candidate's frames to our fraud detection pipelines — deepfake video, voice liveness, gaze behavioral analysis, and audio authenticity — while transcription runs on everyone's audio using the open-source faster-whisper model.

The text authenticity pipeline analyzes what the candidate is saying: it runs an open-source transformer, roberta-base-openai-detector, on every answer to detect AI-generated or scripted responses. It also detects unnatural pauses before polished answers, and — importantly — it emits positive evidence when answers appear spontaneous and genuine, so the authenticity score rises for real candidates instead of staying flat.

All 22 signal sources feed evidence packets into a Bayesian fusion engine that maintains per-participant running log-odds for identity and authenticity, with a 120-second time-decay half-life so stale evidence doesn't dominate indefinitely.

Finally, the presentation layer: a FastAPI backend streams snapshots via WebSocket at about 3 times per second to a vanilla JavaScript SPA that renders the live dashboard with full explainability."

---

## PART 2: LIVE DEMO — UPLOAD & ANALYZE (3 minutes)

"Let me show you this in action. I'm going to upload an interview video and watch Sherlock analyze it live."

*(Open http://localhost:8000 in browser.)*

"You'll notice right away that this is not just a video player — there's a full dashboard underneath."

*(Point to upload section.)*

"Right here, I can upload a video file, paste a YouTube link, or enter a local file path. Let me upload one of our test interview videos."

*(Upload interview_2min.mp4.)*

"As soon as I hit Start Analysis, Sherlock begins ingesting the video — it extracts the audio at 16 kHz mono, feeds it to faster-whisper for real-time transcription, and starts running all pipelines in parallel.

Watch these score bars under the video. **This is updating in real time.** The identity bar has already jumped up because the system seeded itself with the calendar match — it knows who the candidate is supposed to be. But watch the authenticity bar — it's rising as the candidate gives genuine, spontaneous answers."

*(Wait ~20 seconds for scores to accumulate.)*

"See how the identity score is at 99.9%? And authenticity is at 95%? These aren't static numbers — they're continuously updating. Every time the candidate speaks, a small speech-activity signal bumps identity up. Every time they give a specific, personal answer, the text authenticity pipeline emits positive evidence.

Now, let me show you why this matters — the **explainability panel**."

*(Scroll down to the explainability panel.)*

"This is the 'Why This Candidate?' panel. It shows you exactly WHY Sherlock reached its conclusions.

First, the verdict cards: Identity confidence is HIGH at 99.9%. Authenticity is HIGH at 95%. Zero active flags. If anything were wrong, this would change — the levels would drop to MODERATE, LOW, or VERY LOW, and an ambiguity warning would appear.

Below that, the **pipeline status grid**. Seven active pipelines. You can see which ones are producing evidence — right now, text authenticity has produced 9 evidence packets, live transcription has generated 18 segments, and behavioral signals have output 18 packets. The deepfake and voice liveness pipelines are active but haven't detected anything — which is correct for a genuine video.

The **evidence breakdown by source** shows every signal that contributed. Calendar match contributed +3.0 to identity. Human spontaneous text contributed +2.98 to authenticity — that's the positive evidence from genuine answers. Turn-taking contributed +1.44 to identity — because the candidate is answering questions.

And the **top contributing signals** — ranked by impact, with the delta and rationale for each. You can trace every probability number back to the evidence that created it."

*(Switch to the Evidence Room tab.)*

"If I go to the Evidence Room, I can see all 50 most recent evidence packets. I can filter by category — Identity Priors, Behavioral, Semantic LLM, Authenticity. I can search the rationale. I can export the full ledger as JSON. Every piece of evidence has a source, an axis, a delta log-odds, a confidence score, a timestamp, and a human-readable rationale."

*(Switch to the Timeline tab.)*

"The Transcript tab shows every speaker-attributed segment from faster-whisper. If any segments were flagged for AI-generated text or unnatural pauses, they'd be highlighted in red with the flag reason shown."

---

## PART 3: EDGE CASE HANDLING (2 minutes)

"Now let me show you how Sherlock handles the toughest edge cases. These are the exact scenarios from the challenge."

*(Stop the current session. Start a cheating_interview.mp4 video.)*

"Let me start a different video — this one contains scripted, AI-assisted answers."

*(Watch scores.)*

"Notice the difference. The authenticity score is still updating, but it's lower. If the AI text detector finds generated content, the score drops. If there are unnatural pauses before polished answers, that's flagged. The verdict card changes from Genuine to Suspicious or even Likely Cheating."

*(Switch to fixture replay mode.)*

"Let me also show you how Sherlock handles the replay scenarios. I can load any of the 7 edge-case fixtures and step through them in real time."

*(Load the "Device Name Join" scenario.)*

"Here, the candidate joined as 'MacBook Pro' instead of their real name. Watch what happens — the system doesn't care about display names. It uses calendar matching, email domains, turn-taking, and speaking ratio to identify the candidate correctly even though the name is wrong."

*(Load the "Multiple Interviewers" scenario.)*

"In this one, there are two interviewers and a candidate. The candidate is the one receiving questions and giving long answers. The interviewers ask the questions. Turn-taking detects this pattern. The candidate's identity score rises. The interviewers' scores stay low."

---

## PART 4: TRADE-OFFS & FUTURE WORK (90 seconds)

"Let me be honest about the trade-offs.

First, **model availability**. We use open-source models exclusively — faster-whisper for transcription, roberta-base for AI text detection, SpeechBrain for audio anti-spoofing. This means zero API costs and offline capability, but some models — like sentence-transformers for semantic Q/A relevance — require system libraries that aren't always available. The system gracefully degrades when a model can't load.

Second, **speed vs. accuracy**. We run all models on CPU. GPU would be faster, but CPU ensures portability. Transcription runs in ~2-second chunks, which means there's a slight latency between when someone speaks and when you see the transcript. Heavy inference — like the AI text detector — runs in thread pools so it never blocks the real-time audio ingestion.

Third, **single-participant file mode**. When you upload a video, it's treated as a single stream because there's no speaker diarization. In a real meeting with multiple participants, you'd need pyannote or the platform's own speaker attribution. The architecture supports this — the gate already routes per-participant frames — but the file source is simplified.

What would I improve next? Speaker diarization would be top priority. Async YouTube downloads so the UI doesn't block. GPU pipeline execution for deepfake and voice liveness. And a persistent evidence store so you can revisit past interviews.

But what you're seeing right now is a working, end-to-end system: video upload to live transcription to multi-signal fusion to a real dashboard with full explainability. Every decision is traceable. Every score is backed by evidence. Thank you."

---

## QUICK REFERENCE: THINGS TO POINT AT

| Feature | Where to find it |
|---|---|
| Live video player with score bars | Top of dashboard, below the video |
| "Why This Candidate?" panel | Below video scores |
| Pipeline status badges | Explainability panel, second row |
| Evidence breakdown by source | Explainability panel, table |
| Top contributing signals | Explainability panel, ranked list |
| Ambiguity warning | Yellow banner in explainability panel |
| Verdict card | Below score bars (Genuine/Suspicious/Likely Cheating) |
| Flagged segments | Below verdict card (red cards per segment) |
| Evidence Room | Tab "📜 Evidence Room" |
| Transcript | Tab "⏱ Timeline" |
| Upload section | Landing page, prominent section with file/YouTube/path inputs |
| New Analysis button | Top bar, right side (🎥 New Analysis) |
| Scenario replay | Landing page scenario grid, or top bar dropdown |
