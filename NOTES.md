# NOTES — Implementation Assumptions & Interface Analysis

## Task: Real Speech Transcription + LLM Role-Classification

---

## Interface Analysis (Pre-Implementation Review)

### EvidencePacket Schema (from `models.py:69-78`)
```python
@dataclass
class EvidencePacket:
    source: SignalSource          # Enum: must use LLM_ROLE_CLASSIFIER
    axis: SignalAxis              # Enum: must be IDENTITY for role classification
    target_participant_id: str    # Participant ID string
    delta_log_odds: float         # Signed magnitude (positive = supports candidate)
    confidence: float             # 0.0 to 1.0
    rationale: str                # One-line explanation
    timestamp: datetime           # When evidence was produced
    metadata: Dict[str, Any]      # Optional extra data
    evidence_id: str              # Auto-generated UUID
```

**Status: ✅ No mismatches found.** The existing schema supports all required fields.

### TranscriptSegment Schema (from `models.py:153-158`)
```python
@dataclass
class TranscriptSegment:
    participant_id: str           # Speaker ID
    text: str                     # Transcribed text
    start_time: datetime          # Segment start
    end_time: datetime            # Segment end
    is_question: bool             # Whether segment is a question
```

**Status: ✅ No mismatches found.** The schema is compatible with Whisper output.

### SignalSource Enum (from `models.py:11-25`)
- Existing value: `LLM_ROLE_CLASSIFIER = "llm_role_classifier"`
- **Status: ✅ Already exists.** No new enum values needed.

### Fusion Engine Integration (from `fusion.py:42-56`)
- Default weight for `("llm_role_classifier", "identity")`: **1.5**
- **Status: ✅ Compatible.** The fusion engine already has a weight entry for this signal source.

---

## Assumptions Made

### 1. Audio Stream Architecture
**Assumption:** Per AGENT.md §3, the ingestion layer provides "per-participant audio/video streams" separately. Therefore, diarization is not needed — each audio stream is already speaker-separated.

**Implementation:** The `TranscriptionPipeline` transcribes each participant's audio stream independently. No diarization model (pyannote.audio) is required for the base case.

**Fallback:** If streams are NOT pre-separated (e.g., mixed audio from a single microphone), the system would need full diarization. This is documented but not implemented, as the spec explicitly states streams are separate.

### 2. Whisper Model Selection
**Assumption:** The `faster-whisper` library is preferred for performance, with `openai-whisper` as fallback.

**Implementation:** `TranscriptionPipeline` attempts to import `faster_whisper` first, falls back to `whisper` if unavailable.

**Default config:**
- Model size: `base` (good balance of speed/accuracy)
- VAD filter: enabled (reduces hallucinations on silence)
- Word timestamps: enabled when `preserve_disfluencies=True`

### 3. Disfluency Preservation
**Assumption:** Whisper's default text normalization may strip filler words (um, uh, like, etc.) which are critical for the authenticity signal (§4.4a in AGENT.md).

**Implementation:**
- `preserve_disfluencies=True` in config (default)
- `condition_on_previous_text=False` to prevent Whisper from "correcting" disfluencies based on context
- No `no_speech_threshold` manipulation that might filter out short filler utterances

**Risk:** Whisper may still normalize some disfluencies. Production testing with real interview audio is needed to validate.

### 4. LLM Batching Strategy
**Assumption:** Calling the LLM per-utterance is too expensive and rate-limited. Batching by time window is more efficient.

**Implementation:**
- Default batch window: **30 seconds** (`BATCH_WINDOW_SECONDS = 30`)
- Segments are grouped by participant, then by time window
- Each batch produces one evidence packet per participant

**Trade-off:** 30s windows balance cost vs. responsiveness. Shorter windows = more LLM calls but faster belief updates. Longer windows = fewer calls but slower adaptation to role changes.

### 5. Logit Conversion Formula
**Assumption:** The LLM's confidence (0-1) must be converted to log-odds for the fusion engine. The conversion must be:
- Symmetric (candidate vs. interviewer produce equal but opposite log-odds)
- Bounded (prevent extreme values from dominating)
- Monotonic (higher confidence = stronger signal)

**Implementation:**
```python
def confidence_to_log_odds(confidence: float, role: str) -> float:
    # Clamp to [0.5, 0.95] to prevent extreme values
    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
    
    # Standard logit transform
    log_odds = math.log(confidence / (1 - confidence))
    
    # Negate for interviewer role
    if role == "interviewer":
        log_odds = -log_odds
    
    return log_odds
```

**Bounds:**
- `MIN_CONFIDENCE = 0.5` — below this, the signal is too weak to emit
- `MAX_CONFIDENCE = 0.95` — prevents log-odds from exploding (0.95 → +2.94, 0.99 → +4.59)

**Tested:** See `test_semantic.py` for comprehensive tests of this function.

### 6. LLM Structured Output
**Assumption:** OpenRouter's tool-use API (OpenAI-compatible) provides reliable structured output without fragile regex parsing.

**Implementation:**
- Uses OpenRouter's `tools` parameter with a JSON schema (OpenAI-compatible format)
- `tool_choice={"type": "function", "function": {"name": "role_classification"}}` forces tool use
- Parses the `arguments` field of the tool call as JSON

**Model:** `meta-llama/llama-3.1-8b-instruct:free` (free tier on OpenRouter)

**Fallback:** If structured output fails (e.g., API changes), the system logs the error and skips the batch. No fallback to free-text parsing (per constraint: "do not silently swallow errors").

### 7. Error Handling
**Assumption:** LLM API calls can fail (rate limits, timeouts, network errors). The system must not fabricate confidence values.

**Implementation:**
- All exceptions are caught and logged via `logger.error()`
- Failed batches increment `_failed_call_count` for monitoring
- No evidence packet is emitted for failed batches
- `get_failure_stats()` exposes failure metrics

**Constraint compliance:** "Do not silently swallow LLM API errors" — errors are logged, not suppressed.

### 8. Disable Flag
**Assumption:** The LLM signal should be optional for testing and graceful degradation.

**Implementation:**
- `SemanticSignalExtractor(enabled=False)` disables all LLM calls
- When disabled, `extract_all()` returns an empty list immediately
- Logs "LLM signal disabled via config" at INFO level

**Use case:** Demonstrating that the system works without LLM (rule-based-only mode), which is explicitly a bonus point in the challenge.

---

## Interface Mismatches Found

### None
All interfaces matched the existing schema. No modifications to `models.py`, `fusion.py`, or other extractors were required.

---

## Dependencies Added

```
faster-whisper>=1.0.0       # Preferred Whisper implementation
openai>=1.0.0               # OpenAI-compatible client for OpenRouter
soundfile>=0.12.0           # Audio file I/O (for buffer transcription)
numpy>=1.24.0               # Audio array handling
```

**Note:** `openai-whisper` is supported as a fallback if `faster-whisper` is not installed.

**LLM Provider:** OpenRouter (https://openrouter.ai) with free-tier model `meta-llama/llama-3.1-8b-instruct:free`. API key configured via `.env` file (`OPENROUTER_API_KEY`).

---

## Testing Strategy

### Unit Tests (`test_semantic.py`)
- **Logit conversion:** 6 tests covering edge cases, bounds, symmetry
- **Evidence packet schema:** 1 test validating all fields
- **Semantic extractor:** 11 tests covering:
  - Disabled mode
  - No LLM client
  - Short segment filtering
  - Batching behavior
  - Low confidence filtering
  - Invalid role filtering
  - API error handling
  - Evidence packet values
  - Interviewer classification
  - Multiple participants

### Mock Strategy
- `MockLLMClient` provides deterministic responses for testing
- No real API calls in test suite (per constraint)
- `FailingLLMClient` in tests simulates API errors

---

## Future Considerations

### 1. Streaming Transcription
Current implementation transcribes complete audio files. For real-time operation, consider:
- Chunked transcription (e.g., 5-second windows)
- Incremental transcript updates
- Partial segment emission

### 2. Diarization Fallback
If audio streams are NOT pre-separated, add:
- `pyannote.audio` integration for speaker diarization
- Speaker embedding clustering
- Alignment of diarized segments to participant IDs

### 3. LLM Prompt Optimization
Current prompts are functional but not optimized. Consider:
- Few-shot examples for edge cases
- Calibration of confidence thresholds
- A/B testing of prompt variations

### 4. Cost Monitoring
LLM calls cost money. Add:
- Token usage tracking
- Cost estimation per meeting
- Budget limits with graceful degradation

---

## Compliance Checklist

- ✅ Does not modify fusion engine core logic
- ✅ Does not implement gaze detection, code telemetry, or voice embedding
- ✅ Does not silently swallow LLM API errors
- ✅ Config flag to disable LLM signal
- ✅ Produces correctly-formatted evidence packets
- ✅ Uses structured output (tool calling), not free-text parsing
- ✅ Batches segments (30s windows)
- ✅ Logit conversion is a pure, testable function
- ✅ LLM signal remains a weak vote, never final verdict
- ✅ Unit tests for logit conversion and schema validity
- ✅ No real API calls in test suite
