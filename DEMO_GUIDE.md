# Sherlock Demo - Quick Start Guide

## What You Have

A complete, working prototype of the Sherlock Candidate Identification Engine with:

✅ **Bayesian Fusion Engine** - Multi-signal belief tracking  
✅ **14 Signal Extractors** - Identity + authenticity signals  
✅ **Real Transcription** - Whisper-based with disfluency preservation  
✅ **LLM Role Classification** - OpenRouter with Llama 3.1 70B  
✅ **Interactive Demo** - Streamlit operations console  
✅ **7 Edge-Case Scenarios** - All from AGENT.md §8  
✅ **29 Passing Tests** - Comprehensive test coverage  

## How to Run the Demo

### 1. Start the Streamlit App

```bash
streamlit run app.py
```

This opens the interactive operations console in your browser.

### 2. Select a Scenario

Choose from 7 pre-recorded edge cases:
- Normal Interview
- Device Name Join (candidate joins as "MacBook Pro")
- Multiple Interviewers + Silent Observer
- Nickname Join
- Wrong Name in Calendar
- Display Name Change Mid-Call
- Silent Observer

### 3. Watch the System Work

The demo shows:
- **Left Panel**: Participant tiles with live status + scrolling transcript
- **Center Panel**: Belief distribution bars + confidence gauge + authenticity gauge
- **Right Panel**: Evidence ledger + timeline scrubber + signal ablation toggles
- **Bottom Bar**: Feedback controls (confirm/correct candidate)

### 4. Interact with the System

- **Play/Pause/Reset**: Control playback speed (0.5x - 4x)
- **Timeline Scrubber**: Jump to any point in the interview
- **Signal Ablation**: Toggle signal categories on/off to see their impact
- **Explain This**: Click to see top evidence for each participant
- **Feedback**: Confirm or correct the candidate to recalibrate weights
- **Export**: Download the evidence ledger as JSON

## Demo Video Structure (5-10 minutes)

### Opening (1 min)
- Show the challenge document
- Explain the problem: "Which participant is the candidate?"
- Show the architecture diagram

### Live Demo (4-5 min)
1. **Normal Interview** (1 min)
   - Start playback
   - Show belief bars updating in real-time
   - Point out evidence ledger entries
   - Highlight confidence gauge reaching 99%+

2. **Device Name Join** (1.5 min)
   - Switch scenario
   - Show how system handles "MacBook Pro" as display name
   - Explain email matching + behavioral signals
   - Demonstrate robustness to name mismatches

3. **Multiple Interviewers** (1 min)
   - Show 4 participants
   - Highlight interviewer negative priors
   - Show system correctly identifying candidate among noise

4. **Signal Ablation** (1 min)
   - Toggle off "Semantic LLM" category
   - Show how system adapts with remaining signals
   - Demonstrate graceful degradation

### Edge Cases (1-2 min)
- Quickly show 2-3 more scenarios
- Emphasize "AMBIGUOUS" state when evidence is insufficient
- Show timeline scrubbing to review decisions

### Feedback Loop (1 min)
- Demonstrate interviewer correction
- Show weight recalibration indicator
- Explain continuous learning

### Closing (30 sec)
- Summarize key features
- Highlight all 6 bonus points achieved
- Mention test coverage and code quality

## Key Features to Highlight

### 1. Multiple Weak Signals
> "No single rule decides anything. The system combines 14 different signals, each contributing a small update to the belief distribution."

### 2. Confidence Scores
> "Every participant has a live probability score, updated in real-time as evidence accumulates. The system reports 99% confidence when the evidence is strong."

### 3. Explainability
> "Every confidence number is traceable. Click 'Explain this' to see the top 3 evidence items that influenced the decision, with full rationale."

### 4. Continuous Learning
> "When the interviewer confirms or corrects the system, signal weights are recalibrated. The system improves over time."

### 5. Real-Time Operation
> "The system processes signals as they arrive, with 30-second batching for LLM calls to control cost while maintaining responsiveness."

### 6. Graceful Uncertainty
> "When evidence is insufficient, the system explicitly reports 'AMBIGUOUS' instead of guessing. This is critical for production use."

## Technical Details

### Signal Categories

**Identity Priors** (4 signals):
- Calendar match (fuzzy name/email)
- Interviewer negative prior
- Email domain check
- Join timing

**Behavioral** (4 signals):
- Turn-taking graph
- Speaking ratio
- Screen share
- Webcam state

**Semantic LLM** (1 signal):
- Role classification (Llama 3.1 70B via OpenRouter)

**Authenticity** (4 signals):
- Disfluency anomaly
- Pause-fluency pattern
- Coding telemetry
- Gaze detection

### Architecture Highlights

- **Bayesian Fusion**: Log-odds accumulation with proper normalization
- **Separate Streams**: Identity and authenticity tracked independently
- **Evidence Ledger**: Immutable, append-only record of all signals
- **Feedback Loop**: Weight recalibration based on interviewer corrections
- **Ambiguity Detection**: Explicit "AMBIGUOUS" state when gap < 10%

### Test Coverage

- 12 fusion engine tests
- 17 semantic signal tests
- Session replay tests
- All edge cases covered

## Files to Show During Demo

1. **AGENT.md** - Original design document
2. **app.py** - Streamlit frontend (show the dark theme, 4-region layout)
3. **sherlock/fusion.py** - Bayesian fusion engine (show the log-odds update logic)
4. **sherlock/signals/semantic.py** - LLM integration (show structured output, batching)
5. **sherlock/fixtures/** - JSON fixtures for edge cases
6. **NOTES.md** - Implementation assumptions and interface analysis

## Troubleshooting

### "No fixtures found"
```bash
python -m sherlock.generate_fixtures
```

### "OPENROUTER_API_KEY not set"
Add your key to `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Streamlit won't start
```bash
# Kill any existing Streamlit processes
taskkill /F /IM streamlit.exe  # Windows
# or
pkill -f streamlit  # Linux/Mac

# Restart
streamlit run app.py
```

## Next Steps

1. **Record the demo video** following the structure above
2. **Test all scenarios** to ensure smooth playback
3. **Practice the walkthrough** to stay within 5-10 minutes
4. **Prepare for Q&A** - be ready to explain:
   - Why Bayesian fusion over rule-based?
   - How does the system handle missing data?
   - What are the limitations?
   - How would you scale this to production?

## Success Criteria

Your demo should demonstrate:
- ✅ System correctly identifies candidate in normal case
- ✅ System handles edge cases (device names, nicknames, etc.)
- ✅ Confidence scores are well-calibrated
- ✅ Evidence is explainable and traceable
- ✅ System reports ambiguity when appropriate
- ✅ Feedback loop works and recalibrates weights
- ✅ All 6 bonus points are clearly visible

Good luck with your demo!
