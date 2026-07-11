# Test Videos for Sherlock Pipeline

## Real Interview Videos (Downloaded from YouTube)

### 1. interview_2min.mp4 (3.2 MB)
- **Duration**: 2:14
- **Content**: Job Interview practice
- **Source**: https://www.youtube.com/watch?v=ExJZAegsOis
- **Use case**: Test normal interview scenario

### 2. interview_esl_1min.mp4 (5.1 MB)
- **Duration**: 1:45
- **Content**: Job Interview: I Want to Learn (ESL)
- **Source**: https://www.youtube.com/watch?v=naIkpQ_cIt0
- **Use case**: Test with non-native English speaker

### 3. interview_dialogue_2min.mp4 (2.7 MB)
- **Duration**: 2:10
- **Content**: Interviewing for a Job – Everyday English Dialogues
- **Source**: https://www.youtube.com/watch?v=w0YQwglgtTM
- **Use case**: Test dialogue/conversation scenario

## Synthetic Videos (Generated)

### 4. synthetic_interview.mp4 (774 KB)
- **Duration**: 12 seconds
- **Content**: Animated face with moving mouth (simulating speaking)
- **Use case**: Test basic face detection and mouth movement

### 5. synthetic_looking_away.mp4 (527 KB)
- **Duration**: 12 seconds
- **Content**: Animated face positioned to the side (looking away)
- **Use case**: Test gaze detection pipeline

### 6. bbb.mp4 (771 KB)
- **Duration**: ~10 seconds
- **Content**: Big Buck Bunny animation clip
- **Source**: https://www.w3schools.com/html/mov_bbb.mp4
- **Use case**: Test with animated content (no real face)

## How to Test

1. **Start the Streamlit server**:
   ```bash
   python3 -m streamlit run app.py --server.headless true --server.port 8501
   ```

2. **Open browser**: http://localhost:8501

3. **Load a scenario** from the dropdown and click "Load Scenario"

4. **Use Live A/V Analysis**:
   - Expand the " Live A/V Analysis (experimental)" section in the sidebar
   - Enter the path to one of the videos:
     ```
     /mnt/e/Machine Learning/ML Assignment/test_videos/interview_2min.mp4
     ```
   - Click "▶ Start live"

5. **Watch the results**:
   - Confidence scores update in real-time
   - Active flags appear if anomalies are detected
   - Latency metrics show processing speed

## Expected Results

- **Confidence**: Should be 100% (system correctly identifies the candidate)
- **Flags**: May show 0 flags (real detectors need model downloads)
- **Latency**: Should be < 100ms per frame

## Pipeline Status

✅ **Architecture**: Complete and functional
✅ **Identity Resolution**: Working (Bayesian fusion)
✅ **Stream Gating**: Working (only candidate frames analyzed)
✅ **Video Processing**: Working (real videos processed)
️ **Deepfake Detection**: Heuristic only (insightface model needs download)
️ **Gaze Detection**: Heuristic only (MediaPipe API changed in v0.10+)
✅ **Voice Liveness**: Working (resemblyzer embeddings active)

## Next Steps

To enable full pipeline functionality:
1. Download insightface models (requires internet)
2. Update MediaPipe integration for v0.10+ API
3. Add more test videos with different scenarios

