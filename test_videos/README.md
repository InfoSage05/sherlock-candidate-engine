# Test Videos for Sherlock Pipeline

To test the multi-signal Bayesian engine with real human candidates (rather than synthetic patterns or random video files like Big Buck Bunny), use the links and scripts below to populate this folder.

---

## Selected Mock Interview Videos

These YouTube videos represent realistic software engineering technical and behavioral interviews for students:

1. **Google Coding Interview with a College Student**
   * **URL**: [https://www.youtube.com/watch?v=1t1_a1bz0_Y](https://www.youtube.com/watch?v=1t1_a1bz0_Y)
   * **Description**: A full 45-minute mock technical interview conducted by Clément Mihailescu (ex-Google) with a college student candidate. Excellent for testing speaking ratios, turn-taking graphs, and coding cadence.

2. **How to: Work at Google — Example Coding/Engineering Interview**
   * **URL**: [https://www.youtube.com/watch?v=XZzt21dWyJw](https://www.youtube.com/watch?v=XZzt21dWyJw)
   * **Description**: Official mock coding interview video from the *Life at Google* channel demonstrating Google engineers conducting a technical assessment. Very clean audio and high-fidelity video streams.

3. **Mock Coding Interview with a Google Engineer** (Exponent)
   * **URL**: [https://www.youtube.com/watch?v=d_kXpewAOH0](https://www.youtube.com/watch?v=d_kXpewAOH0)
   * **Description**: A realistic, detailed software engineering coding interview from the Exponent channel.

4. **Behavioral Mock Interview (Jobs/Internships)**
   * **URL**: [https://www.youtube.com/watch?v=Feg16qXV9Dj](https://www.youtube.com/watch?v=Feg16qXV9Dj)
   * **Description**: A walkthrough of common behavioral interview questions with structured student answers, perfect for testing response pause latencies and disfluency anomalies.

---

## How to Download These Videos

Because the AI IDE environment runs inside a sandboxed network that blocks direct connections to external media servers like YouTube, you should run the download script from your **local machine's terminal** (outside the IDE sandbox).

### Instructions

1. Open a terminal on your host machine.
2. Navigate to this project folder:
   ```bash
   cd "e:\Machine Learning\ML Assignment"
   ```
3. Run the download script:
   ```bash
   python test_videos/download_interviews.py
   ```

The script will automatically install `yt-dlp` (if not already present) and download low-resolution, lightweight versions of these interviews directly into this `test_videos/` directory for testing!
