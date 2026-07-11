import subprocess
import sys
from pathlib import Path

# Target directory
test_videos_dir = Path(__file__).parent
test_videos_dir.mkdir(exist_ok=True)

# High quality mock interview video URLs and local filenames
videos = {
    "https://www.youtube.com/watch?v=XZzt21dWyJw": "google_mock_interview.mp4",
    "https://www.youtube.com/watch?v=1t1_a1bz0_Y": "student_mock_interview.mp4",
    "https://www.youtube.com/watch?v=kFehFC72YRY": "behavioral_mock_interview.mp4"
}

print("Starting download of mock student/candidate interview videos...")
for url, filename in videos.items():
    dest_path = test_videos_dir / filename
    print(f"\nDownloading {url} -> {dest_path}...")
    
    # We download in the lowest quality MP4 to keep file size small and fast
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "worst[ext=mp4]",
        "-o", str(dest_path),
        url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded: {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {filename}: {e}", file=sys.stderr)

print("\nAll downloads complete!")
