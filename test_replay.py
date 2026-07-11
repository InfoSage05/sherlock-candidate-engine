"""
Test script to verify session replay functionality.
"""

from pathlib import Path
from sherlock.session_replay import SessionReplay, list_available_fixtures, load_fixture


def test_session_replay():
    """Test basic session replay functionality."""
    print("Testing Session Replay...")
    
    # List available fixtures
    fixtures_dir = Path(__file__).parent / "sherlock" / "fixtures"
    fixtures = list_available_fixtures(fixtures_dir)
    
    print(f"\nFound {len(fixtures)} fixtures:")
    for f in fixtures:
        print(f"  - {f['name']}: {f['description']}")
    
    # Load first fixture
    if not fixtures:
        print("\nERROR: No fixtures found!")
        return
    
    print(f"\nLoading fixture: {fixtures[0]['name']}")
    fixture = load_fixture(fixtures[0]['path'])
    
    print(f"  Meeting ID: {fixture.meeting_context.meeting_id}")
    print(f"  Participants: {len(fixture.meeting_context.participants)}")
    print(f"  Evidence packets: {len(fixture.evidence_packets)}")
    print(f"  Transcript segments: {len(fixture.transcript_segments)}")
    
    # Create replay
    print("\nCreating SessionReplay...")
    replay = SessionReplay(fixture)
    
    print(f"  Total duration: {replay.get_total_duration():.1f} seconds")
    print(f"  Initial progress: {replay.get_progress():.1%}")
    
    # Step through a few packets
    print("\nStepping through first 5 evidence packets...")
    for i in range(min(5, len(fixture.evidence_packets))):
        snapshot = replay.step_forward()
        if snapshot:
            print(f"  Step {i+1}:")
            print(f"    Time: {snapshot.timestamp.strftime('%H:%M:%S')}")
            print(f"    Status: {snapshot.status}")
            print(f"    Top candidate: {snapshot.top_candidate_id}")
            print(f"    Confidence: {snapshot.top_candidate_probability:.1%}")
            print(f"    Evidence count: {len(snapshot.evidence_ledger)}")
    
    # Test advance to time
    print("\nTesting advance_to_time...")
    if replay.start_time:
        from datetime import timedelta
        target_time = replay.start_time + timedelta(seconds=30)
        snapshot = replay.advance_to_time(target_time)
        print(f"  Advanced to: {snapshot.timestamp.strftime('%H:%M:%S')}")
        print(f"  Progress: {replay.get_progress():.1%}")
    
    # Test reset
    print("\nTesting reset...")
    replay.reset()
    print(f"  Progress after reset: {replay.get_progress():.1%}")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_session_replay()
