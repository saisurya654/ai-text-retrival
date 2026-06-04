from pathlib import Path

from feedback.edit_learning import FeedbackMemory


def test_feedback_memory_learns_terminology(tmp_path: Path):
    memory = FeedbackMemory(tmp_path / "history.json")
    memory.analyze_and_store("owner", "registered owner", "prefer precise term")
    updated = memory.apply_preferences("The owner is identified.")
    assert "registered owner" in updated
