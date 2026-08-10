from agzamov.local_collaborative_prompt import COLLABORATIVE_CHESS_SYSTEM_PROMPT


def test_collaborative_prompt_keeps_model_and_harness_as_competing_hypotheses():
    prompt = COLLABORATIVE_CHESS_SYSTEM_PROMPT.lower()

    assert "do not default to blaming" in prompt
    assert "model error" in prompt
    assert "harness error" in prompt
    assert "competing hypotheses" in prompt


def test_collaborative_prompt_requires_legality_self_check_before_move():
    prompt = COLLABORATIVE_CHESS_SYSTEM_PROMPT.lower()

    assert "side to move" in prompt
    assert "origin square" in prompt
    assert "movement geometry" in prompt
    assert "intervening squares" in prompt
    assert "own pieces" in prompt
    assert "destination occupancy" in prompt
    assert "king safety" in prompt
    assert "state_id" in prompt


def test_collaborative_prompt_requires_self_audit_before_technical_diagnosis():
    prompt = COLLABORATIVE_CHESS_SYSTEM_PROMPT.lower()

    assert "audit your own proposed move first" in prompt
    assert "only then" in prompt
    assert "concise self-report" in prompt
    assert "not ground truth" in prompt


def test_collaborative_prompt_requires_enemy_king_capture_and_mate_audit():
    prompt = COLLABORATIVE_CHESS_SYSTEM_PROMPT.lower()

    assert "destination is attacked by the enemy king" in prompt
    assert "queen is protected after the move" in prompt
    assert "enemy king can capture" in prompt
    assert "all legal enemy-king replies" in prompt
    assert "check or checkmate" in prompt
    assert "all eight adjacent squares" in prompt
    assert "exhaustive" in prompt
