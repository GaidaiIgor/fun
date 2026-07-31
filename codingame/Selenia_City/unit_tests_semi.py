"""Checks semi planner decisions against minimum-score turn-state regressions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
import Selenia_City.semi as semi
from Selenia_City.test_semi import parse_turn_state

semi.FULL_DEBUG = False

MONTH_1_STATE = """
month 1
resources 5000
module 0 1 20 15
module 1 2 140 15
landing 2 40 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
landing 3 80 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
landing 4 120 45 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
module 5 2 20 75
module 6 1 140 75
"""


class PlannerScoreTests(unittest.TestCase):
    """Checks planner scores for preserved month states."""
    def assert_turn_score(self, turn_state: str, minimum_score: int):
        """Checks the concrete command for turn_state against minimum_score."""
        planner = parse_turn_state(turn_state)
        actions = planner.choose_actions()
        command = ";".join(actions) if actions else "WAIT"
        score = planner.score_state(planner.override_state(command)).score
        self.assertGreaterEqual(score, minimum_score, f"score={score}, minimum={minimum_score}, command={command}")

    def test_month_1(self):
        """Checks the current month-one planner baseline."""
        self.assert_turn_score(MONTH_1_STATE, 10710)


if __name__ == "__main__":
    unittest.main()
