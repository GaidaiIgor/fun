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
semi.OVERRIDE_MONTH = -1

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

MONTH_10_STATE = """
month 10
resources 5324
module 0 1 20 15
module 1 2 140 15
landing 2 40 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
landing 3 80 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
landing 4 120 45 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
module 5 2 20 75
module 6 1 140 75
module 7 3 10 45
landing 8 150 45 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
tube 0 2 1
tube 1 4 1
tube 2 3 1
tube 3 4 1
tube 3 5 1
tube 3 6 1
pod id=1, path=[2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 3, 6, 3, 5, 3, 6, 3, 2, 0, 2]
pod id=2, path=[3, 6, 3, 5, 3, 6, 3, 5, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1]
"""

MONTH_15_STATE = """
month 15
resources 50520
module 0 1 20 15
module 1 2 140 15
landing 2 40 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
landing 3 80 45 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
landing 4 120 45 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
module 5 2 20 75
module 6 1 140 75
module 7 3 10 45
landing 8 150 45 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
tube 0 2 1
tube 1 4 1
tube 2 3 1
tube 3 4 1
tube 3 5 1
tube 3 6 1
teleport 8 7
pod id=1, path=[2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 3, 6, 3, 5, 3, 6, 3, 2, 0, 2]
pod id=2, path=[3, 6, 3, 5, 3, 6, 3, 5, 3, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1]
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

    def test_month_10(self):
        """Checks the current month-ten planner baseline."""
        self.assert_turn_score(MONTH_10_STATE, 14485)

    def test_month_15(self):
        """Checks the current month-fifteen planner baseline."""
        self.assert_turn_score(MONTH_15_STATE, 15690)

    def test_month_15_surplus_auto_pod(self):
        """Checks final route materialization for a surplus AUTO pod."""
        planner = parse_turn_state(MONTH_15_STATE)
        command = semi.OVERRIDE_COMMAND
        semi.OVERRIDE_MONTH = 15
        semi.OVERRIDE_COMMAND = "POD 3 AUTO;POD 4 AUTO;POD 5 AUTO"
        try:
            actions = planner.choose_actions()
        finally:
            semi.OVERRIDE_MONTH = -1
            semi.OVERRIDE_COMMAND = command
        pod_5 = next(action for action in actions if action.startswith("POD 5 "))
        self.assertEqual(pod_5.split()[2:4], ["2", "0"])
        result = planner.score_state(planner.override_state(";".join(actions)))
        self.assertGreaterEqual(result.score, 15095)

if __name__ == "__main__":
    unittest.main()
