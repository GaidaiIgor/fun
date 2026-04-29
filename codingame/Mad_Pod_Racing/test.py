"""Simulates Mad Pod Racing scenarios outside the Codingame server."""

from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib import use as use_matplotlib_backend

use_matplotlib_backend("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
from numpy.typing import NDArray

import main as bot
from main import CHECKPOINT_RADIUS, Pod


MAP_WIDTH = 16000
MAP_HEIGHT = 9000
MAX_TURNS = 120
POD_RADIUS = 90
DIRECTION_ARROW_LENGTH = 225
ARROW_WIDTH = 6
ARROW_HEAD_WIDTH = 60
PREDICTION_LABEL_SIZE = 8
PREDICTION_LABEL_WIDTH = 720
PREDICTION_LABEL_HEIGHT = 260
PREDICTION_LABEL_OFFSETS = ((0, -260), (0, 260), (460, 0), (-460, 0), (460, -260), (-460, -260), (460, 260), (-460, 260), (0, -540), (0, 540),
                            (780, 0), (-780, 0), (780, -420), (-780, -420), (780, 420), (-780, 420), (0, -820), (0, 820))
CHECKPOINTS = [np.array((5001, 5270)), np.array((11498, 6051)), np.array((9095, 1838))]


@dataclass(slots=True)
class TurnSnapshot:
    """Stores one simulated turn.
    :var pod: Pod state at the beginning of the turn.
    :var predictions: Predicted future pod states from this turn.
    :var moves: Planned direction delta and thrust sequence for this turn.
    """
    pod: Pod
    predictions: list[Pod]
    moves: list[float]


@dataclass(slots=True)
class RaceViewer:
    """Displays a simulated race history.
    :var checkpoints: Circuit checkpoints.
    :var history: Simulated turn snapshots.
    :var turn_ind: Currently displayed turn index.
    :var figure: Matplotlib figure.
    :var axes: Map axes.
    :var slider: Turn slider.
    :var previous_button: Previous turn button.
    :var next_button: Next turn button.
    """
    checkpoints: list[NDArray[int]]
    history: list[TurnSnapshot]
    turn_ind: int
    figure: Figure
    axes: Axes
    slider: Slider
    previous_button: Button
    next_button: Button

    @classmethod
    def create(cls, checkpoints: list[NDArray[int]], history: list[TurnSnapshot]) -> RaceViewer:
        """Creates an interactive race viewer.
        :param checkpoints: Circuit checkpoints.
        :param history: Simulated turn snapshots.
        :return: Configured race viewer.
        """
        figure, axes = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.18)
        viewer = cls(checkpoints, history, 0, figure, axes,
                     Slider(plt.axes((0.18, 0.08, 0.62, 0.03)), "Turn", 0, len(history) - 1, valinit=0, valstep=1),
                     Button(plt.axes((0.18, 0.02, 0.18, 0.04)), "Previous turn"),
                     Button(plt.axes((0.62, 0.02, 0.18, 0.04)), "Next turn"))
        viewer.slider.on_changed(viewer.set_turn)
        viewer.previous_button.on_clicked(viewer.previous_turn)
        viewer.next_button.on_clicked(viewer.next_turn)
        viewer.render()
        return viewer

    def set_turn(self, value: float):
        """Sets the displayed turn.
        :param value: Slider value.
        """
        self.turn_ind = int(value)
        self.render()

    def previous_turn(self, event: object):
        """Moves one turn backward.
        :param event: Matplotlib callback event.
        """
        self.slider.set_val(max(0, self.turn_ind - 1))

    def next_turn(self, event: object):
        """Moves one turn forward.
        :param event: Matplotlib callback event.
        """
        self.slider.set_val(min(len(self.history) - 1, self.turn_ind + 1))

    def render(self):
        """Draws the current turn."""
        self.axes.clear()
        setup_axes(self.axes, self.turn_ind, len(self.history))
        draw_checkpoints(self.axes, self.checkpoints)
        draw_history(self.axes, self.history, self.turn_ind)
        draw_predictions(self.axes, self.history[self.turn_ind])
        self.figure.canvas.draw_idle()

    def show(self):
        """Shows the race viewer."""
        plt.show()


def main():
    """Runs the default simulation scenario."""
    RaceViewer.create(CHECKPOINTS, simulate_single_pod_lap(CHECKPOINTS)).show()


def simulate_single_pod_lap(checkpoints: list[NDArray[int]]) -> list[TurnSnapshot]:
    """Simulates one pod completing one lap.
    :param checkpoints: Circuit checkpoints.
    :return: Simulated turn snapshots.
    """
    pod = Pod(0, checkpoints[0].copy(), np.array((0, 0)), 0, 1)
    passed_checkpoints = 0
    history = []
    while passed_checkpoints < len(checkpoints) and len(history) < MAX_TURNS:
        result = bot.optimize_pod_moves(pod, checkpoints)
        history.append(TurnSnapshot(pod, predict_planned_states(pod, checkpoints, result.x), result.x.tolist()))
        future_state = bot.predict_next(pod, checkpoints, result.x[0], result.x[1])
        passed_checkpoints += future_state.passed_checkpoints
        pod = future_state.pod
    history.append(TurnSnapshot(pod, [], []))
    return history


def predict_planned_states(pod: Pod, checkpoints: list[NDArray[int]], moves: list[float] | NDArray[float]) -> list[Pod]:
    """Predicts future pod states from a planned move sequence.
    :param pod: Starting pod state.
    :param checkpoints: Circuit checkpoints.
    :param moves: Alternating direction delta and thrust values.
    :return: Predicted pod states.
    """
    predictions = []
    for move_ind in range(0, len(moves), 2):
        predictions.append(bot.predict_next(predictions[-1] if predictions else pod, checkpoints, moves[move_ind], moves[move_ind + 1]).pod)
    return predictions


def setup_axes(axes: Axes, turn_ind: int, turn_count: int):
    """Sets map axes appearance.
    :param axes: Matplotlib axes.
    :param turn_ind: Current turn index.
    :param turn_count: Total turn count.
    """
    axes.set_xlim(0, MAP_WIDTH)
    axes.set_ylim(MAP_HEIGHT, 0)
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"Turn {turn_ind} / {turn_count - 1}")
    axes.grid(True, color="0.9")


def draw_checkpoints(axes: Axes, checkpoints: list[NDArray[int]]):
    """Draws checkpoint circles.
    :param axes: Matplotlib axes.
    :param checkpoints: Circuit checkpoints.
    """
    for checkpoint_ind, checkpoint in enumerate(checkpoints):
        axes.add_patch(Circle(checkpoint, CHECKPOINT_RADIUS, fill=False, edgecolor="black", linewidth=1.5))
        axes.text(checkpoint[0], checkpoint[1], str(checkpoint_ind), ha="center", va="center")


def draw_history(axes: Axes, history: list[TurnSnapshot], turn_ind: int):
    """Draws all pod states up to the current turn.
    :param axes: Matplotlib axes.
    :param history: Simulated turn snapshots.
    :param turn_ind: Current turn index.
    """
    positions = np.array([snapshot.pod.position for snapshot in history[:turn_ind + 1]])
    axes.plot(positions[:, 0], positions[:, 1], color="black", linewidth=1)
    for state_ind, snapshot in enumerate(history[:turn_ind + 1]):
        draw_pod_state(axes, snapshot.pod, 1 if state_ind == turn_ind else 0.35)


def draw_predictions(axes: Axes, snapshot: TurnSnapshot):
    """Draws predicted future states for the current turn.
    :param axes: Matplotlib axes.
    :param snapshot: Current turn snapshot.
    """
    if not snapshot.predictions:
        return
    positions = np.array([snapshot.pod.position] + [pod.position for pod in snapshot.predictions])
    axes.plot(positions[:, 0], positions[:, 1], color="black", linestyle="--", linewidth=1)
    label_boxes = []
    for prediction_ind, pod in enumerate(snapshot.predictions):
        axes.add_patch(Circle(pod.position, POD_RADIUS, fill=False, edgecolor="black", linestyle="--", linewidth=1))
        draw_pod_arrows(axes, pod, 0.5)
        move_ind = 2 * prediction_ind
        label_position = get_label_position(pod.position, label_boxes)
        axes.plot((pod.position[0], label_position[0]), (pod.position[1], label_position[1]), color="0.4", linestyle=":", linewidth=0.7)
        axes.text(label_position[0], label_position[1],
                  f"({snapshot.moves[move_ind]:.3g}, {snapshot.moves[move_ind + 1]:.3g})", ha="center", va="center", fontsize=PREDICTION_LABEL_SIZE)


def get_label_position(position: NDArray[int], label_boxes: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """Finds a non-overlapping label position near a point.
    :param position: Point to label.
    :param label_boxes: Already occupied label boxes.
    :return: Label center position.
    """
    for offset in PREDICTION_LABEL_OFFSETS:
        label_position = position[0] + offset[0], position[1] + offset[1]
        label_box = (label_position[0] - PREDICTION_LABEL_WIDTH / 2, label_position[1] - PREDICTION_LABEL_HEIGHT / 2,
                     label_position[0] + PREDICTION_LABEL_WIDTH / 2, label_position[1] + PREDICTION_LABEL_HEIGHT / 2)
        if all(not boxes_overlap(label_box, existing_box) for existing_box in label_boxes):
            label_boxes.append(label_box)
            return label_position
    label_boxes.append(label_box)
    return label_position


def boxes_overlap(first_box: tuple[float, float, float, float], second_box: tuple[float, float, float, float]) -> bool:
    """Checks whether two boxes overlap.
    :param first_box: First box as left, top, right, bottom.
    :param second_box: Second box as left, top, right, bottom.
    :return: Whether the boxes overlap.
    """
    return first_box[0] < second_box[2] and first_box[2] > second_box[0] and first_box[1] < second_box[3] and first_box[3] > second_box[1]


def draw_pod_state(axes: Axes, pod: Pod, alpha: float):
    """Draws a pod position and direction.
    :param axes: Matplotlib axes.
    :param pod: Pod state.
    :param alpha: Drawing opacity.
    """
    axes.add_patch(Circle(pod.position, POD_RADIUS, color="black", alpha=alpha))
    draw_pod_arrows(axes, pod, alpha)


def draw_pod_arrows(axes: Axes, pod: Pod, alpha: float):
    """Draws direction arrow for one pod.
    :param axes: Matplotlib axes.
    :param pod: Pod state.
    :param alpha: Drawing opacity.
    """
    draw_arrow(axes, pod.position, get_direction_vector(pod.direction, DIRECTION_ARROW_LENGTH), "red", alpha)


def draw_arrow(axes: Axes, position: NDArray[int], vector: NDArray[float] | tuple[float, float], color: str, alpha: float):
    """Draws one vector arrow.
    :param axes: Matplotlib axes.
    :param position: Arrow start position.
    :param vector: Arrow vector.
    :param color: Arrow color.
    :param alpha: Drawing opacity.
    """
    axes.arrow(position[0], position[1], vector[0], vector[1], color=color, alpha=alpha, width=ARROW_WIDTH, head_width=ARROW_HEAD_WIDTH,
               length_includes_head=True)


def get_direction_vector(direction: float, length: float) -> tuple[float, float]:
    """Computes a direction vector.
    :param direction: Pod direction angle.
    :param length: Vector length.
    :return: Direction vector.
    """
    return math.cos(math.radians(direction)) * length, -math.sin(math.radians(direction)) * length


if __name__ == "__main__":
    main()
