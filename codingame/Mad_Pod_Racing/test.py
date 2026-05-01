"""Simulates Mad Pod Racing outside the Codingame server."""

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
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg
from numpy.typing import NDArray

import main as bot
from main import CHECKPOINT_RADIUS, Pod


MAP_WIDTH = 16000
MAP_HEIGHT = 9000
MAX_TURNS = 120
LANDSCAPE_DIRECTION_STEPS = 145
LANDSCAPE_THRUST_STEPS = 151
POD_RADIUS = 90
DIRECTION_ARROW_LENGTH = 225
ARROW_WIDTH = 6
ARROW_HEAD_WIDTH = 60
CHECKPOINTS = [np.array((11300, 2800)), np.array((7500, 7000)), np.array((6000, 5300))]


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
    :var move_sliders: Move control sliders for the current turn prediction.
    :var updating_controls: Whether move controls are being reset from the current turn.
    """
    checkpoints: list[NDArray[int]]
    history: list[TurnSnapshot]
    turn_ind: int
    figure: Figure
    axes: Axes
    slider: Slider
    previous_button: Button
    next_button: Button
    move_sliders: list[Slider]
    updating_controls: bool

    @classmethod
    def create(cls, checkpoints: list[NDArray[int]], history: list[TurnSnapshot]) -> RaceViewer:
        """Creates an interactive race viewer.
        :param checkpoints: Circuit checkpoints.
        :param history: Simulated turn snapshots.
        :return: Configured race viewer.
        """
        figure, axes = plt.subplots(figsize=(13, 7))
        plt.subplots_adjust(right=0.78, bottom=0.12)
        move_sliders = []
        for move_ind in range(2 * bot.PREDICT_TURNS):
            move_sliders.append(Slider(plt.axes((0.84, 0.86 - 0.08 * move_ind, 0.12, 0.025)),
                                       ("d" if move_ind % 2 == 0 else "t") + str(move_ind // 2),
                                       -bot.MAX_TURN_DEG if move_ind % 2 == 0 else 0, bot.MAX_TURN_DEG if move_ind % 2 == 0 else 100, valinit=0))
        viewer = cls(checkpoints, history, 0, figure, axes,
                     Slider(plt.axes((0.18, 0.06, 0.55, 0.03)), "Turn", 0, len(history) - 1, valinit=0, valstep=1),
                     Button(plt.axes((0.18, 0.01, 0.18, 0.04)), "Previous turn"),
                     Button(plt.axes((0.55, 0.01, 0.18, 0.04)), "Next turn"), move_sliders, False)
        viewer.slider.on_changed(viewer.set_turn)
        viewer.previous_button.on_clicked(viewer.previous_turn)
        viewer.next_button.on_clicked(viewer.next_turn)
        for slider in viewer.move_sliders:
            slider.on_changed(viewer.set_move)
        viewer.sync_move_sliders()
        viewer.render()
        return viewer

    def set_turn(self, value: float):
        """Sets the displayed turn.
        :param value: Slider value.
        """
        self.turn_ind = int(value)
        self.sync_move_sliders()
        self.render()

    def sync_move_sliders(self):
        """Sets move sliders to optimized moves for the displayed turn."""
        self.updating_controls = True
        for move_ind, slider in enumerate(self.move_sliders):
            slider.set_val(self.history[self.turn_ind].moves[move_ind] if move_ind < len(self.history[self.turn_ind].moves) else 0)
        self.updating_controls = False

    def set_move(self, value: float):
        """Updates predictions after a move control changes.
        :param value: Slider value.
        """
        if not self.updating_controls:
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
        draw_predictions(self.axes, self.history[self.turn_ind], self.checkpoints, self.get_selected_moves())
        self.figure.canvas.draw_idle()

    def show(self):
        """Shows the race viewer."""
        plt.show()

    def get_selected_moves(self) -> list[float]:
        """Reads selected moves from move controls.
        :return: Alternating direction delta and thrust values.
        """
        return [slider.val for slider in self.move_sliders]


def main():
    """Runs the default test."""
    laps = 3
    turn = 11

    show_race(CHECKPOINTS, laps)
    # run_optimization(CHECKPOINTS, turn)
    # plot_optimization_landscape_1d(CHECKPOINTS, turn)
    # plot_optimization_landscape_2d(CHECKPOINTS, turn)


def show_race(checkpoints: list[NDArray[int]], laps: int):
    """Shows one simulated pod race.
    :param checkpoints: Circuit checkpoints.
    :param laps: Number of laps to simulate and show.
    """
    RaceViewer.create(checkpoints, simulate_single_pod_lap(checkpoints, laps)).show()


def run_optimization(checkpoints: list[NDArray[int]], turn: int):
    """Runs an optimizer check from turn 14 of the current simulation.
    :param turn: Turn for which to run.
    :param checkpoints: Circuit checkpoints.
    """
    pod = simulate_single_pod_lap(checkpoints, 1)[turn].pod
    guess_moves = bot.get_optimizer_guess_moves()
    result = bot.optimize_pod_moves(pod, checkpoints)
    guess_moves_text = ", ".join(f"{value:.3g}" for value in guess_moves)
    optimized_moves_text = ", ".join(f"{value:.3g}" for value in result.x)
    print(f"guess moves=[{guess_moves_text}]")
    print(f"optimized moves=[{optimized_moves_text}]")


def plot_optimization_landscape_2d(checkpoints: list[NDArray[int]], turn: int):
    """Plots the first-move optimization landscape from turn 14 of the current simulation in 3D.
    :param turn: Turn at which to run.
    :param checkpoints: Circuit checkpoints.
    """
    pod = simulate_single_pod_lap(checkpoints, 1)[turn].pod
    guess_moves = bot.get_optimizer_guess_moves()
    result = bot.optimize_pod_moves(pod, checkpoints)
    direction_deltas = np.linspace(-bot.MAX_TURN_DEG, bot.MAX_TURN_DEG, LANDSCAPE_DIRECTION_STEPS)
    thrusts = np.linspace(0, 100, LANDSCAPE_THRUST_STEPS)
    scores = np.empty((len(thrusts), len(direction_deltas)))
    for thrust_ind, thrust in enumerate(thrusts):
        for direction_ind, direction_delta in enumerate(direction_deltas):
            moves = guess_moves.copy()
            moves[0] = direction_delta
            moves[1] = thrust
            scores[thrust_ind, direction_ind] = bot.predict_turns(pod, checkpoints, moves).get_score(checkpoints)

    figure = plt.figure(figsize=(10, 7))
    axes = figure.add_subplot(111, projection="3d")
    direction_grid, thrust_grid = np.meshgrid(direction_deltas, thrusts)
    surface = axes.plot_surface(direction_grid, thrust_grid, scores, cmap="viridis_r", linewidth=0, antialiased=False, alpha=0.9)
    figure.colorbar(surface, ax=axes, label="Score", shrink=0.65)
    optimized_marker_moves = guess_moves.copy()
    optimized_marker_moves[:2] = result.x[:2]
    axes.scatter(guess_moves[0], guess_moves[1], bot.predict_turns(pod, checkpoints, guess_moves).get_score(checkpoints), color="white",
                 edgecolor="black", marker="o", s=60, label="Guess")
    axes.scatter(optimized_marker_moves[0], optimized_marker_moves[1], bot.predict_turns(pod, checkpoints, optimized_marker_moves).get_score(checkpoints),
                 color="red", edgecolor="black", marker="x", s=80, label="Optimized")
    axes.set_xlabel("Direction delta")
    axes.set_ylabel("Thrust")
    axes.set_zlabel("Score")
    axes.set_title("Optimization landscape")
    axes.legend()
    plt.show()


def plot_optimization_landscape_1d(checkpoints: list[NDArray[int]], turn: int):
    """Plots the optimization landscape against one move coordinate.
    :param checkpoints: Circuit checkpoints.
    :param turn: Turn at which to run.
    """
    coordinate_ind = 8
    coords = np.array((9.32, 99, 6.8, 99.2, 4.28, 99.4, 2.25, 99.5, 0.79, 99.7), dtype=float)
    pod = simulate_single_pod_lap(checkpoints, 1)[turn].pod
    result = bot.optimize_pod_moves(pod, checkpoints)
    coordinate_values = np.linspace(-bot.MAX_TURN_DEG, bot.MAX_TURN_DEG, LANDSCAPE_DIRECTION_STEPS) if coordinate_ind % 2 == 0 \
        else np.linspace(0, 100, LANDSCAPE_THRUST_STEPS)
    scores = []
    for coordinate_value in coordinate_values:
        moves = coords.copy()
        moves[coordinate_ind] = coordinate_value
        scores.append(bot.predict_turns(pod, checkpoints, moves).get_score(checkpoints))

    figure, axes = plt.subplots(figsize=(10, 7))
    optimized_marker_moves = coords.copy()
    optimized_marker_moves[coordinate_ind] = result.x[coordinate_ind]
    axes.plot(coordinate_values, scores, color="black", marker="o", markersize=3)
    axes.scatter(optimized_marker_moves[coordinate_ind], bot.predict_turns(pod, checkpoints, optimized_marker_moves).get_score(checkpoints), color="red",
                 marker="x", s=80, label="Optimized")
    axes.set_xlabel(f"move[{coordinate_ind}]")
    axes.set_ylabel("Score")
    axes.set_title("One-coordinate optimization landscape")
    axes.legend()
    plt.show()


def simulate_single_pod_lap(checkpoints: list[NDArray[int]], laps: int) -> list[TurnSnapshot]:
    """Simulates one pod completing a race.
    :param checkpoints: Circuit checkpoints.
    :param laps: Number of laps to simulate.
    :return: Simulated turn snapshots.
    """
    pod = Pod(0, checkpoints[0].astype(float), np.array((0, 0), dtype=float), 0, 1)
    passed_checkpoints = 0
    history = []
    while passed_checkpoints < len(checkpoints) * laps and len(history) < MAX_TURNS * laps:
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


def draw_predictions(axes: Axes, snapshot: TurnSnapshot, checkpoints: list[NDArray[int]], moves: list[float]):
    """Draws predicted future states for the current turn.
    :param axes: Matplotlib axes.
    :param snapshot: Current turn snapshot.
    :param checkpoints: Circuit checkpoints.
    :param moves: Alternating selected direction delta and thrust values.
    """
    if not snapshot.moves:
        return
    predictions = predict_planned_states(snapshot.pod, checkpoints, moves)
    positions = np.array([snapshot.pod.position] + [pod.position for pod in predictions])
    axes.plot(positions[:, 0], positions[:, 1], color="black", linestyle="--", linewidth=1)
    axes.text(0.02, 0.98, f"score={round(bot.predict_turns(snapshot.pod, checkpoints, moves).get_score(checkpoints))}", color="red",
              transform=axes.transAxes, ha="left", va="top")
    for pod in predictions:
        edgecolor = "red" if any(linalg.norm(checkpoint - pod.position) <= CHECKPOINT_RADIUS for checkpoint in checkpoints) else "black"
        axes.add_patch(Circle(pod.position, POD_RADIUS, fill=False, edgecolor=edgecolor, linestyle="--", linewidth=1))
        draw_pod_arrows(axes, pod, 0.5)


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


def draw_arrow(axes: Axes, position: NDArray[float], vector: NDArray[float] | tuple[float, float], color: str, alpha: float):
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
