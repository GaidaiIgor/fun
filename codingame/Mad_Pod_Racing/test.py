"""Simulates and visualizes Mad Pod Racing behavior outside the Codingame server."""

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
from main import BasePod, CHECKPOINT_RADIUS, GameState, RacerPod


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
    """Stores the state available at the beginning of one simulated turn.
    predictions are the future pods produced from the optimized move vector for this turn, and moves is that direction delta and thrust sequence.
    """
    pod: BasePod
    predictions: list[BasePod]
    moves: list[float]


@dataclass(slots=True)
class RaceViewer:
    """Owns the matplotlib race viewer state.
    The viewer keeps the immutable checkpoints, simulated history, current turn selection, navigation controls and editable move sliders.
    Rendering combines the actual simulated past with predictions recomputed from the current slider values.
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
        """Builds the figure, map axes, turn controls, move sliders and callbacks for an interactive race viewer."""
        figure, axes = plt.subplots(figsize=(13, 7))
        figure.canvas.manager.window.showMaximized()
        plt.subplots_adjust(right=0.78, bottom=0.12)
        move_sliders = []
        for move_ind in range(2 * bot.PREDICT_TURNS):
            move_sliders.append(Slider(plt.axes((0.84, 0.86 - 0.08 * move_ind, 0.12, 0.025)),
                                       ("d" if move_ind % 2 == 0 else "t") + str(move_ind // 2),
                                       -180 if move_ind == 0 else -bot.MAX_TURN_DEG if move_ind % 2 == 0 else 0,
                                       180 if move_ind == 0 else bot.MAX_TURN_DEG if move_ind % 2 == 0 else 100, valinit=0))
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
        """Moves the viewer to a history index, resets move sliders to that turn optimized moves and redraws."""
        self.turn_ind = int(value)
        self.sync_move_sliders()
        self.render()

    def sync_move_sliders(self):
        """Copies optimized moves from the selected snapshot into the sliders.
        The first direction slider allows a full turn only on turn 0; all other direction sliders use the normal turn cap.
        """
        self.updating_controls = True
        self.move_sliders[0].valmin = -180 if self.turn_ind == 0 else -bot.MAX_TURN_DEG
        self.move_sliders[0].valmax = 180 if self.turn_ind == 0 else bot.MAX_TURN_DEG
        self.move_sliders[0].ax.set_xlim(self.move_sliders[0].valmin, self.move_sliders[0].valmax)
        for move_ind, slider in enumerate(self.move_sliders):
            slider.set_val(self.history[self.turn_ind].moves[move_ind] if move_ind < len(self.history[self.turn_ind].moves) else 0)
        self.updating_controls = False

    def set_move(self, value: float):
        """Redraws predictions after a user edits a move slider, unless sliders are currently being synchronized."""
        if not self.updating_controls:
            self.render()

    def previous_turn(self, event: object):
        """Moves the selected turn one step backward through the history."""
        self.slider.set_val(max(0, self.turn_ind - 1))

    def next_turn(self, event: object):
        """Moves the selected turn one step forward through the history."""
        self.slider.set_val(min(len(self.history) - 1, self.turn_ind + 1))

    def render(self):
        """Clears the map and redraws checkpoints, simulated history and the editable prediction path."""
        self.axes.clear()
        setup_axes(self.axes, self.turn_ind, len(self.history))
        draw_checkpoints(self.axes, self.checkpoints)
        draw_history(self.axes, self.history, self.turn_ind)
        draw_predictions(self.axes, self.history[self.turn_ind], self.checkpoints, self.get_selected_moves(), self.turn_ind == 0)
        self.figure.canvas.draw_idle()

    def show(self):
        """Starts the matplotlib event loop for the configured viewer."""
        plt.show()

    def get_selected_moves(self) -> list[float]:
        """Returns slider values as an alternating direction delta and thrust sequence."""
        return [slider.val for slider in self.move_sliders]


def main():
    """Selects which local simulator or diagnostic view to run, with inactive tools left commented for quick switching."""
    laps = 3
    turn = 11

    show_race(CHECKPOINTS, laps)
    # run_optimization(CHECKPOINTS, turn)
    # plot_optimization_landscape_1d(CHECKPOINTS, turn)
    # plot_optimization_landscape_2d(CHECKPOINTS, turn)


def show_race(checkpoints: list[NDArray[int]], laps: int):
    """Simulates a racer-only run for the requested number of laps and opens the interactive viewer."""
    RaceViewer.create(checkpoints, simulate_single_pod(checkpoints, laps)).show()


def run_optimization(checkpoints: list[NDArray[int]], turn: int):
    """Recreates the pod at a simulated turn and prints optimizer seed and result without opening a plot."""
    pod = simulate_single_pod(checkpoints, 1)[turn].pod
    guess_moves = RacerPod.get_optimizer_guess_moves()
    result = pod.optimize_moves(checkpoints)
    guess_moves_text = ", ".join(f"{value:.3g}" for value in guess_moves)
    optimized_moves_text = ", ".join(f"{value:.3g}" for value in result.x)
    print(f"guess moves=[{guess_moves_text}]")
    print(f"optimized moves=[{optimized_moves_text}]")


def plot_optimization_landscape_2d(checkpoints: list[NDArray[int]], turn: int):
    """Plots a score surface over the first direction delta and thrust coordinates for a simulated turn.
    All later move coordinates stay at the optimizer seed. White marks the seed and red marks the optimized first move at its actual score height.
    """
    pod = simulate_single_pod(checkpoints, 1)[turn].pod
    guess_moves = RacerPod.get_optimizer_guess_moves()
    result = pod.optimize_moves(checkpoints)
    direction_deltas = np.linspace(-bot.MAX_TURN_DEG, bot.MAX_TURN_DEG, LANDSCAPE_DIRECTION_STEPS)
    thrusts = np.linspace(0, 100, LANDSCAPE_THRUST_STEPS)
    scores = np.empty((len(thrusts), len(direction_deltas)))
    for thrust_ind, thrust in enumerate(thrusts):
        for direction_ind, direction_delta in enumerate(direction_deltas):
            moves = guess_moves.copy()
            moves[0] = direction_delta
            moves[1] = thrust
            scores[thrust_ind, direction_ind] = bot.predict_turns(pod, checkpoints, moves)[-1].get_score(checkpoints)

    figure = plt.figure(figsize=(10, 7))
    figure.canvas.manager.window.showMaximized()
    axes = figure.add_subplot(111, projection="3d")
    direction_grid, thrust_grid = np.meshgrid(direction_deltas, thrusts)
    surface = axes.plot_surface(direction_grid, thrust_grid, scores, cmap="viridis_r", linewidth=0, antialiased=False, alpha=0.9)
    figure.colorbar(surface, ax=axes, label="Score", shrink=0.65)
    optimized_marker_moves = guess_moves.copy()
    optimized_marker_moves[:2] = result.x[:2]
    axes.scatter(guess_moves[0], guess_moves[1], bot.predict_turns(pod, checkpoints, guess_moves)[-1].get_score(checkpoints), color="white",
                 edgecolor="black", marker="o", s=60, label="Guess")
    axes.scatter(optimized_marker_moves[0], optimized_marker_moves[1], bot.predict_turns(pod, checkpoints, optimized_marker_moves)[-1].get_score(checkpoints),
                 color="red", edgecolor="black", marker="x", s=80, label="Optimized")
    axes.set_xlabel("Direction delta")
    axes.set_ylabel("Thrust")
    axes.set_zlabel("Score")
    axes.set_title("Optimization landscape")
    axes.legend()
    plt.show()


def plot_optimization_landscape_1d(checkpoints: list[NDArray[int]], turn: int):
    """Plots a one-coordinate score slice through an explicit move vector.
    coordinate_ind selects which coordinate is swept across its allowed range. Every other coordinate stays at the value from coords.
    """
    coordinate_ind = 8
    coords = np.array((9.32, 99, 6.8, 99.2, 4.28, 99.4, 2.25, 99.5, 0.79, 99.7), dtype=float)
    pod = simulate_single_pod(checkpoints, 1)[turn].pod
    result = pod.optimize_moves(checkpoints)
    coordinate_values = np.linspace(-bot.MAX_TURN_DEG, bot.MAX_TURN_DEG, LANDSCAPE_DIRECTION_STEPS) if coordinate_ind % 2 == 0 \
        else np.linspace(0, 100, LANDSCAPE_THRUST_STEPS)
    scores = []
    for coordinate_value in coordinate_values:
        moves = coords.copy()
        moves[coordinate_ind] = coordinate_value
        scores.append(bot.predict_turns(pod, checkpoints, moves)[-1].get_score(checkpoints))

    figure, axes = plt.subplots(figsize=(10, 7))
    figure.canvas.manager.window.showMaximized()
    optimized_marker_moves = coords.copy()
    optimized_marker_moves[coordinate_ind] = result.x[coordinate_ind]
    axes.plot(coordinate_values, scores, color="black", marker="o", markersize=3)
    axes.scatter(optimized_marker_moves[coordinate_ind], bot.predict_turns(pod, checkpoints, optimized_marker_moves)[-1].get_score(checkpoints), color="red",
                 marker="x", s=80, label="Optimized")
    axes.set_xlabel(f"move[{coordinate_ind}]")
    axes.set_ylabel("Score")
    axes.set_title("One-coordinate optimization landscape")
    axes.legend()
    plt.show()


def simulate_single_pod(checkpoints: list[NDArray[int]], laps: int) -> list[TurnSnapshot]:
    """Runs a racer-only race by calling the production racer logic and prediction model each turn.
    The returned history stores start-of-turn snapshots until the requested number of laps is completed or the turn cap is reached.
    """
    pod = RacerPod(0, checkpoints[0].astype(float), np.array((0, 0), dtype=float), 0, 1)
    passed_checkpoints = 0
    boosts = 1
    history = []
    while passed_checkpoints < len(checkpoints) * laps and len(history) < MAX_TURNS * laps:
        _, thrust, moves = pod.choose_move(GameState(len(history), laps, checkpoints, [pod], [], boosts))
        future_states = bot.predict_turns(pod, checkpoints, moves, len(history) == 0)
        history.append(TurnSnapshot(pod, [future_state.pod for future_state in future_states], moves.tolist()))
        if thrust == "BOOST":
            boosts -= 1
        passed_checkpoints += future_states[0].passed_checkpoints
        pod = future_states[0].pod
    history.append(TurnSnapshot(pod, [], []))
    return history


def setup_axes(axes: Axes, turn_ind: int, turn_count: int):
    """Configures the map axes to match Codingame coordinates, including y increasing downward."""
    axes.set_xlim(0, MAP_WIDTH)
    axes.set_ylim(MAP_HEIGHT, 0)
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"Turn {turn_ind} / {turn_count - 1}")
    axes.grid(True, color="0.9")


def draw_checkpoints(axes: Axes, checkpoints: list[NDArray[int]]):
    """Draws indexed checkpoint circles at their race coordinates."""
    for checkpoint_ind, checkpoint in enumerate(checkpoints):
        axes.add_patch(Circle(checkpoint, CHECKPOINT_RADIUS, fill=False, edgecolor="black", linewidth=1.5))
        axes.text(checkpoint[0], checkpoint[1], str(checkpoint_ind), ha="center", va="center")


def draw_history(axes: Axes, history: list[TurnSnapshot], turn_ind: int):
    """Draws the simulated trajectory through the selected turn and fades older pod states."""
    positions = np.array([snapshot.pod.position for snapshot in history[:turn_ind + 1]])
    axes.plot(positions[:, 0], positions[:, 1], color="black", linewidth=1)
    for state_ind, snapshot in enumerate(history[:turn_ind + 1]):
        draw_pod_state(axes, snapshot.pod, 1 if state_ind == turn_ind else 0.35)


def draw_predictions(axes: Axes, snapshot: TurnSnapshot, checkpoints: list[NDArray[int]], moves: list[float], first_turn: bool):
    """Draws the dashed future path from the selected turn using the current slider moves.
    Predicted positions inside any checkpoint get red outlines, and the final projected score is shown in the map corner.
    """
    if not snapshot.moves:
        return
    future_states = bot.predict_turns(snapshot.pod, checkpoints, moves[:len(snapshot.moves)], first_turn)
    positions = np.array([snapshot.pod.position] + [future_state.pod.position for future_state in future_states])
    axes.plot(positions[:, 0], positions[:, 1], color="black", linestyle="--", linewidth=1)
    axes.text(0.02, 0.98, f"score={round(future_states[-1].get_score(checkpoints))}", color="red", transform=axes.transAxes, ha="left", va="top")
    for future_state in future_states:
        edgecolor = "red" if any(linalg.norm(checkpoint - future_state.pod.position) <= CHECKPOINT_RADIUS for checkpoint in checkpoints) else "black"
        axes.add_patch(Circle(future_state.pod.position, POD_RADIUS, fill=False, edgecolor=edgecolor, linestyle="--", linewidth=1))
        draw_pod_arrows(axes, future_state.pod, 0.5)


def draw_pod_state(axes: Axes, pod: BasePod, alpha: float):
    """Draws the pod center and its facing direction at a chosen opacity."""
    axes.add_patch(Circle(pod.position, POD_RADIUS, color="black", alpha=alpha))
    draw_pod_arrows(axes, pod, alpha)


def draw_pod_arrows(axes: Axes, pod: BasePod, alpha: float):
    """Draws the red facing-direction arrow for one pod without drawing velocity."""
    draw_arrow(axes, pod.position, get_direction_vector(pod.direction, DIRECTION_ARROW_LENGTH), "red", alpha)


def draw_arrow(axes: Axes, position: NDArray[float], vector: NDArray[float] | tuple[float, float], color: str, alpha: float):
    """Draws one fixed-style matplotlib arrow from a start position along a vector."""
    axes.arrow(position[0], position[1], vector[0], vector[1], color=color, alpha=alpha, width=ARROW_WIDTH, head_width=ARROW_HEAD_WIDTH,
               length_includes_head=True)


def get_direction_vector(direction: float, length: float) -> tuple[float, float]:
    """Converts a bot direction angle and visual length into a screen-space vector."""
    return math.cos(math.radians(direction)) * length, -math.sin(math.radians(direction)) * length


if __name__ == "__main__":
    main()
