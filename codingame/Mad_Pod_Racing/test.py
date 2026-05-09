"""Simulates and visualizes Mad Pod Racing behavior outside the Codingame server."""

from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib import use as use_matplotlib_backend

use_matplotlib_backend("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg
from numpy.typing import NDArray

import main as bot
from main import BasePod, BrutePod, CHECKPOINT_RADIUS, COLLISION_RADIUS, GameState, OPTIMIZER_CHECKPOINT_RADIUS, RacerPod, get_segment_direction

bot.DEBUG = False

MAP_WIDTH = 16000
MAP_HEIGHT = 9000
MAX_TURNS = 120
LANDSCAPE_DIRECTION_STEPS = 145
LANDSCAPE_THRUST_STEPS = 151
POD_RADIUS = 90
POD_COLLISION_RADIUS = COLLISION_RADIUS // 2
DIRECTION_ARROW_LENGTH = 225
ARROW_WIDTH = 6
ARROW_HEAD_WIDTH = 60
CHECKPOINTS = [np.array((11300, 2800)), np.array((7500, 7000)), np.array((6000, 5300))]
RACER_COLOR = "green"
BRUTE_COLOR = "blue"
ENEMY_COLOR = "red"
COLLISION_COLOR = "black"


@dataclass(slots=True)
class EnemyRacerPod(RacerPod):
    """Represents an enemy pod that uses the racer program in local simulations."""


@dataclass(slots=True)
class TurnSnapshot:
    """Stores the state available at the beginning of one simulated turn.
    predictions are the future pods produced from the optimized move vector for this turn, and moves is that direction delta and thrust sequence.
    target_direction is the command direction chosen by that pod for this turn when the turn has a command.
    base_target_pos is the target chosen before racer avoidance.
    """
    pod: BasePod
    predictions: list[BasePod]
    moves: list[float]
    target_direction: float | None = None
    base_target_pos: NDArray[float] | None = None
    thrust: float | str | None = None


@dataclass(slots=True)
class RaceViewer:
    """Owns the matplotlib race viewer state.
    The viewer keeps the immutable checkpoints, simulated history, current turn selection, navigation controls and editable move sliders.
    Rendering combines the actual simulated past with predictions recomputed from the current slider values.
    """
    checkpoints: list[NDArray[int]]
    history: list[TurnSnapshot]
    color: str
    extra_histories: list[tuple[list[TurnSnapshot], str]]
    collision_pos: NDArray[float] | None
    show_collision_radius: bool
    show_predicted_collision_radius: bool
    show_racer_avoidance_area: bool
    show_closest_brute_approach: bool
    turn_ind: int
    figure: Figure
    axes: Axes
    slider: Slider
    previous_button: Button
    next_button: Button
    move_sliders: list[Slider]
    updating_controls: bool

    @classmethod
    def create(cls, checkpoints: list[NDArray[int]], history: list[TurnSnapshot], color: str, extra_histories: list[tuple[list[TurnSnapshot], str]],
               collision_pos: NDArray[float] | None, show_collision_radius: bool, show_predicted_collision_radius: bool = False,
               show_racer_avoidance_area: bool = False, show_closest_brute_approach: bool = False) -> RaceViewer:
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
        viewer = cls(checkpoints, history, color, extra_histories, collision_pos, show_collision_radius, show_predicted_collision_radius,
                     show_racer_avoidance_area, show_closest_brute_approach, 0, figure, axes,
                     Slider(plt.axes((0.18, 0.06, 0.55, 0.03)), "Turn", 0, len(history) - 1, valinit=0, valstep=1),
                     Button(plt.axes((0.18, 0.01, 0.18, 0.04)), "Previous turn"),
                     Button(plt.axes((0.55, 0.01, 0.18, 0.04)), "Next turn"), move_sliders, False)
        viewer.slider.on_changed(viewer.set_turn)
        viewer.previous_button.on_clicked(viewer.previous_turn)
        viewer.next_button.on_clicked(viewer.next_turn)
        figure.canvas.mpl_connect("key_press_event", viewer.handle_key_press)
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

    def handle_key_press(self, event: KeyEvent):
        """Routes left and right arrow key presses to the matching turn navigation actions."""
        if event.key == "left":
            self.previous_turn(event)
        elif event.key == "right":
            self.next_turn(event)

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
        draw_history(self.axes, self.history, self.turn_ind, self.color, self.show_collision_radius)
        for history, color in self.extra_histories:
            draw_history(self.axes, history, min(self.turn_ind, len(history) - 1), color, self.show_collision_radius)
        if self.collision_pos is not None and self.turn_ind == len(self.history) - 1:
            self.axes.scatter(self.collision_pos[0], self.collision_pos[1], color=COLLISION_COLOR, marker="x", s=220, linewidths=3)
        draw_predictions(self.axes, self.history[self.turn_ind], self.checkpoints, self.get_selected_moves(), self.turn_ind == 0, self.color,
                         self.show_predicted_collision_radius, self.show_racer_avoidance_area)
        if self.show_closest_brute_approach:
            draw_closest_brute_approach(self.axes, self.history, self.extra_histories, self.turn_ind, self.get_selected_moves(), self.checkpoints,
                                        self.show_predicted_collision_radius)
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

    # show_race(CHECKPOINTS, laps)
    # show_brute_collision()
    # show_three_pods()
    show_coasting()
    # run_optimization(CHECKPOINTS, turn)
    # plot_optimization_landscape_1d(CHECKPOINTS, turn)
    # plot_optimization_landscape_2d(CHECKPOINTS, turn)


def show_race(checkpoints: list[NDArray[int]], laps: int):
    """Simulates the normal racer-only view from the brute-scenario enemy start and opens the interactive viewer."""
    track_direction = get_segment_direction(checkpoints[0], checkpoints[1])
    pod = RacerPod(0, ((checkpoints[0] + checkpoints[1]) / 2).astype(float), np.array((0, 0), dtype=float), track_direction, 1)
    show_simulation(checkpoints, laps, [pod], 0)


def show_brute_collision():
    """Simulates a brute chasing one racer-logic enemy until their next-turn motion first collides, then opens the interactive viewer."""
    track_direction = get_segment_direction(CHECKPOINTS[0], CHECKPOINTS[1])
    brute = BrutePod(1, CHECKPOINTS[0].astype(float), np.array((0, 0), dtype=float), track_direction, 1)
    enemy = EnemyRacerPod(0, ((CHECKPOINTS[0] + CHECKPOINTS[1]) / 2).astype(float), np.array((0, 0), dtype=float), track_direction, 1)
    show_simulation(CHECKPOINTS, 1, [brute, enemy], 0)


def show_three_pods():
    """Simulates our racer, brute and one enemy racer from the requested three-pod initial state."""
    racer_direction = get_segment_direction(CHECKPOINTS[2], CHECKPOINTS[0])
    brute_direction = get_segment_direction(CHECKPOINTS[0], CHECKPOINTS[2])
    racer = RacerPod(0, ((CHECKPOINTS[2] + CHECKPOINTS[0]) / 2).astype(float), np.array((0, 0), dtype=float), racer_direction, 0)
    brute = BrutePod(1, CHECKPOINTS[0].astype(float), np.array((0, 0), dtype=float), brute_direction, 0)
    enemy = EnemyRacerPod(0, CHECKPOINTS[2].astype(float), np.array((0, 0), dtype=float), racer_direction, 0)
    show_simulation(CHECKPOINTS, 1, [racer, brute, enemy], 0, True, True, True)


def show_coasting():
    """Simulates a brute drifting into the enemy segment end while it turns around with zero thrust."""
    segment_end = CHECKPOINTS[2].astype(float)
    velocity = np.array((250, 0), dtype=float)
    coast_distance = linalg.norm(velocity) * (1 - bot.DRAG ** 10) / (1 - bot.DRAG)
    brute_direction = bot.normalize_angle(get_segment_direction(CHECKPOINTS[2], CHECKPOINTS[1]) + 180)
    brute = BrutePod(1, segment_end - velocity / linalg.norm(velocity) * coast_distance, velocity, brute_direction, 1)
    enemy = EnemyRacerPod(0, CHECKPOINTS[0].astype(float), np.array((0, 0), dtype=float), brute_direction, 1)
    show_simulation(CHECKPOINTS, 1, [brute, enemy], 0)


def show_simulation(checkpoints: list[NDArray[int]], laps: int, pods: list[BasePod], boosts: int, show_predicted_collision_radius: bool = False,
                    show_racer_avoidance_area: bool = False, show_closest_brute_approach: bool = False):
    """Runs the shared simulator for the supplied initial pods and opens the shared interactive viewer."""
    histories, colors, collision_pos = simulate_pods(checkpoints, laps, pods, boosts)
    RaceViewer.create(checkpoints, histories[0], colors[0], list(zip(histories[1:], colors[1:])), collision_pos, len(pods) > 1,
                      show_predicted_collision_radius, show_racer_avoidance_area, show_closest_brute_approach).show()


def simulate_pods(checkpoints: list[NDArray[int]], laps: int, pods: list[BasePod], boosts: int = 1) \
    -> tuple[list[list[TurnSnapshot]], list[str], NDArray[float] | None]:
    """Runs the shared local engine.
    Each pod chooses commands through its own program. All pods advance through the same prediction model and the first collision stops the simulation.
    """
    histories = [[] for _ in pods]
    colors = [BRUTE_COLOR if isinstance(pod, BrutePod) else ENEMY_COLOR if isinstance(pod, EnemyRacerPod) else RACER_COLOR for pod in pods]
    for turn_ind in range(MAX_TURNS * laps):
        commands = []
        for pod_ind, pod in enumerate(pods):
            target_direction, thrust, moves, base_target_pos = choose_pod_command(pod, pods, commands, turn_ind, laps, checkpoints, boosts)
            future_states = bot.predict_turns(pod, checkpoints, moves, turn_ind == 0) if len(moves) else []
            histories[pod_ind].append(TurnSnapshot(pod, [future_state.pod for future_state in future_states], moves.tolist(), target_direction,
                                                   base_target_pos, thrust))
            if thrust == "BOOST":
                boosts -= 1
            commands.append((target_direction, thrust))
        next_pods = [bot.predict_next(pod, checkpoints, bot.normalize_angle(command[0] - pod.direction), command[1], turn_ind == 0).pod
                     for pod, command in zip(pods, commands)]
        collision_pos = get_first_collision_pos(pods, next_pods)
        if collision_pos is not None or any(pod.passed_checkpoints >= len(checkpoints) * laps for pod in next_pods if isinstance(pod, RacerPod)):
            for pod_ind, pod in enumerate(next_pods):
                histories[pod_ind].append(TurnSnapshot(pod, [], []))
            return histories, colors, collision_pos
        pods = next_pods
    for pod_ind, pod in enumerate(pods):
        histories[pod_ind].append(TurnSnapshot(pod, [], []))
    return histories, colors, None


def choose_pod_command(pod: BasePod, pods: list[BasePod], commands: list[tuple[float, float | str]], turn_ind: int, laps: int,
                       checkpoints: list[NDArray[int]], boosts: int) \
    -> tuple[float, float | str, NDArray[float], NDArray[float] | None]:
    """Chooses one command for a simulated pod through that pod program."""
    if isinstance(pod, RacerPod):
        direction, thrust, moves = pod.choose_move(GameState(turn_ind, laps, checkpoints, [pod], [], boosts))
        return direction, thrust, moves, None
    racer_ind = next((other_ind for other_ind, other_pod in enumerate(pods[:len(commands)])
                      if isinstance(other_pod, RacerPod) and not isinstance(other_pod, EnemyRacerPod)), None)
    my_pods = [pod] if racer_ind is None else [pods[racer_ind], pod]
    foe_pods = [other_pod for other_pod in pods if isinstance(other_pod, EnemyRacerPod)]
    game_state = GameState(turn_ind, laps, checkpoints, my_pods, foe_pods, 0)
    enemy = pod.get_lead_enemy(game_state)
    direction, thrust = pod.choose_command(game_state, None if racer_ind is None else commands[racer_ind])
    return direction, thrust, np.array((), dtype=float), pod.get_attack_target(enemy) if pod.is_attackable(enemy) else \
        checkpoints[(enemy.next_checkpoint_ind + 1) % len(checkpoints)]


def get_first_collision_pos(pods: list[BasePod], next_pods: list[BasePod]) -> NDArray[float] | None:
    """Returns the first collision position detected among all pod motion segments, or None when no pair collides this turn."""
    for pod_ind, pod in enumerate(pods):
        for other_ind in range(pod_ind + 1, len(pods)):
            collision_pos = get_collision_pos(pod.position, next_pods[pod_ind].position, pods[other_ind].position, next_pods[other_ind].position)
            if collision_pos is not None:
                return collision_pos
    return None


def get_collision_pos(start_1: NDArray[float], end_1: NDArray[float], start_2: NDArray[float], end_2: NDArray[float]) -> NDArray[float] | None:
    """Returns the midpoint of the two pods at closest synchronized approach when that approach is inside collision distance."""
    relative_position = start_1 - start_2
    relative_velocity = end_1 - start_1 - end_2 + start_2
    if not np.any(relative_velocity):
        closest_time = 0
    else:
        closest_time = np.clip(-np.dot(relative_position, relative_velocity) / np.dot(relative_velocity, relative_velocity), 0, 1)
    if linalg.norm(relative_position + relative_velocity * closest_time) > COLLISION_RADIUS:
        return None
    return (start_1 + (end_1 - start_1) * closest_time + start_2 + (end_2 - start_2) * closest_time) / 2


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


def simulate_single_pod(checkpoints: list[NDArray[int]], laps: int, pod: RacerPod | None = None, boosts: int = 1) -> list[TurnSnapshot]:
    """Runs a one-pod race through the shared simulation engine and returns that pod history."""
    if pod is None:
        pod = RacerPod(0, checkpoints[0].astype(float), np.array((0, 0), dtype=float), 0, 1)
    return simulate_pods(checkpoints, laps, [pod], boosts)[0][0]


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


def draw_history(axes: Axes, history: list[TurnSnapshot], turn_ind: int, color: str, show_collision_radius: bool):
    """Draws the simulated trajectory through the selected turn and fades older pod states."""
    positions = np.array([snapshot.pod.position for snapshot in history[:turn_ind + 1]])
    axes.plot(positions[:, 0], positions[:, 1], color=color, linewidth=1)
    for state_ind, snapshot in enumerate(history[:turn_ind + 1]):
        draw_pod_state(axes, snapshot.pod, 1 if state_ind == turn_ind else 0.35, color, show_collision_radius and state_ind == turn_ind,
                       state_ind > 0 and history[state_ind - 1].thrust == "SHIELD")
    if isinstance(history[turn_ind].pod, BrutePod) and history[turn_ind].target_direction is not None:
        if history[turn_ind].base_target_pos is not None:
            axes.plot((history[turn_ind].pod.position[0], history[turn_ind].base_target_pos[0]),
                      (history[turn_ind].pod.position[1], history[turn_ind].base_target_pos[1]), color=color, linestyle=":", linewidth=1.5)
            axes.scatter(history[turn_ind].base_target_pos[0], history[turn_ind].base_target_pos[1], color=color, marker="x", s=180, linewidths=2.5)
        draw_direction_line(axes, history[turn_ind].pod, history[turn_ind].target_direction, color, 0.55)


def draw_direction_line(axes: Axes, pod: BasePod, direction: float, color: str, alpha: float):
    """Draws a dotted visualization ray from a pod along one command direction."""
    vector = np.array(get_direction_vector(direction, bot.TARGET_DISTANCE))
    axes.plot((pod.position[0], pod.position[0] + vector[0]), (pod.position[1], pod.position[1] + vector[1]), color=color, linestyle=":",
              linewidth=1.5, alpha=alpha)


def draw_predictions(axes: Axes, snapshot: TurnSnapshot, checkpoints: list[NDArray[int]], moves: list[float], first_turn: bool, color: str,
                     show_collision_radius: bool, show_racer_avoidance_area: bool):
    """Draws the dashed future path from the selected turn using the current slider moves.
    Predicted positions inside any checkpoint get red outlines, and the final projected score is shown in the map corner.
    """
    if not snapshot.moves:
        return
    future_states = bot.predict_turns(snapshot.pod, checkpoints, moves[:len(snapshot.moves)], first_turn)
    positions = np.array([snapshot.pod.position] + [future_state.pod.position for future_state in future_states])
    axes.plot(positions[:, 0], positions[:, 1], color=color, linestyle="--", linewidth=1)
    axes.text(0.02, 0.98, f"score={round(future_states[-1].get_score(checkpoints))}", color="red", transform=axes.transAxes, ha="left", va="top")
    if show_racer_avoidance_area and isinstance(snapshot.pod, RacerPod):
        direction_delta = bot.constrain_moves([moves[0], 100], first_turn)[0]
        draw_racer_avoidance_area(axes, snapshot.pod, bot.normalize_angle(snapshot.pod.direction + direction_delta), first_turn, color)
    for future_state in future_states:
        edgecolor = "red" if any(linalg.norm(checkpoint - future_state.pod.position) <= OPTIMIZER_CHECKPOINT_RADIUS for checkpoint in checkpoints) else color
        axes.add_patch(Circle(future_state.pod.position, POD_RADIUS, fill=False, edgecolor=edgecolor, linestyle="--", linewidth=1))
        if show_collision_radius and isinstance(future_state.pod, RacerPod):
            axes.add_patch(Circle(future_state.pod.position, POD_COLLISION_RADIUS, fill=False, edgecolor=color, linestyle="--", linewidth=1, alpha=0.45))
        draw_pod_arrows(axes, future_state.pod, 0.5, color)


def draw_racer_avoidance_area(axes: Axes, pod: BasePod, direction: float, first_turn: bool, color: str):
    """Draws the capsule-shaped avoidance area around the racer predicted avoidance segment."""
    segment_end = bot.predict_turns(pod, None, BrutePod.predict_moves(pod, direction), first_turn)[-1].pod.position
    segment_unit = (segment_end - pod.position) / linalg.norm(segment_end - pod.position)
    perpendicular = np.array((-segment_unit[1], segment_unit[0]))
    segment_angle = math.degrees(math.atan2(segment_unit[1], segment_unit[0]))
    area_points = np.array((pod.position + perpendicular * bot.RACER_AVOID_RADIUS, segment_end + perpendicular * bot.RACER_AVOID_RADIUS,
                            segment_end - perpendicular * bot.RACER_AVOID_RADIUS, pod.position - perpendicular * bot.RACER_AVOID_RADIUS))
    axes.add_patch(Polygon(area_points, closed=True, facecolor=color, edgecolor=color, alpha=0.08, linewidth=1, zorder=0.1))
    axes.add_patch(Wedge(pod.position, bot.RACER_AVOID_RADIUS, segment_angle + 90, segment_angle + 270, facecolor=color, edgecolor=color, alpha=0.08,
                         linewidth=1, zorder=0.1))
    axes.add_patch(Wedge(segment_end, bot.RACER_AVOID_RADIUS, segment_angle - 90, segment_angle + 90, facecolor=color, edgecolor=color, alpha=0.08,
                         linewidth=1, zorder=0.1))


def draw_closest_brute_approach(axes: Axes, history: list[TurnSnapshot], extra_histories: list[tuple[list[TurnSnapshot], str]], turn_ind: int,
                                moves: list[float], checkpoints: list[NDArray[int]], show_collision_radius: bool):
    """Draws a predicted-style brute marker at synchronized closest approach to the racer predicted segment."""
    if not history[turn_ind].moves:
        return
    direction_delta = bot.constrain_moves([moves[0], 100], turn_ind == 0)[0]
    racer_direction = bot.normalize_angle(history[turn_ind].pod.direction + direction_delta)
    for extra_history, color in extra_histories:
        snapshot = extra_history[min(turn_ind, len(extra_history) - 1)]
        if isinstance(snapshot.pod, BrutePod) and snapshot.target_direction is not None:
            brute_moves = BrutePod.predict_moves(snapshot.pod, snapshot.target_direction)
            racer_moves = BrutePod.predict_moves(history[turn_ind].pod, racer_direction)
            brute_segment_end = bot.predict_turns(snapshot.pod, None, brute_moves, turn_ind == 0)[-1].pod.position
            racer_segment_end = bot.predict_turns(history[turn_ind].pod, None, racer_moves, turn_ind == 0)[-1].pod.position
            relative_position = snapshot.pod.position - history[turn_ind].pod.position
            relative_velocity = brute_segment_end - snapshot.pod.position - racer_segment_end + history[turn_ind].pod.position
            if not np.any(relative_velocity):
                closest_time = 0
            else:
                closest_time = np.clip(-np.dot(relative_position, relative_velocity) / np.dot(relative_velocity, relative_velocity), 0, 1)
            position = snapshot.pod.position + (brute_segment_end - snapshot.pod.position) * closest_time
            pod = type(snapshot.pod)(snapshot.pod.ind, position, snapshot.pod.velocity, snapshot.pod.direction, snapshot.pod.next_checkpoint_ind,
                                     snapshot.pod.passed_checkpoints)
            axes.add_patch(Circle(pod.position, POD_RADIUS, fill=False, edgecolor=color, linestyle="--", linewidth=1))
            if show_collision_radius:
                axes.add_patch(Circle(pod.position, POD_COLLISION_RADIUS, fill=False, edgecolor=color, linestyle="--", linewidth=1, alpha=0.45))
            draw_pod_arrows(axes, pod, 0.5, color)


def draw_pod_state(axes: Axes, pod: BasePod, alpha: float, color: str, show_collision_radius: bool, shield: bool):
    """Draws the pod center, collision radius and facing direction at a chosen opacity."""
    axes.add_patch(Circle(pod.position, POD_RADIUS, color=color, alpha=alpha))
    if show_collision_radius:
        axes.add_patch(Circle(pod.position, POD_COLLISION_RADIUS, fill=False, edgecolor=color, linestyle="-" if shield else "--", linewidth=1, alpha=alpha))
    draw_pod_arrows(axes, pod, alpha, color)


def draw_pod_arrows(axes: Axes, pod: BasePod, alpha: float, color: str):
    """Draws the facing-direction arrow for one pod without drawing velocity."""
    draw_arrow(axes, pod.position, get_direction_vector(pod.direction, DIRECTION_ARROW_LENGTH), color, alpha)


def draw_arrow(axes: Axes, position: NDArray[float], vector: NDArray[float] | tuple[float, float], color: str, alpha: float):
    """Draws one fixed-style matplotlib arrow from a start position along a vector."""
    axes.arrow(position[0], position[1], vector[0], vector[1], color=color, alpha=alpha, width=ARROW_WIDTH, head_width=ARROW_HEAD_WIDTH,
               length_includes_head=True)


def get_direction_vector(direction: float, length: float) -> tuple[float, float]:
    """Converts a bot direction angle and visual length into a screen-space vector."""
    return math.cos(math.radians(direction)) * length, -math.sin(math.radians(direction)) * length


if __name__ == "__main__":
    main()
