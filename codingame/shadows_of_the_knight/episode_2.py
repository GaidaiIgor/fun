"""Solves CodinGame Shadows of the Knight episode 2 with exact integer geometry."""

from dataclasses import dataclass, field
from math import hypot
import sys


@dataclass(frozen=True, slots=True)
class Point:
    """
    Stores one building coordinate.

    :var x: Horizontal coordinate.
    :var y: Vertical coordinate.
    """

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Constraint:
    """
    Stores a linear constraint over possible bomb coordinates.

    :var a: Coefficient of the horizontal coordinate.
    :var b: Coefficient of the vertical coordinate.
    :var limit: Right side of the comparison.
    :var equal: Whether the comparison is equality instead of less-than-or-equal.
    """

    a: int
    b: int
    limit: int
    equal: bool


@dataclass(frozen=True, slots=True)
class Segment:
    """
    Stores a vertical interval of possible bomb coordinates for one column.

    :var x: Horizontal coordinate shared by the interval.
    :var y_min: Lowest possible vertical coordinate.
    :var y_max: Highest possible vertical coordinate.
    """

    x: int
    y_min: int
    y_max: int


@dataclass(slots=True)
class CandidateStats:
    """
    Stores exact aggregate data for all currently possible bomb coordinates.

    :var count: Number of possible coordinates.
    :var sum_x: Sum of horizontal coordinates over all possible coordinates.
    :var sum_y: Sum of vertical coordinates over all possible coordinates.
    :var min_x: Smallest possible horizontal coordinate.
    :var max_x: Largest possible horizontal coordinate.
    :var min_y: Smallest possible vertical coordinate.
    :var max_y: Largest possible vertical coordinate.
    :var segments: Vertical intervals containing every possible coordinate.
    """

    count: int
    sum_x: int
    sum_y: int
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    segments: list[Segment]

    def add_segment(self, x: int, y_min: int, y_max: int):
        """
        Adds one vertical segment to the aggregate.

        :param x: Horizontal coordinate shared by the segment.
        :param y_min: Lowest vertical coordinate in the segment.
        :param y_max: Highest vertical coordinate in the segment.
        """

        count = y_max - y_min + 1
        self.count += count
        self.sum_x += x * count
        self.sum_y += (y_min + y_max) * count // 2
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y_min)
        self.max_y = max(self.max_y, y_max)
        self.segments.append(Segment(x, y_min, y_max))


@dataclass(slots=True)
class ThermalSearch:
    """
    Tracks and queries the possible bomb coordinates.

    :var width: Building width.
    :var height: Building height.
    :var previous: Position before the latest jump.
    :var current: Current Batman position.
    :var constraints: Device constraints gathered so far.
    :var excluded: Already visited coordinates that did not end the game.
    """

    width: int
    height: int
    previous: Point
    current: Point
    constraints: list[Constraint] = field(default_factory=list)
    excluded: dict[int, set[int]] = field(default_factory=dict)

    def apply_feedback(self, feedback: str):
        """
        Adds the latest device feedback to the search state.

        :param feedback: Device code returned after the latest jump.
        """

        if feedback == "UNKNOWN":
            return
        self.excluded.setdefault(self.current.x, set()).add(self.current.y)
        dx = self.current.x - self.previous.x
        dy = self.current.y - self.previous.y
        a = -2 * dx
        b = -2 * dy
        distance_delta = self.current.x * self.current.x + self.current.y * self.current.y \
            - self.previous.x * self.previous.x - self.previous.y * self.previous.y
        if feedback == "WARMER":
            self.constraints.append(Constraint(a, b, -1 - distance_delta, False))
        elif feedback == "COLDER":
            self.constraints.append(Constraint(-a, -b, distance_delta - 1, False))
        else:
            self.constraints.append(Constraint(a, b, -distance_delta, True))

    def choose_next(self) -> Point:
        """
        Chooses the next jump by minimizing the largest possible feedback bucket.

        :return: Next coordinate to print.
        """

        stats = self.calculate_stats()
        if stats.count == 1:
            segment = stats.segments[0]
            return Point(segment.x, segment.y_min)
        jumps = self.candidate_jumps(stats)
        return min(jumps, key=lambda point: self.score_jump(point, stats))

    def calculate_stats(self) -> CandidateStats:
        """
        Calculates exact integer candidates as vertical intervals.

        :return: Aggregate candidate information.
        """

        stats = CandidateStats(0, 0, 0, self.width, -1, self.height, -1, [])
        for x in range(self.width):
            y_min = 0
            y_max = self.height - 1
            possible = True
            for constraint in self.constraints:
                remaining = constraint.limit - constraint.a * x
                if constraint.equal:
                    if constraint.b == 0:
                        possible = remaining == 0
                    elif remaining % constraint.b:
                        possible = False
                    else:
                        y = remaining // constraint.b
                        y_min = max(y_min, y)
                        y_max = min(y_max, y)
                elif constraint.b > 0:
                    y_max = min(y_max, remaining // constraint.b)
                elif constraint.b < 0:
                    y_min = max(y_min, ceil_div(remaining, constraint.b))
                elif remaining < 0:
                    possible = False
                if not possible or y_min > y_max:
                    possible = False
                    break
            if possible:
                self.add_column_segments(stats, x, y_min, y_max)
        return stats

    def add_column_segments(self, stats: CandidateStats, x: int, y_min: int, y_max: int):
        """
        Adds a column interval after removing known missed jumps.

        :param stats: Aggregate candidate information to mutate.
        :param x: Horizontal coordinate of the interval.
        :param y_min: Lowest possible vertical coordinate.
        :param y_max: Highest possible vertical coordinate.
        """

        start = y_min
        for y in sorted(self.excluded.get(x, set())):
            if y < start:
                continue
            if y > y_max:
                break
            if start < y:
                stats.add_segment(x, start, y - 1)
            start = y + 1
        if start <= y_max:
            stats.add_segment(x, start, y_max)

    def candidate_jumps(self, stats: CandidateStats) -> list[Point]:
        """
        Builds a compact set of promising jumps to score exactly.

        :param stats: Aggregate candidate information.
        :return: Candidate jump coordinates.
        """

        jumps = []
        seen = set()
        center_x = stats.sum_x / stats.count
        center_y = stats.sum_y / stats.count
        reflected_x = 2 * center_x - self.current.x
        reflected_y = 2 * center_y - self.current.y
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self.append_jump(jumps, seen, reflected_x + dx, reflected_y + dy)
        self.append_ray_jumps(jumps, seen, center_x, center_y)
        self.append_circle_jumps(jumps, seen, center_x, center_y)
        middle_x = (stats.min_x + stats.max_x) // 2
        middle_y = (stats.min_y + stats.max_y) // 2
        for x in (stats.min_x, middle_x, stats.max_x):
            for y in (stats.min_y, middle_y, stats.max_y):
                self.append_jump(jumps, seen, x, y)
        for x in (0, self.width - 1):
            for y in (0, self.height - 1):
                self.append_jump(jumps, seen, x, y)
        if stats.count <= 100:
            for segment in stats.segments:
                for y in range(segment.y_min, segment.y_max + 1):
                    self.append_jump(jumps, seen, segment.x, y)
        return jumps

    def append_ray_jumps(self, jumps: list[Point], seen: set[Point], center_x: float, center_y: float):
        """
        Adds jumps lying on the ray from the current point through the candidate centroid.

        :param jumps: Candidate list to mutate.
        :param seen: Coordinates already present in the candidate list.
        :param center_x: Horizontal centroid of possible bomb coordinates.
        :param center_y: Vertical centroid of possible bomb coordinates.
        """

        vx = center_x - self.current.x
        vy = center_y - self.current.y
        if not vx and not vy:
            return
        limits = []
        if vx > 0:
            limits.append((self.width - 1 - self.current.x) / vx)
        elif vx < 0:
            limits.append(-self.current.x / vx)
        if vy > 0:
            limits.append((self.height - 1 - self.current.y) / vy)
        elif vy < 0:
            limits.append(-self.current.y / vy)
        limit = min(limits)
        for factor in (min(2, limit), limit, max(1, limit / 2)):
            self.append_jump(jumps, seen, self.current.x + factor * vx, self.current.y + factor * vy)

    def append_circle_jumps(self, jumps: list[Point], seen: set[Point], center_x: float, center_y: float):
        """
        Adds jumps whose bisector with the current point passes near the centroid.

        :param jumps: Candidate list to mutate.
        :param seen: Coordinates already present in the candidate list.
        :param center_x: Horizontal centroid of possible bomb coordinates.
        :param center_y: Vertical centroid of possible bomb coordinates.
        """

        radius = hypot(self.current.x - center_x, self.current.y - center_y)
        if not radius:
            return
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            length = hypot(dx, dy)
            self.append_jump(jumps, seen, center_x + radius * dx / length, center_y + radius * dy / length)

    def append_jump(self, jumps: list[Point], seen: set[Point], x: float, y: float):
        """
        Adds one in-building jump candidate if it is useful and new.

        :param jumps: Candidate list to mutate.
        :param seen: Coordinates already present in the candidate list.
        :param x: Desired horizontal coordinate.
        :param y: Desired vertical coordinate.
        """

        point = Point(clamp(round(x), 0, self.width - 1), clamp(round(y), 0, self.height - 1))
        if point != self.current and point not in seen:
            seen.add(point)
            jumps.append(point)

    def score_jump(self, point: Point, stats: CandidateStats) -> tuple[int, int, int, int]:
        """
        Scores a jump by exact worst-case remaining candidate count.

        :param point: Jump to score.
        :param stats: Aggregate candidate information.
        :return: Sortable score where lower is better.
        """

        warmer = 0
        same = 0
        colder = 0
        a = -2 * (point.x - self.current.x)
        b = -2 * (point.y - self.current.y)
        distance_delta = point.x * point.x + point.y * point.y - self.current.x * self.current.x - self.current.y * self.current.y
        for segment in stats.segments:
            base = a * segment.x + distance_delta
            lower_or_same = count_linear_le(base, b, segment.y_min, segment.y_max, 0)
            lower = count_linear_le(base, b, segment.y_min, segment.y_max, -1)
            warmer += lower
            same += lower_or_same - lower
            colder += segment.y_max - segment.y_min + 1 - lower_or_same
        hit = self.is_possible(point)
        if hit:
            warmer -= 1
        distance = (point.x - self.current.x) * (point.x - self.current.x) + (point.y - self.current.y) * (point.y - self.current.y)
        return max(warmer, same, colder), same, -int(hit), -distance

    def is_possible(self, point: Point) -> bool:
        """
        Checks whether a coordinate is still a possible bomb position.

        :param point: Coordinate to check.
        :return: Whether the coordinate satisfies all known information.
        """

        if point.y in self.excluded.get(point.x, set()):
            return False
        for constraint in self.constraints:
            value = constraint.a * point.x + constraint.b * point.y
            if constraint.equal and value != constraint.limit:
                return False
            if not constraint.equal and value > constraint.limit:
                return False
        return True

    def jump(self, point: Point):
        """
        Records the jump just printed.

        :param point: New Batman position.
        """

        self.previous = self.current
        self.current = point


def ceil_div(value: int, divisor: int) -> int:
    """
    Calculates mathematical ceiling division.

    :param value: Dividend.
    :param divisor: Non-zero divisor.
    :return: Smallest integer not less than value divided by divisor.
    """

    return -((-value) // divisor)


def clamp(value: int, low: int, high: int) -> int:
    """
    Clamps an integer into an inclusive interval.

    :param value: Value to clamp.
    :param low: Lowest accepted value.
    :param high: Highest accepted value.
    :return: Clamped value.
    """

    return min(max(value, low), high)


def count_linear_le(base: int, step: int, lower: int, upper: int, limit: int) -> int:
    """
    Counts integer y values satisfying base plus step times y is at most a limit.

    :param base: Constant part of the linear expression.
    :param step: Coefficient of y.
    :param lower: Lowest y to count.
    :param upper: Highest y to count.
    :param limit: Inclusive expression limit.
    :return: Number of satisfying integer y values.
    """

    if step > 0:
        return max(0, min(upper, (limit - base) // step) - lower + 1)
    if step < 0:
        return max(0, upper - max(lower, ceil_div(limit - base, step)) + 1)
    if base <= limit:
        return upper - lower + 1
    return 0


def main():
    """Runs the interactive puzzle loop."""

    lines = sys.stdin
    width, height = map(int, lines.readline().split())
    lines.readline()
    x, y = map(int, lines.readline().split())
    search = ThermalSearch(width, height, Point(x, y), Point(x, y))
    for feedback in lines:
        search.apply_feedback(feedback.strip())
        point = search.choose_next()
        print(f"{point.x} {point.y}", flush=True)
        search.jump(point)


if __name__ == "__main__":
    main()
