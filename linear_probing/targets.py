"""Geometry-derived supervision targets for representation probing."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np


TASK_TYPES = {
    'agent_velocity': 'regression',
    'signed_slip_angle': 'circular_regression',
    'agent_kinetic_energy': 'regression',
    'goal_distance': 'regression',
    'signed_goal_angle': 'circular_regression',
    'nearest_hazard_distance': 'regression',
    'direct_path_blocked': 'classification',
    'direct_path_hazard_count': 'regression',
    'direct_path_signed_clearance': 'regression',
    'minimum_evasive_turn': 'regression',
    'best_avoidance_side': 'classification',
    'safe_path_detour_ratio': 'regression',
    'hazard_dispersion': 'regression',
}

TARGET_DEFINITIONS = {
    'agent_velocity': 'Forward/lateral agent-frame velocimeter components.',
    'signed_slip_angle': (
        'Signed shortest angle from agent heading to planar world velocity; undefined below the '
        'configured speed threshold.'
    ),
    'agent_kinetic_energy': (
        'Agent-subtree generalized kinetic energy 0.5 * qvel.T @ M(q) @ qvel.'
    ),
    'goal_distance': 'Euclidean distance between agent and goal XY centres.',
    'signed_goal_angle': 'Signed shortest angle from agent heading to the goal direction.',
    'nearest_hazard_distance': 'Centre-to-centre distance to the nearest hazard.',
    'direct_path_blocked': (
        'Whether an agent-radius-expanded hazard intersects the agent-to-goal segment.'
    ),
    'direct_path_hazard_count': 'Number of expanded hazards intersecting that segment.',
    'direct_path_signed_clearance': (
        'Minimum hazard-centre distance to that segment minus the expanded hazard radius.'
    ),
    'minimum_evasive_turn': (
        'Smallest signed goal-relative angle whose goal-length segment clears every hazard.'
    ),
    'best_avoidance_side': 'Sign of the minimum evasive turn; undefined on clear direct paths.',
    'safe_path_detour_ratio': (
        'Collision-checked grid-path length divided by straight-line goal distance; exactly one '
        'when the direct segment is clear.'
    ),
    'hazard_dispersion': (
        'Mean nearest-neighbour hazard distance, min-max normalized over this dataset.'
    ),
}

SLIP_SPEED_EPSILON = 1e-4

TASK_GROUPS = {
    'agent_related': {
        'easy': ('agent_velocity',),
        'medium': ('signed_slip_angle',),
        'hard': ('agent_kinetic_energy',),
    },
    'goal_related': {
        'easy': ('goal_distance', 'signed_goal_angle'),
    },
    'hazard_related': {
        'easy': ('nearest_hazard_distance',),
        'medium': (
            'direct_path_blocked',
            'direct_path_hazard_count',
            'direct_path_signed_clearance',
        ),
        'hard': (
            'minimum_evasive_turn',
            'best_avoidance_side',
            'safe_path_detour_ratio',
            'hazard_dispersion',
        ),
    },
}


def validate_planner_config(config: dict[str, Any]) -> None:
    """Reject planner settings that would make geometric labels ambiguous."""
    bounds = np.asarray(config['bounds'], dtype=np.float64)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError('Planner bounds must contain four finite values.')
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError('Planner bounds must satisfy xmin < xmax and ymin < ymax.')
    resolution = float(config['grid_resolution'])
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError('Planner grid resolution must be finite and positive.')
    step = float(config['avoidance_angle_step_degrees'])
    maximum = float(config['maximum_avoidance_angle_degrees'])
    if not 0.0 < step <= maximum <= 180.0:
        raise ValueError('Avoidance angles must satisfy 0 < step <= maximum <= 180 degrees.')


@dataclass(frozen=True)
class Scene:
    """Minimal world state required by all probe targets."""

    agent_xy: np.ndarray
    heading_xy: np.ndarray
    agent_velocity_xy: np.ndarray
    agent_velocity: np.ndarray
    goal_xy: np.ndarray
    hazards_xy: np.ndarray
    agent_radius: float
    hazard_radius: float
    goal_radius: float

    @property
    def expanded_hazard_radius(self) -> float:
        """Hazard radius after accounting for the actor footprint."""
        return self.agent_radius + self.hazard_radius


def signed_angle(source: np.ndarray, target: np.ndarray) -> float:
    """Return the signed shortest angle from ``source`` to ``target`` in radians."""
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    cross = source[0] * target[1] - source[1] * target[0]
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return float(np.arctan2(cross, dot))


def point_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Compute distances from points to a finite line segment."""
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return np.linalg.norm(points - start, axis=1)
    projection = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
    closest = start + projection[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def direct_path_clearances(scene: Scene, direction: np.ndarray | None = None) -> np.ndarray:
    """Return signed hazard clearances along a goal-length path segment."""
    goal_vector = scene.goal_xy - scene.agent_xy
    goal_distance = float(np.linalg.norm(goal_vector))
    if direction is None:
        endpoint = scene.goal_xy
    else:
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        endpoint = scene.agent_xy + direction * goal_distance
    distances = point_segment_distances(scene.hazards_xy, scene.agent_xy, endpoint)
    return distances - scene.expanded_hazard_radius


def hazard_dispersion_raw(hazards_xy: np.ndarray) -> float:
    """Mean nearest-neighbour distance between hazards."""
    if len(hazards_xy) < 2:
        return 0.0
    pairwise = np.linalg.norm(hazards_xy[:, None, :] - hazards_xy[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    return float(np.min(pairwise, axis=1).mean())


def local_targets(scene: Scene, planner_config: dict[str, Any]) -> dict[str, Any]:
    """Compute all targets except the grid-based safe-path detour ratio."""
    goal_vector = scene.goal_xy - scene.agent_xy
    goal_distance = float(np.linalg.norm(goal_vector))
    goal_direction = goal_vector / max(goal_distance, 1e-12)
    clearances = direct_path_clearances(scene)
    blocked = bool(len(clearances) and np.min(clearances) <= 0.0)

    if len(scene.hazards_xy):
        nearest_hazard_distance = float(
            np.min(np.linalg.norm(scene.hazards_xy - scene.agent_xy, axis=1)),
        )
        signed_clearance = float(np.min(clearances))
    else:
        nearest_hazard_distance = np.nan
        signed_clearance = np.nan

    evasive_turn, avoidance_side = minimum_evasive_turn(scene, planner_config, blocked)
    planar_speed = float(np.linalg.norm(scene.agent_velocity_xy))
    slip_angle = (
        signed_angle(scene.heading_xy, scene.agent_velocity_xy)
        if planar_speed > SLIP_SPEED_EPSILON
        else np.nan
    )
    return {
        'nearest_hazard_distance': nearest_hazard_distance,
        'goal_distance': goal_distance,
        'signed_goal_angle': (
            signed_angle(scene.heading_xy, goal_direction) if goal_distance > 1e-8 else np.nan
        ),
        'agent_velocity': scene.agent_velocity,
        'signed_slip_angle': slip_angle,
        'direct_path_blocked': float(blocked),
        'direct_path_hazard_count': float(np.count_nonzero(clearances <= 0.0)),
        'direct_path_signed_clearance': signed_clearance,
        'minimum_evasive_turn': evasive_turn,
        'best_avoidance_side': avoidance_side,
        'hazard_dispersion_raw': hazard_dispersion_raw(scene.hazards_xy),
    }


def minimum_evasive_turn(
    scene: Scene,
    planner_config: dict[str, Any],
    direct_path_blocked: bool,
) -> tuple[float, float]:
    """Find the smallest goal-relative turn yielding a clear goal-length segment.

    Positive angles/labels denote a left (counter-clockwise) turn and negative values denote
    right. The side target is undefined for an already-clear direct path and is stored as NaN.
    """
    if not direct_path_blocked:
        return 0.0, np.nan

    step_degrees = float(planner_config['avoidance_angle_step_degrees'])
    maximum_degrees = float(planner_config['maximum_avoidance_angle_degrees'])
    if not 0.0 < step_degrees <= maximum_degrees <= 180.0:
        raise ValueError('Avoidance angles must satisfy 0 < step <= maximum <= 180 degrees.')
    goal_vector = scene.goal_xy - scene.agent_xy
    goal_direction = goal_vector / max(float(np.linalg.norm(goal_vector)), 1e-12)
    step = np.deg2rad(step_degrees)
    maximum = np.deg2rad(maximum_degrees)
    candidate_angles = np.arange(step, maximum + 1e-12, step)

    for magnitude in candidate_angles:
        candidates = []
        for angle in (-magnitude, magnitude):
            cosine, sine = np.cos(angle), np.sin(angle)
            direction = np.array(
                [
                    cosine * goal_direction[0] - sine * goal_direction[1],
                    sine * goal_direction[0] + cosine * goal_direction[1],
                ],
            )
            clearances = direct_path_clearances(scene, direction)
            minimum_clearance = float(np.min(clearances)) if len(clearances) else np.inf
            if minimum_clearance > 0.0:
                candidates.append((minimum_clearance, float(angle)))
        if candidates:
            _, best_angle = max(candidates)
            return best_angle, float(np.sign(best_angle))
    return np.nan, np.nan


class GridPathPlanner:
    """Small 8-connected grid planner used only to generate supervision labels."""

    _NEIGHBOURS = (
        (-1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, np.sqrt(2.0)),
    )

    def __init__(self, scene: Scene, planner_config: dict[str, Any]) -> None:
        validate_planner_config(planner_config)
        bounds = np.asarray(planner_config['bounds'], dtype=np.float64)
        xmin, xmax, ymin, ymax = bounds
        self.resolution = float(planner_config['grid_resolution'])
        self.bounds = bounds
        self.hazards_xy = scene.hazards_xy
        self.expanded_hazard_radius = scene.expanded_hazard_radius
        self.goal_xy = scene.goal_xy
        epsilon = self.resolution * 1e-9
        self.xs = np.arange(xmin, xmax + epsilon, self.resolution)
        self.ys = np.arange(ymin, ymax + epsilon, self.resolution)
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)
        grid = np.stack((grid_x, grid_y), axis=-1)

        occupied = np.zeros(grid.shape[:2], dtype=bool)
        for hazard in scene.hazards_xy:
            occupied |= np.linalg.norm(grid - hazard, axis=2) <= scene.expanded_hazard_radius

        self.occupied = occupied
        self.goal_index: tuple[int, int] | None = None
        self.distance_to_goal: np.ndarray | None = None

    def _point_in_bounds(self, point: np.ndarray) -> bool:
        xmin, xmax, ymin, ymax = self.bounds
        return bool(xmin <= point[0] <= xmax and ymin <= point[1] <= ymax)

    def _point_is_clear(self, point: np.ndarray) -> bool:
        if not len(self.hazards_xy):
            return True
        return bool(
            np.all(
                np.linalg.norm(self.hazards_xy - point, axis=1)
                > self.expanded_hazard_radius,
            ),
        )

    def _segment_is_clear(self, start: np.ndarray, end: np.ndarray) -> bool:
        distances = point_segment_distances(self.hazards_xy, start, end)
        return bool(np.all(distances > self.expanded_hazard_radius))

    def _point(self, index: tuple[int, int]) -> np.ndarray:
        return np.array([self.xs[index[1]], self.ys[index[0]]])

    def _neighbours(
        self,
        index: tuple[int, int],
    ) -> list[tuple[tuple[int, int], float]]:
        row, column = index
        neighbours = []
        for row_delta, column_delta, multiplier in self._NEIGHBOURS:
            neighbour = row + row_delta, column + column_delta
            if not (0 <= neighbour[0] < len(self.ys) and 0 <= neighbour[1] < len(self.xs)):
                continue
            if self.occupied[neighbour]:
                continue
            if not self._segment_is_clear(self._point(index), self._point(neighbour)):
                continue
            neighbours.append((neighbour, multiplier))
        return neighbours

    def _connector_index(self, point: np.ndarray) -> tuple[int, int] | None:
        """Return the closest free grid node connected by a collision-free segment."""
        if not self._point_in_bounds(point) or not self._point_is_clear(point):
            return None
        free_indices = np.argwhere(~self.occupied)
        points = np.column_stack((self.xs[free_indices[:, 1]], self.ys[free_indices[:, 0]]))
        for candidate in np.argsort(np.linalg.norm(points - point, axis=1)):
            index = tuple(free_indices[candidate])
            if self._segment_is_clear(point, self._point(index)):
                return index
        return None

    def _ensure_distance_field(self) -> bool:
        if self.distance_to_goal is not None:
            return self.goal_index is not None
        self.goal_index = self._connector_index(self.goal_xy)
        if self.goal_index is None:
            self.distance_to_goal = np.full(self.occupied.shape, np.inf)
            return False
        self.distance_to_goal = self._distance_field(self.goal_index, self.goal_xy)
        return True

    def _distance_field(
        self,
        goal_index: tuple[int, int],
        goal_xy: np.ndarray,
    ) -> np.ndarray:
        distances = np.full(self.occupied.shape, np.inf)
        distances[goal_index] = float(np.linalg.norm(self._point(goal_index) - goal_xy))
        queue = [(distances[goal_index], 0, goal_index)]
        sequence = count(1)

        while queue:
            distance, _, (row, column) = heapq.heappop(queue)
            if distance != distances[row, column]:
                continue
            for neighbour, multiplier in self._neighbours((row, column)):
                candidate = distance + multiplier * self.resolution
                if candidate < distances[neighbour]:
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, next(sequence), neighbour))
        return distances

    def detour_ratio(self, agent_xy: np.ndarray, goal_xy: np.ndarray) -> float:
        """Return planned path length divided by straight-line distance."""
        if not np.allclose(goal_xy, self.goal_xy):
            raise ValueError('The planner can only be queried for the goal it was built with.')
        if not self._point_in_bounds(agent_xy) or not self._point_in_bounds(goal_xy):
            raise ValueError('Agent and goal positions must lie inside the configured bounds.')
        direct_distance = float(np.linalg.norm(goal_xy - agent_xy))
        if direct_distance <= 1e-12:
            return 1.0
        if not self._point_is_clear(agent_xy) or not self._point_is_clear(goal_xy):
            return np.nan
        if self._segment_is_clear(agent_xy, goal_xy):
            return 1.0
        if not self._ensure_distance_field():
            return np.nan

        start = self._connector_index(agent_xy)
        if start is None:
            return np.nan

        assert self.distance_to_goal is not None
        grid_distance = float(self.distance_to_goal[start])
        if not np.isfinite(grid_distance):
            return np.nan
        path_length = grid_distance + float(np.linalg.norm(self._point(start) - agent_xy))
        ratio = path_length / direct_distance
        if ratio < 1.0 - 1e-9:
            raise RuntimeError('A planned polyline cannot be shorter than the direct path.')
        return max(1.0, ratio)

    def shortest_path(self, agent_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        """Reconstruct one shortest grid path for qualitative visualization."""
        if not np.allclose(goal_xy, self.goal_xy):
            raise ValueError('The planner can only be queried for the goal it was built with.')
        if not self._point_in_bounds(agent_xy) or not self._point_in_bounds(goal_xy):
            return np.empty((0, 2))
        if self._segment_is_clear(agent_xy, goal_xy):
            return np.asarray((agent_xy, goal_xy))
        if not self._ensure_distance_field():
            return np.empty((0, 2))
        current = self._connector_index(agent_xy)
        assert self.distance_to_goal is not None
        if current is None or not np.isfinite(self.distance_to_goal[current]):
            return np.empty((0, 2))

        path = [np.asarray(agent_xy)]
        current_point = self._point(current)
        if not np.allclose(path[-1], current_point):
            path.append(current_point)
        for _ in range(self.occupied.size):
            if current == self.goal_index:
                if not np.allclose(path[-1], goal_xy):
                    path.append(np.asarray(goal_xy))
                return np.asarray(path)
            neighbours = self._neighbours(current)
            if not neighbours:
                break
            current = min(
                neighbours,
                key=lambda item: self.distance_to_goal[item[0]] + item[1] * self.resolution,
            )[0]
            path.append(self._point(current))
        return np.empty((0, 2))
