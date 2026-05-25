from typing import Optional, Sequence, Tuple

import numpy as np
from gymnasium import logger
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class GoalMarkedVideoRecorder(VideoRecorder):
    """Video recorder that overlays a green star at a configured goal location."""

    def __init__(
        self,
        env,
        goal_location: Sequence[int],
        grid_shape: Optional[Tuple[int, int]] = None,
        marker_color: Tuple[int, int, int] = (57, 158, 118),
        marker_scale: float = 0.35,
        **kwargs,
    ):
        self.goal_location = (int(goal_location[0]), int(goal_location[1]))
        self.grid_shape = grid_shape
        self.marker_color = marker_color
        self.marker_scale = marker_scale
        super().__init__(env, **kwargs)

    def _to_color(self, frame: np.ndarray) -> np.ndarray:
        if np.issubdtype(frame.dtype, np.floating):
            return np.array(self.marker_color, dtype=frame.dtype) / 255.0
        return np.array(self.marker_color, dtype=frame.dtype)

    @staticmethod
    def _draw_line(
        frame: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: np.ndarray,
        thickness: int,
    ) -> None:
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if steps <= 0:
            return
        xs = np.linspace(x0, x1, steps).astype(int)
        ys = np.linspace(y0, y1, steps).astype(int)
        half = max(0, thickness // 2)

        height, width = frame.shape[:2]
        for x, y in zip(xs, ys):
            y_min = max(0, y - half)
            y_max = min(height, y + half + 1)
            x_min = max(0, x - half)
            x_max = min(width, x + half + 1)
            frame[y_min:y_max, x_min:x_max, :3] = color

    def _overlay_goal_star(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame

        out = np.array(frame, copy=True)
        if out.ndim == 2:
            out = np.repeat(out[..., None], 3, axis=2)
        if out.ndim != 3 or out.shape[2] < 3:
            return out

        if self.grid_shape is not None:
            grid_w, grid_h = self.grid_shape
        elif hasattr(self.env, "width") and hasattr(self.env, "height"):
            grid_w, grid_h = int(self.env.width), int(self.env.height)
        else:
            return out

        gx, gy = self.goal_location
        if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
            return out

        height, width = out.shape[:2]
        cell_w = width / float(grid_w)
        cell_h = height / float(grid_h)

        cx = int((gx + 0.5) * cell_w)
        cy = int((gy + 0.5) * cell_h)

        radius = max(2, int(min(cell_w, cell_h) * self.marker_scale))
        diag = max(1, int(radius * 0.7))
        thickness = max(1, radius // 4)
        color = self._to_color(out)

        points = [
            (cx, cy - radius),
            (cx, cy + radius),
            (cx - radius, cy),
            (cx + radius, cy),
            (cx - diag, cy - diag),
            (cx + diag, cy + diag),
            (cx - diag, cy + diag),
            (cx + diag, cy - diag),
        ]

        for px, py in points:
            self._draw_line(out, cx, cy, px, py, color, thickness)

        return out

    def capture_frame(self):
        """Render environment frame and store it with a goal marker overlay."""
        frame = self.env.render()
        if isinstance(frame, list):
            self.render_history += frame
            frame = frame[-1]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return

        if frame is None:
            if self._async:
                return
            logger.warn(
                "Env returned None on render(). Disabling video recorder for this run."
            )
            self.broken = True
            return

        self.recorded_frames.append(self._overlay_goal_star(frame))
