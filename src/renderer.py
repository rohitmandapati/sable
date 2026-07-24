from collections.abc import Sequence

import pygame

from map import Map
from robot import Robot

"""
Ugly PyGame renderer
The robots need pixels
"""

class Renderer:
    def __init__(
        self,
        world_map: Map,
        cell_size: int = 24,
        fps: int = 30,
    ) -> None:
        pygame.init()

        self.world_map = world_map
        self.cell_size = cell_size
        self.fps = fps

        window_width = world_map.width * cell_size
        window_height = world_map.height * cell_size

        self.screen = pygame.display.set_mode(
            (window_width, window_height)
        )

        pygame.display.set_caption("SABLE Grid World")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, max(18, cell_size))

        # Basic colors.
        self.background_color = (30, 30, 30)
        self.free_color = (235, 235, 235)
        self.wall_color = (45, 45, 45)
        self.grid_line_color = (170, 170, 170)

        self.robot_colors = [
            (220, 50, 50),
            (50, 100, 220),
            (50, 180, 80),
            (220, 160, 40),
        ]

    def cell_rectangle(
        self,
        row: int,
        col: int,
    ) -> pygame.Rect:
        """
        Convert a grid cell into its screen rectangle.

        Grid coordinates:
            (row, column)

        Screen coordinates:
            (x, y)
        """
        x = col * self.cell_size
        y = row * self.cell_size

        return pygame.Rect(
            x,
            y,
            self.cell_size,
            self.cell_size,
        )

    def cell_center(
        self,
        position: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Convert (row, column) into the center pixel of that cell.
        """
        row, col = position

        center_x = col * self.cell_size + self.cell_size // 2
        center_y = row * self.cell_size + self.cell_size // 2

        return center_x, center_y

    def draw_map(self) -> None:
        for row in range(self.world_map.height):
            for col in range(self.world_map.width):
                cell_value = self.world_map.grid[row, col]
                rectangle = self.cell_rectangle(row, col)

                if cell_value == 1:
                    color = self.wall_color
                else:
                    color = self.free_color

                pygame.draw.rect(
                    self.screen,
                    color,
                    rectangle,
                )

                pygame.draw.rect(
                    self.screen,
                    self.grid_line_color,
                    rectangle,
                    width=1,
                )

    def draw_trajectory(
        self,
        robot: Robot,
        color: tuple[int, int, int],
    ) -> None:
        if len(robot.trajectory_map) < 2:
            return

        points = [
            self.cell_center(position)
            for position in robot.trajectory_map
        ]

        pygame.draw.lines(
            self.screen,
            color,
            False,
            points,
            width=max(2, self.cell_size // 8),
        )

    def draw_robot(
        self,
        robot: Robot,
        color: tuple[int, int, int],
    ) -> None:
        center = self.cell_center(robot.position)
        radius = max(4, self.cell_size // 3)

        pygame.draw.circle(
            self.screen,
            color,
            center,
            radius,
        )

        # Outline makes the robot visible on any cell color.
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            center,
            radius,
            width=2,
        )

        label = self.font.render(
            robot.robot_id,
            True,
            (255, 255, 255),
        )

        label_position = (
            center[0] - label.get_width() // 2,
            center[1] - radius - label.get_height(),
        )

        self.screen.blit(label, label_position)

    def draw_robots(
        self,
        robots: Sequence[Robot],
    ) -> None:
        for index, robot in enumerate(robots):
            color = self.robot_colors[
                index % len(self.robot_colors)
            ]

            self.draw_trajectory(robot, color)
            self.draw_robot(robot, color)

    def render(self, robots):
        # Service the OS event queue so the window actually paints (required on macOS).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        self.screen.fill(self.background_color)
        self.draw_map()
        self.draw_robots(robots)
        pygame.display.flip()
        self.clock.tick(self.fps)


    def close(self) -> None:
        pygame.quit()