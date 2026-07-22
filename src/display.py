# Antiquated and obsolete display code


# import pygame

# from src.map import Map


# class Display:

#     COLORS = {
#         "free": (235, 235, 235),
#         "obstacle": (40, 40, 40),
#         "seen": (170, 205, 255),
#         "robot": (220, 60, 60),
#         "grid": (205, 205, 205),
#     }

#     def __init__(self, title="Display", cell_size=40, fps=30, show_grid=True):
#         self.title = title
#         self.cell_size = cell_size
#         self.fps = fps
#         self.show_grid = show_grid
#         self.screen = None
#         self.clock = None
#         self._running = False

#     def _ensure_init(self, map: Map):
#         if self.screen is not None:
#             return
#         pygame.init()
#         w = map.width * self.cell_size
#         h = map.height * self.cell_size
#         self.screen = pygame.display.set_mode((w, h))
#         pygame.display.set_caption(self.title)
#         self.clock = pygame.time.Clock()
#         self._running = True

#     def _pump_events(self):
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT or (
#                 event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
#             ):
#                 self.close()
#                 return False
#         return True

#     def render(self, map: Map, robot=None):
#         """Draw one frame. Returns False if the window was closed."""
#         self._ensure_init(map)
#         if not self._running or not self._pump_events():
#             return False

#         cs = self.cell_size
#         seen = set(tuple(s) for s in robot.seen) if robot is not None else set()

#         self.screen.fill(self.COLORS["free"])
#         for row in range(map.height):
#             for col in range(map.width):
#                 if map.grid[row, col] == 1:
#                     color = self.COLORS["obstacle"]
#                 elif (row, col) in seen:
#                     color = self.COLORS["seen"]
#                 else:
#                     continue
#                 pygame.draw.rect(
#                     self.screen, color, (col * cs, row * cs, cs, cs)
#                 )

#         if robot is not None:
#             row, col = robot.position
#             pygame.draw.rect(
#                 self.screen,
#                 self.COLORS["robot"],
#                 (col * cs, row * cs, cs, cs),
#             )

#         if self.show_grid and cs >= 6:
#             grid_color = self.COLORS["grid"]
#             for col in range(map.width + 1):
#                 x = col * cs
#                 pygame.draw.line(self.screen, grid_color, (x, 0), (x, map.height * cs))
#             for row in range(map.height + 1):
#                 y = row * cs
#                 pygame.draw.line(self.screen, grid_color, (0, y), (map.width * cs, y))

#         pygame.display.flip()
#         self.clock.tick(self.fps)
#         return True

#     def show(self, map: Map, robot=None):
#         """Render and block until the window is closed (drop-in for the old API)."""
#         if not self.render(map, robot):
#             return
#         while self._running:
#             if not self.render(map, robot):
#                 break

#     def close(self):
#         if self.screen is not None:
#             pygame.quit()
#             self.screen = None
#         self._running = False
