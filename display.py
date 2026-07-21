import matplotlib.pyplot as plt
from map import Map

class Display:
    def __init__(self, title="Display"):
        pass

    def show(self, map: Map):
        import matplotlib.pyplot as plt

        plt.imshow(
            map.grid,
            cmap="gray_r",
            interpolation="nearest",
            vmin=0,
            vmax=2
        )
        plt.show()


