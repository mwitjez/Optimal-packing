import matplotlib.pyplot as plt

class Plotter2d():
    def __init__(self, bin) -> None:
        self.bin = bin
        self.cmap = plt.cm.get_cmap("hsv", len(self.bin.items))

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.bin.width)
        ax.set_ylim(0, self.bin.height)
        for i, item in enumerate(self.bin.items):
            rect = plt.Rectangle((item.x, item.y), item.width, item.height, facecolor=self.cmap(i), edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        plt.show()
