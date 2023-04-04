import matplotlib.pyplot as plt

class Plotter2d():
    def __init__(self, bin) -> None:
        self.bin = bin

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.bin.width)
        ax.set_ylim(0, self.bin.height)
        for item in self.bin.items:
            rect = plt.Rectangle((item.x, item.y), item.width, item.height, facecolor='r')
            ax.add_patch(rect)
        plt.show()
