from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


class Plotter3d():
    def __init__(self, bin) -> None:
        self.bin = bin
        self.cmap = plt.cm.get_cmap("hsv", len(self.bin.items))

    def plot(self):
        """Plots the bin and the items in it."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.bin.width)
        ax.set_ylim(0, self.bin.height)
        ax.set_zlim3d(0, self.bin.depth)
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        for i, item in enumerate(self.bin.items):
            vertices = self._get_vertices(item)
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]]
            ]
            poly = Poly3DCollection(faces, edgecolors="black", facecolors=self.cmap(i), linewidths=1, alpha=0.5)
            ax.add_collection3d(poly)
        plt.show()

    def _get_vertices(self, item):
        """Returns the vertices of the item."""

        vertices = [
            (item.x, item.y, item.z),
            (item.x + item.width, item.y, item.z),
            (item.x + item.width, item.y + item.height, item.z),
            (item.x, item.y + item.height, item.z),
            (item.x, item.y, item.z + item.depth),
            (item.x + item.width, item.y, item.z + item.depth),
            (item.x + item.width, item.y + item.height, item.z + item.depth),
            (item.x, item.y + item.height, item.z + item.depth)
        ]
        return vertices


class NetworkPlotter3d():
    def __init__(self, cuboids, bin_size) -> None:
        self.bin_size = bin_size
        self.cuboids = cuboids
        self.cmap = plt.cm.get_cmap("hsv", len(self.cuboids))

    def plot(self):
        """Plots the bin and the items in it."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.bin_size[0])
        ax.set_ylim(0, self.bin_size[1])
        ax.set_zlim3d(0, self.bin_size[2]+30)
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        for i, item in enumerate(self.cuboids):
            vertices = self._get_vertices(item)
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]]
            ]
            poly = Poly3DCollection(faces, edgecolors="black", facecolors=self.cmap(i), linewidths=1, alpha=0.5)
            ax.add_collection3d(poly)
        plt.show()

    def _get_vertices(self, item):
        """Returns the vertices of the item."""

        vertices = [
            (item.x, item.y, item.z),
            (item.x + item.width, item.y, item.z),
            (item.x + item.width, item.y + item.height, item.z),
            (item.x, item.y + item.height, item.z),
            (item.x, item.y, item.z + item.depth),
            (item.x + item.width, item.y, item.z + item.depth),
            (item.x + item.width, item.y + item.height, item.z + item.depth),
            (item.x, item.y + item.height, item.z + item.depth)
        ]
        return vertices
