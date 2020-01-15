import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as ppl
import maxflow
from maxflow.fastmin import aexpansion_grid

'''g = maxflow.Graph[int](2, 2)
nodes = g.add_nodes(2)
g.add_edge(nodes[0], nodes[1], 1, 2)
g.add_tedge(nodes[0], 2, 5)
g.add_tedge(nodes[1], 9, 4)

flow = g.maxflow()
print("max flow: {}, node0: {}, node1: {}".format(flow, g.get_segment(nodes[0]), g.get_segment(nodes[1])))
'''

'''img = imread("D:/s5/lena/a2.png")

g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes(img.shape)
g.add_grid_edges(nodeids, 50)
g.add_grid_tedges(nodeids, img, 255 - img)

g.maxflow()
sgs = g.get_grid_segments(nodeids)
print(sgs)

img2 = np.int_(np.logical_not(sgs))
print(img2)
ppl.imshow(img2)
ppl.show()
'''

D = np.asarray([
    [
        [5, 1, 5],
        [5, 5, 1]
    ], [
        [5, 1, 5],
        [5, 1, 5]
    ]
])

V = np.asarray([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

print(aexpansion_grid(D, V))
