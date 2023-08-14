from py3dbp import Packer, Bin, Item
import numpy as np

packer = Packer()

packer.add_bin(Bin('', 10, 10, 10, 4))
packer.add_item(Item('', 1, 1, 1, 1))
packer.add_item(Item('', 1, 2, 2, 1))




packer.pack()

for b in packer.bins:
    print(":::::::::::", b.string())

    print("FITTED ITEMS:")
    for item in b.items:
        print("====> ", item.string())

    print("UNFITTED ITEMS:")
    for item in b.unfitted_items:
        print("====> ", item.string())

    print("***************************************************")
    print("***************************************************")
