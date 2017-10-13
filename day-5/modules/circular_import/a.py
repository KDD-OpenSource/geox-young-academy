print("a before import", __name__)
from b import B
print("a after import")

A = 1

print("B = {}".format(B))

if __name__ == "__main__":
    # only run if module is executed
    # from command line
    print("executing file", __file__)
