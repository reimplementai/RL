# cache the best rollouts

import pickle
import numpy as np

class GenCache:
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self.items = []
        self.index = 0
        self.weights = np.array([])

    def insert(self, x, weight):
        n = len(self.items)
        if n >= self.maxsize:
            insert_index = self.weights.argmin()
            current_min = self.weights[insert_index]
            if weight > current_min:
                self.items[insert_index] = x
                self.weights[insert_index] = weight
            return
            
        self.items.append(x)
        self.weights = np.append(self.weights, weight)

    def get(self):
        if len(self.items) <= self.index:
            return None, None
        roll = self.items[self.index]
        r = self.weights[self.index]
        self.index = (self.index + 1) % len(self.items)
        return roll, r

    def getstate(self):
        s = f"items:   {self.items}\n" \
            f"weights: {self.weights}\n" \
            f"index:   {self.index}\n" \
            f"maxsize: {self.maxsize}\n"
        return s
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            c = pickle.load(f)
            self.maxsize = c.maxsize
            self.items = c.items
            self.index = c.index
            self.weights = c.weights
        
if __name__ == "__main__":
    cache = GenCache(4)
    r = [1, 1, 2, 3, 4, -2, 1, 3, 6, 2, 1, 4, 10]

    print(f"inserting r = {r}")
    for i in range(len(r)):
        cache.insert(i, r[i])
        print()
        print(i)
        print(cache.getstate())

    print(f"\n\naccessing items = {cache.items}")
    for i in range(10):
        x = cache.get()
        print(f"\n{i}: got {x}")
        print(cache.getstate())

    cache.save("cachetest")
    c = GenCache()
    c.load("cachetest")
    print("original:")
    print(cache.getstate())
    print("saved:")
    print(c.getstate())

