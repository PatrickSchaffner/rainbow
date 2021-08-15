import numpy as np
from numpy.random import default_rng as rng

from .rolling import RollingReplayBuffer


class PrioritizedReplayBuffer(RollingReplayBuffer):
    
    def __init__(self, *args, alpha=0.5, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self._alpha = alpha
        self._priority = SumTree(self.capacity)
    
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
    
    def store_transition(self, state, action, reward, next_state, done, priority):
        idx = super(PrioritizedReplayBuffer, self).store_transition(state, action, reward, next_state, done)
        self._priority.update(idx, priority)
        return idx
    
    def sample_batch(self, size:int, replace:bool=False, restore:bool=True):
        if size > self.size:
            raise ValueError(f"Cannot sample {size:d} observations from buffer of size {self.size:d}.")
        p = rng.uniform(low=0, high=1, size=(size,)) < self.alpha
        n = sum(p)
        batch = np.empty((size,), dtype=np.int)
        batch[p] = self._priority.sample(n, replace=replace, restore=restore)
        batch[np.logical_not(p)] = np.random.choice(self.size, size=(size - n,), replace=replace)
        return self._s0[batch], self._a[batch], self._r[batch], self._s1[batch], self._d[batch], batch
    
    def update_priority(self, batch, priority):
        for (leaf, value) in zip(batch, priority):
            self._tree.update(leaf, value)
    
    def _collect_save_items(self):
        data = super(PrioritizedReplayBuffer, self)._collect_save_items()
        data['p'] = self._priority.get_leafs()[:self.size]
        data['alpha'] = self.alpha
        return data
    
    def _restore_save_items(self, p=None, alpha=None, **kwargs):
        super(PrioritizedReplayBuffer, self)._restore_save_items(**kwargs)
        self._priority.set_leafs(p)
        self._alpha = alpha
        

class SumTree:
    
    def __init__(self, capacity:int):
        self._capacity = capacity
        self._tree = np.zeros((2*self.capacity-1,), dtype=np.float64)

    @property
    def capacity(self):
        return self._capacity
    
    @property
    def total(self):
        return self._tree[0]
    
    def get_leafs(self):
        return self._tree[(self.capacity-1):]
    
    def set_leafs(self, values):
        low = self.capacity - 1
        up = low + len(values)
        self._tree[low:up] = values
        self._tree[up:] = 0
        for node in reversed(range(low)):
            left = node * 2 + 1
            right = left + 1
            self._tree[node] = self._tree[left] + self._tree[right]
    
    def update(self, index:int, value:float):
        self._propagate(index + self.capacity - 1, value)
    
    def sample(self, size:int, replace:bool=True, restore:bool=True, eps:float=1e-13):
        nodes = np.empty((size,), dtype=np.int)
        delta = np.empty((size,), dtype=np.float64)
        remove = not replace
        for i in range(size):
            val = rng.uniform(low=0, high=self.total)
            (nodes[i], delta[i]) = self._find(0, val, remove=remove)
        if remove and restore:
            for (leaf, value) in zip(nodes, delta):
                self._propagate(leaf, value)
        return nodes - self.capacity + 1
    
    def _propagate(self, node:int, value:float):
        self._tree[node] = value
        if node > 0:
            other = node - 1 + 2 * (node % 2)
            value += self._tree[other]
            parent = (node - 1 ) // 2
            self._propagate(parent, value)
    
    def _find(self, node:int, value:float, remove:bool=False):
        left = node * 2 + 1
        if left >= len(self._tree):
            delta = self._tree[node]
            if remove:
                self._tree[node] = 0
            return node, delta
        if value < self._tree[left]:
            (leaf, delta) = self._find(left, value)
        else:
            right = left + 1
            value -= self._tree[left]
            (leaf, delta) = self._find(right, value)
        if remove:
            self._tree[node] -= delta
        return leaf, delta
