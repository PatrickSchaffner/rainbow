import numpy as np

from .core import NumpyReplayBuffer


class RollingReplayBuffer(NumpyReplayBuffer):
    
    def __init__(self, *args, **kwargs):
        super(RollingReplayBuffer, self).__init__(*args, **kwargs):
        self._size = 0
        self._counter = 0
    
    @property
    def size(self):
        return self._size
    
    def store_transition(self, state, action, reward, next_state, done):
        idx = self._counter % self.capacity
        self._counter += 1
        self._size = min(self._counter, self.capacity)
        self._s0[idx] = state
        self._a[idx] = action
        self._r[idx] = reward
        self._s1[idx] = next_state
        self._d[idx] = done
        return idx
    
    def sample_batch(self, size:int, replace:bool=False):
        idx = np.random.choice(self.size, size=(size,), replace=replace)
        return self._s0[idx], self._a[idx], self._r[idx], self._s1[idx], self._d[idx]
    
    def _collect_save_items(self):
        return {
            's0': self._s0[:self.size],
            'a' : self._a[self.size],
            'r' : self._r[self.size],
            's1': self._s1[self.size],
            'd' : self._d[self.size],
            'c' : self._counter}
    
    def _restore_save_items(self,
                            s0=None,
                            a=None,
                            r=None,
                            s1=None,
                            d=None,
                            c=None):
        self._size = np.shape(s0)[0]
        self._s0[:self.size] = s0
        self._a[:self.size] = a
        self._r[:self.size] = r
        self._s1[:self.size] = s1
        self._d[:self.size] = d
        self._counter = c if np.isscalar(c) else c.item()
