from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    
    @property
    @abstractmethod
    def capacity(self):
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def size(self):
        raise NotImplementedError()
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        raise NotImplementedError()
    
    @abstractmethod
    def sample_buffer(self, size):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, filename):
        raise NotImplementedError()
    
    @abstractmethod
    def load(self, filename):
        raise NotImplementedError()
    
    def __len__(self):
        return self.size


class NumpyReplayBuffer(ReplayBuffer):
    
    def __init__(self,
                 capacity=2**14,
                 state_shape=(),
                 state_dtype=np.float64,
                 action_shape=(),
                 action_dtype=np.float64,
                 reward_dtype=np.float64):
        self._capacity = capacity
        self._s0 = np.empty((capacity,)+state_shape,  dtype=state_dtype)
        self._a  = np.empty((capacity,)+action_shape, dtype=action_dtype)
        self._r  = np.empty((capacity,),              dtype=reward_dtype)
        self._s1 = np.empty((capacity,)+state_shape,  dtype=state_dtype)
        self._d  = np.empty((capacity,),              dtype=np.bool)
    
    @property
    def capacity(self):
        return self._capacity
    
    def save(self, filename, compress=False):
        save_func = np.savez_compressed if compress else np.savez
        save_func(filename, **self._collect_save_items)
    
    def load(self, filename):
        with np.load(filename) as f:
            self._restore_save_items(**f)
    
    def _collect_save_items(self):
        return {'s0': self._s0,
                'a' : self._a,
                'r' : self._r,
                's1': self._s1,
                'd' : self._d}
    
    def _restore_save_items(self, s0=None, a=None, r=None, s1=None, d=None):
        self._s0 = data['s0']
        self._a  = data['a']
        self._r  = data['r']
        self._s1 = data['s1']
        self._d  = data['d']
