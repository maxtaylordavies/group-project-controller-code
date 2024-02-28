"""
Experience replay implementations
"""

from collections import namedtuple

import h5py
import numpy as np
import torch


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)


class ReplayBuffer:
    """Replay buffer to sample experience/ transition tuples from

    :attr capacity (int): total capacity of the replay buffer
    :attr memory (Transition):
        Each component of the transition tuple is represented by a zero-initialised np.ndarray of
        floats with dimensionality (total buffer capacity, component dimensionality)
    :attr writes (int): number of experiences/ transitions already added to the buffer
    """

    def __init__(self, capacity: int):
        """Constructor for a ReplayBuffer initialising an empty buffer (without memory

        :param capacity (int): total capacity of the replay buffer
        """
        self.capacity = int(capacity)
        self.memory = Transition(
            states=None, actions=None, next_states=None, rewards=None, done=None
        )
        self.writes = 0

    def init_memory(self, transition: Transition):
        """Initialises the memory with zero-entries

        :param transition (Transition): transition(s) to take the dimensionalities from
        """
        for t in transition:
            assert t.ndim == 1  # sanity check

        self.memory = Transition(
            *[np.zeros([self.capacity, t.size], dtype=t.dtype) for t in transition]
        )

    def push(self, *args):
        """Adds transitions to the memory

        Note:
            overwrites first transitions stored once the capacity limit is reached

        :param *args: arguments to create transition from
        """
        if self.memory.states is None:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes = self.writes + 1

    def sample(self, batch_size: int, device: str = "cpu") -> Transition:
        """Samples batch of experiences from the replay buffer

        :param batch_size (int): size of the batch to be sampled and returned
        :param device (str): PyTorch device to cast to (for potential GPU support)
        :return (Transition): batch of experiences of given batch size
        """
        samples = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[
                torch.from_numpy(np.take(d, samples, axis=0)).to(device)
                for d in self.memory
            ]
        )
        return batch

    def __len__(self):
        """Gives the length of the buffer"""
        return min(self.writes, self.capacity)

    def save(self, filename: str):
        """Saves the replay buffer to a file

        :param filename (str): filename to save the replay buffer to
        """
        with h5py.File(filename, "w") as f:
            f.create_dataset(
                "observations",
                data=self.memory.states[: self.writes],
                compression="gzip",
            )
            f.create_dataset(
                "actions", data=self.memory.actions[: self.writes], compression="gzip"
            )
            f.create_dataset(
                "rewards", data=self.memory.rewards[: self.writes], compression="gzip"
            )
            f.create_dataset(
                "terminations", data=self.memory.done[: self.writes], compression="gzip"
            )
            f.flush()


def load_saved_replay_data(filename: str):
    with h5py.File("../replay_buffer.hdf5", "r") as f:
        observations = f["observations"][()]
        actions = f["actions"][()]
        rewards = f["rewards"][()]
        terminations = f["terminations"][()]
    return observations, actions, rewards, terminations
