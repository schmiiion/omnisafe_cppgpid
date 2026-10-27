"""Expert demonstrations as plain NumPy arrays, without a dataset framework."""

from pathlib import Path
import pickle

import numpy as np


def load_expert_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NPZ observations/actions or a directory of trusted benchmark pickles.

    NPZ arrays may be (transitions, features) or (episodes, steps, features).
    Benchmark pickles contain one episode each, with original_observations
    (or observations) and actions. Inputs must use raw observations/native actions.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        observations, actions = [], []
        for file in sorted(path.iterdir()):
            if file.suffix not in ('.pkl', '.pickle'):
                continue
            with file.open('rb') as stream:
                episode = pickle.load(stream)
            key = 'original_observations' if 'original_observations' in episode else 'observations'
            observations.append(episode[key])
            actions.append(episode['actions'])
        obs, acs = np.concatenate(observations), np.concatenate(actions)
    else:
        with np.load(path, allow_pickle=False) as dataset:
            key = 'original_observations' if 'original_observations' in dataset else 'observations'
            obs, acs = dataset[key], dataset['actions']
    obs = np.asarray(obs, dtype=np.float32).reshape(-1, obs.shape[-1])
    acs = np.asarray(acs, dtype=np.float32).reshape(-1, acs.shape[-1])
    if len(obs) == 0 or len(obs) != len(acs):
        raise ValueError('Expert data must contain matching, nonempty pre-action observations/actions.')
    return obs, acs
