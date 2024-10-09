import os
import pathlib
from glob import glob
from importlib import import_module
import inspect
import pickle
import json
import numpy as np
import pandas as pd
from torch.utils.data import (
    Dataset as torchDataset,
    Subset as torchSubset,
)


class PhasingDataset(torchDataset):

    _freq_key = 'freqs'
    _value_keys = ['phases']
    _label_keys = ['m1', 'm2', 's1z', 's2z']
    _theory_keys = ['b_ppe', 'gamma_bar']
    _theory_spec = 'dpsi_bar'

    def __init__(self, waveform_dataset, n_ppe=None, norm_fac={}):
        self._len = len(waveform_dataset)
        theory_keys = self._theory_keys
        if n_ppe is not None:
            spec = [self._theory_spec+f'_{i}' for i in range(2,n_ppe)]
            theory_keys = self._theory_keys + spec
        self._theory = self.transform_theory(waveform_dataset[theory_keys].values)
        self._labels = self.transform_labels(waveform_dataset[self._label_keys].values)
        if n_ppe is not None:
            freqs = self.transform_values(waveform_dataset[self._freq_key].values)
            self._values = self.calculate_phases(freqs, self._theory[:,0], self._theory[:,1], self._theory[:,2:])
            self._values = self._values.reshape(self._len, 1, -1)
        else:
            self._values = self.transform_values(waveform_dataset[self._value_keys].values)
        for th, norm in norm_fac.items():
            mask = self._theory[:,0] == float(th)
            self._values[mask] /= norm
        self._theory = np.asarray(self._theory, dtype=np.float64)
        self._labels = np.asarray(self._labels, dtype=np.float64)
        self._values = np.asarray(self._values, dtype=np.float64)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self._values[idx], self._labels[idx], self._theory[idx]

    @staticmethod
    def transform_labels(labels):
        m1, m2, s1z, s2z = np.asarray(labels).T
        mask = m1 < m2
        m1[mask], m2[mask] = m2[mask], m1[mask]
        s1z[mask], s2z[mask] = s2z[mask], s1z[mask]
        mc = component_masses_to_chirp_mass(m1, m2)
        eta = component_masses_to_symmetric_mass_ratio(m1, m2)
        mtot = m1 + m2
        q = m2 / m1
        chis = 0.5 * (s1z + s2z)
        chia = 0.5 * (s1z - s2z)
        # return np.asarray([np.log(mc), np.log(0.25/eta-1), chis, chia]).T
        return np.asarray([np.log(mc), q, chis, chia]).T

    @staticmethod
    def transform_values(values):
        if values.ndim == 1:
            values = np.vstack(values)
        else:
            values = [np.vstack(v) for v in values]
            values = np.asarray(values)
        return values

    @staticmethod
    def transform_theory(theory):
        return np.asarray(theory, dtype=float)

    @staticmethod
    def calculate_phases(freqs, b, gamma_bar, dpsi_bars):
        v = (np.pi * freqs) ** (1/3)
        phases = gamma_bar[:,None] * v ** b[:,None]
        for i,dpb in enumerate(dpsi_bars.T):
            phases += gamma_bar[:,None] * dpb[:,None] * v ** (b[:,None] + i + 2)
        return phases


def component_masses_to_chirp_mass(mass_1, mass_2):
    return (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2

def component_masses_to_mass_ratio(mass_1, mass_2):
    return mass_2 / mass_1

def component_masses_to_symmetric_mass_ratio(mass_1, mass_2):
    symmetric_mass_ratio = (mass_1 * mass_2) / (mass_1 + mass_2) ** 2
    return np.minimum(symmetric_mass_ratio, 0.25)

def component_masses_to_total_mass(mass_1, mass_2):
    return mass_1 + mass_2

def chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio):
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(
                    chirp_mass=chirp_mass, mass_ratio=mass_ratio)
    mass_1, mass_2 = total_mass_and_mass_ratio_to_component_masses(
                    total_mass=total_mass, mass_ratio=mass_ratio)
    return mass_1, mass_2

def chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio):
    with np.errstate(invalid="ignore"):
        return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio ** 0.6

def chirp_mass_and_total_mass_to_symmetric_mass_ratio(chirp_mass, total_mass):
    return (chirp_mass / total_mass) ** (5 / 3)

def total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass):
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return mass_1, mass_2

def symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio):
    temp = (1 / symmetric_mass_ratio / 2 - 1)
    return temp - (temp ** 2 - 1) ** 0.5
    


class DatasetManager(object):
    """
    Bundles a dataset and its split
    Also keeps record of how to reproduce them
    """

    def __init__(
            self, filenames, root_dir=None,
            sample_indices=None, subset_indices=None, 
            sample_size=None, subset_split=[0.8, 0.1, 0.1], random_state=None, 
            dataset_type=PhasingDataset, dataset_kwargs={},
            comment="",):
        # DEFAULT BEHAVIOR: use all data, randomly split by [0.8, 0.1, 0.1]
        # FOR REPRODUCTION: provide both sample_indices and subset_indices

        if type(filenames) is str:
            if root_dir is None:
                filepaths = glob(filenames)
            else:
                filepaths = glob(os.path.join(root_dir, filenames))
                filenames = [os.path.relpath(fp, root_dir) for fp in filepaths]
        else:
            if root_dir is None:
                filepaths = filenames
            else:
                filepaths = [os.path.join(root_dir, fn) for fn in filenames]
        # if type(filenames) is str:
        #     filenames = glob(filenames, root_dir=root_dir)
        # if root_dir is not None:
        #     filepaths = [os.path.join(root_dir, fn) for fn in filenames]
        # else:
        #     filepaths = filenames
        if type(dataset_type) is str:
            dataset_type = eval(dataset_type)

        reproduce = sample_indices is not None and subset_indices is not None
        if reproduce:
            # sanity check
            assert len(sample_indices) == len(filepaths)
            assert sum(len(indices) for indices in sample_indices) \
                        == sum(len(indices) for indices in subset_indices)

            dataframe = []
            for indices, fp in zip(sample_indices, filepaths):
                df = pd.read_pickle(fp)
                dataframe.append(df.iloc[indices,:].copy())
                del df
            dataframe = pd.concat(dataframe, ignore_index=True)
            dataset = dataset_type(dataframe, **dataset_kwargs)
            subsets = [torchSubset(dataset, indices) for indices in subset_indices]

        else:
            if random_state is None or type(random_state) is int:
                random_state = np.random.default_rng(random_state)
            if sample_size is None:
                sample_size = sum(subset_split)
                subset_split = [x/sample_size for x in subset_split]
            # sanity check
            subset_split_sum = sum(subset_split)
            subset_split_type = np.asarray(subset_split).dtype
            subset_split_as_fracs = subset_split_type == float and subset_split_sum == 1.
            subset_split_as_sizes = subset_split_type == int
            sample_size_as_frac = type(sample_size) == float and sample_size < 1.
            sample_size_as_size = type(sample_size) == int
            assert subset_split_as_fracs or subset_split_as_sizes
            assert sample_size_as_frac or sample_size_as_size
            if sample_size_as_size and subset_split_as_sizes:
                assert sample_size == subset_split_sum
            # If subset split is given in sizes, but sample size is given in fraction,
            # then size matching is also necessary, but should be delayed till sampling is done

            dataframe = []
            sample_indices = []
            if sample_size_as_size:
                sample_norm = sum(len(pd.read_pickle(fp)) for fp in filepaths)
            else:
                sample_norm = 1.
            for fp in filepaths:
                df = pd.read_pickle(fp)
                size = int(len(df)*sample_size/sample_norm)
                indices = random_state.choice(len(df), size, replace=False)
                sample_indices.append(indices)
                dataframe.append(df.iloc[indices,:].copy())
                del df
            dataframe = pd.concat(dataframe, ignore_index=True)
            dataset = dataset_type(dataframe, **dataset_kwargs)
            # complete size check
            if sample_size_as_frac and subset_split_as_sizes:
                assert len(dataframe) == subset_split_sum

            subsets = []
            subset_indices = []
            if subset_split_as_sizes:
                subset_sizes = subset_split
            else:
                subset_sizes = [int(x*len(dataframe)) for x in subset_split]
                subset_sizes[0] = len(dataframe) - sum(subset_sizes[1:])
            perm = random_state.permutation(len(dataframe))
            for size in subset_sizes:
                indices = perm[:size].copy()
                subset_indices.append(indices)
                subsets.append(torchSubset(dataset, indices))
                perm = perm[size:]

        self._root_dir = root_dir
        self._filenames = filenames
        self._sample_indices = sample_indices
        self._dataset_type = dataset_type
        self._dataset_kwargs = dataset_kwargs
        self._comment = comment
        self._subsets = subsets


    @property
    def dataset(self):
        return self._subsets[0].dataset
    @property
    def subsets(self):
        return self._subsets

    @property
    def recipe(self):
        return {
            'comment': self._comment,
            'dataset_type': self._dataset_type,
            'dataset_kwargs': self._dataset_kwargs,
            'root_dir': self._root_dir,
            'filenames': self._filenames,
            'sample_indices': self._sample_indices,
            'subset_indices': [sub.indices for sub in self._subsets],
        }

    @property
    def brief_recipe(self):
        # neat for log writing
        # may be used to make datasets of the same sizes
        # but the exact choice of indices cannot be reproduced this way
        return {
            'comment': self._comment,
            'dataset_type': self._dataset_type,
            'dataset_kwargs': self._dataset_kwargs,
            'root_dir': self._root_dir,
            'filenames': self._filenames,
            'subset_split': [len(sub) for sub in self._subsets],
        }

    def save_recipe(self, path, comment=None):
        # add comment if needed
        recipe = self.recipe
        if comment is not None:
            recipe['comment'] = comment
        with open(path, 'w') as f:
            json.dump(recipe, f, indent=4, cls=DatasetRecipeJSONEncoder)

    @classmethod
    def from_recipe(cls, path, root_dir=None):
        # careful about reseting root_dir
        # this key is designed for cross-platform applications
        # eg different machines each maintaining a copy of the data
        with open(path, 'r') as f:
            recipe = json.load(f)
            recipe = recursively_decode_dataset_recipe_json(recipe)
        if root_dir is not None:
            recipe['root_dir'] = root_dir
        return cls(**recipe)


class DatasetRecipeJSONEncoder(json.JSONEncoder):
    # check eg bilby json encoder for template
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if inspect.isfunction(obj):
            return {
                "__function__": True,
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        if inspect.isclass(obj):
            return {
                "__class__": True,
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        return json.JSONEncoder.default(self, obj)
    
def decode_dataset_recipe_json(dct):
    if dct.get("__function__", False) or dct.get("__class__", False):
        default = ".".join([dct["__module__"], dct["__name__"]])
        return getattr(import_module(dct["__module__"]), dct["__name__"], default)
    return dct

def recursively_decode_dataset_recipe_json(dct):
    dct = decode_dataset_recipe_json(dct)
    if isinstance(dct, dict):
        for key in dct:
            if isinstance(dct[key], dict):
                dct[key] = recursively_decode_dataset_recipe_json(dct[key])
    return dct
