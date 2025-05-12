from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Callable, Literal

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Mappers:
    action: Callable | None = None
    observation: dict[str, Callable | None] = field(default_factory=dict)
    task: dict[str, Callable | None] = field(default_factory=dict)


@dataclass
class Statistic:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    p99: np.ndarray
    p01: np.ndarray

    @classmethod
    def from_dict(cls, data: dict) -> "Statistic":
        return cls(
            mean=np.array(data["mean"]),
            std=np.array(data["std"]),
            min=np.array(data["min"]),
            max=np.array(data["max"]),
            p99=np.array(data["p99"]),
            p01=np.array(data["p01"]),
        )


@dataclass(frozen=True)
class Statistics:
    action: Statistic
    num_actions: int
    num_episodes: int
    proprio: Statistic | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Statistics":
        return cls(
            action=Statistic.from_dict(data["action"]),
            num_actions=data["num_actions"],
            num_episodes=data["num_episodes"],
            proprio=Statistic.from_dict(data["proprio"]) if "proprio" in data else None,
        )


class H5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        obs_seq_len: int = 1,
        obs_seq_stride: int = 1,
        action_seq_len: int = 1,
        mappers: Mappers = Mappers(),
        training: bool = True,
        valid_masks: list[bool] | None = None,
        repeat: int = 1,
        mode: Literal["default", "statistic", "no_obs"] = "default",
    ):
        self.h5_path = h5_path
        self.obs_seq_len = obs_seq_len
        self.obs_seq_stride = obs_seq_stride
        self.action_seq_len = action_seq_len
        self.mappers = mappers
        self.training = training
        self.valid_masks = valid_masks
        self.repeat = repeat
        self.mode = mode

        self._h5_fp: h5py.File | None = None
        self._statistics: Statistics | None = None
        self._episode_start_end_idxs: list[tuple[int, int, int, int, int]] | None = None
        self._actions: np.ndarray | None = None
        self._language_instructions: list[str] | None = None
        self._tasks: dict[str, np.ndarray] = {}
        self.cached_chunk_indices: dict[str, list[tuple[int, int]]] = {}

    @property
    def h5_fp(self) -> h5py.File:
        if self._h5_fp is None:
            self._h5_fp = h5py.File(self.h5_path, "r")
        return self._h5_fp

    @property
    def statistics(self) -> Statistics:
        if self._statistics is None:
            statistics_str = self.h5_fp.get("statistics", None)
            if statistics_str is None:
                raise ValueError("Statistics not found in HDF5 file")
            statistics_str = np.array(statistics_str).item()
            assert isinstance(statistics_str, bytes)
            self._statistics = Statistics.from_dict(json.loads(statistics_str.decode("utf-8")))
        return self._statistics

    @property
    def episode_start_end_idxs(self) -> list[tuple[int, int, int, int, int]]:
        if self._episode_start_end_idxs is None:
            episode_start_end_idxs = self.h5_fp.get("episode_start_end_idxs", None)
            if episode_start_end_idxs is None:
                raise ValueError("Episode start and end indices not found in HDF5 file")
            episode_start_end_idxs = episode_start_end_idxs[:]
            assert episode_start_end_idxs.shape[1] == 5
            episode_start_end_idxs = episode_start_end_idxs.tolist()
            episode_start_end_idxs = [
                (
                    episode_start_end_idxs[0],
                    episode_start_end_idxs[1],
                    episode_start_end_idxs[2],
                    episode_start_end_idxs[3],
                    episode_start_end_idxs[4],
                )
                for i, episode_start_end_idxs in enumerate(episode_start_end_idxs)
                if episode_start_end_idxs[1] - episode_start_end_idxs[0] >= self.action_seq_len and (
                    self.valid_masks is None or self.valid_masks[i]
                )
            ]
            self._episode_start_end_idxs = episode_start_end_idxs
        return self._episode_start_end_idxs

    @property
    def actions(self) -> np.ndarray:
        if self._actions is None:
            actions = self.h5_fp.get("actions", None)
            if actions is None:
                raise ValueError("Actions not found in HDF5 file")
            actions = actions[:]
            self._actions = actions
        return self._actions

    @property
    def language_instructions(self) -> list[str] | None:
        if self._language_instructions is None:
            tasks_group = self.h5_fp.get("tasks", None)
            if tasks_group is None:
                raise ValueError("Tasks group not found in HDF5 file")
            language_instructions = tasks_group.get("language_instruction", None)
            if language_instructions is not None:
                language_instructions = language_instructions[:]
                language_instructions = language_instructions.tolist()

                def decode(t: bytes | list[bytes]) -> str | list[str]:
                    if isinstance(t, bytes):
                        return t.decode("utf-8")
                    else:
                        return [t_i.decode("utf-8") for t_i in t]

                language_instructions = [decode(t) for t in language_instructions]
            self._language_instructions = language_instructions
        return self._language_instructions

    def get_tasks(self, key: str) -> np.ndarray | None:
        if key not in self._tasks:
            tasks_group = self.h5_fp.get("tasks", None)
            if tasks_group is None:
                raise ValueError("Tasks group not found in HDF5 file")
            if key not in tasks_group:
                v = None
            else:
                v = tasks_group[key][:]

                if key == "language_instruction":
                    def decode(t: bytes | list[bytes]) -> str | tuple[str, ...]:
                        if isinstance(t, bytes):
                            return t.decode("utf-8")
                        else:
                            return (t_i.decode("utf-8") for t_i in t)

                    v = [decode(t) for t in v.tolist()]
            self._tasks[key] = v
        return self._tasks[key]

    def __len__(self):
        return len(self.episode_start_end_idxs) * self.repeat

    def __getitem__(self, idx: int):
        assert isinstance(idx, int)
        episode_idx = idx % len(self.episode_start_end_idxs)

        (
            start_idx,
            end_idx,
            chunk_idx,
            chunk_start_idx,
            chunk_end_idx,
        ) = self.episode_start_end_idxs[episode_idx]

        episode_len = end_idx - start_idx
        assert episode_len >= self.action_seq_len

        if self.mode == "statistic":
            step_idx = 0
            action_seq_len = episode_len
        elif self.training:
            # [0, episode_len - self.action_seq_len + 1)
            step_idx = np.random.randint(0, episode_len - self.action_seq_len + 1)
            action_seq_len = self.action_seq_len
        else:
            # Use fixed seed for deterministic validation/test indices
            rng = np.random.default_rng(idx)
            step_idx = rng.integers(0, episode_len - self.action_seq_len + 1)
            action_seq_len = self.action_seq_len

        actions = self.actions[start_idx + step_idx : start_idx + step_idx + action_seq_len]
        assert actions.shape[0] == action_seq_len

        if self.mappers.action is not None:
            actions = self.mappers.action(actions, statistic=self.statistics.action)

        obs_group = self.h5_fp.get("observations", {})
        obs = {}

        if self.mode == "statistic":
            obs_selected_idxs = list(range(chunk_start_idx, chunk_end_idx))
        else:
            obs_selected_idxs = []
            cur = step_idx
            while cur >= 0 and len(obs_selected_idxs) < self.obs_seq_len:
                obs_selected_idxs.append(chunk_start_idx + cur)
                cur -= self.obs_seq_stride
            obs_selected_idxs = list(reversed(obs_selected_idxs))

        for key, mapper in self.mappers.observation.items():
            if self.mode == "no_obs":
                continue

            if (key.startswith("rgb_") or key.startswith("depth_")) and self.mode == "statistic":
                continue

            g = obs_group.get(key, None)
            if g is None:
                if mapper is not None:
                    obs[key] = mapper(None, statistic=None)
                    continue
                else:
                    raise ValueError(f"Observation group not found for key: {key}")
            assert mapper is None or callable(mapper)

            chunk_key = f"{chunk_idx:06d}"
            chunk = g.get(chunk_key, None)
            if chunk is None:
                raise ValueError(f"Chunk not found for key: {key}")

            if key.startswith("rgb_"):
                chunk_indices_key = f"{chunk_key}_indices"
                indices_cache_key = f"{key}_{chunk_key}"
                if indices_cache_key not in self.cached_chunk_indices:
                    indices = g.get(chunk_indices_key, None)
                    if indices is None:
                        raise ValueError(f"Indices not found for key: {key}")
                    indices = indices[:]
                    self.cached_chunk_indices[indices_cache_key] = [
                        (start, end)
                        for start, end in indices.tolist()
                    ]
                indices = self.cached_chunk_indices[indices_cache_key]

                byte_indices = []
                img_start_end_idxs = []
                for idx in obs_selected_idxs:
                    byte_indices_i = list(range(indices[idx][0], indices[idx][1]))
                    img_start_end_idxs.append((len(byte_indices), len(byte_indices) + len(byte_indices_i)))
                    byte_indices += byte_indices_i

                all_img_bytes = chunk[byte_indices]
                imgs = []
                for start, end in img_start_end_idxs:
                    img_bytes = all_img_bytes[start:end]
                    img = Image.open(io.BytesIO(img_bytes))
                    imgs.append(np.array(img))
                v = np.stack(imgs, axis=0)
            elif key.startswith("depth_"):
                raise NotImplementedError("Depth images are not implemented")
            elif key.startswith("proprio"):
                v = chunk[obs_selected_idxs]
            else:
                raise ValueError(f"Unknown observation key: {key}")

            if self.mode == "default":
                if v.shape[0] < self.obs_seq_len:
                    # Pad with the first value
                    v = np.concatenate([v[:1]] * (self.obs_seq_len - v.shape[0]) + [v], axis=0)
                assert v.shape[0] == self.obs_seq_len, f"{v.shape[0]} != {self.obs_seq_len}"

            if mapper is None:
                obs[key] = v
            else:
                statistic = self.statistics.proprio if key == "proprio" else None
                obs[key] = mapper(v, statistic=statistic)

        tasks_g = self.h5_fp.get("tasks", {})

        tasks = {}
        for key, mapper in self.mappers.task.items():
            assert mapper is None or callable(mapper)

            if key in tasks_g:
                v = tasks_g[key][start_idx + step_idx]

                if key == "language_instruction":
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    else:
                        v = [v_i.decode("utf-8") for v_i in v]

                if mapper is None:
                    tasks[key] = v
                else:
                    tasks[key] = mapper(v)
            else:
                assert mapper is not None
                tasks[key] = mapper(None)

        return {
            "actions": actions,
            "observations": obs,
            "tasks": tasks,
        }
