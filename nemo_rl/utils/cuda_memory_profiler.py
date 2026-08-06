# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bounded, opt-in CUDA allocator profiling at logical phase boundaries."""

import os
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch


class CudaMemoryPhaseProfiler:
    """Measure phase-local peaks and optionally dump allocator history."""

    def __init__(self, role: str, rank: int, env_prefix: str = "NRL") -> None:
        self.role = role
        self.rank = rank
        self.prefix = f"{env_prefix}_CUDA_MEMORY_"
        self.enabled = os.environ.get(f"{self.prefix}PROFILE", "0") == "1"
        self.output_dir = Path(
            os.environ.get(f"{self.prefix}SNAPSHOT_DIR", "/logs/cuda_memory_snapshots")
        )
        self.max_entries = int(
            os.environ.get(f"{self.prefix}SNAPSHOT_MAX_ENTRIES", "50000")
        )
        self.max_snapshots = int(
            os.environ.get(f"{self.prefix}SNAPSHOT_MAX_PER_PHASE", "1")
        )
        self.snapshot_ranks = self._parse_ranks(
            os.environ.get(f"{self.prefix}SNAPSHOT_RANKS", "0")
        )
        self.active_phase: Optional[str] = None
        self.history_active = False
        self.phase_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _parse_ranks(value: str) -> Optional[set[int]]:
        value = value.strip().lower()
        if value == "all":
            return None
        return {int(rank.strip()) for rank in value.split(",") if rank.strip()}

    def _should_snapshot(self, phase: str) -> bool:
        rank_selected = self.snapshot_ranks is None or self.rank in self.snapshot_ranks
        return rank_selected and self.phase_counts[phase] < self.max_snapshots

    def start(self, phase: str) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        if self.active_phase == phase:
            return
        if self.active_phase is not None:
            self.stop(self.active_phase)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.active_phase = phase
        if self._should_snapshot(phase):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=self.max_entries,
            )
            self.history_active = True

    def stop(self, phase: Optional[str] = None) -> None:
        if not self.enabled or self.active_phase is None:
            return
        if phase is not None and phase != self.active_phase:
            return

        torch.cuda.synchronize()
        gib = 1024**3
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        current_phase = self.active_phase
        occurrence = self.phase_counts[current_phase]
        snapshot_path = None
        if self.history_active:
            safe_role = self.role.replace("/", "_")
            snapshot_path = self.output_dir / (
                f"{safe_role}-{socket.gethostname()}-pid{os.getpid()}-rank{self.rank}-"
                f"{current_phase}-{occurrence}-{time.time_ns()}.pickle"
            )
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)
            self.history_active = False

        print(
            f"[cuda-memory-phase] role={self.role} rank={self.rank} "
            f"phase={current_phase} occurrence={occurrence} "
            f"allocated={torch.cuda.memory_allocated() / gib:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f}GiB "
            f"peak_allocated={torch.cuda.max_memory_allocated() / gib:.2f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved() / gib:.2f}GiB "
            f"device_used={(total_bytes - free_bytes) / gib:.2f}GiB "
            f"device_total={total_bytes / gib:.2f}GiB snapshot={snapshot_path}",
            flush=True,
        )
        self.phase_counts[current_phase] += 1
        self.active_phase = None
