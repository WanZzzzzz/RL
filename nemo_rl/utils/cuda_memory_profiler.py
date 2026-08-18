# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in CUDA allocator profiling at logical phase boundaries."""

import os
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch


class CudaMemoryPhaseProfiler:
    """Measure phase-local peaks and optionally dump allocator snapshots."""

    def __init__(self, role: str, rank: int, env_prefix: str = "NRL") -> None:
        prefix = f"{env_prefix}_CUDA_MEMORY_"
        self.role = role
        self.rank = rank
        self.enabled = os.environ.get(f"{prefix}PROFILE", "0") == "1"
        self.output_dir = Path(
            os.environ.get(f"{prefix}SNAPSHOT_DIR", "/logs/cuda_memory_snapshots")
        )
        self.max_entries = int(
            os.environ.get(f"{prefix}SNAPSHOT_MAX_ENTRIES", "50000")
        )
        self.stacks = os.environ.get(f"{prefix}SNAPSHOT_STACKS", "all")
        self.max_snapshots = int(
            os.environ.get(f"{prefix}SNAPSHOT_MAX_PER_PHASE", "2")
        )
        self.snapshot_ranks = self._parse_ranks(
            os.environ.get(f"{prefix}SNAPSHOT_RANKS", "0")
        )
        self.active_phase: Optional[str] = None
        self.start_allocated = 0
        self.start_reserved = 0
        self.start_device_used = 0
        self.history_active = False
        self.phase_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _parse_ranks(value: str) -> Optional[set[int]]:
        value = value.strip().lower()
        if value == "all":
            return None
        return {int(rank.strip()) for rank in value.split(",") if rank.strip()}

    def _rank_selected(self) -> bool:
        return self.snapshot_ranks is None or self.rank in self.snapshot_ranks

    def _start_history(self, phase: str) -> None:
        if (
            not self._rank_selected()
            or self.phase_counts[phase] >= self.max_snapshots
        ):
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks=self.stacks,
                max_entries=self.max_entries,
            )
            self.history_active = True
        except Exception as exc:
            print(
                f"[cuda-memory-phase] role={self.role} rank={self.rank} "
                f"phase={phase} history_start_error={exc!r}",
                flush=True,
            )

    def _snapshot_path(self, phase: str, occurrence: int, suffix: str = "") -> Path:
        safe_role = self.role.replace("/", "_")
        safe_suffix = suffix.replace("/", "_").replace(" ", "_")
        suffix_part = f"-{safe_suffix}" if safe_suffix else ""
        return self.output_dir / (
            f"{safe_role}-{socket.gethostname()}-pid{os.getpid()}-rank{self.rank}-"
            f"{phase}-{occurrence}{suffix_part}-{time.time_ns()}.pickle"
        )

    def _stop_history(self) -> None:
        if not self.history_active:
            return
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as exc:
            print(
                f"[cuda-memory-phase] role={self.role} rank={self.rank} "
                f"history_stop_error={exc!r}",
                flush=True,
            )
        self.history_active = False

    def dump_active_phase_on_error(self, reason: str) -> Optional[Path]:
        """Dump active history without synchronizing a failed CUDA context."""
        if not self.enabled or self.active_phase is None:
            return None
        failed_phase = self.active_phase
        occurrence = self.phase_counts[failed_phase]
        if not self.history_active:
            self.phase_counts[failed_phase] += 1
            self.active_phase = None
            return None
        snapshot_path = self._snapshot_path(failed_phase, occurrence, reason)
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            print(
                f"[cuda-memory-phase-error] role={self.role} rank={self.rank} "
                f"phase={failed_phase} occurrence={occurrence} "
                f"reason={reason} snapshot={snapshot_path}",
                flush=True,
            )
            return snapshot_path
        except Exception as exc:
            print(
                f"[cuda-memory-phase-error] role={self.role} rank={self.rank} "
                f"phase={failed_phase} occurrence={occurrence} "
                f"reason={reason} snapshot_error={exc!r}",
                flush=True,
            )
            return None
        finally:
            self._stop_history()
            self.phase_counts[failed_phase] += 1
            self.active_phase = None

    def start(self, phase: str) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        if self.active_phase == phase:
            return
        if self.active_phase is not None:
            self.stop(self.active_phase)

        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        self.start_allocated = torch.cuda.memory_allocated()
        self.start_reserved = torch.cuda.memory_reserved()
        self.start_device_used = total_bytes - free_bytes
        torch.cuda.reset_peak_memory_stats()
        self.active_phase = phase
        self._start_history(phase)

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
            snapshot_path = self._snapshot_path(current_phase, occurrence)
            try:
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
            except Exception as exc:
                print(
                    f"[cuda-memory-phase] role={self.role} rank={self.rank} "
                    f"phase={current_phase} snapshot_error={exc!r}",
                    flush=True,
                )
                snapshot_path = None
            finally:
                self._stop_history()

        print(
            f"[cuda-memory-phase] role={self.role} rank={self.rank} "
            f"phase={current_phase} occurrence={occurrence} "
            f"start_allocated={self.start_allocated / gib:.2f}GiB "
            f"start_reserved={self.start_reserved / gib:.2f}GiB "
            f"start_device_used={self.start_device_used / gib:.2f}GiB "
            f"allocated={torch.cuda.memory_allocated() / gib:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f}GiB "
            f"peak_allocated={torch.cuda.max_memory_allocated() / gib:.2f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved() / gib:.2f}GiB "
            f"peak_allocated_delta="
            f"{(torch.cuda.max_memory_allocated() - self.start_allocated) / gib:.2f}GiB "
            f"device_used={(total_bytes - free_bytes) / gib:.2f}GiB "
            f"device_total={total_bytes / gib:.2f}GiB snapshot={snapshot_path}",
            flush=True,
        )
        self.phase_counts[current_phase] += 1
        self.active_phase = None
