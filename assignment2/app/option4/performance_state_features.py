from __future__ import annotations

import torch


def derive_performance_state_features(
    piano_roll: torch.Tensor,
    max_age_frames: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Derive extra note-state features from the existing 3-channel piano roll.

    Input:
        piano_roll: [B, 3, T, 88]
            channel 0: active
            channel 1: onset
            channel 2: velocity_onset

    Output:
        enhanced: [B, 6, T, 88]
            channel 0: active
            channel 1: onset
            channel 2: velocity_onset
            channel 3: offset
            channel 4: local_note_age
            channel 5: active_velocity

    Notes:
        This is a window-local approximation. If a note was already active
        before the current 4-second window, its true age and velocity are not
        fully recoverable from the cached 3-channel piano roll.
    """
    if piano_roll.ndim != 4:
        raise ValueError(f"Expected piano_roll [B, C, T, P], got {piano_roll.shape}")

    if piano_roll.shape[1] < 3:
        raise ValueError(f"Expected at least 3 channels, got {piano_roll.shape[1]}")

    active = piano_roll[:, 0].clamp(0.0, 1.0)          # [B, T, P]
    onset = piano_roll[:, 1].clamp(0.0, 1.0)           # [B, T, P]
    velocity_onset = piano_roll[:, 2].clamp(0.0, 1.0)  # [B, T, P]

    # offset[t] = active[t-1] - active[t], clipped to positive values.
    offset = torch.zeros_like(active)
    offset[:, 1:, :] = torch.relu(active[:, :-1, :] - active[:, 1:, :])

    batch_size, frames, pitches = active.shape

    note_age = torch.zeros_like(active)
    active_velocity = torch.zeros_like(active)

    age_state = torch.zeros(
        (batch_size, pitches),
        device=piano_roll.device,
        dtype=piano_roll.dtype,
    )
    velocity_state = torch.zeros_like(age_state)

    max_age = float(max(1, max_age_frames))

    for t in range(frames):
        cur_active = (active[:, t, :] > eps).to(piano_roll.dtype)
        cur_onset = (onset[:, t, :] > eps).to(piano_roll.dtype)
        cur_velocity = velocity_onset[:, t, :]

        # Reset age on onset. Otherwise accumulate while active.
        age_state = torch.where(
            cur_onset > 0,
            torch.ones_like(age_state),
            age_state + cur_active,
        )
        age_state = age_state * cur_active

        note_age[:, t, :] = (age_state / max_age).clamp(0.0, 1.0)

        # Propagate onset velocity across the active duration.
        velocity_state = torch.where(
            cur_velocity > eps,
            cur_velocity,
            velocity_state,
        )
        velocity_state = velocity_state * cur_active

        active_velocity[:, t, :] = velocity_state

    enhanced = torch.stack(
        [
            active,
            onset,
            velocity_onset,
            offset,
            note_age,
            active_velocity,
        ],
        dim=1,
    )

    return enhanced.contiguous()
