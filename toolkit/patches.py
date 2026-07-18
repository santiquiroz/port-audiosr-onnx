"""Windows shims required before importing audiosr 0.0.7."""

import sys


def apply_torchaudio_soundfile_patch():
    # torchaudio 2.x on Windows has no TorchCodec wheel; route load/save through soundfile.
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np

    def _load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def _save(path, tensor, sample_rate, *args, **kwargs):
        array = tensor.detach().cpu().numpy()
        if array.ndim == 2:
            array = array.T
        sf.write(str(path), array, sample_rate)

    torchaudio.load = _load
    torchaudio.save = _save


def apply_all():
    apply_torchaudio_soundfile_patch()
    print("[patches] torchaudio->soundfile applied", file=sys.stderr)
