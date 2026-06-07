import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.preprocess_dataset import (
    DEFAULT_DATASET_ROOT,
    METADATA_FILENAME,
    SAMPLE_BUNDLE_FILENAME,
    SOURCE_DOWNMIX_SIGNAL_FILENAME,
    SOURCE_MONO_SIGNAL_FILENAME,
    SOURCE_STEREO_SIGNAL_FILENAME,
    TARGET_SIGNAL_FILENAME,
    sample_dir_from_stream_hash,
)
from stereo2spatial.common.channel_layouts import channel_labels_for_layout
from stereo2spatial.inference.audio import write_audio_channels_first

def parse_stream_hash(raw_hash: str) -> str:
    value = raw_hash.strip().lower()
    if not re.fullmatch(r"[0-9a-f]+", value):
        raise ValueError(f"Invalid stream hash: {raw_hash!r}")
    return value


def resolve_sample_dir(
    sample_dir: Optional[str],
    stream_hash: Optional[str],
    dataset_root: Path,
) -> Path:
    if sample_dir:
        return Path(sample_dir).resolve(strict=False)
    if stream_hash:
        return sample_dir_from_stream_hash(
            dataset_root=dataset_root,
            stream_hash=parse_stream_hash(stream_hash),
        )
    raise ValueError("Provide one of: --sample-dir OR --stream-hash")


def load_metadata(sample_dir: Path) -> dict:
    metadata_path = sample_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid metadata payload: {type(payload)}")
    return payload


def torch_load_cpu(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_signal(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() in {2, 3}:
        return tensor.contiguous()
    raise ValueError(f"{name} must have shape [C,S] or [C,P,T], got {tuple(tensor.shape)}")


def unpatch_signal(signal_cpt: torch.Tensor, sample_count: int | None) -> torch.Tensor:
    if signal_cpt.dim() == 2:
        audio = signal_cpt.contiguous()
        if sample_count is not None:
            return audio[:, :sample_count].contiguous()
        return audio
    if signal_cpt.dim() != 3:
        raise ValueError(f"signal must have shape [C,S] or [C,P,T], got {tuple(signal_cpt.shape)}")
    audio = signal_cpt.permute(0, 2, 1).reshape(signal_cpt.shape[0], -1).contiguous()
    if sample_count is not None:
        return audio[:, :sample_count].contiguous()
    return audio


def write_qc_wav(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    channel_order: list[str] | None = None,
) -> None:
    write_audio_channels_first(
        audio_path=path,
        audio=audio.float(),
        sample_rate=sample_rate,
        channel_order=channel_order,
    )


def safe_stem(raw_path: str) -> str:
    stem = Path(raw_path).stem if raw_path else "sample"
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_")
    return cleaned or "sample"


def load_signals(sample_dir: Path) -> dict[str, torch.Tensor]:
    bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
    if bundle_path.exists():
        bundle = torch_load_cpu(bundle_path)
        if not isinstance(bundle, dict):
            raise TypeError(f"Invalid sample bundle payload: {type(bundle)}")
        required = {
            "target_signal",
            "source_stereo_signal",
        }
        missing = [key for key in required if key not in bundle]
        if missing:
            raise KeyError(f"Bundle missing keys: {missing}")
        signals = {
            "target_signal": to_signal(bundle["target_signal"], "target_signal"),
            "source_stereo_signal": to_signal(
                bundle["source_stereo_signal"], "source_stereo_signal"
            ),
        }
        if "source_mono_signal" in bundle:
            signals["source_mono_signal"] = to_signal(
                bundle["source_mono_signal"], "source_mono_signal"
            )
        if "source_downmix_signal" in bundle:
            signals["source_downmix_signal"] = to_signal(
                bundle["source_downmix_signal"], "source_downmix_signal"
            )
        return signals

    split_paths = {
        "target_signal": sample_dir / TARGET_SIGNAL_FILENAME,
        "source_stereo_signal": sample_dir / SOURCE_STEREO_SIGNAL_FILENAME,
    }
    missing_files = [str(path) for path in split_paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing signal artifacts and no bundle found:\n  - "
            + "\n  - ".join(missing_files)
        )
    source_mono_path = sample_dir / SOURCE_MONO_SIGNAL_FILENAME
    if source_mono_path.exists():
        split_paths["source_mono_signal"] = source_mono_path
    source_downmix_path = sample_dir / SOURCE_DOWNMIX_SIGNAL_FILENAME
    if source_downmix_path.exists():
        split_paths["source_downmix_signal"] = source_downmix_path
    return {
        name: to_signal(torch_load_cpu(path), name) for name, path in split_paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write one processed waveform sample into QC WAVs: target multichannel, "
            "source stereo, source mono when present, and source downmix when present."
        )
    )
    parser.add_argument(
        "--sample-dir",
        default=None,
        help="Path to processed sample dir (.../samples/xx/yy/<stream_hash>).",
    )
    parser.add_argument(
        "--stream-hash",
        default=None,
        help="Stream hash to resolve sample dir from --dataset-root.",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder for QC WAVs. Defaults to <sample-dir>/_qc_waveforms.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve(strict=False)
    sample_dir = resolve_sample_dir(args.sample_dir, args.stream_hash, dataset_root)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    metadata = load_metadata(sample_dir)
    signals = load_signals(sample_dir)

    sample_rate = int(metadata.get("sample_rate", 48000))
    sample_count = metadata.get("input_samples")
    input_samples = int(sample_count) if isinstance(sample_count, int) else None
    target_layout = str(metadata.get("target_layout", "unknown_layout"))
    target_channel_labels_raw = metadata.get("target_channel_labels")
    target_channel_labels = (
        [str(label) for label in target_channel_labels_raw]
        if isinstance(target_channel_labels_raw, list)
        else channel_labels_for_layout(
            target_layout,
            int(signals["target_signal"].shape[0]),
        )
    )
    stream_hash = str(metadata.get("stream_hash", sample_dir.name))
    source_path = str(metadata.get("source_path", ""))

    out_dir = (
        Path(args.out_dir).resolve(strict=False)
        if args.out_dir
        else sample_dir / "_qc_waveforms"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    target_audio = unpatch_signal(signals["target_signal"], input_samples)
    source_stereo_audio = unpatch_signal(signals["source_stereo_signal"], input_samples)
    source_mono_audio = (
        unpatch_signal(signals["source_mono_signal"], input_samples)
        if "source_mono_signal" in signals
        else None
    )
    source_downmix_audio = (
        unpatch_signal(signals["source_downmix_signal"], input_samples)
        if "source_downmix_signal" in signals
        else None
    )

    prefix = safe_stem(source_path) + "__" + stream_hash[:12]
    target_suffix = re.sub(r"[^0-9A-Za-z]+", "_", target_layout).strip("_").lower()
    if not target_suffix:
        target_suffix = "target"

    target_wav = out_dir / f"{prefix}__waveform_{target_suffix}.wav"
    stereo_wav = out_dir / f"{prefix}__waveform_stereo.wav"
    mono_wav = out_dir / f"{prefix}__waveform_mono.wav"
    downmix_wav = out_dir / f"{prefix}__waveform_downmix.wav"

    write_qc_wav(
        target_wav,
        target_audio,
        sample_rate,
        channel_order=target_channel_labels,
    )
    write_qc_wav(stereo_wav, source_stereo_audio, sample_rate)
    if source_mono_audio is not None:
        write_qc_wav(mono_wav, source_mono_audio, sample_rate)
    if source_downmix_audio is not None:
        write_qc_wav(downmix_wav, source_downmix_audio, sample_rate)

    audio_shapes = {
        "target": [int(x) for x in target_audio.shape],
        "source_stereo": [int(x) for x in source_stereo_audio.shape],
    }
    output_files = {
        "target": str(target_wav),
        "source_stereo": str(stereo_wav),
    }
    if source_mono_audio is not None:
        audio_shapes["source_mono"] = [int(x) for x in source_mono_audio.shape]
        output_files["source_mono"] = str(mono_wav)
    if source_downmix_audio is not None:
        audio_shapes["source_downmix"] = [int(x) for x in source_downmix_audio.shape]
        output_files["source_downmix"] = str(downmix_wav)
    report = {
        "sample_dir": str(sample_dir),
        "stream_hash": stream_hash,
        "source_path": source_path,
        "sample_rate": sample_rate,
        "target_layout": target_layout,
        "target_channel_labels": target_channel_labels,
        "input_samples": input_samples,
        "signal_shapes": {
            name: [int(x) for x in tensor.shape] for name, tensor in signals.items()
        },
        "audio_shapes": audio_shapes,
        "output_files": output_files,
    }
    report_path = out_dir / f"{prefix}__waveform_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print("Wrote QC waveforms:")
    print(f"  - {target_wav}")
    print(f"  - {stereo_wav}")
    if source_mono_audio is not None:
        print(f"  - {mono_wav}")
    if source_downmix_audio is not None:
        print(f"  - {downmix_wav}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
