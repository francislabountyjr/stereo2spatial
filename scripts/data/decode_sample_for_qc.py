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
    SOURCE_DOWNMIX_LATENT_FILENAME,
    SOURCE_MONO_LATENT_FILENAME,
    SOURCE_STEREO_LATENT_FILENAME,
    TARGET_LATENT_FILENAME,
    sample_dir_from_stream_hash,
)
from stereo2spatial.codecs.ear_vae import (
    decode_channels_independent,
    get_default_device,
    load_vae,
    vae_decode,
)

try:
    import soundfile as sf
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing dependency: soundfile. Install with `pip install soundfile`."
    ) from error


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
        return json.load(handle)


def to_cdt(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.contiguous()
    raise ValueError(f"{name} must have shape [C,D,T], got {tuple(tensor.shape)}")


def to_bdt(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.contiguous()
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).contiguous()
    raise ValueError(
        f"{name} must have shape [D,T] or [B,D,T], got {tuple(tensor.shape)}"
    )


def reduce_stereo_to_mono(stereo: torch.Tensor, mode: str) -> torch.Tensor:
    if stereo.dim() != 2 or stereo.shape[0] != 2:
        raise ValueError(f"Expected stereo [2,S], got {tuple(stereo.shape)}")
    if mode == "mean":
        return stereo.mean(dim=0, keepdim=True)
    if mode == "left":
        return stereo[0:1, :]
    if mode == "right":
        return stereo[1:2, :]
    raise ValueError("mono reduction must be one of: mean, left, right")


def decode_stereo_like(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    use_chunked_decode: bool,
    chunk_size_frames: int,
    overlap_frames: int,
    show_progress: bool,
    device: torch.device,
) -> torch.Tensor:
    decoded = vae_decode(
        vae=vae,
        pred_latents=to_bdt(latents, "stereo_like_latents"),
        use_chunked_decode=use_chunked_decode,
        chunk_size_frames=chunk_size_frames,
        overlap_frames=overlap_frames,
        offload_wav_to_cpu=True,
        normalize_audio=False,
        return_cpu_list=False,
        show_progress=show_progress,
        device=device,
    )
    if decoded.dim() == 3:
        return decoded[0].contiguous()
    if decoded.dim() == 2:
        return decoded.contiguous()
    raise ValueError(f"Unexpected decoded shape: {tuple(decoded.shape)}")


def tensor_to_soundfile_array(audio_cs: torch.Tensor) -> torch.Tensor:
    if audio_cs.dim() != 2:
        raise ValueError(f"Expected [C,S], got {tuple(audio_cs.shape)}")
    return audio_cs.transpose(0, 1).contiguous().cpu().numpy()


def safe_stem(raw_path: str) -> str:
    stem = Path(raw_path).stem if raw_path else "sample"
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_")
    return cleaned or "sample"


def load_latents(sample_dir: Path) -> dict[str, torch.Tensor]:
    bundle_path = sample_dir / SAMPLE_BUNDLE_FILENAME
    if bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu")
        if not isinstance(bundle, dict):
            raise TypeError(f"Invalid sample bundle payload: {type(bundle)}")
        required = {
            "target_latent",
            "source_stereo_latent",
            "source_mono_latent",
            "source_downmix_latent",
        }
        missing = [key for key in required if key not in bundle]
        if missing:
            raise KeyError(f"Bundle missing keys: {missing}")
        return {
            "target_latent": bundle["target_latent"],
            "source_stereo_latent": bundle["source_stereo_latent"],
            "source_mono_latent": bundle["source_mono_latent"],
            "source_downmix_latent": bundle["source_downmix_latent"],
        }

    split_paths = {
        "target_latent": sample_dir / TARGET_LATENT_FILENAME,
        "source_stereo_latent": sample_dir / SOURCE_STEREO_LATENT_FILENAME,
        "source_mono_latent": sample_dir / SOURCE_MONO_LATENT_FILENAME,
        "source_downmix_latent": sample_dir / SOURCE_DOWNMIX_LATENT_FILENAME,
    }
    missing_files = [str(path) for path in split_paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing latent artifacts and no bundle found:\n  - "
            + "\n  - ".join(missing_files)
        )
    return {
        name: torch.load(path, map_location="cpu") for name, path in split_paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode one processed sample into QC WAVs: target multichannel, "
            "source stereo, source mono, and source downmix."
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
        help="Output folder for decoded WAVs. Defaults to <sample-dir>/_qc_decodes.",
    )
    parser.add_argument(
        "--vae-checkpoint-path",
        required=True,
        help="Path to EAR_VAE checkpoint.",
    )
    parser.add_argument(
        "--vae-config-path",
        default=None,
        help="Optional path to VAE config JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for decoding (for example: cuda, cpu). Defaults to auto.",
    )
    parser.add_argument(
        "--disable-chunked-decode",
        action="store_true",
        help="Disable chunked VAE decode.",
    )
    parser.add_argument(
        "--decode-chunk-size-frames",
        type=int,
        default=2048,
        help="Chunk size in latent frames for chunked decode.",
    )
    parser.add_argument(
        "--decode-overlap-frames",
        type=int,
        default=256,
        help="Overlap in latent frames for chunked decode.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show decode progress bars.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve(strict=False)
    sample_dir = resolve_sample_dir(args.sample_dir, args.stream_hash, dataset_root)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    metadata = load_metadata(sample_dir)
    latents = load_latents(sample_dir)

    sample_rate = int(metadata.get("sample_rate", 48000))
    target_layout = str(metadata.get("target_layout", "unknown_layout"))
    mono_reduction = str(metadata.get("mono_reduction", "mean")).strip().lower()
    stream_hash = str(metadata.get("stream_hash", sample_dir.name))
    source_path = str(metadata.get("source_path", ""))

    out_dir = (
        Path(args.out_dir).resolve(strict=False)
        if args.out_dir
        else sample_dir / "_qc_decodes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    decode_device = torch.device(args.device) if args.device else get_default_device()
    print(f"Loading VAE on device={decode_device} ...")
    vae = load_vae(
        vae_checkpoint_path=args.vae_checkpoint_path,
        config_path=args.vae_config_path,
        device=decode_device,
        torch_dtype=torch.float32,
    )
    print("VAE loaded.")

    use_chunked_decode = not args.disable_chunked_decode

    target_latent = to_cdt(latents["target_latent"], "target_latent")
    source_stereo_latent = to_bdt(
        latents["source_stereo_latent"], "source_stereo_latent"
    )
    source_mono_latent = to_bdt(latents["source_mono_latent"], "source_mono_latent")
    source_downmix_latent = to_bdt(
        latents["source_downmix_latent"], "source_downmix_latent"
    )

    print("Decoding target multichannel latents ...")
    target_audio = decode_channels_independent(
        vae=vae,
        channel_latents=target_latent,
        use_chunked_decode=use_chunked_decode,
        chunk_size_frames=args.decode_chunk_size_frames,
        overlap_frames=args.decode_overlap_frames,
        offload_wav_to_cpu=True,
        reduction="mean",
        show_progress=args.show_progress,
        device=decode_device,
    )

    print("Decoding source stereo latent ...")
    source_stereo_audio = decode_stereo_like(
        vae=vae,
        latents=source_stereo_latent,
        use_chunked_decode=use_chunked_decode,
        chunk_size_frames=args.decode_chunk_size_frames,
        overlap_frames=args.decode_overlap_frames,
        show_progress=args.show_progress,
        device=decode_device,
    )

    print("Decoding source mono latent ...")
    source_mono_stereo_audio = decode_stereo_like(
        vae=vae,
        latents=source_mono_latent,
        use_chunked_decode=use_chunked_decode,
        chunk_size_frames=args.decode_chunk_size_frames,
        overlap_frames=args.decode_overlap_frames,
        show_progress=args.show_progress,
        device=decode_device,
    )
    source_mono_audio = reduce_stereo_to_mono(source_mono_stereo_audio, mono_reduction)

    print("Decoding source downmix latent ...")
    source_downmix_audio = decode_stereo_like(
        vae=vae,
        latents=source_downmix_latent,
        use_chunked_decode=use_chunked_decode,
        chunk_size_frames=args.decode_chunk_size_frames,
        overlap_frames=args.decode_overlap_frames,
        show_progress=args.show_progress,
        device=decode_device,
    )

    prefix = safe_stem(source_path) + "__" + stream_hash[:12]
    target_suffix = re.sub(r"[^0-9A-Za-z]+", "_", target_layout).strip("_").lower()
    if not target_suffix:
        target_suffix = "target"

    target_wav = out_dir / f"{prefix}__decoded_{target_suffix}.wav"
    stereo_wav = out_dir / f"{prefix}__decoded_stereo.wav"
    mono_wav = out_dir / f"{prefix}__decoded_mono.wav"
    downmix_wav = out_dir / f"{prefix}__decoded_downmix.wav"

    sf.write(
        target_wav,
        tensor_to_soundfile_array(target_audio),
        sample_rate,
        subtype="FLOAT",
    )
    sf.write(
        stereo_wav,
        tensor_to_soundfile_array(source_stereo_audio),
        sample_rate,
        subtype="FLOAT",
    )
    sf.write(
        mono_wav,
        tensor_to_soundfile_array(source_mono_audio),
        sample_rate,
        subtype="FLOAT",
    )
    sf.write(
        downmix_wav,
        tensor_to_soundfile_array(source_downmix_audio),
        sample_rate,
        subtype="FLOAT",
    )

    report = {
        "sample_dir": str(sample_dir),
        "stream_hash": stream_hash,
        "source_path": source_path,
        "sample_rate": sample_rate,
        "target_layout": target_layout,
        "mono_reduction": mono_reduction,
        "decoded_shapes": {
            "target": [int(x) for x in target_audio.shape],
            "source_stereo": [int(x) for x in source_stereo_audio.shape],
            "source_mono": [int(x) for x in source_mono_audio.shape],
            "source_downmix": [int(x) for x in source_downmix_audio.shape],
        },
        "output_files": {
            "target": str(target_wav),
            "source_stereo": str(stereo_wav),
            "source_mono": str(mono_wav),
            "source_downmix": str(downmix_wav),
        },
    }
    report_path = out_dir / f"{prefix}__decode_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print("Wrote QC decodes:")
    print(f"  - {target_wav}")
    print(f"  - {stereo_wav}")
    print(f"  - {mono_wav}")
    print(f"  - {downmix_wav}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
