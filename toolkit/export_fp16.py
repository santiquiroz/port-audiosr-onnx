"""Convert an exported fp32 pack (artifacts/) into the fp16 pack published as
release models-fp16-v1.0, plus its manifest (precision, required_files with the
.data of every graph, SHA-256 of each asset and of the fp32 sources).

Graph inputs/outputs stay float32, so the numpy driver is unchanged. fp16 is for
GPU (DirectML): the CPU EP has few fp16 kernels, so CPU users keep the fp32 pack.

Usage: python toolkit/export_fp16.py <fp32_dir> <fp16_dir>
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx
from onnxruntime.transformers import float16
from onnxruntime.transformers.onnx_model import OnnxModel

# RMSNorm/GroupNorm stay in fp32 or the output is silently NaN: Pow squares the
# activations and ReduceMean sums the squares past fp16's 65504 ceiling.
OP_BLOCK_LIST = ("Pow", "ReduceMean", "Sqrt", "Div")
MANIFEST = "manifest.json"
GRAPH_SUFFIX = ".onnx"
DATA_SUFFIX = ".onnx.data"
FP16_NOTES = {
    "converted_from": "models-v1.0 (fp32)",
    "converter": "onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)",
    "op_block_list": list(OP_BLOCK_LIST),
    "io_dtype": "float32 (graphs cast internally, so the driver is unchanged)",
    "not_for_cpu": "The CPU EP has far fewer fp16 kernels and the ones it has are often "
                   "slower. Use the fp32 release when running on CPU.",
}


def convert_graph(src: Path, dst: Path) -> None:
    converted = float16.convert_float_to_float16(
        onnx.load(str(src)), keep_io_types=True, op_block_list=list(OP_BLOCK_LIST)
    )
    # Stale fp32 value_info makes the fp16 graph fail type checks at load time.
    del converted.graph.value_info[:]
    drop_duplicate_nodes(converted.graph)
    wrapper = OnnxModel(converted)
    # The converter appends its graph_input_cast nodes after their consumers.
    wrapper.topological_sort()
    save_external(wrapper.model, dst)


def duplicate_node_indices(nodes) -> list[int]:
    seen = set()
    duplicates = []
    for i, node in enumerate(nodes):
        key = node.SerializeToString(deterministic=True)
        if key in seen:
            duplicates.append(i)
        seen.add(key)
    return duplicates


def drop_duplicate_nodes(graph: onnx.GraphProto) -> None:
    # Newer converters (seen in ORT 1.30, not in 1.24.4) name the Cast after the
    # tensor, so a tensor feeding two blocked nodes gets two identical Casts.
    for i in reversed(duplicate_node_indices(graph.node)):
        del graph.node[i]


def save_external(model: onnx.ModelProto, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = dst.with_name(dst.name + ".data")
    # onnx.save appends to an existing external-data file instead of truncating it.
    dst.unlink(missing_ok=True)
    data.unlink(missing_ok=True)
    onnx.save(model, str(dst), save_as_external_data=True,
              all_tensors_to_one_file=True, location=data.name)


def is_graph(name: str) -> bool:
    return name.endswith(GRAPH_SUFFIX)


def is_graph_file(name: str) -> bool:
    return is_graph(name) or name.endswith(DATA_SUFFIX)


def fp16_required_files(fp32_required: list[str]) -> list[str]:
    required = []
    for name in fp32_required:
        if name.endswith(DATA_SUFFIX):
            continue
        required.append(name)
        if is_graph(name):
            required.append(name + ".data")
    return required


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hashes(folder: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_of(folder / name) for name in names}


def fp16_manifest(fp32_manifest: dict, asset_sha: dict, source_sha: dict) -> dict:
    return {
        **fp32_manifest,
        "required_files": fp16_required_files(fp32_manifest["required_files"]),
        "precision": "fp16",
        "fp16": {**FP16_NOTES, "source_sha256": source_sha},
        "sha256": asset_sha,
    }


def load_manifest(folder: Path) -> dict:
    return json.loads((folder / MANIFEST).read_text())


def convert_pack(src: Path, dst: Path) -> dict:
    fp32 = load_manifest(src)
    required = [n for n in fp32["required_files"] if n != MANIFEST]
    dst.mkdir(parents=True, exist_ok=True)
    for name in required:
        copy_or_convert(src, dst, name)
    source_sha = hashes(src, [n for n in required if is_graph_file(n)])
    manifest = fp16_manifest(fp32, hashes(dst, fp16_required_files(required)), source_sha)
    (dst / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def copy_or_convert(src: Path, dst: Path, name: str) -> None:
    if name.endswith(DATA_SUFFIX):
        return
    if is_graph(name):
        print(f"[fp16] converting {name}...")
        convert_graph(src / name, dst / name)
        return
    shutil.copy2(src / name, dst / name)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="fp32 pack (with manifest.json)")
    ap.add_argument("dst", type=Path, help="output folder for the fp16 pack")
    args = ap.parse_args(argv)
    if args.src.resolve() == args.dst.resolve():
        ap.error("src and dst must differ: converting in place would delete the fp32 graphs")
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest = convert_pack(args.src, args.dst)
    print(f"[fp16] wrote {len(manifest['sha256'])} assets + {MANIFEST} to {args.dst}")


if __name__ == "__main__":
    main()
