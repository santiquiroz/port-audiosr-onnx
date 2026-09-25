import json
import re

import export_fp16
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from manifest import sha256_of
from onnx import TensorProto, helper, numpy_helper

CPU = ["CPUExecutionProvider"]
WIDTH = 64
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def rms_norm_graph():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((WIDTH, WIDTH)).astype(np.float32) * 0.1
    initializers = [
        numpy_helper.from_array(weight, "w"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(1e-5, np.float32), "eps"),
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"]),
        helper.make_node("Pow", ["h", "two"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["mean", "eps"], ["var"]),
        helper.make_node("Sqrt", ["var"], ["rms"]),
        helper.make_node("Div", ["h", "rms"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "rms_norm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, WIDTH])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, WIDTH])],
        initializers,
    )
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])


def save_fp32(model, path):
    onnx.save(model, str(path), save_as_external_data=True,
              all_tensors_to_one_file=True, location=path.name + ".data")


@pytest.fixture
def fp32_pack(tmp_path):
    src = tmp_path / "fp32"
    src.mkdir()
    save_fp32(rms_norm_graph(), src / "tiny.onnx")
    np.save(src / "table.npy", np.arange(8, dtype=np.float64))
    manifest = {
        "model": "synthetic",
        "required_files": ["table.npy", "tiny.onnx", "tiny.onnx.data", "manifest.json"],
    }
    (src / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return src


def run_cpu(path, x):
    return ort.InferenceSession(str(path), providers=CPU).run(None, {"x": x})[0]


def elem_types(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    typed = list(graph.value_info) + list(graph.input) + list(graph.output)
    types = {v.name: v.type.tensor_type.elem_type for v in typed}
    types.update({t.name: t.data_type for t in graph.initializer})
    return types


def test_converted_graph_matches_fp32_on_cpu(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")
    x = np.random.default_rng(1).standard_normal((4, WIDTH)).astype(np.float32)

    expected = run_cpu(fp32_pack / "tiny.onnx", x)
    actual = run_cpu(dst / "tiny.onnx", x)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)


def test_converted_graph_really_is_fp16(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")

    model = onnx.load(str(dst / "tiny.onnx"))
    weight = next(t for t in model.graph.initializer if t.dims == [WIDTH, WIDTH])
    assert weight.data_type == TensorProto.FLOAT16


def test_block_listed_nodes_stay_float32(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")

    model = onnx.load(str(dst / "tiny.onnx"))
    types = elem_types(model)
    blocked = [n for n in model.graph.node if n.op_type in export_fp16.OP_BLOCK_LIST]
    assert {n.op_type for n in blocked} == set(export_fp16.OP_BLOCK_LIST)
    for node in blocked:
        for name in list(node.input) + list(node.output):
            assert types[name] == TensorProto.FLOAT, (node.op_type, name)


def test_tensor_feeding_two_blocked_nodes_gets_a_single_cast(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")

    model = onnx.load(str(dst / "tiny.onnx"), load_external_data=False)
    names = [n.name for n in model.graph.node if n.name]
    outputs = [o for n in model.graph.node for o in n.output]
    assert len(names) == len(set(names))
    assert len(outputs) == len(set(outputs))


def test_value_info_is_dropped(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")

    model = onnx.load(str(dst / "tiny.onnx"), load_external_data=False)
    assert len(model.graph.value_info) == 0


def test_converting_twice_does_not_append_external_data(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")
    first = (dst / "tiny.onnx.data").stat().st_size

    export_fp16.convert_graph(fp32_pack / "tiny.onnx", dst / "tiny.onnx")

    assert first > 0
    assert (dst / "tiny.onnx.data").stat().st_size == first


def test_fp16_required_files_add_data_next_to_each_graph():
    fp32 = ["alphas_cumprod.npy", "ddpm.onnx", "ddpm.onnx.data", "mel_basis.npy",
            "vae_decoder.onnx", "vocoder.onnx", "manifest.json"]

    assert export_fp16.fp16_required_files(fp32) == [
        "alphas_cumprod.npy", "ddpm.onnx", "ddpm.onnx.data", "mel_basis.npy",
        "vae_decoder.onnx", "vae_decoder.onnx.data",
        "vocoder.onnx", "vocoder.onnx.data", "manifest.json",
    ]


def test_convert_pack_writes_fp16_manifest_with_hashes(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_pack(fp32_pack, dst)

    manifest = json.loads((dst / "manifest.json").read_text())
    assert manifest["precision"] == "fp16"
    assert manifest["model"] == "synthetic"
    hashed = [f for f in manifest["required_files"] if f != "manifest.json"]
    assert set(manifest["sha256"]) == set(hashed)
    for name in hashed:
        assert (dst / name).exists()
        assert SHA256_HEX.match(manifest["sha256"][name])
    assert manifest["sha256"]["tiny.onnx"] == sha256_of(dst / "tiny.onnx")


def test_convert_pack_records_fp32_source_hashes(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"
    export_fp16.convert_pack(fp32_pack, dst)

    manifest = json.loads((dst / "manifest.json").read_text())
    sources = manifest["fp16"]["source_sha256"]
    assert set(sources) == {"tiny.onnx", "tiny.onnx.data"}
    assert sources["tiny.onnx.data"] == sha256_of(fp32_pack / "tiny.onnx.data")
    assert manifest["fp16"]["op_block_list"] == list(export_fp16.OP_BLOCK_LIST)


def test_convert_pack_leaves_fp32_manifest_untouched(fp32_pack, tmp_path):
    before = (fp32_pack / "manifest.json").read_text()

    export_fp16.convert_pack(fp32_pack, tmp_path / "fp16")

    assert (fp32_pack / "manifest.json").read_text() == before


def test_main_converts_pack_from_cli(fp32_pack, tmp_path):
    dst = tmp_path / "fp16"

    export_fp16.main([str(fp32_pack), str(dst)])

    assert json.loads((dst / "manifest.json").read_text())["precision"] == "fp16"


def test_main_rejects_same_source_and_destination(fp32_pack):
    with pytest.raises(SystemExit) as exc:
        export_fp16.main([str(fp32_pack), str(fp32_pack)])
    assert exc.value.code != 0
