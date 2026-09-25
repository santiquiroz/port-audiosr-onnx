import hashlib
import json

import manifest
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

SCALE_FACTOR = 0.3342401385307312


def identity_graph(opset, weight=None):
    initializers = [] if weight is None else [numpy_helper.from_array(weight, "w")]
    nodes = [helper.make_node("Add" if initializers else "Identity",
                              ["x", "w"] if initializers else ["x"], ["y"])]
    graph = helper.make_graph(
        nodes, "synthetic",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
        initializers,
    )
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", opset)])


def save_with_data(model, path):
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=path.name + ".data", size_threshold=0)


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def art_dir(tmp_path):
    save_with_data(identity_graph(18, np.ones(4, np.float32)), tmp_path / "ddpm.onnx")
    onnx.save(identity_graph(17), str(tmp_path / "vocoder.onnx"))
    save_with_data(identity_graph(18, np.zeros(4, np.float32)), tmp_path / "ddpm_fp16.onnx")
    np.save(tmp_path / "alphas_cumprod.npy", np.linspace(1, 0, 8))
    np.save(tmp_path / "mel_basis.npy", np.eye(4, dtype=np.float32))
    np.save(tmp_path / "ddpm_in0.npy", np.zeros(4, np.float32))
    return tmp_path


def test_required_files_ignore_foreign_files(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    assert "ddpm_fp16.onnx" not in built["required_files"]
    assert "ddpm_fp16.onnx.data" not in built["required_files"]
    assert "ddpm_in0.npy" not in built["required_files"]


def test_required_files_list_known_graphs_constants_and_existing_data(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    assert built["required_files"] == [
        "alphas_cumprod.npy", "ddpm.onnx", "ddpm.onnx.data", "mel_basis.npy",
        "vae_decoder.onnx", "vae_feature_extract.onnx", "vocoder.onnx", "manifest.json",
    ]


def test_opsets_are_read_per_graph(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    assert built["opsets"] == {"ddpm": 18, "vocoder": 17}
    assert "opset" not in built


def test_precision_is_fp32(art_dir):
    assert manifest.build_manifest(art_dir, SCALE_FACTOR)["precision"] == "fp32"


def test_sha256_covers_existing_assets(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    present = ["alphas_cumprod.npy", "ddpm.onnx", "ddpm.onnx.data", "mel_basis.npy",
               "vocoder.onnx"]
    assert built["sha256"] == {name: sha256_of(art_dir / name) for name in present}


def test_scale_factor_and_static_config(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    assert built["scale_factor"] == SCALE_FACTOR
    assert built["sampling_rate"] == 48000
    assert built["latent"]["frames_per_second"] == 12.5
    assert set(built["graphs"]) == set(manifest.GRAPHS)


def test_manifest_is_json_serializable(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    assert json.loads(json.dumps(built)) == built


def test_missing_files_names_absent_required_assets(art_dir):
    built = manifest.build_manifest(art_dir, SCALE_FACTOR)

    assert manifest.missing_files(art_dir, built) == ["vae_decoder.onnx",
                                                      "vae_feature_extract.onnx"]


def test_opset_ignores_non_default_domains(tmp_path):
    model = identity_graph(17)
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(model, str(tmp_path / "vocoder.onnx"))

    assert manifest.graph_opset(tmp_path / "vocoder.onnx") == 17
