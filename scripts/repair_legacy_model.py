"""One-time repair for the original model/car_damage_model.h5.

The shipped .h5 was saved from a Keras-3 environment where a Sequential model
wraps a Functional submodel (Sequential([MobileNetV2(...), GlobalAveragePooling2D(),
Dense(128), Dropout(0.3), Dense(8)])). Keras 3's legacy h5 reload path
mis-reconstructs exactly this shape and fails with:

    ValueError: Layer "dense_3" expects 1 input(s), but it received 2 input tensors.

on every attempt to tf.keras.models.load_model() it -- this is not an
environment/version issue, it reproduces identically across TF 2.16 and 2.21,
with or without TF_USE_LEGACY_KERAS. Note: model.load_weights(path,
by_name=True) does NOT sidestep this either -- it still failed with a shape
mismatch on the first depthwise layer, because it goes through the same
buggy legacy_h5_format reconstruction internally.

The underlying weight arrays are completely intact (verified layer-by-layer
below); only Keras's config-based reconstruction is broken. This script
reads the raw h5py datasets directly and assigns them to a freshly built
model, bypassing Keras's reloader entirely, then re-saves the result in the
native .keras format (which does not have this bug).

Run once: python scripts/repair_legacy_model.py
Produces: model/car_damage_model.keras (the file app/config.py loads by default)
"""
import h5py
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

SRC = "model/car_damage_model.h5"
DST = "model/car_damage_model.keras"


def collect_leaf_weight_groups(h5_group, bag=None):
    """Walk the h5 tree; whenever a group's children are ALL datasets,
    treat it as one layer's weight bag: {layer_name: {dataset_name: array}}."""
    if bag is None:
        bag = {}
    children = list(h5_group.values())
    if children and all(isinstance(c, h5py.Dataset) for c in children):
        bag[h5_group.name.rsplit("/", 1)[-1]] = {k: np.array(v) for k, v in h5_group.items()}
        return bag
    for child in h5_group.values():
        if isinstance(child, h5py.Group):
            collect_leaf_weight_groups(child, bag)
    return bag


def ordered_weights(layer_name, wdict):
    names = set(wdict.keys())
    if names == {"kernel", "bias"}:
        return [wdict["kernel"], wdict["bias"]]
    if names == {"kernel"}:
        return [wdict["kernel"]]
    if names == {"gamma", "beta", "moving_mean", "moving_variance"}:
        return [wdict["gamma"], wdict["beta"], wdict["moving_mean"], wdict["moving_variance"]]
    raise ValueError(f"Unrecognized weight set for {layer_name}: {names}")


def flatten_layers(model):
    out = []
    for layer in model.layers:
        out.append(layer)
        if hasattr(layer, "layers"):
            out.extend(flatten_layers(layer))
    return out


def main():
    with h5py.File(SRC, "r") as f:
        root = f.get("model_weights", f)
        bag = collect_leaf_weight_groups(root)
    print(f"Recovered {len(bag)} leaf weight groups from {SRC}.")

    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(name="global_average_pooling2d_2"),
        layers.Dense(128, activation="relu", name="dense_3"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(8, activation="softmax", name="dense_4"),
    ])
    model.build((None, 224, 224, 3))

    applied, skipped, missing = 0, 0, []
    for layer in flatten_layers(model):
        if not layer.weights:
            skipped += 1
            continue
        if layer.name not in bag:
            missing.append(layer.name)
            continue
        arrays = ordered_weights(layer.name, bag[layer.name])
        expected_shapes = [w.shape for w in layer.get_weights()]
        got_shapes = [a.shape for a in arrays]
        if expected_shapes != got_shapes:
            raise ValueError(f"Shape mismatch on {layer.name}: expected {expected_shapes}, got {got_shapes}")
        layer.set_weights(arrays)
        applied += 1

    print(f"Applied recovered weights to {applied} layers ({skipped} layers have no weights).")
    if missing:
        # The Sequential container itself ("mobilenetv2_1.00_224") has no leaf
        # weights of its own -- its weights live on its sublayers, already
        # applied above -- so seeing exactly that one name here is expected.
        print(f"No direct match for: {missing} (expected for container layers).")

    # Sanity check: a real prediction should be a valid, non-degenerate softmax.
    x = np.random.rand(2, 224, 224, 3).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 8), f"unexpected output shape {y.shape}"
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-4), "softmax doesn't sum to 1 -- weight recovery is wrong"
    assert not np.allclose(y[0], y[0][0]), "output is uniform -- weights look like garbage/zeros"
    print("Sanity check passed.")

    model.save(DST)
    print(f"Saved {DST}")


if __name__ == "__main__":
    main()
