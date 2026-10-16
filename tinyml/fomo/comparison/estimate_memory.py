"""Pre-board G3 estimate: peak activation memory of each candidate's INT8 graph.

The same estimator runs on every model: tensor lifetimes come from operator order, and buffers are
placed with a greedy first-fit planner (largest first), the strategy of TFLite Micro's
GreedyMemoryPlanner. Activations count 1 byte per element (INT8). Weights are excluded (they stay in
flash), as are runtime-specific persistent and scratch allocations, so this is a lower bound to compare
candidates, not a device measurement; RecordingMicroInterpreter / ESP-DL logs on the board replace it.

TFLite models need TensorFlow (.venv); ONNX models need onnx (.venv-espdet).
Usage: estimate_memory.py <model.tflite|model.onnx> [...]
"""
import json
from pathlib import Path
import sys


def tflite_graph(path):
    from tensorflow.lite.python import schema_py_generated as schema
    model = schema.Model.GetRootAsModel(Path(path).read_bytes(), 0)
    graph = model.Subgraphs(0)
    sizes, constant = {}, set()
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        elements = 1
        for d in range(tensor.ShapeLength()):
            elements *= max(int(tensor.Shape(d)), 1)
        sizes[index] = elements
        if model.Buffers(tensor.Buffer()).DataLength() > 0:
            constant.add(index)
    ops = []
    for o in range(graph.OperatorsLength()):
        op = graph.Operators(o)
        ins = [op.Inputs(i) for i in range(op.InputsLength()) if op.Inputs(i) >= 0]
        outs = [op.Outputs(i) for i in range(op.OutputsLength())]
        ops.append((ins, outs))
    inputs = [graph.Inputs(i) for i in range(graph.InputsLength())]
    outputs = [graph.Outputs(i) for i in range(graph.OutputsLength())]
    return sizes, constant, ops, inputs, outputs


def onnx_graph(path):
    import onnx
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    graph = model.graph
    sizes = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        dims = value.type.tensor_type.shape.dim
        elements = 1
        for d in dims:
            elements *= d.dim_value if d.dim_value > 0 else 1
        sizes[value.name] = elements
    constant = {i.name for i in graph.initializer}
    constant |= {o for n in graph.node if n.op_type == "Constant" for o in n.output}
    ops = [([i for i in n.input if i], list(n.output)) for n in graph.node if n.op_type != "Constant"]
    inputs = [i.name for i in graph.input if i.name not in constant]
    outputs = [o.name for o in graph.output]
    missing = [t for ins, outs in ops for t in ins + outs if t not in sizes and t not in constant]
    if missing:
        raise ValueError(f"Shape inference left {len(missing)} tensors without shapes, e.g. {missing[:3]}")
    return sizes, constant, ops, inputs, outputs


def plan(sizes, constant, ops, inputs, outputs):
    first, last = {}, {}
    for t in inputs:
        first[t] = last[t] = 0
    for step, (ins, outs) in enumerate(ops):
        for t in outs:
            if t not in constant:
                first.setdefault(t, step)
                last[t] = max(last.get(t, step), step)
        for t in ins:
            if t not in constant:
                first.setdefault(t, step)
                last[t] = step
    for t in outputs:
        last[t] = len(ops)
    buffers = sorted(first, key=lambda t: (-sizes[t], str(t)))
    placed = []  # (offset, size, first, last)
    peak = 0
    for t in buffers:
        size, start, end = sizes[t], first[t], last[t]
        live = sorted((o, s) for o, s, a, b in placed if not (b < start or a > end))
        offset = 0
        for o, s in live:
            if offset + size <= o:
                break
            offset = max(offset, o + s)
        placed.append((offset, size, start, end))
        peak = max(peak, offset + size)
    largest = max(sizes[t] for t in buffers)
    return {"peak_activation_bytes": int(peak), "largest_tensor_bytes": int(largest),
            "activation_tensors": len(buffers), "operators": len(ops)}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        graph = tflite_graph(path) if path.endswith(".tflite") else onnx_graph(path)
        print(json.dumps({"model": path, **plan(*graph)}))


if __name__ == "__main__":
    main()
