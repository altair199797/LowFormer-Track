import os, sys
import onnx
import argparse
from onnx import helper
import copy

def fix_model(model_path, new_model_path):
    model = onnx.load(model_path)
    nodes_to_remove = []
    params = {tensor.name: tensor for tensor in model.graph.initializer}
    for node in model.graph.node:
        if not len(node.input):
            continue
        original_param_name = node.input[0]
        if node.op_type == "Identity" and original_param_name in params:
            new_param_tensor = copy.deepcopy(params[original_param_name])
            new_param_tensor.name = node.output[0]
            model.graph.initializer.append(new_param_tensor)

            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    onnx.save(model, new_model_path)

def main():
    fold = ["Wonnxmodels", "Wonnxmodelsv2", "Wonnxmodelsv3", "Wonnxmodelsv4"]
    fold = fold[3]
    fold = "/home/moritz/Research/SMAT/tracking/onnx_models"
    models = [i for i in os.listdir(fold) if ".onnx" in i]

    newfold = fold + "_fix"
    if not os.path.exists(newfold):
        os.makedirs(newfold)
    for model_name in models:
        try:
            fix_model(os.path.join(fold, model_name), os.path.join(newfold, model_name))
        except Exception as e:
            print("didn't work for", model_name, e)
    
if __name__ == "__main__":
    main()