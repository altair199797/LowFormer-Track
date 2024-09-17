import os, sys, yaml, torch
from copy import deepcopy
from cls import efficientvit_cls_b0, lowformer_cls_b1, efficientvit_cls_b2, efficientvit_cls_b3, efficientvit_cls_l1, efficientvit_cls_l2, efficientvit_cls_l3, LowFormerCls


def load_config(filename: str) -> dict:
    """Load a yaml file."""
    filename = os.path.realpath(os.path.expanduser(filename))
    return yaml.load(open(filename), Loader=SafeLoaderWithTuple)

def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        if key in config and isinstance(partial_config[key], dict) and isinstance(config[key], dict):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]
    return config

class SafeLoaderWithTuple(yaml.SafeLoader):
    """A yaml safe loader with python tuple loading capabilities."""

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoaderWithTuple.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderWithTuple.construct_python_tuple)


def setup_exp_config(config_path: str, recursive=True, opt_args: dict or None = None) -> dict:
    # load config
    if not os.path.isfile(config_path):
        raise ValueError(config_path)

    fpaths = [config_path]
    if recursive:
        extension = os.path.splitext(config_path)[1]
        while os.path.dirname(config_path) != config_path:
            config_path = os.path.dirname(config_path)
            fpath = os.path.join(config_path, "default" + extension)
            if os.path.isfile(fpath):
                fpaths.append(fpath)
        fpaths = fpaths[::-1]

    default_config = load_config(fpaths[0])
    exp_config = deepcopy(default_config)
    for fpath in fpaths[1:]:
        partial_update_config(exp_config, load_config(fpath))
    # update config via args
    if opt_args is not None:
        partial_update_config(exp_config, opt_args)

    return exp_config



def create_cls_model(name: str, pretrained=True, weight_url: str or None = None, **kwargs) -> LowFormerCls:
    model_dict = {
        "b0": efficientvit_cls_b0,
        "b1": lowformer_cls_b1,
        "b2": efficientvit_cls_b2,
        "b3": efficientvit_cls_b3,
        #########################
        "l1": efficientvit_cls_l1,
        "l2": efficientvit_cls_l2,
        "l3": efficientvit_cls_l3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)


    try:
        if pretrained:
            weight_url = weight_url #or REGISTERED_CLS_MODEL.get(name, None)
            if weight_url is None:
                raise ValueError(f"Do not find the pretrained weight of {name}.")
            else:
                weight = load_state_dict_from_file(weight_url)
                model.load_state_dict(weight)
    except Exception as e:
        print("Model weights could not be loaded!!!!!!!!!!!!!!!!!!!",e)
    return model

def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if "epoch" in checkpoint:
        print("checkpoint from epoch %d and its best validation result is %.3f" % (checkpoint["epoch"],checkpoint["best_val"]))
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def get_lowformer(config_path="Wymodelgetter/configs/b1.yaml", checkpoint_path="Wymodelgetter/checkpoints/b1/evalmodel.pt", less_layers=0):
    
    config = setup_exp_config(config_path, recursive=True, opt_args=None)
    
    model = create_cls_model(weight_url=checkpoint_path, pretrained=True, less_layers=less_layers, torchscriptsave=False, **config["net_config"])
    
    model = model.backbone
    model.max_stage_id = 4
    # print(model)
    inp = torch.randn(1,3,224,224)
    out = model(inp)
    print(out.keys())
    print(out[list(out.keys())[-1]].shape)
    return model

if __name__ == "__main__":
    get_lowformer()