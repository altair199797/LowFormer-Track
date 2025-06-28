import os,sys 

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.tracker.VOT_lowformer_track import lowformer_track_imple


def build_tam():
    from TAM_segmentor.EfficientTAM.efficient_track_anything.build_efficienttam import (
    build_efficienttam,
    )
    from TAM_segmentor.EfficientTAM.efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)
    
    checkpoint = "/home/moritz/Research/SMAT/TAM_segmentor/EfficientTAM/checkpoints/efficienttam_ti_512x512.pt"
    model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"
    
    # checkpoint = "/home/moritz/Research/SMAT/TAM_segmentor/EfficientTAM/checkpoints/efficienttam_s.pt"
    # model_cfg = "configs/efficienttam/efficienttam_s.yaml"
    
    
    # import hydra
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize_config_module(model_cfg, version_base='1.2')
    
    predictor = build_efficienttam(model_cfg, checkpoint)
    predictor = EfficientTAMImagePredictor(predictor)


    return predictor


lowformtrack = lowformer_track_imple()
print(lowformtrack)



tammodel =build_tam()
print(tammodel)