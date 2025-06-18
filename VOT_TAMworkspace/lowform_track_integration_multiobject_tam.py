import os, sys
import cv2

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
    
from lib.test.tracker.VOT_lowformer_track import lowformer_track_imple
import vot, torch
import numpy as np

def _mask_from_rect(rect, output_size):
    '''
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    '''
    mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    x0 = max(int(round(rect[0])), 0)
    y0 = max(int(round(rect[1])), 0)
    x1 = min(int(round(rect[0] + rect[2])), output_size[0])
    y1 = min(int(round(rect[1] + rect[3])), output_size[1])
    mask[y0:y1, x0:x1] = 1
    return mask


def _rect_from_mask(mask):
    '''
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    '''
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]



def main():
    # tracker = lowformer_track_imple()
    
    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()
    imagefile = handle.frame()
    
    if not imagefile:
        sys.exit(0)

    image = cv2.imread(imagefile)#, cv2.IMREAD_GRAYSCALE)
    
    tam_predictor = build_tam()
    
    trackers = [lowformer_track_imple().initialize(image,_rect_from_mask(object)) for object in objects] 
    # tracker.initialize(image, selection)
    
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        
        image = cv2.imread(imagefile)#, cv2.IMREAD_GRAYSCALE)
        
        # results = [_mask_from_rect(tracker.track(image)[0], (image.shape[1], image.shape[0])) for tracker in trackers]
        results = [predict_tam(tam_predictor,tracker.track(image)[0], image) for tracker in trackers]
        
        
        # region, confidence = tracker.track(image)
        # region = vot.Rectangle(region[0], region[1], region[2],region[3])
        # print(region, confidence)
        
        handle.report(results)
        # handle.report(region, confidence)
        # print("it")
        
        
        
def build_tam():
    from TAM_segmentor.EfficientTAM.efficient_track_anything.build_efficienttam import (
    build_efficienttam,
    )
    from TAM_segmentor.EfficientTAM.efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)
    
    checkpoint = "/home/moritz/Research/SMAT/TAM_segmentor/EfficientTAM/checkpoints/efficienttam_s.pt"
    model_cfg = "/home/moritz/Research/SMAT/TAM_segmentor/EfficientTAM/efficient_track_anything/configs/efficienttam/efficienttam_s.yaml"
    model_cfg = "configs/efficienttam/efficienttam_s.yaml"
    
    # import hydra
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize_config_module(model_cfg, version_base='1.2')
    
    predictor = build_efficienttam(model_cfg, checkpoint)
    predictor = EfficientTAMImagePredictor(predictor)


    return predictor

def predict_tam(predictor, bbox, image):
    print("predict tam initiated", image.shape, predictor.device, bbox)
    predictor.set_image(image)
    print("image set")
    # to xyxy
    bbox = np.array(bbox)
    bbox[2:]+=bbox[:2]
    
    # bbox 4-array numpy, xyxy
    masks, quality, _ = predictor.predict(box=bbox, multimask_output=True)
    print("prediction done!")
    max_index = np.argmax(quality)
    ret_mask = masks[max_index,:,:]
    ret_mask = ret_mask.astype(np.uint8)
    print("result is returned now...", ret_mask.shape, ret_mask.dtype)
    
    return ret_mask
    
if __name__ == "__main__":
    # build_tam()
    
    main()
    