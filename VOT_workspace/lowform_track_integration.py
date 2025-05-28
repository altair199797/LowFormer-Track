import os, sys
import cv2

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
    
from lib.test.tracker.VOT_lowformer_track import lowformer_track_imple
import vot

def main():
    tracker = lowformer_track_imple()
    
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    
    if not imagefile:
        sys.exit(0)

    image = cv2.imread(imagefile)#, cv2.IMREAD_GRAYSCALE)
    tracker.initialize(image, selection)
    
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.imread(imagefile)#, cv2.IMREAD_GRAYSCALE)
        region, confidence = tracker.track(image)
        region = vot.Rectangle(region[0], region[1], region[2],region[3])
        print(region, confidence)
        handle.report(region, confidence)
        # print("it")
    
if __name__ == "__main__":
    main()
    