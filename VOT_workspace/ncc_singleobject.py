#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy
import collections

# from ncc_tracker import NCCTrackerImpl
import cv2
import numpy as np
import vot


class NCCTrackerImpl(object):

    def __init__(self, image, region):
        if isinstance(region, np.ndarray):
            region = self._rect_from_mask(region)
            region = vot.Rectangle(region[0], region[1], region[2], region[3])

        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

    def track(self, image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]], 0

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]], max_val

    def _rect_from_mask(self, mask):
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

    def _mask_from_rect(self, rect, output_size):
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

class NCCTracker(object):
    def __init__(self, image, region):
        self.ncc_ = NCCTrackerImpl(image, region)

    def track(self, image):
        pred_region, max_val = self.ncc_.track(image)
        return vot.Rectangle(pred_region[0], pred_region[1], pred_region[2], pred_region[3]), max_val

handle = vot.VOT("rectangle")
selection = handle.region()
print("region extracted")
imagefile = handle.frame()
print("frame extracted")
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
tracker = NCCTracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    region, confidence = tracker.track(image)
    handle.report(region, confidence)