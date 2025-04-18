import numpy as np
import os
import json
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class TREK150Dataset(BaseDataset):
    """
    TREK 150
    """
    def __init__(self, attribute=None):
        super().__init__()
        self.base_path = self.env_settings.trek150_path
        self.sequence_list = self._get_sequence_list()

        self.att_dict = None

        if attribute is not None:
            self.sequence_list = self._filter_sequence_list_by_attribute(attribute, self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/full_occlusion/{}_full_occlusion.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        # out_of_view_label_path = '{}/out_of_view/{}_out_of_view.txt'.format(self.base_path, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible =  np.ones(ground_truth_rect.shape[0]) #np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/img/'.format(self.base_path, sequence_name)

        # frames_list = ['{}/frame_{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        frames_list = [os.path.join(frames_path,i) for i in sorted(os.listdir(frames_path), key= lambda x: int(x.replace(".jpg","").replace("frame_","")))]
        

        # print(frames_list)
        # print(ground_truth_rect.shape)
        # assert False
        # ground_truth_rect = ground_truth_rect.reshape(-1, 4)
        # ground_truth_rect[:,2] -= ground_truth_rect[:,0]   
        
        
        return Sequence(sequence_name, frames_list, 'trek150', ground_truth_rect, target_visible=target_visible)

    def get_attribute_names(self, mode='short'):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        names = self.att_dict['att_name_short'] if mode == 'short' else self.att_dict['att_name_long']
        return names

    def _load_attributes(self):
        assert False
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'dataset_attribute_specs', 'avist_attributes.json'), 'r') as f:
            att_dict = json.load(f)
        return att_dict

    def _filter_sequence_list_by_attribute(self, att, seq_list):
        if self.att_dict is None:
            self.att_dict = self._load_attributes()

        if att not in self.att_dict['att_name_short']:
            if att in self.att_dict['att_name_long']:
                att = self.att_dict['att_name_short'][self.att_dict['att_name_long'].index(att)]
            else:
                raise ValueError('\'{}\' attribute invalid.')

        return [s for s in seq_list if att in self.att_dict[s]]

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)  # frames start from 1

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open(os.path.join(self.base_path,"sequences.txt"),"r") as read_file:
            sequence_list = [i.replace("\n","") for i in read_file.readlines()]
            
        return sequence_list
