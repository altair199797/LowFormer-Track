import _init_paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

import os, sys



parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--dataset', default="otb", type=str, help='Name of config file.')
parser.add_argument('--datasetadd', default="", type=str, help='Name of config file.')

parser.add_argument('--runid', default=None, type=int, help='Name of config file.')
parser.add_argument('--track_arch', default="lowformer", type=str, help='Name of config file.')
parser.add_argument('--add', default="", type=str, help='Name of config file.')
parser.add_argument('--config', default="", type=str, help='Name of config file.')


# parser.add_argument('tracker_param', type=str, help='Name of config file.')
# # parser.add_argument('--subset_clips_file', type=str, default=None)
# # parser.add_argument('--runid', type=int, default=None)
# # parser.add_argument('--skipmissing', action='store_true', default=False,help='')
# # parser.add_argument('--protocol', type=str, default="")    
args = parser.parse_args()

trackers = []

chosen_model = "mobilevitv2_256_128x1_ep300"
chosen_model = "mobilevitv2_256_128x1_ep300_coco"
chosen_model = "mobilevitv2_256_128x1_ep300_mine"
chosen_model = args.track_arch
chosen_model += "_256_128x1_ep300_"+args.dataset+ args.datasetadd +"_b15" + args.add

if len(args.config):
    chosen_model = args.config
    args.track_arch = args.config.split("_")[0]
    

trackers.extend(trackerlist(name=args.track_arch+'_track', parameter_name=chosen_model, dataset_name=args.dataset, run_ids=args.runid))
trackers[0].results_dir =  os.path.join(trackers[0].results_dir,args.dataset.replace("_test","").lower())

dataset = get_dataset(args.dataset+("_val" if args.dataset.lower()=="got10k" else ""))
#dataset = get_dataset('egotracks_val')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, args.dataset, merge_results=True, plot_types=('success',"prec"), force_evaluation=True)

    
# python tracking/analysis_results.py --dataset lasot --config mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit_typembv2
