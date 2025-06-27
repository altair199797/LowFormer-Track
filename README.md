# LowFormer-Track - ADAPTATION OF SMAT Tracker


## Thanks to the authors
The code is based upon the paper Separable Self and Mixed Attention Transformers for Efficient Object Tracking paper. Thanks for the authors to share their code!

## VOT Integration

The tracker integration is in `VOT_TAMworkspace/lowform_track_integration_multiobject_tam.py`.
The checkpoint can be found [here](https://www.dropbox.com/t/ZNPnw3eI6xU6SC66). It only needs to be put in the folder `output/checkpoints/train/mobilevitv2_track/mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit_b15`, where the `here_comes_the_checkpoint.txt` file is.
But checkpoint path can also be customized in `lib/test/tracker/VOT_lowformer_track.py` at line 51.

## Installation

Install the dependency packages using the environment file `smat_pyenv.yml`.

Generate the relevant files:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, modify the datasets paths by editing these files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Training

* Set the path of training datasets in `lib/train/admin/local.py`
* Place the pretrained backbone model under the `pretrained_models/` folder
* For data preparation, please refer to [this](https://github.com/botaoye/OSTrack/tree/main)
* Uncomment lines `63, 67, and 71` in the [base_backbone.py](https://github.com/goutamyg/SMAT/blob/main/lib/models/mobilevit_track/base_backbone.py) file. 
Long story short: The code is opitmized for high inference speed, hence some intermediate feature-maps are pre-computed during testing. However, these pre-computations are not feasible during training. 
* Run
```
python tracking/train.py --script mobilevitv2_track --config mobilevitv2_256_128x1_ep300 --save_dir ./output --mode single
```
* The training logs will be saved under `output/logs/` folder

## Pretrained tracker model
The pretrained tracker model can be found [here](https://drive.google.com/drive/folders/1TindIEwu82IvtozwL4XQFrSnFE2Z6W4y)

## Tracker Evaluation

* Update the test dataset paths in `lib/test/evaluation/local.py`
* Place the [pretrained tracker model](https://drive.google.com/drive/folders/1TindIEwu82IvtozwL4XQFrSnFE2Z6W4y) under `output/checkpoints/` folder 
* Run
```
python tracking/test.py --tracker_name mobilevitv2_track --tracker_param mobilevitv2_256_128x1_ep300 --dataset got10k_test or trackingnet or lasot
```
* Change the `DEVICE` variable between `cuda` and `cpu` in the `--tracker_param` file for GPU and CPU-based inference, respectively  
* The raw results will be stored under `output/test/` folder

## Tracker demo
To evaluate the tracker on a sample video, run
```
python tracking/video_demo.py --tracker_name mobilevitv2_track --tracker_param mobilevitv2_256_128x1_ep300 --videofile *path-to-video-file* --optional_box *bounding-box-annotation*
```


## Acknowledgements
* We use the Separable Self-Attention Transformer implementation and the pretrained `MobileViTv2` backbone from [ml-cvnets](https://github.com/apple/ml-cvnets). Thank you!
* Our training code is built upon [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking)
* To generate the evaluation metrics for different datasets (except, server-based GOT-10k and TrackingNet), we use the [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit)
* Thanks to the Authors of SMAT. The citation of their paper is found below!
## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{gopal2024separable,
  title={Separable self and mixed attention transformers for efficient object tracking},
  author={Gopal, Goutam Yelluru and Amer, Maria A},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6708--6717},
  year={2024}
}
```
