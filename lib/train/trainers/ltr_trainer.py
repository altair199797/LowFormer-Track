import os
import datetime
from collections import OrderedDict
import shutil

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size

# related to evaluating got10k_val dataset
from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker_got10kval import Tracker
from lib.train.eval_ao.dataset import DatasetBuilder

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # related to computing aor and sr for got10_val datasets
        dataset_name = 'got10k'
        anno_path = os.path.join(self.settings.env.workspace_dir, 'got10k_val_anno')
        self.dataset_builder_obj = DatasetBuilder(dataset_name, anno_path)

        self.count = 0

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        # try:
        #     self.actor.net.backbone.training = loader.training
        # except:
        #     print("didn't find it!")
        # print("!!!!!!!!!!!!!")
        # print(loader.training)
        # print(self.actor.net.backbone.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            
            
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # print("Before",loader.training, i, len(loader), i==len(loader))
            # compute the AOR and SR values for got10k-val dataset (currently supported under single-GPU training only)
            if loader.training is False and i == len(loader) \
                    and self.settings.local_rank == -1 and self.epoch % 5 == 0 :

                # 0. Save the model to a temporary folder
                tmp_folder = os.path.join('tmp_folder', self.settings.cfg_file.split('/')[-1].split('.')[0])
                tmp_folder_path = os.path.join(self._checkpoint_dir.split('checkpoints')[0], tmp_folder)
                if os.path.isdir(tmp_folder_path) is False:
                    os.makedirs(tmp_folder_path)
                state = {'net_type': type(self.actor.net).__name__,
                         'net': self.actor.net.state_dict()}

                tmp_model_path = '{}/{}.tmp'.format(tmp_folder_path, type(self.actor.net).__name__)
                torch.save(state, tmp_model_path)
                file_path = '{}/{}.pth.tar'.format(tmp_folder_path, type(self.actor.net).__name__)
                os.rename(tmp_model_path, file_path)

                # 1. Initialize the got10k_val dataset
                dataset = get_dataset('got10k_val')

                # 2. Initialize the tracker
                trackers = [Tracker(name=self.settings.config_name.split('_')[0] + '_track',
                                    parameter_name=self.settings.config_name,
                                    dataset_name='got10k_val',
                                    run_id=None)]
                
                # 3. Evaluate the model on got10k_val dataset. Erase if got10k_val directory already exists from the
                # previous iteration
                if os.path.exists(os.path.join(tmp_folder_path, 'got10k')):
                    shutil.rmtree(os.path.join(tmp_folder_path, 'got10k'))
                run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=1)

                # 4. Compute AOR, SR values
                self.dataset_builder_obj.compute_test_results(tmp_folder_path)
                ao_, sr_0p50_ = self.dataset_builder_obj.summarize_tracker_results()
                # print("val out:",ao_, sr_0p50_)
                # 5. Write the results to tensorboard
                stats['ValMetric/AOR'] = ao_
                stats['ValMetric/SR_0.5'] = sr_0p50_
                torch.cuda.synchronize()
            else:
                pass
                # stats.pop('ValMetric/AOR')
                # stats.pop('ValMetric/SR_0.5')
                

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)
            # if loader.training:
            #     break
            

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self):
        print("New Epoch started!")
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
