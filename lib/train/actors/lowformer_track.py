from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class LowFormerTrackActor(BaseActor):
    """ Actor for training MobileViT-Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['template_images']) == 1
        # assert len(data['search_images']) == 1

        # print(data["template_images"][0].shape) # torch.Size([128, 3, 128, 128])
        # print(data["search_images"][0].shape) # torch.Size([128, 3, 256, 256])
        
        
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)
            

        # search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        # print(search_img.shape, data['search_images'][0].shape)
        # print((template_list[0] - template_list[1]).abs().mean())
        
        # assert False
        
        assert len(template_list) == 2
        merged_image = torch.cat([data['search_images'][0],torch.cat(template_list,dim=-1)],dim=-2)
        
        
        out_dict = self.net(merged_image=merged_image)
        # print(self.net)
        # assert False
        
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        # print(pred_boxes.shape)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        # print(pred_boxes_vec.shape, gt_boxes_vec.shape)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute location loss
        if 'score_map' in pred_dict and not pred_dict["score_map"] is None :
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        if "grid_map" in pred_dict and not pred_dict["grid_map"] is None:
            loss_fn = torch.nn.CrossEntropyLoss()
            grid_cells_n = self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
            temp_indices = ((gt_boxes_vec[:,1]*grid_cells_n).int()*grid_cells_n + (gt_boxes_vec[:,0]*grid_cells_n).int()).long()
            gt_map = torch.nn.functional.one_hot(temp_indices, num_classes=grid_cells_n*grid_cells_n).float()
            # print( temp_indices,  gt_bbox, gt_map.shape, pred_dict["grid_map"].shape)
            # print(pred_dict["grid_map"])
            # print("IAM HERE")
            location_loss = loss_fn(pred_dict["grid_map"],gt_map) * 0.1
        
        
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss