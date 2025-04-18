import os, sys, torch, argparse, time, psutil
import numpy as np
import ptflops, onnx
import onnxruntime as ort

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
    
# from lib.models.mobilevit_track.lowformer_track import show_params_flops

from lib.test.evaluation.tracker import Tracker

def setup_onnx_gpu(model_path, inp, out):
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
    
    session = ort.InferenceSession(model_path, sess_options, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": True})], verbose=True) # ,"CPUExecutionProvider"
    ro = ort.RunOptions()
    ro.add_run_config_entry("gpu_graph_id", "1")
    io_binding = session.io_binding()
    # print(inp)
    # print(out)
    x_ortvalue = ort.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
    
    y_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,1,4), dtype=np.float32), 'cuda', 0)
    io_binding.bind_ortvalue_output('pred_boxes', y_ortvalue)
    
    io_binding.bind_ortvalue_input('input', x_ortvalue)
    
    return io_binding, ro, y_ortvalue, session
    
    session.run_with_iobinding(io_binding, ro)
    ort_outs = y_ortvalue.numpy()

def testrun_it(model, image_sizes=(384,256), iterations=4000, batch_size=1, cpu=False,  args=None):
    # device = "cpu" if cpu else "cuda:0"
    inp = torch.randn(batch_size, 3, image_sizes[0], image_sizes[1]).cuda()
    model.eval()
    # Transform Model
    if args.optit:
        model.eval()
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
    if args.onnx:
        model_path = os.path.join("tracking","onnx_models",args.config+".onnx")
        # model_path = os.path.join("tracking","onnx_models","temp"+".onnx")
        downsize = int(image_sizes[-1]/16)
        # {'pred_boxes': torch.Size([1, 1, 4]), 'score_map': torch.Size([1, 1, 14, 14]), 'size_map': torch.Size([1, 2, 14, 14]), 'offset_map': torch.Size([1, 2, 14, 14])}  
        out_dict = {'pred_boxes': torch.Size([1, 1, 4]), 'score_map': torch.Size([1, 1, downsize, downsize]), 'size_map': torch.Size([1, 2, downsize, downsize]), 'offset_map': torch.Size([1, 2, downsize, downsize])}#, 'grid_map': None}
            
        if not os.path.exists(model_path):
            out_names = list(out_dict.keys())
            dyn_axes = {name:{0:"batch_size"} for name in out_names}
            dyn_axes["input"] =  {0: "batch_size"}
            os.makedirs(os.path.join("tracking","onnx_models"), exist_ok=True)
            torch.onnx.export(model, inp.detach(), model_path, do_constant_folding=True, opset_version=13, input_names=["input"], output_names=out_names, dynamic_axes=dyn_axes)
            
            onnx_model =  onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
        if True:
            from onnxsim import simplify as simplify_func
            onnx_model = onnx.load_model(model_path)
            onnx_model, success = simplify_func(onnx_model)
            assert success 
            onnx.save(onnx_model, model_path)
        
    
    ## Run
    timings = []
    if args.onnx:
        inp = inp.cpu().numpy().astype(np.float32)
        if cpu:
            ort_session = ort.InferenceSession(model_path)
            
            for i in range(10):
                outputs = ort_session.run(None,{"input": inp},)
            for i in range(iterations):
                start_time = time.time()
                outputs = ort_session.run(None,{"input": inp},)
                timings.append(1000*(time.time()-start_time)/inp.shape[0])
        else:
            io_binding, ro, y_ortvalue, session = setup_onnx_gpu(model_path, inp, out_dict)
            # session.run_with_iobinding(io_binding, ro)
            # ort_outs = y_ortvalue.numpy()    
            
            for i in range(5):
                session.run_with_iobinding(io_binding, ro)
            
            # print(inp.shape)
            for i in range(iterations):
                start_time = time.time()
                session.run_with_iobinding(io_binding, ro)
                timings.append(time.time() - start_time)
                ort_outs = y_ortvalue.numpy()
                print(ort_outs)    
    else:
        with torch.inference_mode():
            model.eval()
            model.cuda()
            # warmup
            for i in range(5):
                out = model(inp)
            
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # print(inp.shape)
            for i in range(iterations):
                if cpu:
                    start_time = time.time()
                else:
                    starter.record()
                out = model(inp)
                if cpu:
                    timings.append(time.time() - start_time)
                else:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender)/inp.shape[0])
                # print(timings[-1])
        
    timings = np.array(timings)
    print("tracker time:", timings[-1],"| median:",np.median(timings), "mean:", np.mean(timings))
    
    return np.median(timings)


def init_model(args):
    tracker_name = args.config.split("_")[0] +"_track"
    tracker_param = args.config
    dataset_name = "lasot"
    run_id = None
    tracker_fw = Tracker(tracker_name, tracker_param, dataset_name, run_id, testit=True, args=None)
    tracker = tracker_fw.get_tracker().network
    return tracker 

def short_test(tracker):
    inp = torch.randn(1,3,384,256).cuda()
    out = tracker(inp)
    print({key:(value.shape if not value is None else None) for key,value in out.items()})
    # print(out["max_score"])

class TrackerWrapper(torch.nn.Module):
    
    def __init__(self, net, args, nobb=False, base_size=128):
        super().__init__()
        self.net = net
        self.nobb = nobb
        
        z = self.net.backbone.conv_1.forward(torch.randn(1,3,base_size,base_size).cuda())

        # layer_1 (i.e., MobileNetV2 block) output
        z = self.net.backbone.layer_1.forward(z)

        # layer_2 (i.e., MobileNetV2 with down-sampling + 2 x MobileNetV2) output
        z = self.net.backbone.layer_2.forward(z)
        self.z = z.detach()
        
        if self.nobb:
            x = torch.randn(1,3,base_size*2,base_size*2).cuda()
            self.x, self.z = self.net.backbone(x=x, z=self.z)
        
        if not "lowformit" in args.config:
            net.box_head.enable_coreml_compatible_fn = True
            net.box_head.register_buffer(
                name="unfolding_weights",
                tensor=net.box_head._compute_unfolding_weights().cuda(),
                persistent=False,
            )
        
    def forward(self, x):
        search_ind = int((x.shape[2]/3)*2)
        # templ = int(x.shape[2]/3)
        if self.nobb:
            x,z = self.net.neck(self.x, self.z)
            feat_fused = self.net.feature_fusor(z,x)
            out = self.net.forward_head(feat_fused, None)
            return out
        # print(x[:,:,:search_ind,:].shape, self.z.shape)
        out = self.net(search=x[:,:,:search_ind,:], template=self.z)
        
        # {'pred_boxes': torch.Size([1, 1, 4]), 'score_map': torch.Size([1, 1, 14, 14]), 'size_map': torch.Size([1, 2, 14, 14]), 'offset_map': torch.Size([1, 2, 14, 14])}  
        # print({key:out[key].shape for key in out})
        return out
      

def show_params_flops(model, tmpsize, searsize):
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        inp_size = (3, tmpsize+searsize, searsize)
        print("Testing for total size of ",inp_size)
        macs, params = get_model_complexity_info(model, inp_size, as_strings=False,
                                    print_per_layer_stat=False, verbose=False)
        print("MMACS: %d  |  PARAMETERS (M): %.2f" % (macs/1_000_000, params/1_000_000))

# lowformer_256_128x1_ep300_lasot_coco_b3_lffv3_convhead # 15
# mobilevitv2_256_128x1_ep300_mine
def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    # parser.add_argument('--tracker_name', type=str, default='mobilevitv2_track', help='Name of tracking method.')
    # parser.add_argument("--force_eval", action="store_true", default=False)
    # parser.add_argument('--ckpos', type=int, default=-1)
    parser.add_argument('--config', type=str, default='lowformer_256_128x1_ep300_lasot_coco_got_b15_lffv3_convhead', help='Name of tracking method.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--optit", action="store_true", default=False)
    parser.add_argument("--onnx", action="store_true", default=False)
    parser.add_argument('--basesize', type=int, default=128)
    
    args = parser.parse_args()
    
    base_size = args.basesize
        
    torch.cuda.set_device(args.gpu)
    
    
    tracker = init_model(args)
    if not "lowformer" in args.config:
        tracker = TrackerWrapper(tracker, args, base_size=base_size)
    # short_test(tracker)
    show_params_flops(tracker, base_size, base_size*2)
    
    with torch.no_grad():
        testrun_it(tracker, image_sizes=(base_size*3, base_size*2),args=args)
    
    

if __name__ == '__main__':
    main()
