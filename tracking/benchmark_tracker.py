import os, sys, torch, argparse, time, psutil
import numpy as np
import ptflops, onnx
import onnxruntime as ort

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation.tracker import Tracker

def setup_onnx_gpu(model_path, inp, out):
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)
    
    session = ort.InferenceSession(model_path, sess_options, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": True})], verbose=True)
    ro = ort.RunOptions()
    ro.add_run_config_entry("gpu_graph_id", "1")
    io_binding = session.io_binding()

    x_ortvalue = ort.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
    y_ortvalue = ort.OrtValue.ortvalue_from_numpy(out, 'cuda', 0)
    io_binding.bind_ortvalue_output('output', y_ortvalue)
    io_binding.bind_ortvalue_input('input', x_ortvalue)
    
    return io_binding, ro, y_ortvalue, session
    
    session.run_with_iobinding(io_binding, ro)
    ort_outs = y_ortvalue.numpy()

def testrun_it(model, image_sizes=(384,256), iterations=4000, batch_size=1, cpu=False, optit=False, onnx_bool=False, args=None):
    # device = "cpu" if cpu else "cuda:0"
    inp = torch.randn(batch_size, 3, image_sizes[0], image_sizes[1]).cuda()
    
    # Transform Model
    if optit:
        model.eval()
        model = torch.jit.script(model)#, example_inputs=[inp])    
        model = torch.jit.optimize_for_inference(model)
    if onnx_bool:
        model_path = os.path.join("tracking","onnx_models","temp"+".onnx")
        out_dict = {'pred_boxes': torch.Size([2, 1, 4]), 'score_map': torch.Size([2, 1, 16, 16]), 'size_map': torch.Size([2, 2, 16, 16]), 'offset_map': torch.Size([2, 2, 16, 16]), 'max_score': torch.Size([2, 1])}#, 'grid_map': None}
            
        if not os.path.exists(model_path):
            out_names = list(out_dict.keys())
            dyn_axes = {name:{0:"batch_size"} for name in out_names}
            dyn_axes["input"] =  {0: "batch_size"}
            torch.onnx.export(model, inp, model_path, do_constant_folding=True,opset_version=16, input_names=["input"], output_names=out_names, dynamic_axes=dyn_axes)
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
    if onnx_bool:
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
                print(ort_outs.shape)    
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
    
    def __init__(self, net, args):
        super().__init__()
        self.net = net

        
        z = self.net.backbone.conv_1.forward(torch.randn(1,3,128,128).cuda())

        # layer_1 (i.e., MobileNetV2 block) output
        z = self.net.backbone.layer_1.forward(z)

        # layer_2 (i.e., MobileNetV2 with down-sampling + 2 x MobileNetV2) output
        z = self.net.backbone.layer_2.forward(z)
        self.z = z
        
    def forward(self, x):
        search_ind = int((x.shape[2]/3)*2)
        # templ = int(x.shape[2]/3)
        return self.net(search=x[:,:,:search_ind,:], template=self.z)


# lowformer_256_128x1_ep300_lasot_coco_b3_lffv3_convhead # 15
# mobilevitv2_256_128x1_ep300_mine
def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    # parser.add_argument('--tracker_name', type=str, default='mobilevitv2_track', help='Name of tracking method.')
    # parser.add_argument("--force_eval", action="store_true", default=False)
    # parser.add_argument('--ckpos', type=int, default=-1)
    parser.add_argument('--config', type=str, default='lowformer_256_128x1_ep300_lasot_coco_got_b15_lffv3_convhead', help='Name of tracking method.')
    parser.add_argument('--gpu', type=int, default=0)
    
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    
    
    tracker = init_model(args)
    if not "lowformer" in args.config:
        tracker = TrackerWrapper(tracker, args)
    short_test(tracker)
    
    testrun_it(tracker)
    
    

if __name__ == '__main__':
    main()
