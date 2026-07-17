import os, sys, subprocess


datasets = ["LaSOT","avist", "otb", "trek150","nfs","got10k","uav"]
# datasets = [datasets[-1]]
# datasets = datasets[5:6]
models = ["mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit_e2","mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit_b15", "mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit", "mobilevitv2_256_128x1_ep300_lasot_got_coco"]
models = models[0:1]
datasets = datasets[4:5]

def eval_all():
    # python tracking/test.py --tracker_name lowformer_track --tracker_param lowformer_256_128x1_ep300_lasot_b15 --dataset LaSOT --force_eval --threads 5
    for dataset in datasets:
        for model in models:
            basecmd =  "python tracking/test.py  --tracker_name mobilevitv2_track  --dataset " 
            basecmd += dataset +("_val" if "got10k" in dataset else "") + "  "
            if not model ==  "mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit_b15":
                basecmd += " --runid 5 "
            basecmd +=  " --tracker_param " + model
            basecmd += " --threads 1 "
            
            os.system(basecmd)


def analyze_all():
    for dataset in datasets:
        for model in models:
            basecmd =  "python tracking/analysis_results.py --dataset " 
            basecmd += dataset + "  "
            if not model ==  "mobilevitv2_256_128x1_ep300_lasot_got_coco_lowformit_b15":
                basecmd += " --runid 5 "
            basecmd +=  "--config " + model
            os.system(basecmd)  
            # try:
            #     proc = subprocess.Popen([basecmd], stdout=subprocess.PIPE, shell=True)
            #     (out, err) = proc.communicate()
            # except:
            #     print("dataset",dataset,"failed with model",model)
            
            # if not isinstance(out,str):
            #     out = out.decode("utf-8")

            # print("OUT:",out)
            # out = [i for i in out.split("\n") if len(i)]
            # for i in out:
            #     print(i)
            

def main():
    eval_all()
    analyze_all()

if __name__ == "__main__":
    main()
