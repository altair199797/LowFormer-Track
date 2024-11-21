import os, sys, torch, torchvision





def main():
    model = torchvision.ops.FeaturePyramidNetwork([160, 320], 160)
    print(model)
    inp = {"feat0": torch.randn(10,160,16,16), "feat1":torch.randn(10,320,8,8)}

    out = model(inp)
    print([(k, v.shape) for k, v in out.items()])
    

def main2():
    path = "moritz@158.110.40.61:/home/moritz/Research/efficientvit-master/requirements.txt"
    with open(path, "r") as read_file:
        lines = read_file.readlines()
    print(lines)

if __name__ == "__main__":
    main2()
    