import os, sys, torch, torchvision





def main():
    model = torchvision.ops.FeaturePyramidNetwork([160, 320], 160)
    print(model)
    inp = {"feat0": torch.randn(10,160,16,16), "feat1":torch.randn(10,320,8,8)}

    out = model(inp)
    print([(k, v.shape) for k, v in out.items()])
    
    
if __name__ == "__main__":
    main()
    