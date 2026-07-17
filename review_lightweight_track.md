
# Title
The proposed approach closely resembles the SMAT single-object tracking framework, and offers limited originality or technical contribution beyond that baseline.

# Paper Summary
## Method
The paper introduces an efficient RGB-T tracker based on the SMAT framework, which performs single-object tracking on multi-modal videos (RGB and infrared (IR)). Their approach is best characterized by the fact that the model merges tokens of the different modalities in a portion of the attention layers. They further introduce a fusion equation that combines the feature maps of both modalities channelwise with learned weights.
## Results
The paper evaluates the methods on common RGB-T (thermal) single-object tracking benchmarks. The proposed model is in most part not able to improve upon the speed-accuracy trade-off of the previous state-of-the-art approach. 


# Strengths
* The related work is well structured and formulated. The paper compares  its architecture with recent state-of-the-art models for multi-modal tracking and highlights the differences.
* The evaluation procedure is sound and the datasets/benchmarks with their accompanied metrics are well explained.
* The language level is very high and the approach is generally well explained.
* The tables offer good insight and are well formatted.


# Weaknesses

## Major
* The paper clearly uses the SMAT [1] framework for single object tracking as a baseline and consistently sells the contributions of SMAT as their own (see minor weaknesses). The paper's approach only slightly differs from SMAT and does not highlight the similarities in any way. The only similarity mentioned is that the paper utilizes the SMAT prediction head. 
* The architecture is explained in great detail, though it mostly describes the architecture of SMAT [1] and MobileViTv2 [2]. This e.g. occurs in the following lines: 131-137 (describes MobileVitv2), 154-167 (basically just describes that SMAT is executed for RGB and IR), 167-205 (same as before), 205-217 (only differing in the fact, that the paper concatenates the tokens of IR and RGB in the attention layer), 220-225 (executing SMAT on each modality, otherwise exactly the same), 245-248 (describing prediction head of SMAT), 262-271 (describing SMAT architecture and setup)

--> In summary, most of the method in the paper is a description of the SMAT architecture and its MobileViTv2 backbone.
* The fusion in equation (4) only relies on fixed learned channelwise fusion weights, which are independent of the input. It lacks any sophistication and could probably be replaced by an addition, yielding similar results. This is the case, since the feature maps are anyway processed by channelwise weights right before the final fusion in equation (4). If it did not include a sigmoid actication, the channelwise combination could be omitted, leading to the exact same result. It is questionable if the sigmoid activation yields any benefit. 
Additionally, since the combination weights are not forced to sum to 1 channelwise, it could lead to overfitting/instability.
Since this represents the main novel contribution of the paper, an ablation study about it would have been crucial.
* The paper's result in table 1 are not convincing. The proposed approach fares considerably worse compared to SUTrack_Tiny [3], while only achieving 20% more frames per second (fps). The paper additionally argues that it is more parameter-efficient than SU_Track for example, however the SU_track model is also able to predict on several other modalities, making it a more general model which explains the increased parameter count. In my view a lower parameter count alone is not an convincing argument anyway, especially as none of the contributions of the paper are responsible for that and it solely is due to the efficiency of the SMAT framework, which the paper is based on.
* The low resolution of the figures significantly impacts readability, making the text within them difficult to discern.




## Minor
* Abstract: When the backbone model is mentioned in lines 17/18, it would be better to be specific. The paper could directly call it MobileViTv2 or don't mention it at all in the abstract.
* Abstract: Simply using an efficient backbone (Mobile Vision Transformer) is in itself not a contribution (lines 23-25). 
* In lines 49-51: "Our backbone follows a ...". Since the paper just leverages MobileVitv2 off-the-shelf, it is not fitting to call it "our".
* In lines 51-53: "In addition, we introduce a progressive interaction design that jointly models template and search features through separable attention..." This approach was proposed by the SMAT [1] framework and is not introduced by this paper. 
* In lines 118-121: "Our model extends the MobileViTv2...". The paper extends SMAT and not only MobileVitv2. Its approach is not inspired by SMAT, but uses SMAT with few adaptations.
* In lines 129-131: "...followed by a point-wise convolution, which increases the channel dimension while reducing spatial resolution." Does the pointwise convolution really reduce the resolution? Probably the depthwise convolution is strided, and the pointwise convolution has stride=1.
* The ablation study is comparably shallow. Pure RGB models faring worse than IR models is obvious and not a novel insight, as it was studied before by compared approaches. On the other side further ablations would have been interesting, not just removing whole layers. Possible ablations: always applying cross-modal fusion in attention, never applying cross-modal fusion in attention (but keeping layers in contrast to what the paper does), alternatives for equation (4).   
* The paper lacks any name for the proposed model.


[1] Gopal, Goutam Yelluru, and Maria A. Amer. "Separable self and mixed attention transformers for efficient object tracking." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2024.

[2] Mehta, Sachin, and Mohammad Rastegari. "Separable self-attention for mobile vision transformers." arXiv preprint arXiv:2206.02680 (2022).

[3] Chen, Xin, et al. "Sutrack: Towards simple and unified single object tracking." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 2. 2025.

# Preliminary Recommendation
Reject


# Justification of Preliminary Review
The contribution of the paper is very shallow and is not clearly highlighted. Most of the method section is a description of the SMAT single object tracking framework. The only contributions are the following: fusion in equation (4), that in layer_4 the tokens are concatenated for the attention operation and the "cross-modal feature fusion" which is just the repetition of layer_4. Additionally, the fusion in equation (4) does probably not yield any benefit and could be replaced by an addition. Further the concatenation of tokens in layer_4 and the "cross-modal feature fusion" just extends the idea of SMAT who proposed the Mixed-Attention which concatenates template and search tokens.
Finally, the speed-accuracy trade-off of the proposed model does not exceed the state-of-the-art approaches.




# Confidence Level
4