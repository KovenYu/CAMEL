This is a project page associating to our work on unsupervised metric learning for Re-ID, associating to the paper

https://arxiv.org/abs/1708.08062

H.-X. Yu, A. Wu, W.-S. Zheng, "Cross-view Asymmetric Metric Learning for Unsupervised Person Re-identification", In ICCV, 2017.

This page contains the demo code for our model CAMEL and the package to construct the dataset ExMarket.

# Feature

In the folder ./Feature is the code for feature extraction.
Here we provide the pre-trained JSTL model (without fine-tune or domain guided dropout) to extract features for different datasets apart from the test ones.
Note that the model was pre-trained using the full training set [4], i.e., VIPeR, CUHK01, CUHK03, PRID, 3DPeS, i-LIDS and Shinpuhkan.

Our implementation is based on matconvnet: https://github.com/vlfeat/matconvnet

# CAMEL

In the folder ./CAMEL is the DEMO code for our linear metric learning model CAMEL.
We provide an interface of CAMEL here and the full code is coming soon.
One can easily use it following main.m.
See main.m for details.

We also prepared a supervised version of CAMEL in main_supervised.m,
which runs much faster than CAMEL and can be a weak baseline in comparison
with supervised models.

# ExMarket

In the folder ./ExMarket is the package which contains the MATLAB code for constructing the ExMarket Dataset and evaluation.
To construct the ExMarket dataset, please follow the steps below:

1. Download the Market-1501 dataset [1] from
http://www.liangzheng.org/Project/project_reid.html

2. Download the MARS dataset [2] from
http://www.liangzheng.org/Project/project_mars.html

3. Unzip them to the same directory with IMDBmaking.m, and run IMDBmaking. 

If you use the dataset, please kindly cite [1] and [2] and our paper [3].

We also provide a demo for evaluation in main.m. 
The example feature ``ExMarket_JSTL64.mat'' was extracted using the JSTL model provided in ./Feature [4]. 
If you use the feature or the model, please also kindly cite [4].

# Reference

[1] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian. Scalable person re-identification: A benchmark. In ICCV, 2015.

[2]  L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, and Q. Tian. Mars: A video benchmark for large-scale person re-identification. In ECCV, 2016.

[3] H.-X. Yu, A. Wu and W.-S. Zheng, "Cross-view Asymmetric Metric Learning for Unsupervised Person Re-identification", In ICCV, 2017.

[4] T. Xiao, H. Li, W. Ouyang, and X. Wang. Learning deep feature representations with domain guided dropout for person re-identification. In CVPR, 2016.
