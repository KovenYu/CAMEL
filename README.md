This page contains the demo code for our model CAMEL and the package to construct the dataset ExMarket. If you have any problem, please feel free to contact us. My Email address: xKoven@gmail.com

H.-X. Yu, A. Wu, W.-S. Zheng, "[Cross-view Asymmetric Metric Learning for Unsupervised Person Re-identification](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yu_Cross-View_Asymmetric_Metric_ICCV_2017_paper.pdf)", In ICCV, 2017.

# Results on large popular datasets

Dataset| Rank-1| Rank-5| Rank-10| MAP
-|-|-|-|-
Market-1501| 54.45| 73.10| 79.69| 26.31
DukeMTMC-reID| 40.26| 57.59| 64.09| 19.81

# CAMEL

In the folder ./CAMEL is the DEMO code on the Market-1501 dataset.
Please see main.m for details.

Also note that a different MATLAB version may lead to a result that is a little bit different from the result reported in the paper, because of several random procedures in the algorithm and during the testing. The reported result (in this demo, 54.5% rank-1 accuracy for Market-1501) was obtained using MATLAB R2014a.

We also prepared a supervised version of CAMEL in main_supervised.m,
which runs much faster than CAMEL and can be a weak baseline in comparison
with supervised models.

# Feature

In the folder ./Feature is the code for feature extraction.
Here we provide the pre-trained JSTL model (without fine-tune or domain guided dropout) to extract features for different datasets apart from the test ones.
Note that the model was pre-trained using the full training set [4], i.e., VIPeR, CUHK01, CUHK03, PRID, 3DPeS, i-LIDS and Shinpuhkan.

Our implementation is based on matconvnet: https://github.com/vlfeat/matconvnet

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
The example feature ''ExMarket_JSTL64.mat'' was extracted using the JSTL model provided in ./Feature [4]. 
If you use the feature or the model, please also kindly cite [4].

# Reference

[1] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian. Scalable person re-identification: A benchmark. In ICCV, 2015.

[2]  L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, and Q. Tian. Mars: A video benchmark for large-scale person re-identification. In ECCV, 2016.

[3] H.-X. Yu, A. Wu and W.-S. Zheng, "Cross-view Asymmetric Metric Learning for Unsupervised Person Re-identification", In ICCV, 2017.

[4] T. Xiao, H. Li, W. Ouyang, and X. Wang. Learning deep feature representations with domain guided dropout for person re-identification. In CVPR, 2016.
