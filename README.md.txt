数据集文件：
XXX_train_data.npy：训练数据集
XXX_train_label.npy：训练标签
XXX_test_data.npy：测试数据集
XXX_test_label.npy：测试标签

使用包的源代码：
evalGP_main：IDGP种群初始化、种群更新操作
feature_extractors：特征提取函数
feature_function：IDGP图像相关算子函数
gp_restrict：强GP函数，约束每个结点的输出输出类型
sift_feature：SIFT函数
algo_iegp：PUIDGP种群初始化、种群更新操作
felgp_functions：PUIDGP图像相关算子函数

主要源代码：
IDGP_main：IDGP源代码
pca_IDGP：在数据训练过程中加入PCA的IDGP源代码
FELGP_main：New_IDGP源代码
FELGP_unique：在New_IDGP中加入Unique_Pop的源代码
PCA_FELGP：PUIDGP源代码
