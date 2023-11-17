# 添加目标/评估模型步骤

1. 继承在`base.py`中的`BaseTargetModel`类，并实现`get_feature_dim`方法，该方法返回最后一个特征（最后一个线性层的输入）的维度
2. 在`forward`函数的开头调整输入图片大小
3. 使用`modelresult.py`中的`ModelResult`类封装输出
    + result: 模型的输出
    + feat: forward函数生成的中间特征列表。最后一个线性层的输入必须在该列表中
    + addition_info: 其它信息的字典，可选。
4. 把该模型添加到`get_models.py`的`get_model`函数