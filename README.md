# ModelInversionAttackBox
A toolbox for model inversion attacks.




# Git 规范

+ 禁止在main分支开发，请另开分支，确保能跑通后开pull request，再使用squash merge到main分支，建议由另一个人review代码
+ 不要上传数据集和预训练文件等大文件
+ commit规范：https://www.conventionalcommits.org/en/v1.0.0/
  + 常用：feat, fix, refactor
+ 分支命名
  + 开发分支dev-xxx
  + debug分支fix-xxx



# 文件夹使用

## attack、defense

存放各类攻击、防御方法

使用参数用@dataclass表示，参照[llama/llama/model.py at main · facebookresearch/llama (github.com)](https://github.com/facebookresearch/llama/blob/main/llama/model.py)，充分使用ide自动补全和检查功能



## models

存放各类target、eval的模型

为统一使用，把模型输出使用如下类进行封装，如下，在`models/modelresult.py`

```python
@dataclass
class ModelResult:
    result: torch.Tensor
    feat: list[torch.Tensor]
```



## <work_dir>

由用户指定，中间产物生成位置，开发时默认为`./cache`

每种攻击/防御方法生成位置`<work_dir>/<method>/`



## <result_dir>

由用户指定，结果输出位置，开发时默认为`./result`

每种攻击/防御方法生成位置`<result_dir>/<method>/`

## dev_scripts

存放用于开发测试的脚本，开发时常量写在`development_config.py`中