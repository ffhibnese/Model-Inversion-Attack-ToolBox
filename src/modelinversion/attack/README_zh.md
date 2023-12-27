# 攻击流程


攻击主要使用两个类，`AttackConfig` 与 `Attacker`。

## 攻击参数 `AttackConfig`

该类必须继承`./base.py`中的`BaseAttackConfig`，用于攻击算法使用的参数。`BaseAttackConfig`已给出各算法之间的通用参数，各算法对应的参数只需要添加特定的参数。注意：
1. 由于dataclass的限制，继承后的各算法参数必须要有默认值
2. 根据dataclass的特性，如果希望用户能够在创建的时候指定改参数，请添加该参数的类型注释；否则，请勿添加类型注释

## 攻击模块 `Attacker`

该类必须继承`./base.py`中的`BaseAttacker`，用于实现完整的攻击与评估流程。

初始化接收参数为`AttackConfig`，完成folder_manager的创建以及准备target / eval模型.
`attack`函数接收`batch_size`和`target_labels`两个参数，是攻击的主函数，攻击过程如下：
+ 调用`prepare_attack_models`方法，准备攻击所需要的模型
+ 按照攻击的目标类以及批次大小，划分每次攻击所需要的目标类，调用`attack_step`函数完成攻击
+ 攻击结果信息
`evaluation`函数为评估的主函数

各算法必须实现的抽象方法如下：

1. 返回区分该方法不同参数的字符串，中间结果以及最终结果保存在`<cache_dir>/<tag>/`和`<result_dir>/<tag>/`中。
```python
def get_tag(self) -> str:
    pass
```

2. 准备攻击所用的模型。
```python
def prepare_attack_models(self):
    pass
```

3. 攻击过程。iden为攻击的目标类id，按batch_size划分好。
```python
def attack_step(self, iden):
    pass
```


