[中文|[English](README_en.md)]

# Tiger
A **Tig**ht-fisted Optimiz**er**， 一个“抠”到极致的优化器！

## 简介

Tiger、Lion和AdamW的对比：

<img src="https://raw.githubusercontent.com/bojone/tiger/main/Tiger-Lion-AdamW.png" width=100%>

从[Lion](https://kexue.fm/archives/9473)的视角，Tiger是一个简化版的Lion（beta1=beta2）；从[SignSGD](https://arxiv.org/abs/1802.04434)的视角，Tiger是一个带有动量和weight decay的SignSGD。

Tiger只用到了动量来构建更新量，根据[《隐藏在动量中的梯度累积：少更新几步，效果反而更好？》](https://kexue.fm/archives/8634)的结论，此时我们不新增一组参数来“无感”地实现梯度累积！这也意味着在我们有梯度累积需求时，Tiger已经达到了显存占用的最优解，这也是“Tiger”这个名字的来源（**Tig**ht-fisted Optimiz**er**，抠门的优化器，不舍得多花一点显存）。

- 博客1：https://kexue.fm/archives/9512
- 博客2：https://kexue.fm/archives/8634

## 特性

- 不逊色于[AdamW](https://arxiv.org/abs/1711.05101)和[LAMB](https://arxiv.org/abs/1904.00962)的效果；
- 梯度累积下的显存需求最小化；
- 类似[LAMB](https://arxiv.org/abs/1904.00962)的分参数学习率自适应；
- 简单的预防模型崩溃到NaN的策略；
- 可以模拟任意schedule的分段现行学习率。

## 使用

目前的实现在tensorflow 1.15下开发，估计tensorflow 2.x的前几个版本也能用，后面的版本由于没有使用经验，不确定能否可用。

参考代码：

```python
from tiger import Tiger

optimizer = Tiger(
    learning_rate=1.76e-3,  # 全局相对学习率lr
    beta=0.965,  # beta参数
    weight_decay=0.01,  # 权重衰减率
    grad_accum_steps=4,  # 梯度累积步数
    lr_schedule={
        40000: 1,  # 前40k步，学习率从0线性增加到1*lr【即warmup步数为40000/4步】
        160000: 0.5,  # 40k-160k步，学习率从1*lr线性降低到0.5*lr
        640000: 0.1,  # 160k-640k步，学习率从0.5*lr线性降低到0.1*lr
        1280000: 0.01,  # 640k-1280k步，学习率从0.1*lr线性降低到0.01*lr，并在之后保持不变
    }
)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

## 鸣谢

特别感谢Lion的充分实验，以及Lion作者们的友好交流。

## 引用

```
@misc{tigeropt,
  title={Tiger: A Tight-fisted Optimizer},
  author={Jianlin Su},
  year={2023},
  howpublished={\url{https://github.com/bojone/tiger}},
}
```

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
