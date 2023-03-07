[中文|[English](README_en.md)]

# Tiger
A **Tig**ht-fisted Optimiz**er**， 一个“抠”到极致的优化器！

## 简介

<img src="https://raw.githubusercontent.com/bojone/tiger/main/Tiger-Lion-AdamW.png" width=70%>

从[Lion](https://kexue.fm/archives/9473)的视角，Tiger是一个简化版的Lion（beta1=beta2）；从[SignSGD](https://arxiv.org/abs/1802.04434)的视角，Tiger是一个带有动量和weight decay的SignSGD。

Tiger只用到了动量来构建更新量，根据[《隐藏在动量中的梯度累积：少更新几步，效果反而更好？》](https://kexue.fm/archives/8634)的结论，此时我们不新增一组参数来“无感”地实现梯度累积！这也意味着在我们有梯度累积需求时，Tiger已经达到了显存占用的最优解，这也是“Tiger”这个名字的来源（**Tig**ht-fisted Optimiz**er**，抠门的优化器，不舍得多花一点显存）。

- 博客1：https://kexue.fm/archives/9512
- 博客2：https://kexue.fm/archives/8634

## 特性

## 鸣谢

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
