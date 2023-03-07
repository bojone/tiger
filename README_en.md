[[中文](README.md)|English]

# Tiger

A **Tig**ht-fisted Optimiz**er**, an optimizer that is extremely budget-conscious!

## Features

- Achieves comparable performance to [AdamW](https://arxiv.org/abs/1711.05101) and [LAMB](https://arxiv.org/abs/1904.00962).
- Minimizes memory requirements when using gradient accumulation.
- Adaptive learning rates per parameter, similar to [LAMB](https://arxiv.org/abs/1904.00962).
- Simple strategy to prevent the model from collapsing to NaN.
- Can simulate any lr schedule with piecewise linear learning rates.

## Introduction

Tiger, Lion, and AdamW comparison:

<img src="https://raw.githubusercontent.com/bojone/tiger/main/Tiger-Lion-AdamW.png" width=100%>

From the perspective of [Lion](https://kexue.fm/archives/9473), Tiger is a simplified version of Lion (beta1=beta2). From the perspective of [SignSGD](https://arxiv.org/abs/1802.04434), Tiger is a SignSGD with momentum and weight decay.

Tiger only uses momentum to build the update procedure. According to the conclusion of "[Gradient Accumulation Hidden in Momentum](https://kexue.fm/archives/8634)," we don't need to add another group of parameters to "unobtrusively" implement gradient accumulation! This also means that when we have a gradient accumulation requirement, Tiger has already achieved the best solution for memory usage, which is why this optimizer is called Tiger (**Tig**ht-fisted Optimiz**er**)!

- Blog 1: https://kexue.fm/archives/9512
- Blog 2: https://kexue.fm/archives/8634

## Usage

The current implementation is developed under tensorflow 1.15, and is likely to work in the first few versions of tensorflow 2.x. Later versions have not been tested and their compatibility is unknown.

Reference code:

```python
from tiger import Tiger

optimizer = Tiger(
    learning_rate=1.76e-3,  # Global relative learning rate lr
    beta=0.965,  # beta parameter
    weight_decay=0.01,  # Weight decay rate
    grad_accum_steps=4,  # Gradient accumulation steps
    lr_schedule={
        40000: 1,  # For the first 40k steps, linearly increase the learning rate from 0 to 1*lr (i.e. warm-up steps are 40000/4 steps)
        160000: 0.5,  # For steps 40k-160k, linearly decrease the learning rate from 1*lr to 0.5*lr
        640000: 0.1,  # For steps 160k-640k, linearly decrease the learning rate from 0.5*lr to 0.1*lr
        1280000: 0.01,  # For steps 640k-1280k, linearly decrease the learning rate from 0.1*lr to 0.01*lr, and then keep it constant
    }
)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

## Acknowledgments

Special thanks to Lion's thorough experiments and the friendly communication with Lion's authors.

## Citation

```
@misc{tigeropt,
  title={Tiger: A Tight-fisted Optimizer},
  author={Jianlin Su},
  year={2023},
  howpublished={\url{https://github.com/bojone/tiger}},
}
```

## Communication

QQ group: 808623966, WeChat group: add the robot WeChat account spaces_ac_cn.
