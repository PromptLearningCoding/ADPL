# Code for ADPL

Code for the paper 'ADPL: Attentive Dual Modality Prompt Learning for Vision-Language Understanding'

## How to install datasets

Please follow the instructions at [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare all datasets.

## Training and Evaluation

(1)Base-to-Novel class generalization setting

> ```shell
> # seed=1
> # trains and evaluates on base classes
> bash scripts/adpl/base2new_train_adpl.sh imagenet 1
> # evaluates on novel classes
> bash scripts/adpl/base2new_test_adpl.sh imagenet 1
> 
> # seed=2
> # trains and evaluates on base classes
> bash scripts/adpl/base2new_train_adpl.sh imagenet 2
> # evaluates on novel classes
> bash scripts/adpl/base2new_test_adpl.sh imagenet 2
> 
> # seed=3
> # trains and evaluates on base classes
> bash scripts/adpl/base2new_train_adpl.sh imagenet 3
> # evaluates on novel classes
> bash scripts/adpl/base2new_test_adpl.sh imagenet 3
> ```

(2) Cross-Dataset Transfer

```shell
# seed=1 
bash scripts/adpl/xd_train_adpl.sh imagenet 1
# seed=2 
bash scripts/adpl/xd_train_adpl.sh imagenet 2
#seed=3 
bash scripts/adpl/xd_train_adpl.sh imagenet 3
```

 (3)Domain Generalization

```shell
for SEED in 1 2 3
do
    bash scripts/adpl/xd_test_adpl.sh imagenetv2 ${SEED}
    bash scripts/adpl/xd_test_adpl.sh imagenet_sketch ${SEED}
    bash scripts/adpl/xd_test_adpl.sh imagenet_a ${SEED}
    bash scripts/adpl/xd_test_adpl.sh imagenet_r ${SEED}
done
```

