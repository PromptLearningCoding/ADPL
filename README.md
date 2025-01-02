# Code for ADPL

Code for the paper 'ADPL: Attentive Dual-Modality Prompt Tuning for Enhanced Vision-Language Understanding'

## Installation

Install dassl library.

```shell
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/INSTALL.md)

## Data preparation

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```shell
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– ucf101/
|–– ...
```

Please follow the instructions at [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare all datasets.

## Training and Evaluation

**(1) Base-to-Novel class generalization setting**

```shell
# dataset:imagenet
# seed=1
# trains and evaluates on base classes
bash scripts/adpl/base2new_train_adpl.sh imagenet 1
# evaluates on novel classes
bash scripts/adpl/base2new_test_adpl.sh imagenet 1

# seed=2
# trains and evaluates on base classes
bash scripts/adpl/base2new_train_adpl.sh imagenet 2
# evaluates on novel classes
bash scripts/adpl/base2new_test_adpl.sh imagenet 2

# seed=3
# trains and evaluates on base classes
bash scripts/adpl/base2new_train_adpl.sh imagenet 3
# evaluates on novel classes
bash scripts/adpl/base2new_test_adpl.sh imagenet 3

# dataset:dtd
# seed=1
# trains and evaluates on base classes
bash scripts/adpl/base2new_train_adpl.sh dtd 1
# evaluates on novel classes
bash scripts/adpl/base2new_test_adpl.sh dtd 1
# ...
```

**Averaging results over 3 seeds:**

Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```shell
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– ADPL/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– ADPL/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

**(2) Cross-Dataset Transfer**

- Firstly, train ADPL on imagenet in few-shot manner (for all 3 seeds).

```shell
# seed=1 
bash scripts/adpl/xd_train_adpl.sh imagenet 1
# seed=2 
bash scripts/adpl/xd_train_adpl.sh imagenet 2
# seed=3 
bash scripts/adpl/xd_train_adpl.sh imagenet 3
```

- Now evaluate imageNet model on downstream datasets.

```shell
for SEED in 1 2 3
do
    bash scripts/adpl/xd_test_adpl.sh caltech101 ${SEED}
    bash scripts/adpl/xd_test_adpl.sh oxford_pets ${SEED}
    bash scripts/adpl/xd_test_adpl.sh stanford_cars ${SEED}
done
```

 **(3) Domain Generalization**

We use imagenet trained ADPL model for domain generalization experiments. The steps are similar to above cross-dataset experiments, however, model is evaluated on imagenet variants.

- Evaluate imageNet model on variants of imagenet (domain shift datasets).

```shell
for SEED in 1 2 3
do
    bash scripts/adpl/xd_test_adpl.sh imagenetv2 ${SEED}
    bash scripts/adpl/xd_test_adpl.sh imagenet_sketch ${SEED}
    bash scripts/adpl/xd_test_adpl.sh imagenet_a ${SEED}
    bash scripts/adpl/xd_test_adpl.sh imagenet_r ${SEED}
done
```

## Acknowledgements

Our overall experimental pipeline is based on [CoOp, CoCoOp](https://github.com/KaiyangZhou/CoOp) ,[Maple](https://github.com/muzairkhattak/multimodal-prompt-learning) repository.
