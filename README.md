# Description

Repository to train mmrotate model on ssd-tiny dataset.

## References

- <https://github.com/open-mmlab/mmrotate/blob/main/demo/MMRotate_Tutorial.ipynb>

## Manual installation

```
conda create --name mmrotate python=3.10 -y
conda activate mmrotate
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

```
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
conda activate mmrotate
pip install .
```
