# Description

Repository to train mmrotate model on ssd-tiny dataset.

## References

- <https://github.com/open-mmlab/mmrotate/blob/main/demo/MMRotate_Tutorial.ipynb>

## Manual installation

```bash
conda create --name mmrotate-azure python=3.8 -y
conda activate mmrotate-azure
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

```bash
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
conda activate mmrotate-azure
pip install .
```

```bash
conda env export > environment.yml
```

Then add this under pip within environment.yml file: `- --extra-index-url https://download.pytorch.org/whl/cu121`
