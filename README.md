# jEyeRL
## Ocular biomechanical RL environment

The env uses [OpenSIm RL](https://osim-rl.kidzinski.com/). It includes an ocular environment and an agent trained using Deep Deterministic Policy Gradients method to perform saccades. The agent was able to match the desired eye position with a mean deviation angle of 3.5°±1.25°. 

The proposed DRL environment is based on OpenAI (Brockman et al., 2016), OpenSim (Kidziński et al., 2018a; Seth et al., 2018) and
ocular biomechanics (Iskander et al., 2018d, 2019, 2018b). 

[Watch the RL agent learning to perform saccades](https://ars.els-cdn.com/content/image/1-s2.0-S0021929022000021-mmc1.mp4)


## Setup 

[osim-rl](https://github.com/stanfordnmbl/osim-rl#getting-started)

```
conda create -n opensim-rl -c kidzik -c conda-forge opensim python=3.6.1
source activate opensim-rl
pip install osim-rl
conda install -c anaconda scipy
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
git clone https://github.com/jIskCoder/jEyeRL.git

```
# Running

```
source activate opensim-rl
cd jEyeRL
python jeye.py
```


## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@article{ISKANDER2022110943,
title = {An ocular biomechanics environment for reinforcement learning},
journal = {Journal of Biomechanics},
volume = {133},
pages = {110943},
year = {2022},
issn = {0021-9290},
doi = {https://doi.org/10.1016/j.jbiomech.2022.110943},
url = {https://www.sciencedirect.com/science/article/pii/S0021929022000021},
author = {Julie Iskander and Mohammed Hossny},
}
```
```bibtex
@article{iskander2018ocular,
  title={An ocular biomechanic model for dynamic simulation of different eye movements},
  author={Iskander, J and Hossny, Mohammed and Nahavandi, Saeid and Del Porto, L},
  journal={Journal of biomechanics},
  volume={71},
  pages={208--216},
  year={2018},
  publisher={Elsevier}
}
```
