# Noise Inspector for Stochastic Gradient Descent 

Author: Guney Tombak (gtombak@ethz.ch)  
Supervisor: Mr. Thomas Allard  
Professor: Dr. Helmut Bölcskei  
Chair for Mathematical Information Science, D-ITET, ETH Zurich  

## Setup

All dependencies can be fulfilled by creating a conda environment using environment.yml  

```shell
conda env create -f environment.yml && conda activate sgdn
```

For GPU implementation use also:

```shell
conda install pytorch torchvision torchaudio cudatoolkit=X.Y -c pytorch -c conda-forge
```

with a cuda version X.Y. compatible with your GPU.

Before running, you should log in to your wandb (Weights and Biases) account:

```shell
wandb login
```

## Usage

The parameters of the run can be configured by editing `config.py`.  
The parameters as a tuple creates a new branch for the search tree.  

## Results by Weights and Biases

The results are saved in both your local device in the folder named `wandb` and also the Weights and Biases cloud. You can inspect the results directly on web or to use the local files, please check the [documentation](https://docs.wandb.ai/guides/track/public-api-guide) and `visualization/plot_results.ipynb`.

