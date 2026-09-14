# Pre-Training 

The training is adapted from [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) (e48b0c45) 

## Software environment 
- gcc 10.3.0
- Python 3.8
- Pytorch 1.14.0
- Deepspeed 0.7.3
- ROCM 5.1-5.4
- libfabric 1.15.2.0 
- aws-rccl-plugin 66b3b31

## Build 
- Setup environment 
```bash
odule load PrgEnv-gnu
module load gcc/10.3.0
module load rocm/5.1.0
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p miniconda
```
- build PyTorch
```bash
git clone --recursive -b IFU-master-2022-11-22 https://github.com/ROCmSoftwarePlatform/pytorch
python tools/amd_build/build_amd.py
USE_ROCM=1 MAX_JOBS=4 python setup.py bdist_wheel
```
- build DeepSpeed
```bash
git clone https://github.com/microsoft/DeepSpeed
DS_BUILD_FUSED_LAMB=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1  DS_BUILD_UTILS=1 pip install .
```
- build GPT-NeoX
```bash
git clone https://github.com/EleutherAI/gpt-neox.git
pip install -r requirements/requirements.txt
```

## Configuration  
The example configuration for [forge-l](./configs/forge-l.yml)
- input data setup 
```json
  "data-path": "/lustre/orion/stf218/scratch/junqi/frontier/data/core-aminer-mag-arxiv-scopus/tokens/all_text_document",
  "vocab-file": "/lustre/orion/stf218/scratch/junqi/frontier/data/core-aminer-mag-arxiv-scopus/tokens/all_vocab.json",
  "tokenizer_type": "HFTokenizer"
```
  (these are this project's paths on Frontier's shared Lustre filesystem, which Lux can also reach; point them at your own tokenized corpus)
- parallelism setup 
```json
   "model-parallel-size": 2
```
- model setup
```json
   "num-layers": 48,
   "hidden-size": 6144,
   "num-attention-heads": 48,
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "layernorm",
   "pos-emb": "rotary"
```
- optimizer setup
```json
   "optimizer": {
     "type": "Adam",
     "params": {
       "lr": 0.0001,
       "betas": [0.9, 0.95],
       "eps": 1.0e-8
     }
   },
   "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 450000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 450000000,
    "contiguous_gradients": true
  }
```
- SLURM hostfile
```json
   "hostfile": "logs/forge-l/hostfile"
```
  written by `job.sb` from the SLURM node list before each run (see Run below).

## Run
This branch's [job script](./job.sb) and [configuration](./configs/lux.yml) target OLCF's Lux cluster.

`job.sb` sources a Lux environment (ROCm 7.2.4, mpich, and a Python env with PyTorch/DeepSpeed/ROCm flash-attention) via:
```bash
source /lustre/orion/world-shared/stf218/junqi/lux/lux_env.sh
```
That path and the `#SBATCH -A stf218` account are specific to this project's allocation -- point them at your own environment script and account before submitting.

```bash
sbatch job.sb
```

By default this runs `configs/forge-l.yml` with the Lux launcher config `configs/lux.yml` (override the model with `MODEL=<name> sbatch job.sb`, or the secondary config with `SECONDARY_CONFIG=<name>.yml sbatch job.sb`). `job.sb` builds `logs/${MODEL}/hostfile` from the SLURM node list and launches via DeepSpeed's `slurm` launcher.

## Community benchmark 
The community benchmarks are evaluated by 
```python
python ./deepy.py evaluate.py -d configs ${MODEL}.yml frontier.yml --eval_tasks sciq arc_easy arc_challenge piqa hendrycksTest-college_physics hendrycksTest-college_chemistry hendrycksTest-college_medicine hendrycksTest-college_computer_science hendrycksTest-sociology openbookqa
```
