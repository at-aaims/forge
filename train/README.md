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
The example configuration for [forge-mat](./configs/forge-mat.yml)
- input data setup 
```json
  "data-path": "data/mat/tokens/mat_text_document",
  "vocab-file": "data/mat/tokens/mat_vocab.json",
  "tokenizer_type": "HFTokenizer"
```
- parallelism setup 
```json
   "pipe-parallel-size": 1,
   "model-parallel-size": 1
```
- model setup
```json
   "num-layers": 24,
   "hidden-size": 2064,
   "num-attention-heads": 24,
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "layernorm",
   "pos-emb": "rotary"
```
- optimizer setup
```json
   "optimizer": {
     "type": "Lamb",
     "params": {
       "lr": 0.012,
       "betas": [0.9, 0.999],
       "eps": 1.0e-8,
     }
   },
   "zero_optimization": {
    "stage": 1,
    "allgather_partitions": True,
    "allgather_bucket_size": 500000000,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": True,
    "cpu_offload": False
  }
```

## Run
The example [job script](./job.sb) and Frontier [configuration](./configs/frontier.yml) can be used to run on Frontier 
```bash
sbatch job.sb
```

## Community benchmark 
The community benchmarks are evaluated by 
```python
python ./deepy.py evaluate.py -d configs ${MODEL}.yml frontier.yml --eval_tasks sciq arc_easy arc_challenge piqa hendrycksTest-college_physics hendrycksTest-college_chemistry hendrycksTest-college_medicine hendrycksTest-college_computer_science hendrycksTest-sociology openbookqa
```
