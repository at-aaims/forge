# MatGPT: A Controlled Comparisons of LLM Architectures on Materials Science Texts   

## Contributions   
-	A compute-efficient way to design LLM architecture
-	Open releases of MatGPT in both GPT-Neox and LLaMA architectures 
-	State-of-the-art in band gap prediction benchmark 

### MatGPT models 
|   Model   | #Params | #Tokens |                                          Link                                         |
|:---------:|:-------:|:-------:|:-------------------------------------------------------------------------------------:|
| llama-1.7B|  1.76B  |   15B   | [download](https://www.dropbox.com/scl/fo/v7sz1sc2c04d11tfcotpz/h?rlkey=jvlfbo24o3otliplhd5abyn65&dl=0) |
| neox-1.7B |  6.7B   |   15B   | [download](https://www.dropbox.com/scl/fo/8fnm51brxw0e7euvsjxmm/h?rlkey=7jmalp9pkmrdzcu62ocu4xpt0&dl=0) |
| neox-6.7B |  6.7B   |   15B   | [download](https://www.dropbox.com/scl/fo/6w8sosiky0rwfw5da5vsm/h?rlkey=o0ifv3xfumtxqkmamdmby4zm6&dl=0) |


### Data sources
- CORE: https://core.ac.uk/documentation/dataset  (core_2020-12-20)
- MAG: https://www.microsoft.com/en-us/research/project/open-academic-graph/  (v2-1)
- Aminer: https://www.microsoft.com/en-us/research/project/open-academic-graph/ (v2-1)
- Arixv: https://huggingface.co/datasets/arxiv_dataset 
- Scopus: 6M abstracts for the [DOIs](https://www.dropbox.com/s/8uxxaptavgxi7r9/dois.txt?dl=0I) extracted via Scopus API
- HF Tokenized data can be [download](https://www.dropbox.com/scl/fo/wldsm8e6fm1wsw87xnepy/h?rlkey=22n9p0yg9be7rh5l9lrahe09e&dl=0)

### Pre-processing 
- Steps on preprocessing [CORE](./preprocess/core/README.md), [MAG and Aminer](./preprocess/oag/README.md)
- Steps on domain [partitioning](./preprocess/domain-partitioning/README.md)

### Training 
- Software envrionment, configurations, and steps on [pre-training](./train/README.md) 

## Reference
```text
@misc{yin2024comparative,
      title={Comparative Study of Large Language Model Architectures on Frontier}, 
      author={Junqi Yin and Avishek Bose and Guojing Cong and Isaac Lyngaas and Quentin Anthony},
      year={2024},
      eprint={2402.00691},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
