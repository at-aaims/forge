# FORGE: Pre-training Open Foundation Model for Science 

## Contributions   
-	Best practices for the end-to-end pre-training LLMs for science on HPC 
-	Open releases of a set of foundation models (and domain datasets) on scientific corpus  
-	Propose scientific related down-stream benchmarks for evaluating LLMs for science
-	Provide heuristics for large-batch training and communication requirment
-	Evaluate current practices and share our observations 

### FORGE models 
|   Model   | #Params | #Tokens |                                          Link                                         |
|:---------:|:-------:|:-------:|:-------------------------------------------------------------------------------------:|
| Forge-bio |  1.44B  |   38B   | [download](https://www.dropbox.com/sh/41sqapgza3ok9q9/AADLgwTiHVU26ZeW_UQ8apyta?dl=0) |
| Forge-che |  1.44B  |   41B   | [download](https://www.dropbox.com/sh/1jn3n7099r8pzt8/AAAO6sOpFYG-G_qFI6C6CXVVa?dl=0) |
| Forge-eng |  1.44B  |   29B   | [download](https://www.dropbox.com/sh/ueki0n6y3v8gtkw/AAB6-3ml9slcbOonk6ccdD4Ua?dl=0) |
| Forge-mat |  1.44B  |   15B   | [download](https://www.dropbox.com/sh/ngrr3bjulc76944/AABpm_OxA-GQPWzIPM4KpVKOa?dl=0) |
| Forge-phy |  1.44B  |   32B   | [download](https://www.dropbox.com/sh/jxux4tplw5aw7kw/AAAdk334IEMbY7HJlJrWVzyfa?dl=0) |
| Forge-soc |  1.44B  |   90B   | [download](https://www.dropbox.com/sh/54tuyslytqhpq1z/AAAc65c3TQWo2MyPoSiPxKI2a?dl=0) |
|  Forge-s1 |  1.44B  |   10B   | [download](https://www.dropbox.com/sh/kr5otsr07e56kse/AAB8o5_tFAF1HxkQpuVwSprLa?dl=0) |
|  Forge-s2 |  1.44B  |   20B   | [download](https://www.dropbox.com/sh/2wdw9nz4xw5905y/AAB8ckmsZ-do3LEV-e9MafdAa?dl=0) |
|  Forge-s3 |  1.44B  |   30B   | [download](https://www.dropbox.com/sh/muvwrhzebv60mzm/AADhFBQATT7CKqNTtDQcskr9a?dl=0) |
|  Forge-s4 |  1.44B  |   257B  | [download](https://www.dropbox.com/sh/byr1ydik5n1ucod/AADOu_9C6AwVPTThTUFQ7yQba?dl=0) |
|  Forge-m1 |   13B   |   30B   | [download](https://www.dropbox.com/sh/lgoq8z0aw72mtjw/AACUaW83vjUMvlmnQXBftRf-a?dl=0) |
|  Forge-m2 |   13B   |   143B  | [download](https://www.dropbox.com/sh/unc9zjvw34h9v00/AAAsVofwizrkxpTbjY7HMsipa?dl=0) |
|  Forge-l  |  25.6B  |   125B  | [download](https://www.dropbox.com/sh/wgl7fe7i8situkm/AACDWmIWzoPR5Nt6MZrMMDbYa?dl=0) |

### Data sources
- CORE: https://core.ac.uk/documentation/dataset  (core_2020-12-20)
- MAG: https://www.microsoft.com/en-us/research/project/open-academic-graph/  (v2-1)
- Aminer: https://www.microsoft.com/en-us/research/project/open-academic-graph/ (v2-1)
- Arixv: https://huggingface.co/datasets/arxiv_dataset 
- Scopus: 6M abstracts for the [DOIs](https://www.dropbox.com/s/8uxxaptavgxi7r9/dois.txt?dl=0I) extracted via Scopus API

### Example usages
- Forge models can be used using standard Hugging Face API
```python
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
model = GPTNeoXForCausalLM.from_pretrained("path_to_forge_model")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("path_to_forge_model")
prompt = "high entropy alloy applications include"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids,
                            do_sample=True,
                            temperature=0.7,
                            max_length=100)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
```
```text
high entropy alloy applications include high strength steels, alloys, composites, as well some metal alloys. In recent years, there has been much interest the use of such materials for manufacturing parts, components, machinery. For example, automotive sector an increasing number applications. most widely used is steels.
```

### Pre-processing 
- Steps on preprocessing [CORE](./preprocess/core/README.md), [MAG and Aminer](./preprocess/oag/README.md)
- Steps on domain [partitioning](./preprocess/domain-partitioning/README.md)

### Training 
- Software envrionment, configurations, and steps on [pre-training](./train/README.md) 

### Scientific downstream tasks
- [Domain subject and material phase classifications](./downstream/classification/README.md)
- [Energy regression](./downstream/regression/README.md)

### Raw performance data and plots 
- The raw performance data including computation performance, loss, downstream evaluations, etc are [available](https://www.dropbox.com/sh/9uagepbj5fuzwhj/AADTMXzaf3Iwo69t1Qo-USWIa?dl=0)
- The jupyter notebook to plot is also [provided](./plots.ipynb)
