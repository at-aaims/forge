# FORGE: Pre-training Open Foundation Model for Science 

## Contributions   
-	Best practices for the end-to-end pre-training LLMs for science on HPC 
-	Fist open releases of a set of foundation models (and domain datasets) on scientific corpus  
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
- Scopus: 6M abstracts for the [DOIs] (https://www.dropbox.com/s/8uxxaptavgxi7r9/dois.txt?dl=0I) extracted via Scopus API

