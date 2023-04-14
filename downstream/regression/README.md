# Regression 
The [simulation data](./FePt_lsms.csv) is obtained from https://www.osti.gov/dataexplorer/biblio/dataset/1198813 

The [embedding data](./FePt_emb_matscibert.csv) is generated for each model

The regression task uses a linear model to predict the energy, and we compare the cross-validated MSE on simulation data only with combined simulation data and LLM embeddings

- Apply LLM embeddings to improve regression, e.g.
```
python llm-regression.py --data FePt_lsms.csv --emb FePt_emb_matscibert.csv

MSE on simulation data:  0.22071666644181817
MSE on simulation data + LLM embedding:  0.20490496034455158
Improvement: 7%
``` 

