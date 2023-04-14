# Partition the data by scientific domains 

## define domain labels
```python
labels = {'material':0,
          'physics':1,
          'chemistry':2,
          'cs':3,
          'medical':4,
          'socialscience':5
          }
```
## use fine-tuned SciBERT model 
```python
model = torch.load(<path-to-finetuned-model>)
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

with torch.no_grad():
        for inputs, labels in loader:
            # preprocess the input
            mask = inputs['attention_mask'].to(device)
            input_id = inputs['input_ids'].squeeze(1).to(device)
            result = model(input_id, mask)
            domains = result[0].argmax(dim=1).cpu().detach().numpy()
```

## Run the partitioning
```bash
srun -n8 -N2 --gpus-per-task 1 python partition.py --model <path-to-finetuned-model> --data_dir <path-to-data>
```
| data |     domain    |    files    |
|:----:|:-------------:|:-----------:|
|  MAG |    physics    |  5,206,984  |                
|      |   chemistry   |  18,910,292 |                
|      |    material   |  15,447,301 |                
|      |       cs      |  22,089,618 |                
|      |    medical    |  28,678,821 |                
|      | socialscience |  13,815,190 |                
|      |      all      | 104,148,206 | 
| CORE |    physics    |  4,777,087  |                
|      |   chemistry   |  9,023,452  |                
|      |    material   |  2,837,809  |                
|      |       cs      |  6,519,304  |                
|      |    medical    |  10,170,512 |                
|      | socialscience |  18,931,258 |                
|      |      all      |  52,259,422 |                
|Aminer|    physics    |   4,028,448 |                
|      |   chemistry   | 12,489,929  |                
|      |    material   |  3,197,740  |                
|      |       cs      |  7,840,826  |                
|      |    medical    |  15,081,601 |                
|      | socialscience |  4,499,049  |                
|      |      all      |  47,137,593 |                
