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
