# MAG/Aminer data preprocessing

## STEPS
-       unzip all files into mag directory 
```text
mag_papers_0.txt   mag_papers_19.txt  mag_papers_28.txt  mag_papers_37.txt  mag_papers_5.txt
mag_papers_10.txt  mag_papers_1.txt   mag_papers_29.txt  mag_papers_38.txt  mag_papers_6.txt
mag_papers_11.txt  mag_papers_20.txt  mag_papers_2.txt   mag_papers_39.txt  mag_papers_7.txt
mag_papers_12.txt  mag_papers_21.txt  mag_papers_30.txt  mag_papers_3.txt   mag_papers_8.txt
mag_papers_13.txt  mag_papers_22.txt  mag_papers_31.txt  mag_papers_40.txt  mag_papers_9.txt
mag_papers_14.txt  mag_papers_23.txt  mag_papers_32.txt  mag_papers_41.txt  
mag_papers_15.txt  mag_papers_24.txt  mag_papers_33.txt  mag_papers_42.txt
mag_papers_16.txt  mag_papers_25.txt  mag_papers_34.txt  mag_papers_43.txt
mag_papers_17.txt  mag_papers_26.txt  mag_papers_35.txt  mag_papers_44.txt
mag_papers_18.txt  mag_papers_27.txt  mag_papers_36.txt  mag_papers_4.txt

```
-       run preprocess.lsf with 45 ranks to extract the abstracts into cleandata directory in parallel 
```python
with jl.open("mag/mag_papers_" + str(rank) + ".txt") as reader:
    for obj in reader:
        if "indexed_abstract" in obj.keys():
            abstract += 1
            if abstract % 2000 == 0:
                #print("Writing 2000 objects")
                json_writer.write_all(abstract_buffer)
                abstract_buffer = []
            try:
                plain_text = json.loads(obj["indexed_abstract"])
                plain_text = " ".join(plain_text["InvertedIndex"].keys())
                abstract_buffer.append({"id" : obj["id"], "title": obj["title"], "abstract" : plain_text})
            except:
                err_count += 1
        total += 1
```
-       convert abstracts to webdataset format into webdata_abstracts directory 

```
python wd_writer.py 
```

## DATA Directory
   
