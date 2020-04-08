# Dataset 
There are two sets of data: train & test, stored in "data" repository. 
Student are given only train data. There are totally 4 map corresponding to different dataset_id (0, ..., 3)

# Run on Train dataset
```
python main.py --split_name train --dataset_id <0, 1 or 2, or 3>
```

# Run on Test dataset
```
python main.py --split_name test  --dataset_id 0 
```

# Generate figures 
To generate figures, run
```
python gen_figures.py --split_name <train or test>  --dataset_id <0, 1, or 2, 3> 
```

# Log files
Log file & images are all stored in "logs" repository
