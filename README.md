# cost-aware-policy-learning
Repository for the paper Cost-Aware Near-Optimal Policy Learning

## Downloading the dataset
Download the voting data from: https://github.com/gsbDBI/ExperimentData/blob/master/Social/ProcessedData/socialpressnofact.csv

## Voting simulation
First, run
```bash
python generate_voting_data.py
```
for data processing. 

Then, run the following for simulating the experiment. 
```bash
python simulation_voting.py --random_seed=40
```
## Court appearance simulation
```bash
python simulation_rideshare.py
```
