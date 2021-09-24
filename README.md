# "Normative disagreement as a challenge forCooperative AI"
Anonymized code

# Install

Better to be installed in a Virtual Machine. No GPU needed.
You need around 30 Go to install all the dependencies 
(Torch, TF, Gym, OpenSpiel, Ray, RLLib, etc.) 
and up to 100 Go to 
store all the training runs with their logs. 
Tested with Ubuntu 18-04.

```
# 1) Create a new conda virtual env:
conda create -n submission_v_env
# (Conda installation isntruction: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

# 2) Run:
conda activate submission_v_env
bash install_part_1.sh

# 3) Shutdown and restart the (virtual) machine

# 4) After restarting, run:
conda activate submission_v_env
bash install_part_2.sh
```

# Run the experiments

## Base games experiments (Section 4) 

When needed, we use the CLI arg `env` to specify the 
environment and the CLI arg `train_n_replicates` to 
specified the number of replicates to train. 
A minimum of 2 replicates is needed. 
At least 30 replicates are used in the paper.
Finally, it is better to have 1 cpu available 
for each replicate to train.


### amTFT with one of (IPD, IAsymBoS, CG, ABCG):

```
# Matrix games experiments: 
#   Training time with 1 cpu per replicate ~ 30 min 
#   (more than 1 cpu per replicates doesn't speed up the training)
python submission/base_game_experiments/amtft_various_env.py --env IteratedPrisonersDilemma --train_n_replicates 40
python submission/base_game_experiments/amtft_various_env.py --env IteratedAsymBoS --train_n_replicates 40

# Coin game experiments: 
#   Training time with 1 cpu per replicate ~ 12h
#   (more than 1 cpu per replicates doesn't speed up the training)
python submission/base_game_experiments/amtft_various_env.py --env CoinGame --train_n_replicates 40
python submission/base_game_experiments/amtft_various_env.py --env ABCoinGame --train_n_replicates 40
```

You can find the raw results under `~/ray_results/amTFT/{date}/{time}/eval/{date_2nd}/{time_2nd}/`



### meta-amTFT with one of (IAsymBoS, IAsymBoSandPD):

```
# Matrix games experiments: 
# First: you need to train the base amTFT policies (see section just above)
# Second: you need to replace the ckeckpoints's paths in the 
# _get_saved_base_amTFT_paths function by the paths to your own checkpoints.  
#   Evaluating time with 1 cpu per replicate ~ 30 min 
#   (more than 1 cpu per replicates doesn't speed up the training)
python submission/base_game_experiments/meta_amtft_various_env.py --env IteratedAsymBoS --train_n_replicates 40
python submission/base_game_experiments/meta_amtft_various_env.py --env IteratedAsymBoSandPD --train_n_replicates 40
python submission/base_game_experiments/meta_amtft_various_env.py --env CoinGame --train_n_replicates 40
python submission/base_game_experiments/meta_amtft_various_env.py --env ABCoinGame --train_n_replicates 40
```


### LOLA-Exact with one of (IPD, IAsymBoS, IAsymBoSandPD):

```
# Training time with 1 cpu per replicate < 30 min 
# (more than 1 cpu per replicates doesn't speed up the training)
python submission/base_game_experiments/lola_exact_official.py --env IteratedPrisonersDilemma --train_n_replicates 30
python submission/base_game_experiments/lola_exact_official.py --env IteratedAsymBoS --train_n_replicates 30
python submission/base_game_experiments/lola_exact_official.py --env IteratedAsymBoSandPD --train_n_replicates 30
```

You can find the raw results under `~/ray_results/LOLA_Exact/{date}/{time}/eval/{date_2nd}/{time_2nd}/`


### LOLA-PG with one of (CG, ABCG):

LOLA-PG uses a lot of RAM. 
Be careful not to run out of memory!
Better to run these experiments on a Virtual Machine 
with lot of RAM.

```
# Training time with 1 cpu per replicate ~ 12h
# (increasing the number of cpu available per replicates speeds up the training)
python submission/base_game_experiments/lola_pg_official.py --env CoinGame --train_n_replicates 40
python submission/base_game_experiments/lola_pg_official.py --env ABCoinGame --train_n_replicates 40
```

You can find the raw results under `~/ray_results/LOLA_PG/{date}/{time}/eval/{date_2nd}/{time_2nd}/`

## Produce the bar and scatter plots


**With the provided data:**  
We provide the data we used to create the plots. 
You may still need to 
change the `prefix` variable to fit to path where 
this code is located (see point 2  below). 

**With new data:**
If you want to create the plots 
with new data, you need to provide the paths to 
each of the new base game experiments.
To do that, you need to make the following modifications in 
the file `submission/plots/plot_bar_chart_from_saved_results.py`:
1) inside the `_get_inputs` function, change the `files_data` 
   variable to contain the 
   paths to the new base game experiments 
2) inside the `_get_inputs` function, change the `prefix` 
   variable to `~/ray_results` 
   (or what fit your case). 

**Generate the plots:**  

```
# Bar plot
python submission/plots/plot_bar_chart_from_saved_results.py
 
# Scatter plots
python submission/plots/plot_scatter_figs_from_saved_results.py

# Exploitation vs cross-play plot (only with the provided data)
python submission/plots/plot_exploitability_tradeoff.py

# Strategies produced by each meta solver
python submission/plots/plot_meta_policies.py
```