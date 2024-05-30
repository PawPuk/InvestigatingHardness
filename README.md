# Figure 2 - Sample complexity experiments

## Description

In the Figure 2 of our work we have shown simple experiments on 4 toy datasets. All the code for that is available in 
the `sample_complexity_experiments.py`. This program works as following:

1. Define the class manifolds, together with their neighbourhoods and sample the training data from those manifolds. 
Everything  is hard coded. In the future, it would be interesting to implement an algorithm for neighbourhood generation 
based on  density, so that it can be generalized to setting more similar to the real-world one.
2. Loop through neighbourhoods. For each neighbourhood $\mathcal{N}$.
   1. Remove all samples from $\mathcal{N}$ that are in the training set.
   2. Train ten networks on this training set.
   3. Evaluate the trained models on a test set obtained by randomly generating a thousand samples from $\mathcal{N}$.
   4. If the models did not achieve an average accuracy of over $99\%$, then sample a single point from $\mathcal{N}$ 
   into the training dataset and retrain the models from scratch.
   5. Repeat the above five times for statistical significance.
3. The average (over five runs) of the numbers of samples required from $\mathcal{N}$ to achieve an average of 
over $99\%$ accuracy is our estimate for sample complexity for $\mathcal{N}$.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy

Make sure you have the required libraries installed. You can install them using pip:

```bash
pip install torch torchvision matplotlib tqdm numpy
```

## Running the Code

To run [sample_complexity_experiment.py](sample_complexity_experiment.py), use the following command in your terminal:

```bash
python sample_complexity_experiment.py --dataset [DATASET] 
```

### Parameters

- `--dataset`: Specifies which dataset to use in the experiment. This must be an integer value. The possible choices are
1, 2, 3, 4. The default choice is 1 as it's the simplest dataset.

## Expected Results

When running `python sample_complexity_experiment`, the following result is observed:

<img src="Figures/1_a_0.5_samples_3_t_99_init_2_opt_ADAM_lr_0.01_epochs_100_runs_5_networks_10.png" width="400">

As mentioned in our paper, we notice a bias in which the model performs better on the orange crosses (right) than green 
circles (left). This means that majority of the paths along the loss landscape taken by the model result in the decision
functions that are curved towards the left side. In our initial experiments we did not find an experimental setting in
which this bias would be reversed (further experiments are required).

# Figure 3 - Hardness-based within-class data imbalance on CIFAR10

Generating Figure 3 is a three-step process. We firstly have to compute the confidences using `compute_confidences.py`, 
and then run `investigate_hardness_common_case.py` and `investigate_hardness_edge_case.py`, making sure that we specify
that we want those programs to be run on CIFAR10 (add `--dataset_name CIFAR10`). `compute_confidences.py` works as 
follows:

1. Load the entire dataset specified by `dataset_name` and normalize it. We are combining training and test data.
2. Initialize 20 models (5 instances of 4 different architectures differing based on the dataset used).
3. Iterate through models.
   1. Train model on the entire dataset.
   2. Compute and save confidences and energies for every data sample

The results are saved in the `Results/Confidences` folder.