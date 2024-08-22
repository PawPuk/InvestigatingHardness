import argparse
import matplotlib.pyplot as plt
import numpy as np
import utils as u


def load_results(dataset_name: str, varying_param: str, fixed_value: float):
    """
    Load the results for a fixed parameter value while varying another parameter.

    :param dataset_name: The name of the dataset (e.g., MNIST).
    :param varying_param: The parameter being varied ('oversampling_factor' or 'undersampling_ratio').
    :param fixed_value: The fixed value for the other parameter.
    :return: A dictionary containing the results for each combination of training and test sets.
    """
    varying_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    for value in varying_values:
        if varying_param == 'oversampling_factor':
            file_name = f"{u.ACCURACIES_SAVE_DIR}{dataset_name}_osf_{value}_usr_{fixed_value}.pkl"
        else:
            file_name = f"{u.ACCURACIES_SAVE_DIR}{dataset_name}_osf_{fixed_value}_usr_{value}.pkl"
        results[value] = u.load_data(file_name)
    return results, varying_values


def plot_results(results, varying_values, fixed_param_name, varying_param, fixed_value):
    """
    Plot the results with the varying parameter on the x-axis and accuracy on the y-axis.

    :param results: The results for each value of the varying parameter.
    :param varying_values: The values of the varying parameter.
    :param fixed_param_name: The name of the fixed parameter.
    :param fixed_value: The fixed value of that parameter.
    """
    # Define styles and colors for clarity
    styles = ['solid', 'dashed', 'dotted']
    markers = ['o', 's', '^']
    colors = ['red', 'green', 'blue']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over the training sets (Hard, Easy, All)
    for i, train_set in enumerate(['Hard', 'Easy', 'All']):
        # Iterate over the test sets (Hard, Easy, All)
        for j, test_set in enumerate(['hard', 'easy', 'all']):
            # Extract the accuracies for the current training/test set combination
            means = [results[val][train_set][test_set][0] for val in varying_values]
            stds = [results[val][train_set][test_set][1] for val in varying_values]
            # Plot the mean accuracies with a line
            ax.plot(varying_values, means, label=f'Train: {train_set}, Test: {test_set}',
                    linestyle=styles[i], marker=markers[i], color=colors[j])

            # Fill the area between (mean - std) and (mean + std) with a semi-transparent color
            ax.fill_between(varying_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                            color=colors[j], alpha=0.2)

    ax.set_xlabel(f'{varying_param.capitalize()} Values')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy vs {varying_param.capitalize()} (Fixed {fixed_param_name}: {fixed_value})')
    ax.legend()
    plt.grid(True)
    plt.show()


def main(dataset_name: str, fixed_oversampling_factor: float, fixed_undersampling_ratio: float):
    # Check for argument validity
    if fixed_oversampling_factor is not None and fixed_undersampling_ratio is not None:
        raise ValueError("Please provide only one parameter (either oversampling_factor or undersampling_ratio).")
    if fixed_oversampling_factor is None and fixed_undersampling_ratio is None:
        raise ValueError("Please provide one parameter (either oversampling_factor or undersampling_ratio).")

    # Determine the varying parameter
    if fixed_oversampling_factor is not None:
        varying_param = 'undersampling_ratio'
        fixed_param_name = 'oversampling_factor'
        fixed_value = fixed_oversampling_factor
    else:
        varying_param = 'oversampling_factor'
        fixed_param_name = 'undersampling_ratio'
        fixed_value = fixed_undersampling_ratio

    # Load results
    results, varying_values = load_results(dataset_name, varying_param, fixed_value)
    # Plot results
    plot_results(results, varying_values, fixed_param_name, varying_param, fixed_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot accuracy as a function of a single parameter.')
    parser.add_argument('--dataset_name', type=str, default='MNIST', help='Name of the dataset (e.g., MNIST, CIFAR10).')
    parser.add_argument('--fixed_oversampling_factor', type=float,
                        help='Fixed value for oversampling_factor (set this or undersampling_ratio).')
    parser.add_argument('--fixed_undersampling_ratio', type=float,
                        help='Fixed value for undersampling_ratio (set this or oversampling_factor).')

    args = parser.parse_args()
    main(args.dataset_name, args.fixed_oversampling_factor, args.fixed_undersampling_ratio)
