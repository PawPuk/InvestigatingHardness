import argparse

from tqdm import tqdm

from utils import initialize_models, load_data_and_normalize, train, transform_datasets_to_dataloaders, save_data


def main(dataset_name: str, subset_size: int, runs: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    loader = transform_datasets_to_dataloaders(dataset)
    all_epoch_radii = []
    for _ in tqdm(range(runs), desc='Investigating the dynamics of the radii of class manifolds for distinctly '
                                    'initialized networks'):
        models, optimizers = initialize_models('MNIST')
        model, optimizer = models[0], optimizers[0]
        epoch_radii = train(dataset_name, model, loader, optimizer, True, 250)
        all_epoch_radii.append(epoch_radii)
    save_data(all_epoch_radii, f'Results/Radii_over_epoch/all_epoch_radii_{subset_size}{dataset_name}.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Investigate the dynamics of the radii of class manifolds for distinctly initialized networks.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset to load. It has to be available in torchvision.datasets.')
    parser.add_argument('--subset_size', type=int, default=20000,
                        help='Size of the subset to use for the analysis.')
    parser.add_argument('--runs', type=int, default=3,
                        help='Indicates how many repetitions of the experiment we want to perform to achieve '
                             'statistical significance.')
    args = parser.parse_args()
    main(args.dataset_name, args.subset_size, args.runs)
