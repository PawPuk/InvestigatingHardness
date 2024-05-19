import pickle

from tqdm import tqdm

from utils import initialize_model, train_stop_at_inversion, load_data_and_normalize, transform_datasets_to_dataloaders

inversion_points_list = []

for run in tqdm(range(25)):
    dataset = load_data_and_normalize('MNIST', 70000)
    loader = transform_datasets_to_dataloaders(dataset)
    inversion_points = {}
    while set(inversion_points.keys()) != set(range(10)):
        model, optimizer = initialize_model()
        if len(inversion_points.keys()) > 0:
            print(f'Have to restart because not all stragglers were found (found {inversion_points.keys()}).')
        _, inversion_points = train_stop_at_inversion(model, loader, optimizer)
    inversion_points_list.append(inversion_points)

# Save the ten dictionaries of inversion points for later use
with open('Results/inversion_points_list.pkl', 'wb') as f:
    pickle.dump(inversion_points_list, f)
