import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from binaural_dataset import BinauralDataLoader
from loss import BinauralOneSourceLoss
from trainer import BinauralOneSource


def _print_and_flush(msg):
    print(msg)
    sys.stdout.flush()


def main():
    # 1. load params
    with open('params.json') as json_file:
        params = json.load(json_file)
    print("Training parameters: ", params)
    
    batch_size = params["training"]["batch_size"]

    results = {}
    # Load loss
    loss = BinauralOneSourceLoss()
    trainer = BinauralOneSource(params, loss)
    
    dataset_paths = params["path_binaural_test"]
    dataset_paths = [
        os.path.join(dataset_paths, dataset_path)
        for dataset_path in os.listdir(dataset_paths)
    ]
    for dataset_path in dataset_paths:
        if not os.path.isdir(dataset_path) or dataset_path.endswith(".DS_Store"):
            continue
        print("Testing on dataset: ", dataset_path)
        params["path_binaural_test"] = dataset_path
        dataset_test = BinauralDataLoader(params, split='test',mode='Noisy')
        # %% Network training
        
        loss_test, rmsae_test = trainer.test_epoch(dataset_test, 1)
        # azi_error = 1 - 2*np.arccos(loss_test)/np.pi
        azi_error = np.arccos(1 - loss_test)/np.pi
        # breakpoint()
        print("Test loss: ", loss_test)
        print("Azimuth Error: ", azi_error*180)

        results[dataset_path.split("/")[-1]] = azi_error
        
    
    print(results)
    # breakpoint()
    keys = list(results.keys())
    keys = sorted(keys, key=lambda x: int(x[:-2]))
    
    # Save results in a .mat file after sorting them by the keys
    sorted_results = dict(sorted(results.items(), key=lambda x: int(x[0][:-2])))

# Extract keys and values
    keys_list = list(sorted(keys, key=lambda x: int(x[:-2])))
    values_list = list([value * 180 for value in sorted_results.values()])

# Save keys and values lists to a .mat file
    sio.savemat('sorted_results_rvrb_IsoDir.mat', {'keys_list': keys_list, 'values_list': values_list})

        
    # res = list({key: results[key] for key in keys}.values())
    # Keys are strings representing integers with 'dB' in the end, so we need to sort them as integers
    
      # Keys are strings representing integers with 'dB' in the end, so we need to sort them as integers
  
    plt.figure(figsize=(18.5,10.5), dpi=200)
    plt.bar(keys, [results[key]*180 for key in keys])
    plt.xlabel("SNR", fontsize=18)
    plt.ylabel("RMS Error (deg)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig("results_c_lsnr_Isodir_rvrb.png")
if __name__ == '__main__':
    main()
