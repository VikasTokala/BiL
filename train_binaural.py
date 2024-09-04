import json
import os
import sys
import torch

from datetime import datetime

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
    lr = params["training"]["lr"]
    nb_epoch = params["training"]["nb_epochs"]

    model_name = params["model"]  # Only for the output filenames, change it also in Network declaration cell

    # Load loss
    loss = BinauralOneSourceLoss(mode='cosine')

    trainer = BinauralOneSource(params, loss)
    
    if torch.cuda.is_available():
        trainer.cuda()
     
        
        
   
    dataset_train = BinauralDataLoader(params, split='train')
    dataset_val = BinauralDataLoader(params, split='validation')
    # %% Network training

    print('Training network...')
    best_epoch = 0
    best_val_metric = float('inf')
    start_time_str = datetime.now().strftime('%m-%d_%Hh%Mm')
    run_dir = f'logs/{model_name}_{start_time_str}'
    os.makedirs(run_dir, exist_ok=True)
    # Save params
    with open(os.path.join(run_dir, 'params.json'), 'w') as json_file:
        json.dump(params, json_file, indent=4)

    for epoch_idx in range(1, nb_epoch + 1):
        _print_and_flush('\nEpoch {}/{}:'.format(epoch_idx, nb_epoch))

        if epoch_idx in params["training"]["lr_decrease_epochs"]:
            # Decrease the learning rate
            print('\nDecreasing learning rate')
            lr /= params["training"]["lr_decrease_factor"]
        
        trainer.train_epoch(dataset_train,
                              batch_size,
                            lr=lr,
                            epoch=epoch_idx
        )
        loss_test, rmsae_test = trainer.test_epoch(dataset_val, batch_size)
        _print_and_flush('Test loss: {:.4f}, Test rmsae: {:.2f}deg'.format(loss_test, rmsae_test))

        # Save best model
        if rmsae_test < best_val_metric:
            best_val_metric = rmsae_test
            _print_and_flush('New best model found at epoch {}, saving...'.format(epoch_idx))
            
            last_best_epoch = best_epoch
            best_epoch = epoch_idx

            best_model_path = f'{run_dir}/best_ep{best_epoch}.bin'
            trainer.save_checkpoint(best_model_path)
            if last_best_epoch > 0:
                last_best_model_path = f'{run_dir}/best_ep{last_best_epoch}.bin'
                os.remove(last_best_model_path)

    print('\nTraining finished\n')

    # %% Save model
    _print_and_flush('Saving model...')
    
    trainer.save_checkpoint(f'{run_dir}/last.bin')
    _print_and_flush('Model saved.\n')


if __name__ == '__main__':
    main()
