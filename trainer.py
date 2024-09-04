import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm, trange

from utils import sph2cart, cart2sph
from models.binaural import BinauralLocalizer, BinauralFeatureExtractor


class BinauralOneSource(torch.nn.Module):
    """Abstract class to the routines to train the one source tracking models and perform inferences."""

    def __init__(self, params, loss, print_model=False):
        """
        model: Model to work with
        """

        super().__init__()

        model = BinauralLocalizer(
            params["nb_gcc_bins"], params["neural_srp"]
        )
        
        feature_extractor = BinauralFeatureExtractor(params)
        checkpoint_path = params["model_checkpoint_path"]

        self.model = model
        self.feature_extractor = feature_extractor

        self.cuda_activated = False
        self.loss = loss
        if checkpoint_path != "":
            self.load_checkpoint(checkpoint_path)
        
        if print_model:
            print(self.model)
            n_params = sum(p.numel() for p in self.model.parameters())
            print("Number of parameters: {}".format(n_params))

    def cuda(self):
        """Move the model to the GPU and perform the training and inference there."""
        self.model.cuda()
        self.cuda_activated = True

        if self.feature_extractor is not None:
            self.feature_extractor.cuda()
        
    def cpu(self):
        """Move the model back to the CPU and perform the training and inference here."""
        self.model.cpu()
        self.cuda_activated = False

    def forward(self, batch):
        if self.feature_extractor is not None:
            batch = self.feature_extractor(batch)
        return self.model(batch)

    def load_checkpoint(self, path):  
        print(f"Loading model from checkpoint {path}")
        state_dict = torch.load(path, map_location=torch.device('cpu'))        
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, path):
        print(f"Saving model to checkpoint {path}")
        torch.save(self.model.state_dict(), path)

    def extract_features(self, mic_sig_batch=None, labels=None):
        """Basic data transformation which
        moves the data to GPU tensors and applies the VAD.
        Override this method to apply more transformations."""

        output = {
            "network_input": {},
            "network_target": {},
        }

        # 1. Apply transform for mic signals
        if isinstance(mic_sig_batch, np.ndarray):
            mic_sig_batch = torch.from_numpy(mic_sig_batch.astype(np.float32))

        if self.cuda_activated:
            mic_sig_batch = mic_sig_batch.cuda()

        output["network_input"]["signal"] = mic_sig_batch

        if labels is not None:
            DOAw_batch = torch.from_numpy(
                labels.astype(np.float32)
            )

            if self.cuda_activated:
                DOAw_batch = DOAw_batch.cuda()

            output["network_target"] = DOAw_batch

        if self.feature_extractor is not None:
            output["network_input"] = self.feature_extractor(
                output["network_input"]
            )

        return output

    def predict_batch(self, mic_sig_batch, labels=None):
        """Predict the DOA for a batch of trajectories."""
        data = self.extract_features(mic_sig_batch, labels)
        x_batch = data["network_input"]

        model_output = self.model(x_batch)

        # A model must output either a DOA in cartesian or spherical coordinates
        if "doa_sph" not in model_output:
            model_output["doa_sph"] = cart2sph(model_output["doa_cart"])
        if "doa_cart" not in model_output:
            model_output["doa_cart"] = sph2cart(model_output["doa_sph"])

        if labels is not None:
            # Prepare the targets for the loss function
            DOA_batch_cart = data["network_target"]

            targets = {
                "doa_cart": DOA_batch_cart,
            }

            return model_output, targets
        else:
            return model_output

    def train_epoch(
        self,
        dataset,
        batch_size,
        lr=0.0001,
        shuffle=True,
        epoch=None
    ):
        """Train the model with an epoch of the dataset."""

        self.model.train()  # set the model in "training mode"
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if shuffle:
            dataset.shuffle()
        bar = tqdm(dataset.get_batch(), total=len(dataset))
        for i, (mic_sig_batch, labels) in enumerate(bar):

            optimizer.zero_grad()

            model_output, labels = self.predict_batch(mic_sig_batch, labels)

            loss = self.loss(model_output, labels)

            loss["loss"].backward()
            optimizer.step()
            
            if i % 100 == 0:
                bar.set_description(
                    f"Epoch {epoch} - loss: {loss['loss'].item():.6f}"
                )
            del model_output, loss, mic_sig_batch, labels

    def test_epoch(self, dataset, batch_size, sum_loss=True, return_labels=False):
        """Test the model with an epoch of the dataset."""
        self.model.eval()  # set the model in "testing mode"
        with torch.no_grad():
            loss_data = 0
            rmsae_data = 0

            if not sum_loss:
                loss_data = []
                rmsae_data = []
            
            if return_labels:
                labels_list = []

            nb_batches = len(dataset) // batch_size
            bar = tqdm(dataset.get_batch(), total=len(dataset))
            for i, (mic_sig_batch, labels) in enumerate(bar):
                model_output, targets = self.predict_batch(
                    mic_sig_batch, labels)

                loss = self.loss(model_output, targets)

                if sum_loss:
                    loss_data += loss["loss"].item()
                else:
                    loss_data.append(loss["loss"].cpu().item())
                
                if return_labels:
                    labels_list.append(labels)
 
            if sum_loss:
                loss_data /= nb_batches
                rmsae_data /= nb_batches
            else:
                loss_data = np.array(loss_data)
                rmsae_data = np.array(rmsae_data)

            out = loss_data, rmsae_data

            if return_labels:
                out = out + (labels_list,)
            
            return out
