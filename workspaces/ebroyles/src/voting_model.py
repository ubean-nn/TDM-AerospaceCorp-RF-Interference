import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from constants import LOGS_ROOT

class MyVoter:

    def __init__(self, trained_models, num_classes, samples_label, foldername, filename):
        """
        @params
        trained_models: np.array([MyTrainedModel, ...]) (num_trained_models,)
        num_classes: int
        samples_label: np.array() (num_samples,)
        foldername: String "my_data/my_voter/{foldername}/{filename}.csv" 

        @self
        probs: np.array() (num_models, num_samples, num_classes)
        weights: np.array() (num_models,)
        """
        self.trained_models = trained_models
        self.num_models = trained_models.size
        self.num_classes = num_classes
        self.num_samples = samples_label.size
        self.samples_label = samples_label
        self.folderpath = os.path.join("my_data/my_voter", foldername)
        self.filepath = os.path.join(self.folderpath, filename)
        for tm in trained_models:
            assert tm.num_classes == self.num_classes, "Number of classes must match for all trained models"
            assert tm.num_samples == self.num_samples, "Number of samples must match for all trained models"

        self.probs = np.zeros((self.num_models, self.num_samples, self.num_classes))
        self.weights = self.get_even_weights()

    def set_probs(self):
        """
        @params
        save: bool
        """
        for i,tm in enumerate(self.trained_models):
            self.probs[i] = tm.evaluate()

    def load_probs(self):
        """
        Alternative to set_probs if data already exists
        """
        pass

    def save_csv(self):
        os.makedirs(self.folderpath, exist_ok=True)
        cols = {}
        for m in range(self.num_models):
            for lbl in range(self.num_classes):
                col_name = f"m{m}_L{lbl}"
                cols[col_name] = self.probs[m, :, lbl]

        cols["label"] = self.samples_label
        df = pd.DataFrame(cols)
        df.to_csv(self.filepath, index=False)

    def get_acc(self):
        """
        @brief
        Evaluates the accuracy by summing the probabilities for each model multiplied by its weights. 
        Computes the accuracy if allowed multiple guesses -> [acc with 1 guess, 2 guess, ..., num_classes gueses]

        @returns:
        acc: np.array() (num_classes,) == (allowed_guesses,)
        """
        allowed_guesses = self.num_classes
        num_correct_after_guess = np.zeros(allowed_guesses)
        for s in range(self.num_samples):
            combo_probs = np.sum(self.probs[:, s, :] * self.weights[:, None], axis=0) #rowwise
            ranked_labels = np.argsort(combo_probs)[::-1] # descending
            true_label = self.samples_label[s]
            rank = np.argmax(ranked_labels == true_label)
            if rank < allowed_guesses:
                num_correct_after_guess[rank:] += 1
        acc = num_correct_after_guess / self.num_samples
        return acc
    
    def get_models_acc(self):
        """
        Finds the accuracy for eacch model (weights are not used)

        @returns
        models_acc: np.array() (num_models, num_classes) == (num_models, num_guesses)
        """
        allowed_guesses = self.num_classes
        models_acc = np.zeros((self.num_models, allowed_guesses))
        for m in range(self.num_models):
            num_correct_after_guess = np.zeros(allowed_guesses)
            for s in range(self.num_samples):
                probs = self.probs[m, s, :]
                ranked_labels = np.argsort(probs)[::-1]  # descending
                true_label = self.samples_label[s]
                rank = np.argmax(ranked_labels == true_label)
                if rank < allowed_guesses:
                    num_correct_after_guess[rank:] += 1
            models_acc[m] = num_correct_after_guess / self.num_samples
        return models_acc

    def get_confusion_matrix(self, allowed_guesses):
        """
        @params
        allowed_guesses: int "0->num_classes"

        @returns
        conf_matrix: confustion_matrix
        """
        y_true, y_pred = [], []
        for s in range(self.num_samples):
            combo_probs = np.sum(self.probs[:, s, :] * self.weights[:, None], axis=0)
            ranked_labels = np.argsort(combo_probs)[::-1]
            true_label = self.samples_label[s]
            top_k = ranked_labels[:allowed_guesses]
            if true_label in top_k: pred_label = true_label
            else: pred_label = ranked_labels[0]
            y_true.append(true_label)
            y_pred.append(pred_label)
        return confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))

    def diplay_confusion_matrix(self, allowed_guesses=3):
        """
        @params
        allowed_guesses: int "0->num_classes"

        @outputs
        confusion matrix
        """
        title = f"Confusion Matrix with allowed_guesses={allowed_guesses}"
        cm = self.get_confusion_matrix(allowed_guesses)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(self.num_classes))
        disp.plot()
        plt.title(title)
        plt.show()

    """
    WEIGHTS
    """
    def set_weights(self, weights):
        """
        @params
        weigths: np.array() (num_trained_models,)
        """
        self.weights = weights

    def get_even_weights(self):
        return np.ones(self.num_models)
    
class MyTrainedModel:

    """
    @brief
    Wraps a trained_model and is provided with correctly formated samples_features for the trained_model. 
    This does not check the label it instead returns the probabilities for each label to be used by MyVoter.
    """

    def __init__(self, trained_model, num_classes, samples_features):
        """
        @params
        trained_model: MyNN | MyLinear
        num_classes: int "used for code clarity here"
        samples_features: np.array() (num_samples, num_features) "samples to get probs for"
        samples_label: np.array() (num_samples,)
        """
        self.trained_model = trained_model
        self.samples_features = samples_features
        self.num_classes = num_classes
        self.num_samples, self.num_features = samples_features.shape
        assert self.num_classes == self.trained_model.num_classes, "Number of classes must match (meaning of classes also)"
        assert self.num_features == self.trained_model.num_features, "Provided data has a different number of features than trained_model used"
        self.trained_model.set_ready_to_eval()

    def evaluate(self):
        """
        @returns
        samples_probs: np.array() (num_samples, num_classes)
        """
        samples_probs = np.zeros(self.num_samples)
        for sample_idx in range(self.num_samples):
            samples_probs[sample_idx] = self.evaluate_sample(sample_idx)
        return samples_probs

    def evaluate_sample(self, sample_idx):
        """
        @params
        sample_idx: int

        @returns
        sample_probs: np.array() (num_classes,)
        """
        features = self.samples_features[sample_idx]
        sample_probs = self.trained_model.evaluate_sample(features)
        return sample_probs

class MyNN(pl.LightningModule):
    
    def __init__(self, model_name, max_epochs, num_classes, num_features, model_body, loss_fn, log_every_n_steps=10, lr=1e-3):
        """
        @params
        model_name: String "used as logger folder name"
        num_classes: int 
        num_features: int
        max_epochs: int
        model_body: nn.Sequential(...)
        loss_fn: nn.CrossEntropyLoss | other
        log_every_n_steps: int
        lr: float "learning rate"
        """
        super().__init__()
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.num_features = num_features
        self.model_body = model_body
        self.loss_fn = loss_fn
        logger = CSVLogger(LOGS_ROOT, name=model_name)
        self.trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", log_every_n_steps=log_every_n_steps, logger=self.logger)
        self.save_hyperparameters(ignore=["model_body", "loss_fn", "model_name", "max_epochs", "num_classes", "num_features", "trainer"])
    
    def fit(self, loader):
        """
        @params
        loader: Loader()
        """
        start = time.time()
        self.trainer.fit(self, loader)
        print(f"Fit Time (s): {time.time() - start}")

    def set_ready_to_eval(self):
        self.eval()
        return True
    
    def evaluate_sample(self, features):
        """
        @params
        features: np.array() (num_features,)

        @returns
        probs: np.array() (num_classes,)
        """
        #convert faeatues into a tensor
        X = torch.from_numpy(features)
        logits = self(X)
        probs_tensor = torch.softmax(logits)
        return probs_tensor.numpy()

    def forward(self, X):
        """
        DONT CALL THIS, USED BY LIGHTING
        @Pparams
        X: Tensor() (num_features,)
        """
        return self.model_body(X)
    
    def training_step(self, batch, batch_idx):
        """
        DONT CALL THIS, USED BY LIGHTING
        @params
        batch: Tensor() (batchsize, num_features)?
        batch_idx: int

        @Returns
        loss: float
        """
        X, y = batch
        logits = self(X)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean() #this finds the idx of the label with smallest logits
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """
        DONT CALL THIS, USED BY LIGHTING
        """
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
        
class MyLinear:

    """
    @brief
    Given some samples of features find the median and standard deviation across each feature for each class. 
    Evaluation:
    * features -> zscores across all features for each class -> sum zscores across labels (MAYBE TRY AVERAGE ALSO?)
    * turn summed zscores into probabilities for each label (closer to 0 is better)
    """

    def __init__(self, num_classes, num_features, use_abs_zscore=True):
        """
        @params
        num_classes: int
        num_features: int
        use_abs_zscore: bool "use false to reward having some negetive zscores and some positive" (DO NOT USE FALSE, I HAVE NOT FIGURED OUT A SCORE FOR THIS YET)

        @self
        classes_features_median: np.array() (num_classes, num_features,)
        classes_features_std: np.array() (num_classes, num_features,) "standard deviation"
        """
        self.num_classes = num_classes
        self.num_features = num_features
        self.use_abs_zscore = use_abs_zscore
        self.classes_features_median = np.zeros((num_classes, num_features))
        self.classes_features_std = np.zeros((num_classes, num_features))

    def fit(self, samples_features, samples_label):
        """
        @params
        samples_features: np.array() (num_samples, num_features)
        samples_label: np.array() (num_samples,)
        """
        for i in range(self.num_classes):
            labeli_samples_features = samples_features[samples_label == i]
            self.classes_features_median[i] = np.median(labeli_samples_features, axis=0) #rowwise
            self.classes_features_std[i] = np.std(labeli_samples_features, axis=0) #rowwise
        self.classes_features_std[self.classes_features_std < 1e-8] = 1e-8 #avoid divide by zero

    def set_ready_to_eval(self):
        return True
    
    def evaluate_sample(self, features):
        """"
        @brief
        to compute the 

        @params
        features: np.array() (num_features,)

        @returns
        probs: np.array (num_classes)
        """
        zscores = (features - self.classes_features_median) / self.classes_features_std
        zscores = abs(zscores) if self.use_abs_zscore else zscores
        sum_zscores = np.sum(zscores, axis=1) #colwise -> (num_classes,)
        probs = MyLinear.get_inverse_softmax_probs(sum_zscores)
        return probs
    
    @staticmethod
    def get_inverse_softmax_probs(scores):
        """
        DO NOT USE WITH NEGATIVE NUMBERS (they are given 100%)

        @brief
        takes scores that exists on 0->inf and outputs a probability (0->1) that is exponentially larger when close to 0

        @params
        scores: np.array (num_classes,) "0->inf"

        @returns
        probs: np.array (num_classes,) "0->1"
        """
        probs = 1/np.exp(scores) / (np.sum(1/np.exp(scores)))
        return probs
    






    

