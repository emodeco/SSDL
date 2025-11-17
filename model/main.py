# Import necessary libraries for data processing, machine learning, and visualization
import os
import warnings
import numpy as np
import scipy.io
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from keras.models import Model
from keras.layers import Input, LSTM, TimeDistributed, Dense, RepeatVector
from keras.callbacks import EarlyStopping as KEarlyStopping
import tensorflow as tf

import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping as SEarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


# Set random seeds 
np.random.seed(Config.RANDOM_SEED)
tf.random.set_seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.RANDOM_SEED)

# Determine computation device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


def aggregate_feature_by_poslist(feature, poslist):
    """
    Aggregate features by position list, grouping consecutive timepoints into trials.
    
    Args:
        feature: Feature matrix to aggregate
        poslist: List of positions indicating trial boundaries
        
    Returns:
        agg: Aggregated features (mean per trial)
        segments: List of original segments for each trial
    """
    feature = feature.T
    pos = np.squeeze(poslist).astype(int)
    start = 0
    agg, segments = [], []
    for end in pos:
        seg = feature[start:end].T
        segments.append(seg)
        agg.append(seg.mean(axis=1))
        start = end
    agg = np.stack(agg, axis=0)
    return agg, segments


def create_seq2seq(x, sequence_length=5, num_shift=1):
    """
    Create sequence-to-sequence pairs for LSTM training.
    Each input sequence predicts the next sequence shifted by one timestep.
    
    Args:
        x: Input time series data
        sequence_length: Length of input/output sequences
        num_shift: Number of timesteps to shift between sequences
        
    Returns:
        inputs: Input sequences
        targets: Target sequences (shifted by one timestep)
    """
    num_points = x.shape[0]
    inputs = []
    targets = []
    for p in np.arange(0, num_points, num_shift):
        if p + sequence_length + 1 >= num_points:
            break
        inputs.append(x[p: p + sequence_length, :])
        targets.append(x[p + 1: p + sequence_length + 1, :])
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets


def ensure_minimum_window(x, sequence_length):
    """
    Ensure minimum window size by repeating mean values if data is too short.
    
    Args:
        x: Input data with shape [T, k]
        sequence_length: Required minimum sequence length
        
    Returns:
        window: Input window (repeated mean)
        window: Target window (repeated mean)
    """
    if x.ndim != 2:
        raise ValueError("ensure_minimum_window expects shape [T, k]")
    mean_vec = x.mean(axis=0, keepdims=True)
    window = np.repeat(mean_vec, repeats=sequence_length, axis=0)
    return window[None, ...], window[None, ...]


def build_trial_mats_with_mask_and_scale(trial_segments, mask, scaler=None, fit=False):
    """
    Build trial matrices by applying feature mask and scaling.
    
    Args:
        trial_segments: List of trial segment matrices
        mask: Boolean mask for feature selection
        scaler: StandardScaler instance (if None, will be created)
        fit: Whether to fit the scaler on training data
        
    Returns:
        selected_trials: List of selected and scaled trial matrices
        scaler: Fitted StandardScaler instance
    """
    selected_trials = []
    if fit:
        # Collect all timepoints across all trials for fitting
        all_timepoints = []
        for seg in trial_segments:
            sel = seg[mask, :].T
            all_timepoints.append(sel)
        all_timepoints = np.concatenate(all_timepoints, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_timepoints)

    # Apply mask and scaling to each trial
    for seg in trial_segments:
        sel = seg[mask, :].T
        sel = scaler.transform(sel)
        selected_trials.append(sel)
    return selected_trials, scaler


def build_windows_from_trials(trial_mats, sequence_length=5, num_shift=1, ensure_min=True):
    """
    Build sequence windows from trial matrices for LSTM training.
    
    Args:
        trial_mats: List of trial matrices
        sequence_length: Length of sequences
        num_shift: Shift between sequences
        ensure_min: Whether to ensure minimum window size
        
    Returns:
        X_all: Concatenated input sequences
        Y_all: Concatenated target sequences
    """
    X_list, Y_list = [], []
    for mat in trial_mats:
        Xw, Yw = create_seq2seq(mat, sequence_length=sequence_length, num_shift=num_shift)
        if Xw.shape[0] == 0 and ensure_min:
            Xw, Yw = ensure_minimum_window(mat, sequence_length)
        X_list.append(Xw)
        Y_list.append(Yw)
    X_all = np.concatenate(X_list, axis=0) if len(X_list) > 0 else np.zeros((0, sequence_length, trial_mats[0].shape[1]))
    Y_all = np.concatenate(Y_list, axis=0) if len(Y_list) > 0 else np.zeros((0, sequence_length, trial_mats[0].shape[1]))
    return X_all, Y_all


def encode_trial_means(encoder, trial_mats, sequence_length=5, num_shift=1, drop_first_n=10):
    """
    Encode trial matrices using LSTM encoder and compute mean encoding per trial.
    
    Args:
        encoder: Trained LSTM encoder model
        trial_mats: List of trial matrices to encode
        sequence_length: Sequence length for encoding
        num_shift: Shift between sequences
        drop_first_n: Number of initial encodings to drop before averaging
        
    Returns:
        encoded_means: Mean encoding for each trial
    """
    encoded_means = []
    for mat in trial_mats:
        Xw, _ = create_seq2seq(mat, sequence_length=sequence_length, num_shift=num_shift)
        if Xw.shape[0] == 0:
            Xw, _ = ensure_minimum_window(mat, sequence_length)
        enc = encoder.predict(Xw, verbose=0)
        if enc.shape[0] > drop_first_n:
            enc = enc[drop_first_n:]
        encoded_means.append(enc.mean(axis=0))
    return np.array(encoded_means)


class LSTMSeq2Seq:
    """
    LSTM-based sequence-to-sequence autoencoder for feature encoding.
    """
    def __init__(self, encoding_dim=32, sequence_length=5, epochs=100, batch_size=32, verbose=0,
                 es_patience=10, es_min_delta=1e-4, dropout=0.3, activation='tanh', optimizer='adam', loss='mse', validation_split=0.1):
        """
        Initialize LSTM sequence-to-sequence model.
    
        """
        self.encoding_dim = encoding_dim
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.dropout = dropout
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.validation_split = validation_split
        self.model = None
        self.encoder = None

    def _build(self, num_features):
        """
        Build the LSTM sequence-to-sequence model architecture.
        
        Args:
            num_features: Number of input features
        """
        visible = Input(shape=(self.sequence_length, num_features))
        encoder_lstm = LSTM(self.encoding_dim, activation=self.activation, dropout=self.dropout)(visible)
        decoder = RepeatVector(self.sequence_length)(encoder_lstm)
        decoder = LSTM(self.encoding_dim, activation=self.activation, dropout=self.dropout, return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(num_features))(decoder)
        self.model = Model(inputs=visible, outputs=decoder)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        # Create encoder model for feature extraction
        self.encoder = Model(inputs=visible, outputs=encoder_lstm)

    def fit_on_windows(self, X_seq, y_seq, X_val=None, Y_val=None):
        """
        Train the model on sequence windows.

        """
        if X_seq.shape[0] == 0:
            return self
        num_features = X_seq.shape[2]
        self._build(num_features)

        # Set up early stopping callback
        es_cb = KEarlyStopping(
            monitor='val_loss',
            patience=self.es_patience,
            min_delta=self.es_min_delta,
            restore_best_weights=True,
            verbose=0
        )

        # Train with validation data if provided, otherwise use validation split
        if X_val is not None and Y_val is not None and Y_val.shape[0] > 0:
            self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_data=(X_val, Y_val),
                callbacks=[es_cb]
            )
        else:
            self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_split=self.validation_split,
                callbacks=[es_cb]
            )
        return self


class MLPRegressorModule(nn.Module):
    """
    Multi-layer perceptron regressor module using PyTorch.
    """
    def __init__(self, input_dim, hidden_ratio=4, dropout=0.3):
        """
        Initialize MLP regressor.
        
        Args:
            input_dim: Input dimension
            hidden_ratio: Ratio to determine hidden layer size (hidden_dim = input_dim / hidden_ratio)
            dropout: Dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim / hidden_ratio))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(int(input_dim / hidden_ratio), 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output prediction
        """
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Load data from MATLAB file
    data = scipy.io.loadmat(Config.DATA_FILE)
    feature = data[Config.DATA_FIELDS['feature']][:, :]
    arousal = data[Config.DATA_FIELDS['arousal']][:, :]
    pos = data[Config.DATA_FIELDS['poslist']][:, :]

    # Aggregate features by trial positions
    agg_feature, trial_segments = aggregate_feature_by_poslist(feature, pos)
    print("Mean feature per trial shape:", agg_feature.shape)
    print("trial_segments[0] shape (first trial original segment):", trial_segments[0].shape)

    # Transpose arousal if needed
    if arousal.shape[0] == 1:
        arousal = np.transpose(arousal)

    # Filter out trials with specific filter value
    alltrial = [i for i in range(arousal.shape[0]) if arousal[i] != Config.FILTER_VALUE]
    agg_feature = agg_feature[alltrial, :]
    arousal = arousal[alltrial, :]
    trial_segments = [trial_segments[i] for i in alltrial]

    n_trials, n_features_total = agg_feature.shape
    idx = np.arange(n_trials)

    # Define parameter grid for hyperparameter search
    param_grid = {
        'encoding_dim': Config.LSTM_ENCODING_DIM,
        'k': [min(k, n_features_total) for k in Config.FEATURE_SELECTION_K]
    }
    param_list = list(ParameterGrid(param_grid))

    # Set up nested cross-validation: outer fold for final evaluation
    outer_kf = KFold(n_splits=Config.OUTER_FOLDS, shuffle=Config.CV_SHUFFLE, random_state=Config.CV_RANDOM_STATE)
    all_val_pred, all_val_true = [], []

    outer_fold_ccs = []

    # Outer cross-validation loop
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(idx)):
        print(f"\n=== Outer Fold {outer_fold + 1} ===")

        # Split data for outer fold
        X_outer_train, X_outer_test = agg_feature[outer_train_idx], agg_feature[outer_test_idx]
        y_outer_train_raw, y_outer_test_raw = arousal[outer_train_idx], arousal[outer_test_idx]
        trial_segments_train = [trial_segments[i] for i in outer_train_idx]
        trial_segments_test = [trial_segments[i] for i in outer_test_idx]

        # Inner cross-validation for hyperparameter tuning
        inner_kf = KFold(n_splits=Config.INNER_FOLDS, shuffle=Config.CV_SHUFFLE, random_state=Config.CV_RANDOM_STATE)
        best_score = np.inf
        best_params = None

        # Grid search over parameter combinations
        for params in param_list:
            inner_scores = []

            # Inner cross-validation loop
            for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):
                # Split inner fold data
                X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
                y_inner_train_raw, y_inner_val_raw = y_outer_train_raw[inner_train_idx], y_outer_train_raw[inner_val_idx]
                trials_inner_train = [trial_segments_train[i] for i in inner_train_idx]
                trials_inner_val = [trial_segments_train[i] for i in inner_val_idx]

                # Standardize target values
                y_scaler_inner = StandardScaler()
                y_scaler_inner.fit(y_inner_train_raw)
                y_inner_train_z = y_scaler_inner.transform(y_inner_train_raw).astype(np.float32)
                y_inner_val_z = y_scaler_inner.transform(y_inner_val_raw).astype(np.float32)

                # Drop initial timepoints from trial segments
                trials_inner_train_dropped = []
                for seg in trials_inner_train:
                    if seg.shape[1] > Config.DROP_FIRST_N_POINTS:
                        seg_dropped = seg[:, Config.DROP_FIRST_N_POINTS:]
                    else:
                        seg_dropped = seg
                    trials_inner_train_dropped.append(seg_dropped)
                
                # Compute mean features for feature selection
                X_inner_train_dropped = []
                for seg in trials_inner_train_dropped:
                    mean_feat = seg.mean(axis=1)
                    if mean_feat.ndim == 0:
                        mean_feat = np.array([mean_feat])
                    elif mean_feat.ndim > 1:
                        mean_feat = mean_feat.flatten()
                    X_inner_train_dropped.append(mean_feat)
                
                # Ensure all features have the same shape (pad or truncate)
                if len(X_inner_train_dropped) > 0:
                    first_shape = X_inner_train_dropped[0].shape
                    for i, feat in enumerate(X_inner_train_dropped):
                        if feat.shape != first_shape:
                            if feat.shape[0] < first_shape[0]:
                                padded = np.zeros(first_shape)
                                padded[:feat.shape[0]] = feat
                                X_inner_train_dropped[i] = padded
                            else:
                                X_inner_train_dropped[i] = feat[:first_shape[0]]
                
                X_inner_train_for_selection = np.stack(X_inner_train_dropped, axis=0)

                # Feature selection using SelectKBest
                selector = SelectKBest(score_func=f_regression, k=params['k'])
                selector.fit(X_inner_train_for_selection, y_inner_train_raw.ravel())
                mask = selector.get_support()

                # Build and scale trial matrices with selected features
                train_mats, scaler_lstm = build_trial_mats_with_mask_and_scale(
                    trials_inner_train, mask, scaler=None, fit=True
                )
                val_mats, _ = build_trial_mats_with_mask_and_scale(
                    trials_inner_val, mask, scaler=scaler_lstm, fit=False
                )

                # Create sequence windows for LSTM training
                Xw_train, Yw_train = build_windows_from_trials(
                    train_mats, sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, ensure_min=True
                )
                Xw_val, Yw_val = build_windows_from_trials(
                    val_mats, sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, ensure_min=True
                )

                # Train LSTM sequence-to-sequence model
                lstm_seq2seq = LSTMSeq2Seq(
                    encoding_dim=params['encoding_dim'],
                    sequence_length=Config.SEQUENCE_LENGTH,
                    epochs=Config.LSTM_EPOCHS,
                    batch_size=Config.LSTM_BATCH_SIZE,
                    verbose=0,
                    es_patience=Config.LSTM_ES_PATIENCE,
                    es_min_delta=Config.LSTM_ES_MIN_DELTA,
                    dropout=Config.LSTM_DROPOUT,
                    activation=Config.LSTM_ACTIVATION,
                    optimizer=Config.LSTM_OPTIMIZER,
                    loss=Config.LSTM_LOSS,
                    validation_split=Config.LSTM_VALIDATION_SPLIT
                )
                lstm_seq2seq.fit_on_windows(Xw_train, Yw_train, X_val=Xw_val, Y_val=Yw_val)

                # Encode trial means using trained LSTM encoder
                X_inner_train_encoded_trial_mean = encode_trial_means(
                    lstm_seq2seq.encoder, train_mats,
                    sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, drop_first_n=Config.DROP_FIRST_N
                )
                X_inner_val_encoded_trial_mean = encode_trial_means(
                    lstm_seq2seq.encoder, val_mats,
                    sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, drop_first_n=Config.DROP_FIRST_N
                )

                # Train MLP regressor on encoded features
                mlp = NeuralNetRegressor(
                    MLPRegressorModule,
                    module__input_dim=params['encoding_dim'],
                    module__hidden_ratio=Config.MLP_HIDDEN_RATIO,
                    module__dropout=Config.MLP_DROPOUT,
                    max_epochs=Config.MLP_MAX_EPOCHS,
                    lr=Config.MLP_LR,
                    batch_size=Config.MLP_BATCH_SIZE_TRAIN,
                    optimizer=torch.optim.Adam,
                    optimizer__weight_decay=Config.MLP_WEIGHT_DECAY,
                    verbose=0,
                    device=device,
                    train_split=ValidSplit(Config.MLP_TRAIN_SPLIT, random_state=Config.MLP_TRAIN_SPLIT_RANDOM_STATE),
                    callbacks=[
                        SEarlyStopping(
                            monitor='valid_loss',
                            patience=Config.MLP_ES_PATIENCE,
                            threshold=Config.MLP_ES_THRESHOLD,
                            threshold_mode=Config.MLP_ES_THRESHOLD_MODE,
                            lower_is_better=True,
                            load_best=True
                        )
                    ]
                )
                mlp.fit(
                    X_inner_train_encoded_trial_mean.astype(np.float32),
                    y_inner_train_z.reshape(-1, 1)
                )
                
                # Predict on validation set and inverse transform
                y_val_pred_z = mlp.predict(X_inner_val_encoded_trial_mean.astype(np.float32)).reshape(-1, 1)
                y_val_pred = y_scaler_inner.inverse_transform(y_val_pred_z).ravel()
                y_val_true = y_inner_val_raw.ravel()

                # Evaluate performance
                mse = mean_squared_error(y_val_true, y_val_pred)
                inner_scores.append(mse)
                try:
                    cc, _ = pearsonr(y_val_true, y_val_pred)
                    print(f"Parameters {params}, cc: {cc:.{Config.PRINT_CC_DECIMALS}f}")
                except Exception:
                    print(f"Parameters {params}, cc: nan")

            # Select best parameters based on mean MSE
            mean_score = float(np.nanmean(inner_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        print(f"Outer Fold {outer_fold + 1} best parameters: {best_params}")

        # Prepare outer training data with best parameters
        # Drop initial timepoints
        trial_segments_train_dropped = []
        for seg in trial_segments_train:
            if seg.shape[1] > Config.DROP_FIRST_N_POINTS:
                seg_dropped = seg[:, Config.DROP_FIRST_N_POINTS:]
            else:
                seg_dropped = seg
            trial_segments_train_dropped.append(seg_dropped)
        
        # Compute mean features for feature selection
        X_outer_train_dropped = []
        for seg in trial_segments_train_dropped:
            mean_feat = seg.mean(axis=1)
            if mean_feat.ndim == 0:
                mean_feat = np.array([mean_feat])
            elif mean_feat.ndim > 1:
                mean_feat = mean_feat.flatten()
            X_outer_train_dropped.append(mean_feat)
        
        # Ensure consistent feature shapes
        if len(X_outer_train_dropped) > 0:
            first_shape = X_outer_train_dropped[0].shape
            for i, feat in enumerate(X_outer_train_dropped):
                if feat.shape != first_shape:
                    if feat.shape[0] < first_shape[0]:
                        padded = np.zeros(first_shape)
                        padded[:feat.shape[0]] = feat
                        X_outer_train_dropped[i] = padded
                    else:
                        X_outer_train_dropped[i] = feat[:first_shape[0]]
        
        X_outer_train_for_selection = np.stack(X_outer_train_dropped, axis=0)

        # Feature selection with best k
        selector = SelectKBest(score_func=f_regression, k=best_params['k'])
        selector.fit(X_outer_train_for_selection, y_outer_train_raw.ravel())
        mask = selector.get_support()

        # Build and scale trial matrices
        train_mats, scaler_lstm = build_trial_mats_with_mask_and_scale(
            trial_segments_train, mask, scaler=None, fit=True
        )
        test_mats, _ = build_trial_mats_with_mask_and_scale(
            trial_segments_test, mask, scaler=scaler_lstm, fit=False
        )

        # Create sequence windows
        Xw_train, Yw_train = build_windows_from_trials(
            train_mats, sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, ensure_min=True
        )
        
        # Train final LSTM model with best parameters
        lstm_seq2seq = LSTMSeq2Seq(
            encoding_dim=best_params['encoding_dim'],
            sequence_length=Config.SEQUENCE_LENGTH,
            epochs=Config.LSTM_EPOCHS,
            batch_size=Config.LSTM_BATCH_SIZE,
            verbose=0,
            es_patience=Config.LSTM_ES_PATIENCE,
            es_min_delta=Config.LSTM_ES_MIN_DELTA,
            dropout=Config.LSTM_DROPOUT,
            activation=Config.LSTM_ACTIVATION,
            optimizer=Config.LSTM_OPTIMIZER,
            loss=Config.LSTM_LOSS,
            validation_split=Config.LSTM_VALIDATION_SPLIT
        )
        lstm_seq2seq.fit_on_windows(Xw_train, Yw_train)

        # Encode trial means
        X_train_encoded_trial_mean = encode_trial_means(
            lstm_seq2seq.encoder, train_mats,
            sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, drop_first_n=Config.DROP_FIRST_N
        )
        X_test_encoded_trial_mean = encode_trial_means(
            lstm_seq2seq.encoder, test_mats,
            sequence_length=Config.SEQUENCE_LENGTH, num_shift=Config.NUM_SHIFT, drop_first_n=Config.DROP_FIRST_N
        )

        # Standardize target values for outer fold
        y_scaler_outer = StandardScaler()
        y_scaler_outer.fit(y_outer_train_raw)
        y_outer_train_z = y_scaler_outer.transform(y_outer_train_raw).astype(np.float32)

        # Train final MLP regressor
        mlp = NeuralNetRegressor(
            MLPRegressorModule,
            module__input_dim=best_params['encoding_dim'],
            module__hidden_ratio=Config.MLP_HIDDEN_RATIO,
            module__dropout=Config.MLP_DROPOUT,
            max_epochs=Config.MLP_MAX_EPOCHS,
            lr=Config.MLP_LR,
            batch_size=Config.MLP_BATCH_SIZE_TEST,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=Config.MLP_WEIGHT_DECAY,
            verbose=0,
            device=device,
            train_split=ValidSplit(Config.MLP_TRAIN_SPLIT, random_state=Config.MLP_TEST_SPLIT_RANDOM_STATE),
            callbacks=[
                SEarlyStopping(
                    monitor='valid_loss',
                    patience=Config.MLP_ES_PATIENCE,
                    threshold=Config.MLP_ES_THRESHOLD,
                    threshold_mode=Config.MLP_ES_THRESHOLD_MODE,
                    lower_is_better=True,
                    load_best=True
                )
            ]
        )
        mlp.fit(
            X_train_encoded_trial_mean.astype(np.float32),
            y_outer_train_z.reshape(-1, 1)
        )

        # Predict on test set
        y_pred_z = mlp.predict(X_test_encoded_trial_mean.astype(np.float32)).reshape(-1, 1)
        y_pred = y_scaler_outer.inverse_transform(y_pred_z).ravel()
        y_true = y_outer_test_raw.ravel()

        # Store predictions and true values
        all_val_pred.append(y_pred)
        all_val_true.append(y_true)

        # Calculate correlation coefficient for this fold
        try:
            cc1, _ = pearsonr(y_true, y_pred)
        except Exception:
            cc1 = np.nan
        outer_fold_ccs.append(cc1)
        print(f'Outer Fold {outer_fold + 1} result cc: {cc1:.{Config.PRINT_CC_DECIMALS}f}')

    # Concatenate all predictions and true values
    all_val_pred = np.concatenate(all_val_pred)
    all_val_true = np.concatenate(all_val_true)
    
    # Calculate final evaluation metrics
    try:
        cc_all, _ = pearsonr(all_val_true, all_val_pred)
    except Exception:
        cc_all = np.nan
    r2_all = r2_score(all_val_true, all_val_pred)

    # Print final results
    print("\n" + "=" * 50)
    print("Final Evaluation Results")
    print("=" * 50)
    print(f"Outer fold cc mean: {np.nanmean(outer_fold_ccs):.{Config.PRINT_CC_DECIMALS}f} ± {np.nanstd(outer_fold_ccs):.{Config.PRINT_CC_DECIMALS}f}")
    print(f"Overall data cvCC: {cc_all:.{Config.PRINT_CC_DECIMALS}f}")
    print(f"Overall data R2: {r2_all:.{Config.PRINT_CC_DECIMALS}f}")
    print(f"Overall data shape: {all_val_true.shape}")

    # Save results to MATLAB file
    scipy.io.savemat(Config.DATA_FILE + Config.OUTPUT_SUFFIX, 
                     mdict={Config.OUTPUT_FIELDS['label']: all_val_true, 
                            Config.OUTPUT_FIELDS['predict']: all_val_pred})

    # Normalize labels and predictions for visualization
    label_mean = np.mean(all_val_true)
    label_std = np.std(all_val_true)
    
    finallabel = (all_val_true - label_mean) / label_std
    finalpred = (all_val_pred - label_mean) / label_std
    
    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set plot style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 30
    
    # Plot scatter points
    plt.scatter(finallabel, finalpred, s=8, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot diagonal reference line
    plt.plot([-2, 2], [-2, 2], linestyle='--', color='black', linewidth=1)
    
    # Add regression line with confidence interval
    sns.regplot(x=finallabel, y=finalpred, scatter_kws={"marker": ".", "s": 3, "alpha": 0}, 
                ci=95, line_kws={'linewidth': 1}, color='grey')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Z-score normalized label', fontsize=30)
    plt.ylabel('Z-score normalized prediction', fontsize=30)
    plt.show()