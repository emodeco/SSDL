import numpy as np
import scipy.io
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.feature_selection import SelectKBest, f_regression
import torch
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping as SEarlyStopping
from skorch.dataset import ValidSplit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error



if __name__ == '__main__':

    dataname = CONFIG['data_file']
    valence_filter = CONFIG['valence_filter_value']
    outer_cv_splits = CONFIG['outer_cv_splits']
    inner_cv_splits = CONFIG['inner_cv_splits']
    seednum = CONFIG['random_seed']
    sequence_length = CONFIG['sequence_length']
    num_shift = CONFIG['num_shift']
    drop_first_n = CONFIG['drop_first_n']
    encoding_dims = CONFIG['encoding_dims']
    feature_selection_ks = CONFIG['feature_selection_ks']
    lstm_epochs = CONFIG['lstm_epochs']
    lstm_batch_size = CONFIG['lstm_batch_size']
    lstm_es_patience = CONFIG['lstm_es_patience']
    lstm_es_min_delta = CONFIG['lstm_es_min_delta']
    mlp_hidden_dim = CONFIG['mlp_hidden_dim']
    mlp_max_epochs = CONFIG['mlp_max_epochs']
    mlp_lr = CONFIG['mlp_learning_rate']
    mlp_batch_size = CONFIG['mlp_batch_size']
    mlp_weight_decay = CONFIG['mlp_weight_decay']
    mlp_train_split_ratio = CONFIG['mlp_train_split_ratio']
    mlp_train_split_rs = CONFIG['mlp_train_split_random_state']
    mlp_es_patience = CONFIG['mlp_es_patience']
    mlp_es_threshold = CONFIG['mlp_es_threshold']
    mlp_es_threshold_outer = CONFIG['mlp_es_threshold_outer']
    device = CONFIG['device']
    output_suffix = CONFIG['output_suffix']
    

    data = scipy.io.loadmat(dataname)
    feature = data['feature'][:, :]
    valence = data['picv'][:, :]
    pos = data['poslist'][:, :]

    agg_feature, trial_segments = aggregate_feature_by_poslist(feature, pos)
    print("Mean feature per trial shape:", agg_feature.shape)
    print("trial_segments[0] shape (first trial raw segment):", trial_segments[0].shape)

    if valence.shape[0] == 1:
        valence = np.transpose(valence)


    alltrial = [i for i in range(valence.shape[0]) if valence[i] != valence_filter]
    agg_feature = agg_feature[alltrial, :]
    valence = valence[alltrial, :]
    trial_segments = [trial_segments[i] for i in alltrial]

    n_trials, n_features_total = agg_feature.shape
    idx = np.arange(n_trials)


    param_grid = {
        'encoding_dim': encoding_dims,
        'k': [min(k, n_features_total) for k in feature_selection_ks]
    }
    param_list = list(ParameterGrid(param_grid))

    outer_kf = KFold(n_splits=outer_cv_splits, shuffle=True, random_state=seednum)
    all_val_pred, all_val_true = [], []

    outer_fold_ccs = []
    inner_fold_ccs = []     
    inner_fold_mses = []   
    param_selection_mses = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(idx)):
        print(f"\n=== Outer Fold {outer_fold + 1} ===")

        X_outer_train, X_outer_test = agg_feature[outer_train_idx], agg_feature[outer_test_idx]
        y_outer_train_raw, y_outer_test_raw = valence[outer_train_idx], valence[outer_test_idx]
        trial_segments_train = [trial_segments[i] for i in outer_train_idx]
        trial_segments_test = [trial_segments[i] for i in outer_test_idx]

        y_outer_train = y_outer_train_raw
        y_outer_test = y_outer_test_raw

        inner_kf = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=seednum)
        best_score = np.inf
        best_params = None
        best_mean_cc_for_record = None
        best_mean_mse_for_record = None

        for params in param_list:
            inner_scores = []
            inner_ccs = []

            for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):

                X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
                y_inner_train_raw, y_inner_val_raw = y_outer_train_raw[inner_train_idx], y_outer_train_raw[inner_val_idx]
                trials_inner_train = [trial_segments_train[i] for i in inner_train_idx]
                trials_inner_val = [trial_segments_train[i] for i in inner_val_idx]

                y_inner_train = y_inner_train_raw
                y_inner_val = y_inner_val_raw

                selector = SelectKBest(score_func=f_regression, k=params['k'])
                X_inner_train_selected_mean = selector.fit_transform(X_inner_train, y_inner_train.ravel())
                _ = X_inner_train_selected_mean
                mask = selector.get_support()

                train_mats, scaler_lstm = build_trial_mats_with_mask_and_scale(
                    trials_inner_train, mask, scaler=None, fit=True
                )
                val_mats, _ = build_trial_mats_with_mask_and_scale(
                    trials_inner_val, mask, scaler=scaler_lstm, fit=False
                )

                Xw_train, Yw_train, _ = build_windows_from_trials(
                    train_mats, sequence_length=sequence_length, num_shift=num_shift, ensure_min=True
                )
                Xw_val, Yw_val, _ = build_windows_from_trials(
                    val_mats, sequence_length=sequence_length, num_shift=num_shift, ensure_min=True
                )

                lstm_seq2seq = LSTMSeq2Seq(
                    encoding_dim=params['encoding_dim'],
                    sequence_length=sequence_length,
                    epochs=lstm_epochs,
                    batch_size=lstm_batch_size,
                    verbose=0,
                    es_patience=lstm_es_patience,
                    es_min_delta=lstm_es_min_delta
                )
                lstm_seq2seq.fit_on_windows(Xw_train, Yw_train, X_val=Xw_val, Y_val=Yw_val)

                X_inner_train_encoded_trial_mean = encode_trial_means(
                    lstm_seq2seq.encoder, train_mats,
                    sequence_length=sequence_length, num_shift=num_shift, drop_first_n=drop_first_n
                )
                X_inner_val_encoded_trial_mean = encode_trial_means(
                    lstm_seq2seq.encoder, val_mats,
                    sequence_length=sequence_length, num_shift=num_shift, drop_first_n=drop_first_n
                )

                mlp = NeuralNetRegressor(
                    MLPRegressorModule,
                    module__input_dim=params['encoding_dim'],
                    module__hidden_dim=mlp_hidden_dim,
                    max_epochs=mlp_max_epochs,
                    lr=mlp_lr,
                    batch_size=mlp_batch_size,
                    optimizer=torch.optim.Adam,
                    optimizer__weight_decay=mlp_weight_decay,
                    verbose=0,
                    device=device,
                    train_split=ValidSplit(mlp_train_split_ratio, random_state=mlp_train_split_rs),
                    callbacks=[
                        SEarlyStopping(
                            monitor='valid_loss',
                            patience=mlp_es_patience,
                            threshold=mlp_es_threshold,
                            threshold_mode='rel',
                            lower_is_better=True,
                            load_best=True
                        )
                    ]
                )
                mlp.fit(
                    X_inner_train_encoded_trial_mean.astype(np.float32),
                    y_inner_train.ravel().astype(np.float32).reshape(-1, 1)
                )
                y_val_pred = mlp.predict(X_inner_val_encoded_trial_mean.astype(np.float32)).ravel()
                y_val_true = y_inner_val.ravel()

                mse = mean_squared_error(y_val_true, y_val_pred)
                try:
                    cc, _ = pearsonr(y_val_true, y_val_pred)
                except Exception:
                    cc = np.nan

                inner_scores.append(mse)
                inner_ccs.append(cc)
                print(f"Params {params}, Inner MSE: {mse:.4f}, cc: {cc:.4f}")

            mean_score = float(np.nanmean(inner_scores))
            mean_cc = float(np.nanmean(inner_ccs))
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                best_mean_cc_for_record = mean_cc
                best_mean_mse_for_record = mean_score

        print(f"Outer Fold {outer_fold + 1} best params: {best_params}, min MSE: {best_score:.4f}")

        inner_fold_ccs.append(best_mean_cc_for_record)
        inner_fold_mses.append(best_mean_mse_for_record)
        param_selection_mses.append(best_score)

        selector = SelectKBest(score_func=f_regression, k=best_params['k'])
        _ = selector.fit_transform(X_outer_train, y_outer_train.ravel())
        mask = selector.get_support()

        train_mats, scaler_lstm = build_trial_mats_with_mask_and_scale(
            trial_segments_train, mask, scaler=None, fit=True
        )
        test_mats, _ = build_trial_mats_with_mask_and_scale(
            trial_segments_test, mask, scaler=scaler_lstm, fit=False
        )

        Xw_train, Yw_train, _ = build_windows_from_trials(
            train_mats, sequence_length=sequence_length, num_shift=num_shift, ensure_min=True
        )
        lstm_seq2seq = LSTMSeq2Seq(
            encoding_dim=best_params['encoding_dim'],
            sequence_length=sequence_length,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            verbose=0,
            es_patience=lstm_es_patience,
            es_min_delta=lstm_es_min_delta
        )
        lstm_seq2seq.fit_on_windows(Xw_train, Yw_train)

        X_train_encoded_trial_mean = encode_trial_means(
            lstm_seq2seq.encoder, train_mats,
            sequence_length=sequence_length, num_shift=num_shift, drop_first_n=drop_first_n
        )
        X_test_encoded_trial_mean = encode_trial_means(
            lstm_seq2seq.encoder, test_mats,
            sequence_length=sequence_length, num_shift=num_shift, drop_first_n=drop_first_n
        )

        mlp = NeuralNetRegressor(
            MLPRegressorModule,
            module__input_dim=best_params['encoding_dim'],
            module__hidden_dim=mlp_hidden_dim,
            max_epochs=mlp_max_epochs,
            lr=mlp_lr,
            batch_size=mlp_batch_size,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=mlp_weight_decay,
            verbose=0,
            device=device,
            train_split=ValidSplit(mlp_train_split_ratio, random_state=seednum),
            callbacks=[
                SEarlyStopping(
                    monitor='valid_loss',
                    patience=mlp_es_patience,
                    threshold=mlp_es_threshold_outer,
                    threshold_mode='rel',
                    lower_is_better=True,
                    load_best=True
                )
            ]
        )
        mlp.fit(
            X_train_encoded_trial_mean.astype(np.float32),
            y_outer_train.ravel().astype(np.float32).reshape(-1, 1)
        )

        y_pred = mlp.predict(X_test_encoded_trial_mean.astype(np.float32)).ravel()
        y_true = y_outer_test.ravel()

        all_val_pred.append(y_pred)
        all_val_true.append(y_true)

        try:
            cc1, _ = pearsonr(y_true, y_pred)
        except Exception:
            cc1 = np.nan
        outer_fold_ccs.append(cc1)
        print(f'Outer Fold {outer_fold + 1} result cc: {cc1:.4f}')

    all_val_pred = np.concatenate(all_val_pred)
    all_val_true = np.concatenate(all_val_true)
    try:
        cc_all, _ = pearsonr(all_val_true, all_val_pred)
    except Exception:
        cc_all = np.nan
    mse_all = mean_squared_error(all_val_true, all_val_pred)

    print("\n" + "=" * 50)
    print("Final evaluation results")
    print("=" * 50)
    print(f"Outer fold cc mean: {np.nanmean(outer_fold_ccs):.4f} ± {np.nanstd(outer_fold_ccs):.4f}")
    print(f"Inner fold MSE mean (best params): {np.nanmean(inner_fold_mses):.4f} ± {np.nanstd(inner_fold_mses):.4f}")
    print(f"Inner fold cc mean (best params): {np.nanmean(inner_fold_ccs):.4f} ± {np.nanstd(inner_fold_ccs):.4f}")
    print(f"All data cvCC: {cc_all:.4f}")
    print(f"All data MSE: {mse_all:.4f}")
    print(f"All data shape: {all_val_true.shape}")

    scipy.io.savemat(dataname + output_suffix, mdict={'label': all_val_true, 'predict': all_val_pred})
