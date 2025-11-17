import os

class Config:
    # Environment and Random Seeds
    CUDA_VISIBLE_DEVICES = "0"
    RANDOM_SEED = 3407
    
    # Data Configuration. This example is for image task.
    DATA_FILE = '../data/example_data.mat'
    DATA_FIELDS = {
        'feature': 'feature',
        'arousal': 'pica',
        'poslist': 'poslist'
    }
    FILTER_VALUE = 50
    
    # Cross-Validation Configuration
    OUTER_FOLDS = 10
    INNER_FOLDS = 5
    CV_RANDOM_STATE = 3407
    CV_SHUFFLE = True
    
    # Feature Selection Configuration
    DROP_FIRST_N_POINTS = 10
    FEATURE_SELECTION_K = [50, 100, 200, 300, 400]
    # Sequence Configuration
    SEQUENCE_LENGTH = 5
    NUM_SHIFT = 1
    DROP_FIRST_N = 10
    
    # LSTM Seq2Seq Configuration
    LSTM_ENCODING_DIM = [20, 30, 40, 60, 80]
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    LSTM_ES_PATIENCE = 10
    LSTM_ES_MIN_DELTA = 1e-4
    LSTM_DROPOUT = 0.3
    LSTM_ACTIVATION = 'tanh'
    LSTM_OPTIMIZER = 'adam'
    LSTM_LOSS = 'mse'
    LSTM_VALIDATION_SPLIT = 0.1
    
    # MLP Regressor Configuration
    MLP_MAX_EPOCHS = 100
    MLP_LR = 0.01
    MLP_BATCH_SIZE_TRAIN = 1
    MLP_BATCH_SIZE_TEST = 1
    MLP_WEIGHT_DECAY = 1e-3
    MLP_TRAIN_SPLIT = 0.1
    MLP_TRAIN_SPLIT_RANDOM_STATE = 42
    MLP_TEST_SPLIT_RANDOM_STATE = 3407
    MLP_ES_PATIENCE = 10
    MLP_ES_THRESHOLD = 1e-4
    MLP_ES_THRESHOLD_MODE = 'rel'
    MLP_DROPOUT = 0.3
    MLP_HIDDEN_RATIO = 2
    
    # Output Configuration
    OUTPUT_SUFFIX = '_res.mat'
    OUTPUT_FIELDS = {
        'label': 'label',
        'predict': 'predict'
    }
    
    # Print Configuration
    PRINT_CC_DECIMALS = 4