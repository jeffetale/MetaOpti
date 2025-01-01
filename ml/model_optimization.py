# ml/model_optimization.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import logging
from logging_config import setup_comprehensive_logging

setup_comprehensive_logging()

def create_model_with_hp(model_type, input_shape, hp):
        """
        Create model with hyperparameter optimization support
        """
        # Create optimizer with hyperparameters
        optimizer = Adam(
            learning_rate=hp['learning_rate'],
            beta_1=hp['beta_1'],
            beta_2=hp['beta_2']
        )

        if model_type == "direction":
            model = Sequential([
                Dense(hp['hidden_units'], 
                    activation=hp['activation'],
                    input_shape=(input_shape,),
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(hp['hidden_units'] // 2,
                    activation=hp['activation'],
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])

            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        else:
            model = Sequential([
                Dense(hp['hidden_units'],
                    activation=hp['activation'],
                    input_shape=(input_shape,),
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(hp['hidden_units'] // 2,
                    activation=hp['activation'],
                    kernel_regularizer=tf.keras.regularizers.l2(hp['l2_reg'])),
                BatchNormalization(),
                Dropout(hp['dropout_rate']),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer=optimizer,
                loss="mean_squared_error",
                metrics=["mae"]
            )

        return model

def perform_hyperparameter_optimization(X_train_scaled, y_train, model_type):
    logger = logging.getLogger(__name__)
    
    def create_model():
        if model_type == "direction":
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            model.compile(
                optimizer='adam',
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        else:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(16, activation="relu"),
                Dense(1)
            ])
            model.compile(
                optimizer='adam',
                loss="mean_squared_error",
                metrics=["mae"]
            )
        return model

    # Define the parameter grid
    param_grid = {
        'batch_size': [32, 64],
        'epochs': [30, 50]
    }

    # Create the correct wrapper
    if model_type == "direction":
        model_wrapper = KerasClassifier(
            model=create_model,
            verbose=0
        )
        scoring = 'accuracy'
    else:
        model_wrapper = KerasRegressor(
            model=create_model,
            verbose=0
        )
        scoring = 'neg_mean_absolute_error'

    try:
        # Initialize and run RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model_wrapper,
            param_distributions=param_grid,
            n_iter=5,
            cv=3,
            scoring=scoring,
            n_jobs=1,
            verbose=1,
            return_train_score=True,
            random_state=42
        )

        search_start_time = time.time()
        random_search_result = random_search.fit(X_train_scaled, y_train)
        search_time = time.time() - search_start_time

        logger.info(f"\nHyperparameter optimization completed in {search_time:.2f} seconds")
        logger.info(f"Best {model_type} model parameters: {random_search_result.best_params_}")
        logger.info(f"Best {model_type} model score: {random_search_result.best_score_:.4f}")

        # Get the best model
        best_model = create_model()
        best_params = random_search_result.best_params_
        
        # Train the best model with the best parameters
        history = best_model.fit(
            X_train_scaled,
            y_train,
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            verbose=0
        )

        return best_model, best_params

    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        # Create and train model with default parameters
        default_model = create_model()
        default_params = {
            'batch_size': 32,
            'epochs': 50
        }

        history = default_model.fit(
            X_train_scaled,
            y_train,
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            verbose=0
        )

        return default_model, default_params
