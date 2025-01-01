# ml/model_optimization.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

def create_model_with_hp(self, model_type, input_shape, hp):
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

def perform_hyperparameter_optimization(self, X_train_scaled, y_train, model_type):
    """
    Perform hyperparameter optimization with proper model wrapping
    """
    def create_model(hidden_units, dropout_rate, activation, l2_reg, learning_rate, beta_1, beta_2):
        hp = {
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate,
            'activation': activation,
            'l2_reg': l2_reg,
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2
        }
        return self.create_model_with_hp(model_type, X_train_scaled.shape[1], hp)

    # Define the parameter grid
    param_grid = {
        'hidden_units': [32, 64],
        'dropout_rate': [0.2, 0.3],
        'activation': ['relu'],
        'l2_reg': [0.001],
        'learning_rate': [0.001, 0.01],
        'beta_1': [0.9],
        'beta_2': [0.999],
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

        self.logger.info(f"\nHyperparameter optimization completed in {search_time:.2f} seconds")
        self.logger.info(f"Best {model_type} model parameters: {random_search_result.best_params_}")
        self.logger.info(f"Best {model_type} model score: {random_search_result.best_score_:.4f}")

        # Return the best model and parameters
        best_params = random_search_result.best_params_
        best_model = create_model(
            hidden_units=best_params['hidden_units'],
            dropout_rate=best_params['dropout_rate'],
            activation=best_params['activation'],
            l2_reg=best_params['l2_reg'],
            learning_rate=best_params['learning_rate'],
            beta_1=best_params['beta_1'],
            beta_2=best_params['beta_2']
        )

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
        self.logger.error(f"Error during hyperparameter optimization: {str(e)}")
        # Create model with default parameters
        default_params = {
            'hidden_units': 64,
            'dropout_rate': 0.3,
            'activation': 'relu',
            'l2_reg': 0.001,
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999
        }

        default_model = create_model(
            hidden_units=default_params['hidden_units'],
            dropout_rate=default_params['dropout_rate'],
            activation=default_params['activation'],
            l2_reg=default_params['l2_reg'],
            learning_rate=default_params['learning_rate'],
            beta_1=default_params['beta_1'],
            beta_2=default_params['beta_2']
        )

        # Train the default model
        history = default_model.fit(
            X_train_scaled,
            y_train,
            batch_size=32,
            epochs=50,
            verbose=0
        )

        return default_model, default_params
