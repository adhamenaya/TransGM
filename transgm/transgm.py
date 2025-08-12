# -----------------------------------------------------------------------------
# Author: Adham Enaya
# PhD Supervisors: Prof. Chen Zhong and Prof. Micheal Batty
# Affiliation: Centre for Advanced Spatial Analysis, UCL
# Project Title: TransGM: Transferable Gravity Models for Adaptive Urban Policy
# Funding: European Research Council (ERC) under the EU Horizon 2020 program (No. 949670)
# Date: 2025-08-10
# -----------------------------------------------------------------------------
from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
import models.utils
import models.spatial
import numpy as np
import pandas as pd
import utils
import time
import copy

from data import od, poi
from sklearn.preprocessing import MinMaxScaler
import math
scaler = MinMaxScaler()
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import KFold
from scipy.optimize import minimize
import pandas as pd
from collections import defaultdict
import numpy as np
import uuid
from dataset import DataSet
from adaptfunc import AdaptFunc

class TransGM(BaseEstimator, TransformerMixin):
    """
    A machine learning model for flow prediction based on Gravity Model.

    y represents the origin-destination (OD) matrix,
    X consists of:
      - Origin features vector
      - Destination features matrix
      - Distance matrix

    Model parameters (gravity model):
      - beta: Distance decay parameter
      - gamma: Origin attractiveness parameter
      - alpha: Destination attractiveness parameter
    """

    def __init__(self, beta=1.0, gamma=1.0, alpha=1.0, decay_func='exp'):
        """Initialize the model with gravity model parameters."""
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

        self.decay_func = decay_func
        self.src_dataset_obj = None
        self.tgt_dataset_obj = None

        # Default training hyperparameters
        self.training_param_grid = {
            'lambda': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [100, 200],  # Optimization iterations
            'init_strategy': ['random', 'uniform_positive']  # Different initialization strategies
        }

        self.src_model_results = None
        self.trans_model_results = None
        self.adapt_model_results = None

        self.source = None
        self.target = None
        self.divs = {}
        self.datasets = {}

    # Define gradients for optimization
    def gradient(self, params, data, lambda_=0.1):
        X0, X1, X2, y = data
        # Compute gradient of MSE Loss w.r.t gamma
        def grad_mse_loss_gamma(gamma, X1, X2, beta, y, alpha, X0, e):
            predictions = self.pred(gamma, X1, X2, beta, alpha, X0, e)
            grad_f_gamma = X1 / np.expand_dims(X1 @ gamma, axis=-1)

            grad_loss_gamma = np.mean((predictions - y)[:, :, np.newaxis] * grad_f_gamma, axis=(0,1))

            return grad_loss_gamma

        # Compute gradient of MSE Loss w.r.t beta
        def grad_mse_loss_beta(gamma, X1, X2, beta, y, alpha, X0, e):
            predictions = self.pred(gamma, X1, X2, beta, alpha, X0, e)
            if self.decay_func == "exp":
                grad_f_beta = -X2  # Gradient of -beta * X2 with respect to beta
            else:
                grad_f_beta = -np.log(X2)  # Gradient of -beta * log(X2) with respect to beta
            grad_loss_beta = np.mean((predictions - y) * grad_f_beta)
            return grad_loss_beta

        # Compute gradient of MSE Loss w.r.t alpha
        def grad_mse_loss_alpha(gamma, X1, X2, beta, y, alpha, X0, e):
            predictions = self.pred(gamma, X1, X2, beta, alpha, X0, e)
            grad_f_alpha = np.log(X0)  # Gradient of alpha * log(X0) with respect to alpha
            grad_loss_alpha = np.mean((predictions - y) * grad_f_alpha)
            return grad_loss_alpha

        # Compute gradient of MSE Loss w.r.t e
        def grad_mse_loss_e(gamma, X1, X2, beta, y, alpha, X0, e):
            predictions = self.pred(gamma, X1, X2, beta, alpha, X0, e)
            grad_f_e = np.ones_like(X2)  # Gradient of e with respect to e
            grad_loss_e = np.mean((predictions - y) * grad_f_e)
            return grad_loss_e

        # Define hyperparameter grid
        alpha, beta, *gamma_vals, e = params
        gamma = np.array(gamma_vals)

        # Using your existing gradient functions
        grad_gamma = grad_mse_loss_gamma(gamma, X1, X2, beta, y, alpha, X0, e) + 2 * lambda_ * gamma
        grad_beta = grad_mse_loss_beta(gamma, X1, X2, beta, y, alpha, X0, e) + 2 * lambda_ * beta
        grad_alpha = grad_mse_loss_alpha(gamma, X1, X2, beta, y, alpha, X0, e) + 2 * lambda_ * alpha
        grad_e = grad_mse_loss_e(gamma, X1, X2, beta, y, alpha, X0, e) + 2 * lambda_ * e

        return np.hstack([grad_alpha, grad_beta, grad_gamma, grad_e])

    # Prediction function
    def pred(self, gamma, X1, X2, beta, alpha, X0, e=0):
        gamma = np.array(gamma).reshape(-1, 1)  # Ensure gamma is a column vector
        if self.decay_func == "exp":
            return utils.sigmoid(alpha * np.log(X0) + np.log(np.sum(X1 * gamma.T, axis=2)) - beta * X2 + e)
        else:
            return utils.sigmoid(alpha * np.log(X0) + np.log(np.sum(X1 * gamma.T, axis=2)) - beta * np.log(X2) + e)

    def loss(self, params, data, lambda_, log, uuid):
        alpha, beta, *gamma_vals, e = params
        X0, X1, X2, y = data
        gamma = np.array(gamma_vals)

        predictions = self.pred(gamma, X1, X2, beta, alpha, X0, e)

        # L2 regularization + MSE
        pred_error = (
                np.mean((y - predictions) ** 2) +
                lambda_ * np.sum(gamma**2) +
                lambda_ * alpha**2 +
                lambda_ * beta**2 +
                lambda_ * e**2
        )

        # Log parameters if requested
        if log is not None:
            if uuid not in log:
                log[uuid] = []
            log[uuid].append({
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma.copy(),
                'e': e,
                'mse': pred_error
            })

        return pred_error

    def load_data(self, dataset):
        """Load dataset for a specific city."""
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        if not dataset.city:
            raise ValueError("City name cannot be empty")

        # Initialize datasets dict if it doesn't exist
        if not hasattr(self, 'datasets'):
            self.datasets = {}

        self.datasets[dataset.city] = dataset
        print(f"Loaded dataset for {dataset.city}")

    def fit(self, source_city, serial=0, model_sample=1.0):
        self.src_dataset_obj = self.datasets[source_city]
        self.source = source_city

        uuid_val = uuid.uuid4().hex[:8]
        params_log = {}
        run_id = f"{serial}_{uuid_val}"

        print(f"Baseline model {source_city} - Run# {run_id}\n")
        # Get data
        kfX, kfy = self.src_dataset_obj.od_data()

        # For storing results
        cv_results = defaultdict(list)
        best_params = {}
        best_score = -np.inf
        best_model_params = None

        # Nested cross-validation setup
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # ====== Cross-validation ========
        # Outer CV for evaluating generalization performance
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(kfX)):
            print(f"\n--- Outer Fold {outer_fold+1}/5 ---")
            X_train_outer, X_test = kfX.iloc[train_idx], kfX.iloc[test_idx]
            y_train_outer, y_test = kfy[train_idx], kfy[test_idx]

            # Inner CV for hyperparameter tuning
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

            # Track the best parameters for this outer fold
            fold_best_score = -np.inf
            fold_best_params = None

            # Grid search through parameter combinations
            for lambda_ in self.training_param_grid['lambda']:
                for max_iter in self.training_param_grid['max_iter']:
                    for init_strategy in self.training_param_grid['init_strategy']:
                        # Store validation scores for this parameter combination
                        val_scores = []

                        print(f"Evaluating: lambda={lambda_}, max_iter={max_iter}, init={init_strategy}")

                        # Inner CV for hyperparameter tuning
                        for inner_fold, (train_inner_idx, val_idx) in enumerate(inner_cv.split(X_train_outer)):
                            # Get the training and validation sets
                            X_train = X_train_outer.iloc[train_inner_idx]
                            y_train = y_train_outer[train_inner_idx]
                            X_val = X_train_outer.iloc[val_idx]
                            y_val = y_train_outer[val_idx]

                            # Prepare training data for model
                            combined_train = X_train.copy()
                            combined_train['trips'] = y_train
                            combined_train = pd.DataFrame(combined_train)
                            combined_train.set_index([combined_train.columns[0], combined_train.columns[1]], inplace=True)

                            X0, X1, X2, y = self.src_dataset_obj.transform_to_matrices(pd.DataFrame(combined_train))

                            # Initialize model parameters based on strategy
                            if init_strategy == 'random':
                                gamma_init = np.random.rand(11)
                                alpha_init, beta_init, e_init = np.random.rand(), np.random.rand(), np.random.rand()
                            else:  # uniform_positive
                                gamma_init = np.ones(11) * 0.5
                                alpha_init, beta_init, e_init = 0.5, 0.5, 0.5

                            # Initial parameters
                            initial_params = np.hstack([alpha_init, beta_init, gamma_init, e_init])


                            # Define bounds for parameters
                            bounds = [(1e-6, None) for _ in range(len(initial_params))]

                            # Optimize with L-BFGS-B
                            result = minimize(
                                self.loss,
                                initial_params,
                                args=((X0, X1, X2, y), lambda_, params_log, run_id),  # <-- fixed order
                                jac=lambda params, data, lambda_, log, uuid: self.gradient(params, data=data, lambda_=lambda_),
                                method="L-BFGS-B",
                                bounds=bounds,
                                options={
                                    'maxiter': max_iter,
                                    'ftol': 1e-8,
                                    'gtol': 1e-8,
                                    'maxls': 20
                                }
                            )

                            # Extract optimized parameters
                            alpha_opt, beta_opt, *gamma_vals_opt, e_opt = result.x
                            gamma_opt = np.array(gamma_vals_opt)

                            # Evaluate on validation set
                            combined_val = X_val.copy()
                            combined_val['trips'] = y_val
                            combined_val = pd.DataFrame(combined_val)
                            combined_val.set_index([combined_val.columns[0], combined_val.columns[1]], inplace=True)
                            X0_val, X1_val, X2_val, y_val_matrix = self.src_dataset_obj.transform_to_matrices(pd.DataFrame(combined_val))

                            # Get predictions
                            val_pred = self.pred(gamma_opt, X1_val, X2_val, beta_opt, alpha_opt, X0_val, e_opt)

                            # Calculate R² on validation set
                            val_r2 = r2_score(y_val_matrix.flatten(), val_pred.flatten())
                            val_scores.append(val_r2)

                            print(f"Inner fold {inner_fold+1}: val_r2={val_r2:.4f}")

                        # Average validation score for this parameter combination
                        mean_val_score = np.mean(val_scores)
                        cv_results['lambda'].append(lambda_)
                        cv_results['max_iter'].append(max_iter)
                        cv_results['init_strategy'].append(init_strategy)
                        cv_results['val_r2'].append(mean_val_score)

                        print(f"Mean validation R² = {mean_val_score:.4f}")

                        # Update the best parameters if this combination is better
                        if mean_val_score > fold_best_score:
                            fold_best_score = mean_val_score
                            fold_best_params = {
                                'lambda': lambda_,
                                'max_iter': max_iter,
                                'init_strategy': init_strategy
                            }

            # Train final model on entire outer training set with best parameters
            print(f"\nTraining final model with params: {fold_best_params}")
            lambda_ = fold_best_params['lambda']
            max_iter = fold_best_params['max_iter']
            init_strategy = fold_best_params['init_strategy']

            # Prepare full training data
            combined_train = X_train_outer.copy()
            combined_train['trips'] = y_train_outer
            combined_train = pd.DataFrame(combined_train)
            combined_train.set_index([combined_train.columns[0],combined_train.columns[1]], inplace=True)
            X0, X1, X2, y = self.src_dataset_obj.transform_to_matrices(pd.DataFrame(combined_train))

            # Initialize based on best strategy
            if init_strategy == 'random':
                gamma_init = np.random.rand(11)
                alpha_init, beta_init, e_init = np.random.rand(), np.random.rand(), np.random.rand()
            else:  # uniform_positive
                gamma_init = np.ones(11) * 0.5
                alpha_init, beta_init, e_init = 0.5, 0.5, 0.5

            # Initial parameters
            initial_params = np.hstack([alpha_init, beta_init, gamma_init, e_init])

            # Define bounds for parameters
            bounds = [(1e-6, None) for _ in range(len(initial_params))]

            # Optimize with the best parameters
            log = True
            result = minimize(
                self.loss,
                initial_params,
                args=((X0, X1, X2, y), lambda_, params_log, run_id),  # <-- fixed order
                jac=lambda params, data, lambda_, log, uuid: self.gradient(params, data=data, lambda_=lambda_),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    'maxiter': max_iter,
                    'ftol': 1e-8,
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )

            alpha_opt, beta_opt, *gamma_vals_opt, e_opt = result.x
            gamma_opt = np.array(gamma_vals_opt)

            # Evaluate on test set
            combined_test = X_test.copy()
            combined_test['trips'] = y_test
            combined_test = pd.DataFrame(combined_test)
            combined_test.set_index([combined_test.columns[0],combined_test.columns[1]], inplace=True)
            X0_test, X1_test, X2_test, y_test_matrix = self.src_dataset_obj.transform_to_matrices(pd.DataFrame(combined_test))

            # Get predictions
            test_pred = self.pred(gamma_opt, X1_test, X2_test, beta_opt, alpha_opt, X0_test, e_opt)

            # Calculate R² on test set
            test_r2 = r2_score(y_test_matrix.flatten(), test_pred.flatten())
            print(f"Test R² = {test_r2:.4f}")

            # Store this fold's results
            cv_results['outer_fold'].append(outer_fold)
            cv_results['best_params'].append(fold_best_params)
            cv_results['test_r2'].append(test_r2)

            # Update overall best model if this one is better
            if test_r2 > best_score:
                best_score = test_r2
                best_params = fold_best_params
                best_model_params = {'alpha': alpha_opt, 'beta': beta_opt, 'gamma': gamma_opt, 'e': e_opt}

        # Print overall best parameters
        print("\n==== Cross-Validation Summary ====")
        print(f"Best parameters: {best_params}")
        print(f"Best test R²: {best_score:.4f}")

        # Now train the final model on the entire dataset with best parameters
        print("\nTraining final model on entire dataset...")
        params_log = {}

        X_all, y_all = self.src_dataset_obj.od_data()
        combined_all = X_all.copy()
        combined_all['trips'] = y_all

        # Transform the dataset into matrices
        combined_all = pd.DataFrame(combined_all)
        combined_all.set_index([combined_all.columns[0],combined_all.columns[1]], inplace=True)
        X0, X1, X2, y = self.src_dataset_obj.transform_to_matrices(pd.DataFrame(combined_all))

        # Initialize based on best strategy
        if best_params['init_strategy'] == 'random':
            gamma_init = np.random.rand(11)
            alpha_init, beta_init, e_init = np.random.rand(), np.random.rand(), np.random.rand()
        else:  # uniform_positive
            gamma_init = np.ones(11) * 0.5
            alpha_init, beta_init, e_init = 0.5, 0.5, 0.5

        # Initial parameters
        initial_params = np.hstack([alpha_init, beta_init, gamma_init, e_init])

        # Optimize with the best parameters
        lambda_ = best_params['lambda']
        max_iter = best_params['max_iter']

        # Define bounds for parameters
        bounds = [(1e-6, None) for _ in range(len(initial_params))]

        result = minimize(
            self.loss,
            initial_params,
            args=((X0, X1, X2, y), lambda_, params_log, run_id),  # <-- fixed order
            jac=lambda params, data, lambda_, log, run_id: self.gradient(params, data=data, lambda_=lambda_),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-8,
                'gtol': 1e-8,
                'maxls': 20
            }
        )

        # Save logged parameters to a file
        utils.save_params_log(params_log, run_id)

        alpha_opt, beta_opt, *gamma_vals_opt, e_opt = result.x
        gamma_opt = np.array(gamma_vals_opt)

        # Calculate predictions on full dataset
        pred_trips = self.pred(gamma_opt, X1, X2, beta_opt, alpha_opt, X0, e_opt)
        r2 = r2_score(y.flatten(), pred_trips.flatten())

        # Compute outflows (origins)
        pred_orig = np.sum(pred_trips, axis=1).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        pred_orig = pred_orig / np.sum(pred_orig)
        y_orig = np.sum(y, axis=1).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        y_orig = y_orig / np.sum(y_orig)
        out_r2 = r2_score(y_orig.flatten(), pred_orig.flatten())

        # Compute inflows (destinations)
        pred_dest = np.sum(pred_trips, axis=0).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        pred_dest = pred_dest / np.sum(pred_dest)
        y_dest = np.sum(y, axis=0).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        y_dest = y_dest / np.sum(y_dest)
        in_r2 = r2_score(y_dest.flatten(), pred_dest.flatten())


        # Output final results
        mae = np.sum(np.abs(pred_dest - y_dest)) / self.src_dataset_obj.size[0]
        mse = np.sum((pred_dest - y_dest) ** 2) / self.src_dataset_obj.size[0]
        print(f"-------------------------------------")
        print(f"Final model R²: {r2:.4f}")
        # print(f"Final model MAE: {mae:.4f}")
        print(f"Final model MSE: {mse:.4f}")
        print(f"-------------------------------------")

        # Return model results
        self.src_model_results = ModelResult(
            uuid = uuid_val,
            model_name="Baseline_Model",
            serial=serial,
            model_sample=model_sample,
            decay_function=self.decay_func,
            source=self.source,
            target=None,
            init_beta=beta_init,
            init_alpha=alpha_init,
            init_gamma=gamma_init,
            init_e=e_init,
            optimal_beta=beta_opt,
            optimal_alpha=alpha_opt,
            optimal_gamma=gamma_opt,
            optimal_e=e_opt,
            rmse=mse,
            r2=r2,
            in_r2=in_r2,
            out_r2=out_r2,
            dest_mae=mae,
            dest_mse=mse,
            cv_best_params=best_params,
            cv_best_score=best_score,
            pred_trips=pred_trips,
            dist_matrix=X2,
            dest_result=[y_dest, pred_dest],
            orig_result=[y_orig, pred_orig]
        )

        return self # self at this point represent the trained model

    def transfer(self, target_city, serial=0, model_sample=1.0):
        self.target = target_city
        self.tgt_dataset_obj = self.datasets[target_city]

        uuid_val = uuid.uuid4().hex[:8]
        run_id = f"{serial}_{uuid_val}"
        print(f"Naive transfer model {target_city} - Run# {run_id}\n")

        # Now test the final model on the entire dataset with best parameters
        print("\nTesting final model on another city dataset...")

        X_all, y_all = self.tgt_dataset_obj.od_data()
        combined_all = X_all.copy()
        combined_all['trips'] = y_all
        combined_all = pd.DataFrame(combined_all)
        combined_all.set_index([combined_all.columns[0],combined_all.columns[1]], inplace=True)
        # Transform the dataset into matrices
        X0, X1, X2, y = self.tgt_dataset_obj.transform_to_matrices(pd.DataFrame(combined_all))

        params_log = {}
        if run_id not in params_log:
            params_log[run_id] = []

        # No parameter tuning, use the best parameters from the source model
        alpha_opt, beta_opt, *gamma_vals_opt, e_opt = (self.src_model_results.optimal_alpha,
                                                       self.src_model_results.optimal_beta,
                                                       self.src_model_results.optimal_gamma,
                                                       self.src_model_results.optimal_e)

        gamma_opt = np.array(gamma_vals_opt)
        # Log the parameters
        params_log[run_id].append({
            'alpha': alpha_opt,
            'beta': beta_opt,
            'gamma': gamma_opt,
            'e': e_opt
        })

        # Save logged parameters to a file
        utils.save_params_log(params_log, run_id)

        # Calculate predictions on full dataset
        pred_trips = self.pred(gamma_opt, X1, X2, beta_opt, alpha_opt, X0, e_opt)
        r2 = r2_score(y.flatten(), pred_trips.flatten())

        # Compute outflows (origins)
        pred_orig = np.sum(pred_trips, axis=1).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        pred_orig = pred_orig / np.sum(pred_orig)
        y_orig = np.sum(y, axis=1).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        y_orig = y_orig / np.sum(y_orig)
        out_r2 = r2_score(y_orig.flatten(), pred_orig.flatten())

        # Compute inflows (destinations)
        pred_dest = np.sum(pred_trips, axis=0).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        pred_dest = pred_dest / np.sum(pred_dest)
        y_dest = np.sum(y, axis=0).reshape(int(np.sqrt(y.shape[0])), int(np.sqrt(y.shape[0])))
        y_dest = y_dest / np.sum(y_dest)
        in_r2 = r2_score(y_dest.flatten(), pred_dest.flatten())

        mae = np.sum(np.abs(pred_dest - y_dest)) / self.tgt_dataset_obj.size[0]
        mse = np.sum((pred_dest - y_dest) ** 2) / self.tgt_dataset_obj.size[0]

        # Output final results
        print(f"-------------------------------------")
        print(f"Final model R²: {r2:.4f}")
        print(f"Final model MAE: {mae:.4f}")
        print(f"Final model MSE: {mse:.4f}")
        print(f"-------------------------------------")

        # # 1. Prediction error visualization
        # # heatmap of real and predicted inflow (destinations)
        # print("Prediction Error Analysis:")
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # axes[0].imshow(y_dest, interpolation='none', origin='lower')
        # axes[0].set_title('real_destinations')
        # axes[0].axis('off')
        #
        # axes[1].imshow(pred_dest, interpolation='none', origin='lower')
        # axes[1].set_title('pred_destinations')
        # axes[1].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        # Create a deep copy to avoid modifying the original object
        updated_results = copy.deepcopy(self.src_model_results)

        # Update fields
        updated_results.uuid = uuid_val
        updated_results.model_name = f"Naive_Transfer_{target_city}"
        updated_results.target = target_city
        updated_results.model_sample = model_sample
        updated_results.rmse = mse
        updated_results.r2 = r2
        updated_results.in_r2 = in_r2
        updated_results.out_r2 = out_r2
        updated_results.pred_trips = pred_trips
        updated_results.dist_matrix = X2
        updated_results.dest_result = [y_dest, pred_dest]
        updated_results.orig_result = [y_orig, pred_orig]
        self.trans_model_results = updated_results
        return self

    def adapt(self, target_city, adapt_func='log_penalty', serial=0, model_sample=0.5):
        self.target = target_city
        self.tgt_dataset_obj = self.datasets[target_city]

        uuid_val = uuid.uuid4().hex[:8]
        params_log = {}
        run_id = f"{serial}_{uuid_val}"
        print(f"Adapt model {target_city} - Run# {run_id}\n")

        # Now test the final model on the entire dataset with best parameters
        print("\nTesting final model on another city dataset...")

        # Target city data / transfer learning
        X_all, y_all = self.tgt_dataset_obj.od_data()
        combined_all = X_all.copy()
        combined_all['trips'] = y_all
        combined_all = pd.DataFrame(combined_all)
        combined_all.set_index([combined_all.columns[0],combined_all.columns[1]], inplace=True)
        # Transform the dataset into matrices
        X0_tgt, X1_tgt, X2_tgt, y_tgt = self.tgt_dataset_obj.transform_to_matrices(pd.DataFrame(combined_all))

        # Estimate the baseline model on the source city
        # === Print Results ===
        print(f"\n=== Source: {self.source} Results ===")
        print(f"Optimal beta: {round(self.src_model_results.optimal_beta, 5)}")
        print(f"Optimal alpha: {round(self.src_model_results.optimal_alpha, 5)}")
        print(f"Optimal gamma: {[round(a, 5) for a in np.round(self.src_model_results.optimal_gamma, 5)]}")
        print(f"Optimal e: {round(self.src_model_results.init_e, 5)}")
        print(f"\nSource RMSE: {self.src_model_results.rmse:.4f}")
        print(f"Source R²: {self.src_model_results.r2:.4f}")

        # Initialize params_log for this run
        params_log = {}
        if run_id not in params_log:
            params_log[run_id] = []

        # Define transfer learning loss function with regularization
        def trans_loss(params, X1, X2, X0, y_trans, k, gamma_opt, beta_opt, alpha_opt, e_opt, lambda_reg=0.1):
            alpha, beta, *gamma_vals = params[:-1]
            e = params[-1]
            gamma = np.array(gamma_vals)

            # Make predictions
            predictions = self.pred(gamma, X1, X2, beta, alpha, X0, e)

            # Calculate prediction error (MSE)
            # prediction_error = np.mean((y_trans - predictions) ** 2)
            prediction_error = np.mean(np.abs(y_trans - predictions))

            # Log the parameters
            params_log[run_id].append({
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'e': e,
                'mse': prediction_error
            })

            # Calculate regularization terms with feature-specific penalties
            divs = self.get_divs()
            if divs is None or len(divs) == 0:
                raise ValueError("You must provide feature divergence first")

            gamma_penalty_factors = getattr(AdaptFunc, adapt_func)(divs, k=k)
            # print(f"Penalty factors: {gamma_penalty_factors}")
            # L2 regularization on parameter differences from source model
            gamma_reg = np.sum(gamma_penalty_factors * np.abs(gamma - gamma_opt) ** 2)

            beta_reg = lambda_reg * np.abs(beta - beta_opt) ** 2
            alpha_reg = lambda_reg * np.abs(alpha - alpha_opt) ** 2
            e_reg = lambda_reg * np.abs(e - e_opt) ** 2

            # Total loss with regularization
            return (prediction_error
                    + gamma_reg
                    + beta_reg
                    + alpha_reg
                    + e_reg
                    )
            # End of  trans_loss function

        # Set up cross-validation to optimize k (penalty factor)
        best_k = None
        best_lambda = None
        best_r2 = -np.inf

        # Define parameter grid for cross-validation
        k_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]

        # Setup cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Convert to numpy arrays for easier splitting
        X_tgt_index = np.arange(len(y_tgt))

        # Store CV results
        cv_results = defaultdict(list)

        # ===== Cross-validation for hyperparameter tuning =====
        print("Starting cross-validation for hyperparameter tuning...")
        for k in k_values:
            for lambda_reg in lambda_values:
                fold_scores = []

                for train_idx, val_idx in kf.split(X_tgt_index):
                    # Split target data into train and validation sets
                    X0_tgt_train, X0_tgt_val = X0_tgt[train_idx], X0_tgt[val_idx]
                    X1_tgt_train, X1_tgt_val = X1_tgt[train_idx], X1_tgt[val_idx]
                    X2_tgt_train, X2_tgt_val = X2_tgt[train_idx], X2_tgt[val_idx]
                    y_tgt_train, y_tgt_val = y_tgt[train_idx], y_tgt[val_idx]

                    trans_init_params = np.hstack([self.src_model_results.optimal_alpha, self.src_model_results.optimal_beta, self.src_model_results.optimal_gamma, self.src_model_results.optimal_e])

                    # Define bounds to ensure parameters remain positive
                    bounds = [(1e-6, None) for _ in range(len(trans_init_params))]

                    # Optimize with current hyperparameters
                    try:
                        print("optimal_gamma:", np.shape(self.src_model_results.optimal_gamma))
                        transfer_result = minimize(
                            trans_loss,
                            trans_init_params,
                            args=(X1_tgt_train, X2_tgt_train, X0_tgt_train, y_tgt_train, k,
                                  self.src_model_results.optimal_gamma,
                                  self.src_model_results.optimal_beta,
                                  self.src_model_results.optimal_alpha,
                                  self.src_model_results.optimal_e, lambda_reg),
                            bounds=bounds,
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
                        )
                    except Exception as e:
                        print("Shapes:")
                        print("trans_init_params:", np.shape(trans_init_params))
                        print("optimal_gamma:", np.shape(self.src_model_results.optimal_gamma))
                        print("lambda_reg:", lambda_reg)
                        raise

                    # Extract optimized parameters
                    alpha_val, beta_val, *gamma_vals, e_val = transfer_result.x
                    gamma_val = np.array(gamma_vals)

                    # Evaluate on validation set
                    val_pred = self.pred(gamma_val, X1_tgt_val, X2_tgt_val, beta_val, alpha_val, X0_tgt_val, e_val)
                    val_r2 = r2_score(y_tgt_val.flatten(), val_pred.flatten())

                    fold_scores.append(val_r2)

                # Calculate mean R² for this fold
                print(f"Mean R² for k={k}, lambda={lambda_reg}: {np.mean(fold_scores):.4f}")

                # Calculate mean R² across folds
                mean_r2 = np.mean(fold_scores)
                cv_results['k'].append(k)
                cv_results['lambda'].append(lambda_reg)
                cv_results['mean_r2'].append(mean_r2)

                print(f"k={k}, lambda={lambda_reg}: Mean validation R²={mean_r2:.4f}")

                # Update the best parameters if this combination is better
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_k = k
                    best_lambda = lambda_reg

        print(f"\n*********Best hyperparameters: k={best_k}, lambda={best_lambda}, R²={best_r2:.4f}")

        # Train final model on all target data with best hyperparameters
        print("\nTraining final model with best hyperparameters...")

        trans_init_params = np.hstack([self.src_model_results.optimal_alpha, self.src_model_results.optimal_beta, self.src_model_results.optimal_gamma, self.src_model_results.optimal_e])
        bounds = [(1e-6, None) for _ in range(len(trans_init_params))]

        # Optimize the target model with the best hyperparameters and parameters from the source model as initialization
        final_result = minimize(
            trans_loss,
            trans_init_params,
            args=(X1_tgt, X2_tgt, X0_tgt, y_tgt, best_k, self.src_model_results.optimal_gamma, self.src_model_results.optimal_beta, self.src_model_results.optimal_alpha, self.src_model_results.optimal_e, best_lambda),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
        )

        # Save logged parameters to a file
        utils.save_params_log(params_log, run_id)

        # Extract final optimized parameters
        alpha_trans_opt, beta_trans_opt, *gamma_trans_vals, e_trans_opt = final_result.x
        gamma_trans_opt = np.array(gamma_trans_vals)

        print("\n=== Adapt Results ===")
        print(f"Trans beta: {round(beta_trans_opt, 5)}")
        print(f"Trans alpha: {round(alpha_trans_opt, 5)}")
        print(f"Trans gamma: {[round(a, 5) for a in np.round(gamma_trans_opt, 5)]}")
        print(f"Trans e: {round(e_trans_opt, 5)}")

        print(f"Final Loss: {final_result.fun}")
        print(f"Success: {final_result.success}")
        print(f"Iterations: {final_result.nit}")

        final_pred = self.pred(gamma_trans_opt, X1_tgt, X2_tgt, beta_trans_opt, alpha_trans_opt, X0_tgt, e_trans_opt)
        final_r2 = r2_score(y_tgt.flatten(), final_pred.flatten())
        final_rmse = mean_squared_error(y_tgt.flatten(), final_pred.flatten())
        print(f"\nRMSE: {final_rmse:.4f}")
        print(f"Adapted R²: {final_r2:.4f}")

        # Compute outflows (origins)
        final_pred_orig = np.sum(final_pred, axis=1).reshape(int(np.sqrt(y_tgt.shape[0])), int(np.sqrt(y_tgt.shape[0])))
        final_pred_orig = final_pred_orig / np.sum(final_pred_orig)
        final_y_orig = np.sum(y_tgt, axis=1).reshape(int(np.sqrt(y_tgt.shape[0])), int(np.sqrt(y_tgt.shape[0])))
        final_y_orig = final_y_orig / np.sum(final_y_orig)
        final_out_r2 = r2_score(final_y_orig.flatten(), final_pred_orig.flatten())

        # Compute inflows (destinations)
        final_pred_dest = np.sum(final_pred, axis=0).reshape(int(np.sqrt(y_tgt.shape[0])), int(np.sqrt(y_tgt.shape[0])))
        final_pred_dest = final_pred_dest / np.sum(final_pred_dest)
        final_y_dest = np.sum(y_tgt, axis=0).reshape(int(np.sqrt(y_tgt.shape[0])), int(np.sqrt(y_tgt.shape[0])))
        final_y_dest = final_y_dest / np.sum(final_y_dest)
        final_in_r2 = r2_score(final_y_dest.flatten(), final_pred_dest.flatten())


        # Return model results
        self.adapt_model_results = ModelResult(
            uuid=uuid_val,
            model_name="Adapt_Model",
            serial=serial,
            model_sample=model_sample,
            decay_function=self.decay_func,
            source=self.source,
            target=self.target,
            init_beta=self.src_model_results.optimal_beta,
            init_alpha=self.src_model_results.optimal_alpha,
            init_gamma=self.src_model_results.optimal_gamma,
            init_e=self.src_model_results.optimal_e,
            optimal_beta=beta_trans_opt,
            optimal_alpha=alpha_trans_opt,
            optimal_gamma=gamma_trans_opt,
            optimal_e=e_trans_opt,
            rmse=final_rmse,
            r2=final_r2,
            in_r2=final_in_r2,
            out_r2=final_out_r2,
            dest_mae=0,
            dest_mse=0,
            cv_best_params={'k': best_k, 'lambda': best_lambda},
            cv_best_score=best_r2,
            pred_trips=final_pred,
            dist_matrix=X2_tgt,
            dest_result = [final_y_dest, final_pred_dest],
            orig_result = [final_y_orig, final_pred_orig]
        )
        return self

    def load_divs(self, divs):
        """ Load divergence """
        self.divs = divs
        return self

    def get_divs(self):
        return np.array(list(map(float, self.divs.values())))

    def results(self, model):
        if model == 'base':
            return self.src_model_results
        if model == 'trans':
            return self.trans_model_results
        if model == 'adapt':
            return self.adapt_model_results

class ModelResult:
    def __init__(self, model_name, serial, model_sample, decay_function, source, target, init_beta, init_alpha, init_gamma, init_e, optimal_beta, optimal_alpha, optimal_gamma, optimal_e, rmse, r2,
                 in_r2, out_r2, dest_mae, dest_mse, cv_best_params=None, cv_best_score=None, uuid=None, pred_trips=None, dist_matrix=None, dest_result=None, orig_result=None):
        self.uuid = uuid
        self.model_name = model_name
        self.serial = serial
        self.model_sample = model_sample  # float
        self.decay_function = decay_function  # 'exp' or 'pow'
        self.source = source  # 'bir' or 'cov'
        self.target = target  # 'bir' or 'cov'
        self.init_beta = init_beta
        self.init_alpha = init_alpha
        self.init_gamma = init_gamma
        self.init_e = init_e
        self.optimal_beta = optimal_beta
        self.optimal_alpha = optimal_alpha
        self.optimal_gamma = optimal_gamma
        self.optimal_e = optimal_e
        self.rmse = rmse
        self.r2 = r2
        self.in_r2 = in_r2
        self.out_r2 = out_r2
        self.dest_mae = dest_mae
        self.dest_mse = dest_mse

        # baseline cross-validation results
        self.cv_best_params = cv_best_params
        self.cv_best_score = cv_best_score
        self.pred_trips = pred_trips
        self.dist_matrix = dist_matrix
        self.dest_result = dest_result
        self.orig_result = orig_result

    def __repr__(self):
        return (f"ModelResult(model_name={self.model_name}, serial={self.serial}, "
                f"model_sample={self.model_sample}, decay_function={self.decay_function}, "
                f"source={self.source}, target={self.target}, "
                f"init_beta={self.init_beta}, init_alpha={self.init_alpha}, init_gamma={self.init_gamma}, init_e={self.init_e}, "
                f"optimal_beta={self.optimal_beta}, optimal_alpha={self.optimal_alpha}, optimal_gamma={self.optimal_gamma}, optimal_e={self.optimal_e}, "
                f"rmse={self.rmse}, r2={self.r2}, dest_mae={self.dest_mae}, dest_mse={self.dest_mse}) \n cv_best_params={self.cv_best_params}, cv_best_score={self.cv_best_score}")

    def to_dataframe(self):
        data = {
            "model_name": [self.model_name],
            "serial": [self.serial],
            "model_sample": [self.model_sample],
            "decay_function": [self.decay_function],
            "source": [self.source],
            "target": [self.target],
            "init_beta": [self.init_beta],
            "init_alpha": [self.init_alpha],
        }

        # Add init_gamma columns
        for i, val in enumerate(self.init_gamma):
            data[f"init_gamma_{i}"] = [val]

        # Add remaining columns in the specified order
        data.update({
            "init_e": [self.init_e],
            "optimal_beta": [self.optimal_beta],
            "optimal_alpha": [self.optimal_alpha],
        })

        # Add optimal_gamma columns
        for i, val in enumerate(self.optimal_gamma):
            data[f"optimal_gamma_{i}"] = [val]

        data.update({
            "optimal_e": [self.optimal_e],
            "rmse": [self.rmse],
            "r2": [self.r2],
            "dest_mae": [self.dest_mae],
            "dest_mse": [self.dest_mse],
            "cv_best_params": [self.cv_best_params],
            "cv_best_score": [self.cv_best_score]
        })

        return pd.DataFrame(data)

    @staticmethod
    def aggregate_results(models):
        dfs = [model.to_dataframe() for model in models]
        return pd.concat(dfs, ignore_index=True)