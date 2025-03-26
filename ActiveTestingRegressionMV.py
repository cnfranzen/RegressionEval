# Scikit-learn v1.4; Numpy v1.26.4; Pandas v2.2.1; Matplotlib v3.9
# Keras v2.15; TF v2.15; TFP v0.23; KT v1.4.7

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import Sequential  
from keras.layers import Dense
import tensorflow as tf  
import tensorflow_probability as tfp  
from keras.optimizers import RMSprop
import time
import keras_tuner  
from keras.layers import Lambda
from keras import backend as K

tfd = tfp.distributions
tfpl = tfp.layers

def PermaDropout(rate):
   return Lambda(lambda x: K.dropout(x, level=rate))

def quadratic_loss(y, y_hat): 
    return np.power((y - y_hat), 2)

def find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = 30, model_type='boot'):
    
    y_pred = np.zeros((len(X_test), n_reps))

    if model_type == 'refit':
        for i in range(n_reps):
            reg_evaluator.fit(X_train, y_train, epochs=1, batch_size=10, verbose=0)
            y_pred[:,i] = reg_evaluator.predict(X_test).ravel()

    if model_type == 'boot':
        for i in range(n_reps):
            ind = np.random.choice(len(X_train), len(X_train), replace=True) # Randomly choose index number for X number of samples with replacement
            X_pert = X_train[ind] # Input values from training features from matching index
            y_pert = y_train[ind] # Input values from training responses from matching index
            reg_evaluator.fit(X_pert, y_pert) # fit model to bootstrapped data
            y_pred[:,i] = reg_evaluator.predict(X_test).ravel() # Collect all bootstraps in one matrix
    
    if model_type == 'drop':
        for i in range(n_reps):
            reg_evaluator.fit(X_train, y_train, epochs=1, batch_size=10, verbose=0)
            y_pred[:,i] = reg_evaluator.predict(X_test).ravel()

    if model_type == 'prob':
        for i in range(n_reps):
            y_pred[:,i] = reg_evaluator.predict(X_test).ravel()

    if model_type == 'vi':
        for i in range(n_reps):
            y_pred[:,i] = reg_evaluator.predict(X_test).ravel()

    meanVec = y_pred.mean(axis=1)
    varVec = y_pred.var(axis=1)
    
    return meanVec, varVec


def surrogate_sampling_model(X_train, y_train, X_test, y_pred, M, reg_evaluator, n_reps = 30, loss=quadratic_loss, model_type='boot'):
    
    # keep track of selected indices and weights 
    N = X_test.shape[0] # Capture number of test points available
    remaining_idx = np.arange(N, dtype=int) # Create matrix to track the indices for N samples by initializing with 0-(N-1) indices 
    observed_idx = np.array([], dtype=int) # Initialize empty array with integer data type to capture desired test points
    weights = np.array([]) # Initialize empty array to capture weights based on index

    if model_type == 'refit':
        # find the mean and variance associated with the evaluator 
        meanVec, varVec = find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = n_reps, model_type='refit')

    if model_type == 'boot':
        # find the mean and variance associated with the evaluator 
        meanVec, varVec = find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = n_reps, model_type='boot')
    
    if model_type == 'drop':
        # find the mean and variance associated with the evaluator 
        meanVec, varVec = find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = n_reps, model_type='drop')
    
    if model_type == 'prob':
            # find the mean and variance associated with the evaluator 
        meanVec, varVec = find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = n_reps, model_type='prob')
    
    if model_type == 'vi':
        meanVec, varVec = find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = n_reps, model_type='vi')
    
    for i in range(M):
        probs = loss(y_pred[remaining_idx], meanVec[remaining_idx]) + varVec[remaining_idx] # Compute probabilities for each test point based on predicted value minus mean value squared + variance
        pmf = probs / probs.sum() # Normalize by dividing by total sum of prob
        sample = np.random.multinomial(1, pmf) # Choose 1 sample using custom multinomial probability distibution defined by pmf above
        idx = np.where(sample)[0][0] # Capture index of selected sample
        observed_idx = np.append(observed_idx, remaining_idx[idx]) # Put selected index into observed test sample matrix
        weights = np.append(weights, pmf[idx]) # Capture probability distribution for this selection ()
        remaining_idx =  np.delete(remaining_idx, idx) # Remove selected index from remaining test sample indices
        
    return observed_idx, weights # output M selected test points and their associated probability

def run_exp(X_train, y_train, X_test, y_test, y_pred, reg_evaluator, num_exp = 30, n_reps = 100, model_type="boot", plot_title="Bootstrap"):
    N = len(X_test)
    M_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * N
    M_values = M_values.astype(int)
    result_uni = np.zeros((num_exp, len(M_values))) # uniform 
    result_mod_unweighted = np.zeros((num_exp, len(M_values))) # nonuniform, unweighted (biased)
    result_mod_weighted   = np.zeros((num_exp, len(M_values))) # nonuniform, weighted (unbiased)

    for i in range(len(M_values)):    
        M = M_values[i]
        print('---' + str(M) + '---')
        
        for exp in range(num_exp):
            # uniform 
            observed_idx = np.random.choice(N, M, replace=False)
            weights = np.ones(M)/M 
            result_uni[exp, i] = np.sqrt(risk_estimator_partial(y_test[observed_idx], y_pred[observed_idx], N,  
                        weights, quadratic_loss, mode='unweighted'))
            
            # nonuniform, unweighted 
            observed_idx, weights = surrogate_sampling_model(X_train, y_train,
                                            X_test, y_pred, M, reg_evaluator, n_reps = n_reps, loss=quadratic_loss, model_type=model_type)
            
            result_mod_unweighted[exp, i] = np.sqrt(risk_estimator_partial(y_test[observed_idx], y_pred[observed_idx], N,  
                        weights, quadratic_loss, mode='unweighted'))

            # nonuniform, weighted 
            result_mod_weighted[exp, i] = np.sqrt(risk_estimator_partial(y_test[observed_idx], y_pred[observed_idx], N,  
                        weights, quadratic_loss, mode='weighted'))
    
    full_loss = np.sqrt(quadratic_loss(y_test, y_pred).mean())

    # plot results  
    plt.rcParams.update({'font.size': 16, "figure.figsize": (6,3.5)})
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    calc_width = 0.2 * (M_values[1] - M_values[0])
    ax.bar(M_values - calc_width, result_uni.mean(axis=0), yerr= result_uni.std(axis=0).reshape(1,-1), color = 'royalblue', ecolor='darkblue', width = calc_width, capsize=3)
    ax.bar(M_values , result_mod_unweighted.mean(axis=0), yerr= result_mod_unweighted.std(axis=0).reshape(1,-1), color = 'bisque', ecolor='darkorange', width = calc_width, capsize=3)
    ax.bar(M_values + calc_width, result_mod_weighted.mean(axis=0), yerr= result_mod_weighted.std(axis=0).reshape(1,-1), color = 'limegreen', ecolor='darkgreen', width = calc_width, capsize=3)

    plt.axhline(y = full_loss, color = 'r', linestyle = ':')
    ax.set_xticks(M_values)
    ax.set_ylabel('error estimate (mean+/-std)', fontsize=19)
    ax.set_xlabel('test data size $M$', fontsize=19)
    ax.legend(labels=['test error $R$', 'uniform', 'surrogate, unweighted',  'surrogate, weighted'], ncol=2, fontsize=14, framealpha=0.5)
    plt.title(f'{plot_title} Surrogate Sampling')
    plt.show()

def run_exp_weighted(X_train, y_train, X_test, y_test, y_pred, reg_evaluator, num_exp = 30, n_reps = 100, model_type="boot", plot_title="Bootstrap"):
    N = len(X_test)
    M_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * N
    M_values = M_values.astype(int)
    result_uni = np.zeros((num_exp, len(M_values))) # uniform 
    result_mod_weighted = np.zeros((num_exp, len(M_values))) # nonuniform, weighted (unbiased)

    for i in range(len(M_values)):    
        M = M_values[i]
        print('---' + str(M) + '---')
        
        for exp in range(num_exp):
            # uniform 
            observed_idx = np.random.choice(N, M, replace=False)
            weights = np.ones(M)/M 
            result_uni[exp, i] = np.sqrt(risk_estimator_partial(y_test[observed_idx], y_pred[observed_idx], N,  
                        weights, quadratic_loss, mode='unweighted'))
            
            # nonuniform, unweighted 
            observed_idx, weights = surrogate_sampling_model(X_train, y_train,
                                            X_test, y_pred, M, reg_evaluator, n_reps = n_reps, loss=quadratic_loss, model_type=model_type)

            # nonuniform, weighted 
            result_mod_weighted[exp, i] = np.sqrt(risk_estimator_partial(y_test[observed_idx], y_pred[observed_idx], N,  
                        weights, quadratic_loss, mode='weighted'))

    full_loss = np.sqrt(quadratic_loss(y_test, y_pred).mean())

    # plot results  
    plt.rcParams.update({'font.size': 16, "figure.figsize": (6,3.5)})
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    calc_width = (1/3)*(M_values[1] - M_values[0])
    ax.bar(M_values - 0.5*calc_width, result_uni.mean(axis=0), yerr= result_uni.std(axis=0).reshape(1,-1), width = calc_width, color = 'royalblue', ecolor='darkblue', capsize=3)
    ax.bar(M_values + 0.5*calc_width, result_mod_weighted.mean(axis=0), yerr= result_mod_weighted.std(axis=0).reshape(1,-1), width = calc_width, color = 'limegreen', ecolor='darkgreen', capsize=3)

    plt.axhline(y = full_loss, color = 'r', linestyle = ':')
    ax.set_xticks(M_values)
    ax.set_ylabel('error estimate (mean+/-std)', fontsize=19)
    ax.set_xlabel('test data size $M$', fontsize=19)
    ax.legend(labels=['test error $R$', 'uniform', 'surrogate, unweighted',  'surrogate, weighted'], ncol=2, fontsize=14, framealpha=0.5)
    plt.title(f'{plot_title} Surrogate Sampling')
    plt.show()

    # Print results  
    df = pd.DataFrame({'Uniform R': result_uni.mean(axis=0), 
                        'Uniform Stddev': result_uni.std(axis=0), 
                        'Nonuniform R': result_mod_weighted.mean(axis=0), 
                        'Nonuniform Stddev': result_mod_weighted.std(axis=0)}, index = M_values)
    display(df) #type: ignore
    print("Full Test R Value:", full_loss)

def time_weight(X_train, y_train, X_test, y_test, y_pred, reg_evaluator, num_exp = 30, n_reps = 100, model_type="boot", plot_title="Bootstrap"):
    N = len(X_test)    
    M_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * N
    M_values = M_values.astype(int)
    result_mod_weighted = np.zeros((num_exp, len(M_values))) # nonuniform, weighted (unbiased)
    timemat = []

    for i in range(len(M_values)): 
        start = time.time()   
        M = M_values[i]
        print('---' + str(M) + '---')
        
        for exp in range(num_exp):
            # nonuniform, weighted
            observed_idx, weights = surrogate_sampling_model(X_train, y_train,
                                            X_test, y_pred, M, reg_evaluator, n_reps = n_reps, loss=quadratic_loss, model_type=model_type)
    
            result_mod_weighted[exp, i] = np.sqrt(risk_estimator_partial(y_test[observed_idx], y_pred[observed_idx], N,  
                        weights, quadratic_loss, mode='weighted'))
        end = time.time()    
        timemat = np.append(timemat, end - start)

    print(timemat)

def risk_estimator_partial(y_true, y_pred, N,  acq_weights, loss_function, mode='unweighted'):
    '''
    N: total number of samples in D_test 
    M: number of selected samples (y_true, y_pred)
    acq_weights: sampling distribution for each index (M elements)
    mode: "unweighted" or "weighted" 
    '''
    
    M = y_true.shape[0]
    
    if mode == 'unweighted':
        v_i = np.ones(M)
    
    if mode == 'weighted':
        
        m = np.arange(1, M+1) 
        
        v_i = 1 + ((N-M)/(N-m)) * (1/((N-m+1) * acq_weights) - 1)
        
    
    l_i = loss_function(y_true, y_pred)
    
    R = (v_i * l_i).mean()
    
    return R

def create_model(X_train, y_train, model_type, min_layers, max_layers, min_value, max_value, step):

    if model_type == "boot":
        def build_model(hp):
            # Initialize the model
            model = Sequential()            
            model.add(Dense(
                units = hp.Int("units_00", min_value=128, max_value=256, step=64), 
                activation = 'relu', 
                input_shape = (X_train.shape[1],) 
            ))
            
            for ind in range(hp.Int('num_layers', min_layers, max_layers)):
                model.add(Dense(
                    units = hp.Int("units_" + str(ind), min_value=min_value, max_value=max_value, step=step),
                    activation = "relu"
                ))
         
            # Add the output layer with the specified output shape 
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

            return model

        tuner = keras_tuner.Hyperband(
            hypermodel = build_model,
            objective = "mean_squared_error",
            max_epochs = 10,
            factor = 3,
            directory = "tune_dir",
            project_name = "boot")
            
        tuner.search(X_train, y_train, epochs=5)

        best_model = tuner.get_best_hyperparameters(5)  # Pull best hp from 

        reg_evaluator = build_model(best_model[0])

    if model_type == "drop":
        def build_model(hp):
            # Initialize the model
            model = Sequential()            
            model.add(Dense(
                units = hp.Int("units_00", min_value=128, max_value=256, step=64), 
                activation = 'relu', 
                input_shape = (X_train.shape[1],) 
            ))
            model.add(PermaDropout(0.5))
            
            for ind in range(hp.Int('num_layers', min_layers, max_layers)):
                model.add(Dense(
                    units = hp.Int("units_" + str(ind), min_value=min_value, max_value=max_value, step=step),
                    activation = "relu"
                ))
                model.add(PermaDropout(0.5))
            
            # Add the output layer with the specified output shape 
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

            return model

        tuner = keras_tuner.Hyperband(
            hypermodel = build_model,
            objective = "mean_squared_error",
            max_epochs = 10,
            factor = 3,
            directory = "tune_dir",
            project_name = "drop")
            
        tuner.search(X_train, y_train, epochs=5)

        best_model = tuner.get_best_hyperparameters(5)  # Pull best hp from 

        reg_evaluator = build_model(best_model[0])

    if model_type == "prob":
        def build_model(hp):
            # Initialize the model
            model = Sequential()            
            model.add(Dense(
                units = hp.Int("units_00", min_value=128, max_value=256, step=64), 
                activation = 'relu', 
                input_shape = (X_train.shape[1],) 
            ))
            
            for ind in range(hp.Int('num_layers', min_layers, max_layers)):
                model.add(Dense(
                    units = hp.Int("units_" + str(ind), min_value=min_value, max_value=max_value, step=step),
                    activation = "relu"
                ))
            
            # Add the output layer with the specified output shape 
            model.add(Dense(tfpl.IndependentNormal.params_size(event_shape=1)))
            model.add(tfpl.IndependentNormal(event_shape=1))

            model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

            return model

        tuner = keras_tuner.Hyperband(
            hypermodel = build_model,
            objective = "mean_squared_error",
            max_epochs = 10,
            factor = 3,
            directory = "tune_dir",
            project_name = "prob")
        
        tuner.search(X_train, y_train, epochs=5)

        best_model = tuner.get_best_hyperparameters(5)  # Pull best hp from 

        reg_evaluator = build_model(best_model[0])

    return reg_evaluator

def plot_model(X_train, y_train, reg_evaluator):
    history = reg_evaluator.fit(X_train, y_train,
                batch_size=10,
                epochs=50,
                verbose=0)

    # Summary of the model
    reg_evaluator.summary()

    # summarize history for accuracy
    plt.plot(history.history['mean_squared_error'])
    plt.title('Model Accuracy')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
def plot_1D(X_train, y_train, X_test, X_new, reg_evaluator, reg_learner, n_reps=30, model_type="boot", plot_type = "Bootstrap"):
    mean, var = find_mean_var_model(X_train, y_train, X_test, reg_evaluator, n_reps = n_reps, model_type = model_type)

    # Plot Bootstrap Results
    plt.rcParams.update({'font.size': 14, "figure.figsize": (6,3.5)})

    plt.plot(X_train, y_train, "C5o", markersize=5, label='$D_{train}$')
    plt.plot(X_new, reg_learner.predict(X_new[:,np.newaxis]), 'k:', label="trained $f$")
    plt.errorbar(X_test, mean, yerr=var, ls='none', marker='s', c="C9",
                markersize=4, capsize=2, label= 'results for $\mathcal{X}_{test}$ surrogate')
    plt.xlabel("$x$")
    plt.ylabel("$y$", rotation=0)
    plt.legend(fontsize=13)
    plt.ylim([-3.2,1.5])
    plt.title(f'{plot_type} Surrogate')
    plt.show()


# Define prior with 
def prior_trainable(kernel_size, bias_size = 0, dtype=None):
    """
    Description: This function defines our prior distribution for a given dense layer with trainable mean value and var 1
    
    Args: (kernel_size, bias_size, dtype=None)
    kernel_size: number parameters in the dense layer weight matrix

    Returns: Callable object that takes a tensor t into a lambda function that returns an independent normal distribution with trained mean and variance 1 
    """
    n = kernel_size + bias_size
    return Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

# Define variational posterior
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    """
    Description: This function defines our posterior distribution using mean-field for a given dense layer. 
    
    Args: (kernel_size, bias_size, dtype=None)
    kernel_size: number parameters in the dense layer weight matrix

    Returns: A callable object that is a sequential model that takes a tensor as an input and returns a distribution object.
    """
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

def build_vi(X_train, y_train):
    reg_evaluator = Sequential([
    tfpl.DenseVariational(units=2,
                          make_prior_fn = prior_trainable,
                          make_posterior_fn = posterior_mean_field,
                          kl_weight=1/X_train.shape[0], 
                          activation='sigmoid'),
    tfpl.DenseVariational(units=tfpl.IndependentNormal.params_size(1),
                          make_prior_fn=prior_trainable,
                          make_posterior_fn=posterior_mean_field,
                          kl_weight=1/X_train.shape[0]),
    tfpl.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
    ])

    reg_evaluator.compile(loss=negloglik, optimizer=RMSprop(learning_rate=0.01))
    reg_evaluator.fit(X_train, y_train, epochs=10000, verbose=False)
    reg_evaluator.summary()

    return reg_evaluator
    