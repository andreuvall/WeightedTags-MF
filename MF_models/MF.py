import numpy as np
import matplotlib.pyplot as plt
import time
import random
import bisect
import json
import sys
from numpy import linalg as alg
from scipy import sparse
from sklearn import cross_validation as cv
from itertools import product
from collections import defaultdict
from functools import partial
from multiprocessing import Pool


# Read train and test data for each fold
def get_data(collection, dataset, num_folds, alpha):
    # collection: data collection folder
    # dataset: dataset folder
    # num_folds: data splits
    # alpha: weight for the binary ratings

    # Load ratings data
    full_R = np.loadtxt('../data/' + collection + '/' + dataset + '/playcounts.txt', delimiter=",")
    full_R = sparse.coo_matrix((full_R[:, 2], (full_R[:, 0], full_R[:, 1])))
    num_users, num_items = full_R.shape

    # Make data splits balancing users in each fold and prepare data
    splits = cv.StratifiedKFold(full_R.row, n_folds=num_folds, random_state=1)
    data = []
    test_indices = open('test_' + dataset + '_MF.txt', 'wa')
    for train, test in splits:
        # Train data
        R = sparse.csr_matrix((full_R.data[train], (full_R.row[train],
                                                    full_R.col[train])),
                              shape=(num_users, num_items))

        # P = R > 0 is really not needed through the code

        # Weight data
        weights = 1. + alpha * np.log(1. + R.data)
        C = sparse.csr_matrix((weights, R.nonzero()),
                              shape=(num_users, num_items))

        # Test data
        Rt = sparse.coo_matrix((full_R.data[test], (full_R.row[test],
                                                    full_R.col[test])),
                               shape=(num_users, num_items))

        fold_data = {'C': C, 'Rt': Rt}
        data.append(fold_data)

        # Store test indices for further mpr calculation
        np.savetxt(test_indices, test, fmt='%i')

    test_indices.close()
    return data


# RMSE function
def loss_function(C, X, Y):
    # C: data arrays stored in sparse format
    # X, Y: factor matrices

    loss = 0.
    for u, Cu in enumerate(C):
        Cu_dense = Cu.toarray()
        Pu_dense = np.ones(Cu_dense.shape)
        Pu_dense[Cu_dense == 0.] = 0.
        Cu_dense[Cu_dense == 0.] = 1.  # blank cells are in fact 1s in C
        Zu = X[u].dot(Y.T)
        loss += np.sum(Cu_dense * ((Pu_dense - Zu) ** 2))

    return loss


# Objective function
def cost_function(C, X, Y, eta):
    # C: data arrays in sparse format
    # X, Y: factor matrices
    # eta: regularization term

    # Reconstruction error
    loss = loss_function(C, X, Y)

    # Regularization error
    reg_x = (X ** 2).sum()
    reg_y = (Y ** 2).sum()

    return loss + eta * (reg_x + reg_y)


# Train and test a given fold (convenient for parallel cross-validation)
def run_this_fold(experiment, N_values, fold_and_data):
    # experiment: set of parameters for the current experiment
    # N_values: lengths of the recommendation lists
    # fold_and_data: list including fold and data
    #   fold number, used to iterate
    #   data: split of data for the given fold
    fold = fold_and_data[0]
    data = fold_and_data[1]
    results = defaultdict(list)

    print ('\tMF with ' + str(experiment['num_iterations']) +
           ' it. of ALS. Launching fold ' + str(fold + 1) + '...')

    # Train
    X, Y = train_MF(data['C'], False, fold, **experiment)

    # Test
    for N in N_values:
        mpr_num, mpr_den, rank = test_topN(X, Y, data['Rt'], N, False, fold,
                                           experiment)

        # Save results for each fold and each value of N
        this_result = {'mpr_num': mpr_num, 'mpr_den': mpr_den, 'rank': rank,
                       'fold': fold}
        results[N] = this_result

    return results


# Train MF for implicit feedback
def train_MF(C, plot, fold, alpha, eta, num_factors, num_iterations):
    # C: array of weights as a function of R in sparse format
    # plot: should the train error evolution be plotted?
    # fold: integer indicating which fold is being trained
    # alpha: weight for the implicit feedback
    # eta: regularization term
    # num_factors: self descriptive
    # num_iterations: self descriptive

    # Random user and item factors initialization
    np.random.seed(1)
    num_users, num_items = C.shape
    X = np.random.rand(num_users, num_factors)
    Y = np.random.rand(num_items, num_factors)

    # Iterate Alternating Least Squares
    # cost = []  # just for plot

    for iteration in range(num_iterations):
        t0 = time.time()

        # Common terms for all users and items including regularization
        A_common_user = Y.T.dot(Y) + eta * np.eye(num_factors)
        A_common_item = X.T.dot(X) + eta * np.eye(num_factors)

        for u, Cu in enumerate(C):
            # User dedicated part Y.T * (Cu - I) * Y

            # Use only active items for user u to speed-up
            mask = Cu.nonzero()[1]
            Cu_mask = Cu.data
            Cu_mask_I = Cu_mask - np.array([1])
            Y_mask = Y[mask, :]
            # Pu_mask = P.getrow(u).data  # this is all 1, don't need it!

            A_user = Y_mask.T.dot(Cu_mask_I[:, np.newaxis] * Y_mask)
            # b_user = (Y_mask.T * (Cu_mask * Pu_mask)[np.newaxis, :]).sum(1)
            b_user = (Y_mask.T * Cu_mask[np.newaxis, :]).sum(1)
            X[u] = alg.solve(A_common_user + A_user, b_user)

        for i, Ci in enumerate(C.T):
            # Item dedicated part X.T * (Ci - I) * X

            # Use only active users for item i to speed-up
            mask = Ci.nonzero()[1]
            Ci_mask = Ci.data
            Ci_mask_I = Ci_mask - np.array([1])
            X_mask = X[mask, :]
            # Pi_mask = P.getcol(i).data  # this is all 1, don't need it!

            A_item = X_mask.T.dot(Ci_mask_I[:, np.newaxis] * X_mask)
            # b_item = (X_mask.T * (Ci_mask * Pi_mask)[np.newaxis, :]).sum(1)
            b_item = (X_mask.T * Ci_mask[np.newaxis, :]).sum(1)
            Y[i] = alg.solve(A_common_item + A_item, b_item)

        t1 = time.time()
        print ('\t\tTraining MF on fold ' + str(fold) + ', it. ' +
               str(iteration) + ': ' + str(t1 - t0) + 's')
        # cost.append(cost_function(C, X, Y, eta))

    if plot:
        plt.figure()
        plt.title('MF training\n' + 'alpha = ' + str(alpha) + ', eta = ' +
                  str(eta) + ', num_factors = ' + str(num_factors) +
                  ', num_iterations = ' + str(num_iterations))
        plt.plot(cost, label='cost',
                 marker='o', linestyle='--', color='c', linewidth=2)
        plt.xlabel('Iteration Number')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()

    return X, Y


# Test by Mean Percentage Ranking
# Note: The order in the code has to be (1) sample (2) sort. We can not sort
# just once for each user and then sample, because sample breaks the ordering.
def test_topN(X, Y, Rt, N, plot, fold, parameters):
    # X, Y: latent factor arrays
    # Rt: test data
    # N: length of the recommendation
    # plot: should the rank be plotted?
    # fold: integer indicating which fold is being trained
    # parameters: to further pass to plot

    # Initialize values
    mpr_numerator = 0
    rank = Rt.nnz * [None]
    t0 = time.time()

    # Loop over test set
    # print '\t\tTesting by Mean Percentage Ranking at ' + str(N) + '...'
    u_old = -1
    for k, (u, i, rt) in enumerate(zip(Rt.row, Rt.col, Rt.data)):
        if u != u_old:
            Zu = X[u].dot(Y.T)
            u_old = u

        random.seed(1)
        Zu_sample = random.sample(np.hstack((Zu[:i], Zu[(i + 1):])), N)
        Zu_sample.sort()
        # position of Zu[i] in Zu_sample but reversed order
        rank[k] = N - bisect.bisect(Zu_sample, Zu[i])
        mpr_numerator += rt * rank[k] / float(N)

    t1 = time.time()
    print ('\t\tTesting MPR at ' + str(N) + ' on fold ' + str(fold) + ': ' +
           str(t1 - t0) + 's')

    if plot:
        plot_rank(rank, N, **parameters)

    return mpr_numerator, Rt.data.sum(), rank


# Join results of MPR for each fold and each value of N
def join_folds(results, num_folds, N_values, plot, parameters):
    # results: result for each fold
    # num_folds: number of data splits
    # N_values: possible values for the length of the recommendation
    # plot: should the rank be plotted?
    # parameters: to further pass to plot

    out_mpr = defaultdict()
    out_rank = defaultdict()
    for N in N_values:
        # Initialize values
        mpr_num = 0.
        mpr_den = 0.
        rank = []
        print '\tJoining results of MPR at ' + str(N) + ' for each fold...'

        for fold in range(num_folds):
            mpr_num += results[fold][N]['mpr_num']
            mpr_den += results[fold][N]['mpr_den']
            rank += results[fold][N]['rank']

        if plot:
            plot_rank(rank, N, **parameters)

        out_mpr[N] = mpr_num / mpr_den
        out_rank[N] = rank

    return out_mpr, out_rank


# Plot rank density and ecdf
def plot_rank(rank, N, alpha, eta, num_factors, num_iterations):
    # rank: position of each element in the test set
    # N: length of the recommendation

    count, bins = np.histogram(rank, bins=100)
    ecdf = np.cumsum(count) / float(np.sum(count))

    fig, ax1 = plt.subplots()
    plt.title('MF test at Top' + str(N) + '\n' + r'$\alpha = $' + str(alpha) +
              ', $\eta = $' + str(eta) + ', num_factors = ' + str(num_factors) +
              ', num_iterations =' + str(num_iterations))
    ax1.plot(bins[1:], count, label='count',
             linestyle='-', color='b', linewidth=2)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Density [count]')
    ax1.set_ylim([0, max(count)])
    ax1.legend(loc=2)  # top left

    ax2 = ax1.twinx()
    ax2.plot(bins[1:], ecdf, label='ecdf',
             linestyle='--', color='g', linewidth=2)
    ax2.set_ylabel('Cumulative Distribution [%]')
    ax2.set_ylim([0, 1])
    ax2.legend(loc=1)  # top right
    plt.show()


# Go!

# Parameters for all experiments
param = {'alpha': [120.],
         'eta': [100., 1000.],
         'num_factors': [10],
         'num_iterations': [5]}
N_values = [100]
num_folds = 5
if len(sys.argv) > 1:
    collection = sys.argv[1]
    dataset = sys.argv[2]
else:
    collection = 'dummy_collection'
    dataset = 'dummy_dataset'

# Create all possible experiments
param_names = sorted(param)
experiments = [dict(zip(param_names, prod))
               for prod in product(*(param[name] for name in param_names))]
num_experiments = len(experiments)

# Run all experiments
for k, experiment in enumerate(experiments):
    print 'Experiment ' + str(k + 1) + ' out of ' + str(num_experiments)
    t0 = time.time()

    # Data for this experiment
    data_folds = get_data(collection, dataset, num_folds, experiment['alpha'])

    # Pool of workers for parallel num_folds-CV and
    # special function callable through fun(all_param, looping_index)
    pool = Pool(processes=num_folds)
    run_folds = partial(run_this_fold, experiment, N_values)

    # Parallel loop over the folds
    results = pool.map(run_folds, list(enumerate(data_folds)))
    pool.close()
    pool.join()

    # Join CV results and save this experiment's result
    mpr, rank = join_folds(results, num_folds, N_values, False, experiment)
    # if we only want the mpr ...
    experiments[k]['mpr'] = mpr
    # if we want to save rank too, we should instead do...
    # this_experiment = {'mpr': mpr, 'rank': rank}
    # experiments[k].update(this_experiment)

    t1 = time.time()
    print '\ttime elapsed in experiment ' + str(k + 1) + ': ' + str(t1 - t0)

# Save results in json format
print '\tSaving results to file...'
with open('MF_' + dataset + '.json', 'w') as MF_output:
    json.dump(experiments, MF_output)
MF_output.close()
