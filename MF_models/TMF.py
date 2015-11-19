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


# Read all data and metadata and match dimensions
def read_data(collection, dataset):
    # collection: data collection folder
    # dataset: dataset folder

    # Read data and metadata
    R = np.loadtxt('../data/' + collection + '/' + dataset + '/playcounts.txt', delimiter=",")
    R = sparse.coo_matrix((R[:, 2], (R[:, 0], R[:, 1])))
    num_users_R, num_items_R = R.shape

    U = np.loadtxt('../data/' + collection + '/' + dataset + '/user_tags.txt', delimiter=",")
    U = sparse.coo_matrix((U[:, 2], (U[:, 0], U[:, 1])))
    num_users_U, num_tags_U = U.shape

    T = np.loadtxt('../data/' + collection + '/' + dataset + '/item_tags.txt', delimiter=",")
    T = sparse.coo_matrix((T[:, 2], (T[:, 0], T[:, 1])))
    num_items_T, num_tags_T = T.shape

    # Find maximum dataset dimensions
    num_users = max(num_users_R, num_users_U)
    num_items = max(num_items_R, num_items_T)
    num_tags = max(num_tags_U, num_tags_T)

    # Resize data and metadata
    # R.reshape((num_users, num_items))  # not implemented yet...
    # (coo is convenient for further fold preparation)
    R = sparse.coo_matrix((R.data, (R.row, R.col)),
                          shape=(num_users, num_items))

    # U.reshape((num_users, num_user_tags))  # not implemented yet...
    U = sparse.csr_matrix((U.data, (U.row, U.col)),
                          shape=(num_users, num_tags))

    # T.reshape((num_items, num_item_tags))  # not implemented yet...
    T = sparse.csr_matrix((T.data, (T.row, T.col)),
                          shape=(num_items, num_tags))

    data = {'R': R}
    metadata = {'U': U, 'T': T}

    return data, metadata


# Weighting and folds for the ratings data
def get_data(full_R, num_folds, alpha):
    # full_R: ratings in coo format
    # num_folds: data splits
    # alpha: weight for the binary ratings

    # Make data splits balancing users in each fold
    splits = cv.StratifiedKFold(full_R.row, n_folds=num_folds, random_state=1)
    data = []
    test_indices = open('test_' + dataset + '_TMF.txt', 'wa')
    for train, test in splits:
        # Train data (remind R is in coo format)
        R = sparse.csr_matrix((full_R.data[train], (full_R.row[train],
                                                    full_R.col[train])),
                              shape=full_R.shape)

        # P = R > 0 is really not needed through the code

        # Weight data
        weights = 1. + alpha * np.log(1. + R.data)
        W = sparse.csr_matrix((weights, R.nonzero()), shape=full_R.shape)

        # Test data
        Rt = sparse.coo_matrix((full_R.data[test], (full_R.row[test],
                                                    full_R.col[test])),
                               shape=full_R.shape)

        fold_data = {'W': W, 'Rt': Rt}
        data.append(fold_data)

        # Store test indices for further mpr calculation
        np.savetxt(test_indices, test, fmt='%i')

    test_indices.close()
    return data


# RMSE function
def loss_function(W, U, T, P, Q, X, mu_user, mu_item):
    # W, U, T: data arrays stored in sparse format
    # P, Q, X: factor matrices
    # mu_user, mu_item: weights for the user- and item-tags losses

    # Loss in the ratings reconstruction
    loss_ratings = 0.
    for u, Wu in enumerate(W):
        Wu_dense = Wu.toarray()
        Rbu_dense = np.ones(Wu_dense.shape)
        Rbu_dense[Wu_dense == 0.] = 0.
        Wu_dense[Wu_dense == 0.] = 1.  # blank cells are in fact 1s in W
        Zu = P[u].dot(Q.T)
        loss_ratings += np.sum(Wu_dense * ((Rbu_dense - Zu) ** 2))

    # Loss in the user-tags reconstruction
    loss_user_tags = 0.
    for u, Uu in enumerate(U):
        Uu_dense = Uu.toarray()
        loss_user_tags += np.sum(((Uu_dense - P[u].dot(X.T)) ** 2))

    # Loss in the item-tags reconstruction
    loss_item_tags = 0.
    for i, Ti in enumerate(T):
        Ti_dense = Ti.toarray()
        loss_item_tags += np.sum(((Ti_dense - Q[i].dot(X.T)) ** 2))

    return loss_ratings + mu_user * loss_user_tags + mu_item * loss_item_tags


# Objective function
def cost_function(W, U, T, P, Q, X, mu_user, mu_item, eta):
    # W, U, T: data arrays stored in sparse format
    # P, Q, X: factor matrices
    # mu_user, mu_item: weights for the user- and item-tags losses

    # Loss
    loss = loss_function(W, U, T, P, Q, X, mu_user, mu_item)

    # Regularization error
    reg_P = 0
    for Pu in P:
        reg_P += Pu.dot(Pu)

    reg_Q = 0
    for Qi in Q:
        reg_Q += Qi.dot(Qi)

    reg_X = 0
    for Xt in X:
        reg_X += Xt.dot(Xt)

    return loss + eta * (reg_P + reg_Q + reg_X)


# Train and test a given fold (convenient for parallel cross-validation)
def run_this_fold(experiment, N_values, metadata, fold_and_data):
    # experiment: set of parameters for the current experiment
    # N_values: lengths of the recommendation lists
    # metadata: user- and item-tags data
    # fold_and_data: list including fold and data
    #   fold number, used to iterate
    #   data: split of data for the given fold
    fold = fold_and_data[0]
    data = fold_and_data[1]
    results = defaultdict(list)

    print ('\tTMF with ' + str(experiment['num_iterations']) +
           ' it. of ALS. Launching fold ' + str(fold + 1) + '...')

    # Train
    P, Q = train_TMF(data['W'], metadata['U'], metadata['T'], False,
                     fold, **experiment)

    # Test
    for N in N_values:
        mpr_num, mpr_den, rank = test_topN(P, Q, data['Rt'], N, False, fold,
                                           experiment)

        # Save results for each fold and each value of N
        this_result = {'mpr_num': mpr_num, 'mpr_den': mpr_den, 'rank': rank,
                       'fold': fold}
        results[N] = this_result

    return results


# Train TagsMF for implicit feedback
def train_TMF(W, U, T, plot, fold, alpha, mu_user, mu_item, eta,
              num_factors, num_iterations):
    # W: array of weights as a function of R in sparse format
    # U: user tags in sparse format
    # T: item tags in sparse format
    # plot: should the train error evolution be plotted?
    # fold: integer indicating which fold is being trained
    # alpha: weight for the implicit feedback
    # mu_user, mu_item: weights for the user- and item-tags losses
    # eta: regularization term
    # num_factors: self descriptive
    # num_iterations: self descriptive

    # Random factors initialization
    np.random.seed(1)
    num_users, num_items = W.shape
    if U.shape[1] != T.shape[1]:
        sys.exit("Tags are not correctly merged.")
    num_tags = U.shape[1]

    P = np.random.rand(num_users, num_factors)
    Q = np.random.rand(num_items, num_factors)
    X = np.random.rand(num_tags, num_factors)

    # Iterate Alternating Least Squares
    # cost = []  # just for plotting

    for iteration in range(num_iterations):
        t0 = time.time()

        # Common terms for all users and items
        tPP = P.T.dot(P)
        tQQ = Q.T.dot(Q)
        tXX = X.T.dot(X)
        reg = eta * np.eye(num_factors)

        # loop over users (index u)
        for u, Wu in enumerate(W):
            # Use only active items for user u to speed-up
            maskW = Wu.nonzero()[1]
            Wu_mask = Wu.data
            Wu_mask_I = Wu_mask - np.array([1])
            Q_mask = Q[maskW, :]
            # Rbu_mask = Rb.getrow(u).data  # this is all 1, don't need it!

            # Use only active tags for user u to speed-up
            maskU = U.getrow(u).nonzero()[1]
            Uu_mask = U.getrow(u).data
            X_mask = X[maskU, :]

            A_this_user = Q_mask.T.dot(Wu_mask_I[:, np.newaxis] * Q_mask)
            A_user = tQQ + A_this_user + mu_user * tXX + reg

            # b_user = (Q_mask.T * (Wu_mask * Rbu_mask)[np.newaxis, :]).sum(1)
            b_user = (Q_mask.T * Wu_mask[np.newaxis, :]).sum(1)
            b_user_tag = mu_user * (X_mask.T * Uu_mask[np.newaxis, :]).sum(1)

            P[u] = alg.solve(A_user, b_user + b_user_tag)

        # loop over items (index i)
        for i, Wi in enumerate(W.T):
            # Use only active users for item i to speed-up
            maskW = Wi.nonzero()[1]
            Wi_mask = Wi.data
            Wi_mask_I = Wi_mask - np.array([1])
            P_mask = P[maskW, :]
            # Rbi_mask = Rb.getcol(i).data  # this is all 1, don't need it!

            # Use only active tags for item i to speed-up
            maskT = T.getrow(i).nonzero()[1]
            Ti_mask = T.getrow(i).data
            X_mask = X[maskT, :]

            A_this_item = P_mask.T.dot(Wi_mask_I[:, np.newaxis] * P_mask)
            A_item = tPP + A_this_item + mu_item * tXX + reg

            # b_item = (P_mask.T * (Wi_mask * Rbi_mask)[np.newaxis, :]).sum(1)
            b_item = (P_mask.T * Wi_mask[np.newaxis, :]).sum(1)
            b_item_tag = mu_item * (X_mask.T * Ti_mask[np.newaxis, :]).sum(1)

            Q[i] = alg.solve(A_item, b_item + b_item_tag)

        # loop over tags (index t)
        for t, Ut in enumerate(U.T):
            # Use only active users for tag t to speed-up
            maskU = Ut.nonzero()[1]
            Ut_mask = Ut.data
            P_mask = P[maskU, :]

            # Use only active items for tag t to speed-up
            # Note on getcol: Returns a (m x 1) CSR matrix (column vector).
            # Thus, we use nonzero()[0] instead of nonzero()[1]
            maskT = T.getcol(t).nonzero()[0]
            Tt_mask = T.getcol(t).data
            Q_mask = Q[maskT, :]

            A_tag = mu_user * tPP + mu_item * tQQ + reg
            b_tag_user = mu_user * (P_mask.T * Ut_mask[np.newaxis, :]).sum(1)
            b_tag_item = mu_item * (Q_mask.T * Tt_mask[np.newaxis, :]).sum(1)

            X[t] = alg.solve(A_tag, b_tag_user + b_tag_item)

        t1 = time.time()
        print ('\t\tTraining TMF on fold ' + str(fold) + ', it. ' +
               str(iteration) + ': ' + str(t1 - t0) + 's')
        # cost.append(cost_function(W, U, T, P, Q, X, mu_user, mu_item, eta))

    if plot:
        plt.figure()
        plt.title('TagsMF training\nalpha = ' + str(alpha) +
                  ', mu_user = ' + str(mu_user) + ', mu_item = ' +
                  str(mu_item) + ', eta = ' + str(eta) + '\nnum_factors = ' +
                  str(num_factors) + ', num_iterations =' + str(num_iterations))
        plt.plot(cost, label='cost',
                 marker='s', linestyle=':', color='m', linewidth=2)
        plt.xlabel('Iteration Number')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.subplots_adjust(top=0.85)
        plt.show()

    return P, Q


# Test by Mean Percentage Ranking
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
    print ('\t\tTesting MPR at ' + str(N) + ' on fold ' + str(fold + 1) +
           ': ' + str(t1 - t0) + 's')

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
def plot_rank(rank, N, alpha, mu_user, mu_item, eta, num_factors,
              num_iterations):
    # rank: position of each element in the test set
    # N: length of the recommendation

    count, bins = np.histogram(rank, bins=100)
    ecdf = np.cumsum(count) / float(np.sum(count))

    fig, ax1 = plt.subplots()
    plt.title('JoinTagsMF test at Top' + str(N) + '\nalpha = ' + str(alpha) +
              ', mu_user = ' + str(mu_user) + ', mu_item = ' + str(mu_item) +
              ', eta = ' + str(eta) + '\nnum_factors = ' + str(num_factors) +
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
    plt.subplots_adjust(top=0.85)
    plt.show()


# Go!

# Parameters for all experiments
param = {'alpha': [120.],
         'mu_user': [0.5],
         'mu_item': [0.5],
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
data, metadata = read_data(collection, dataset)
for k, experiment in enumerate(experiments):
    print 'Experiment ' + str(k + 1) + ' out of ' + str(num_experiments)
    t0 = time.time()

    # Data for this experiment
    data_folds = get_data(data['R'], num_folds, experiment['alpha'])

    # Pool of workers for parallel num_folds-CV and
    # special function callable through fun(all_param, looping_index)
    pool = Pool(processes=num_folds)
    run_folds = partial(run_this_fold, experiment, N_values, metadata)

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
with open('TFM_' + dataset + '.json', 'w') as TMF_output:
    json.dump(experiments, TMF_output)
TMF_output.close()
