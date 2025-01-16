import numpy as np
import pandas as pd

# use pandas to load dataset
df = pd.read_csv("real_estate_dataset.csv")

# get n_samples n_feats
n_samples, n_features = df.shape

# print number of features
print(f"Number of samples, features, {df.shape}")

# get names of the columns 
columns = df.columns

np.savetxt("column_names.txt", columns, fmt="%s")

# Use Square_Feet ,Garage_Size, Location_Score, Distance_to_Center as features 
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]

# Use price as the target 
y = df['Price']

# print shape of X
print(f"Shape of X: {X.shape}")

# print datatype of X
print(f"Datatype of X: \n{X.dtypes}")

# Get n_features again
n_samples, n_features = X.shape

# Build a linear model to predict price from four features in X
# make an array of coefs of size n_features + 1

coefs = np.ones(n_features + 1)

# predict for each sample in X
predictions_bydefn = X @ coefs[1:] + coefs[0]

# Bias means given all the coeffs are zero, the best we can do is predict the average 
X = np.hstack((np.ones((n_samples,1)), X))

predictions = X @ coefs

# see if all entries in predictions
is_same = np.allclose(predictions_bydefn, predictions)

print(f"Are the predictions the same as predictions_bydefn?  Ans: {is_same}")

# calculating the errors
errors = y - predictions
# calculate relative error in norm 2 sense
rel_errors = errors/y

# calculate mean of squares of errors using a loop
loss_loop = 0
for error in errors:
    loss_loop += error**2
loss_loop /= len(errors)

# calculate mean of squares of errors using matrix operations
loss_matrix = (errors.T @ errors) / len(errors)

# compare the two methods of computing loss
is_diff = np.isclose(loss_loop, loss_matrix)
print(f"Are the losses the same? {is_diff}")

# print the size of errors and its L2 norm
print(f"Size of errors: {errors.size}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}")
print(f"Relative error in norm 2 sense: {np.linalg.norm(rel_errors)}")

# What is my optimization problem
# The optimization problem is to minimize the mean squared error between the predicted prices 
# and the actual prices of the real estate properties. This is called Least Squares Problem.

# Objective function in mathematical form:
# Minimize (1/n) * Σ(y_i - (β_0 + β_1 * x_i1 + β_2 * x_i2 + β_3 * x_i3 + β_4 * x_i4))^2
# where n is the number of samples, y_i is the actual price, and x_ij are the features.

# What is a solution?
# A solution is a se t of coefs that minimizes the objective.

# How do I find a solution?
# By searching for the coefs at which the gradient of the objective function is zero.
# Or I can set gradient of objective function to zero and solve for the coeffs.

# Write the loss matrix in terms of the data and coeffs
# The loss matrix can be written as:
loss_matrix = (1/n_samples) * (y - X @ coefs).T @ (y - X @ coefs)

# calculate gradient of the loss wrt the coeffs.
grad_matrix = -(2/n_samples) * X.T @ (y - X @ coefs)

# set grad matrix to zero and get coeffs
# X.T @ X @ coefs = X.T @ y 
# coefs  = {X.T @ X} ^ {-1} @ X.T @ y
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# save the text in file coefs.txt
np.savetxt("coefs.txt", coefs, delimiter=",")

# predict and find errors
predictions_model = X @ coefs
errors_model = y - predictions_model

# print L2 norm of the errors_model

rel_errors_model = errors_model/y
print(f"rel_errros_model L2 norm, {np.linalg.norm(rel_errors_model, ord=2)}")

# Use all features except 'Price' as X
X = df.drop(columns=['Price'])

# Add bias term
X = np.hstack((np.ones((n_samples, 1)), X))

# Calculate coefficients using normal equation
coefs_all = np.linalg.inv(X.T @ X) @ X.T @ y

# Predict and find errors
predictions_all = X @ coefs_all
errors_all = y - predictions_all

# Calculate relative errors
rel_errors_all = errors_all / y

# Print L2 norm of the relative errors
print(f"Relative errors norm for all features: {np.linalg.norm(rel_errors_all, ord=2)}")

# Save the coefficients for all features in coefs_all.txt
np.savetxt("coefs_all.txt", coefs_all, delimiter=",")

# Calculate the rank of X.T @ X
rank_XTX = np.linalg.matrix_rank(X.T @ X)

# Print the rank to console
print(f"Rank of X.T @ X: {rank_XTX}")

# Solve using matrix decomposition 
# Qr factorization 
# QR factorization of X.T @ X
Q, R = np.linalg.qr(X)

# Print the shapes of Q and R
print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")

# Save the R matrix to R.csv
np.savetxt("R.csv", R, delimiter=",")

# Checking Q.T Q is identity
sol = Q.T @ Q
np.savetxt("sol.csv", sol, delimiter=",")

# X.T @ X = R.T @ R 
# X.T @ y = R.T @ Q.T @ y
# R*coeffs = Q.T @ y
# b = Q.T @ y

b = Q.T @ y
print(f"Shape of b: {b.shape}")
print(f"Shape of R: {R.shape}")

n_samples, n_features = X.shape
coefs_qr_loop = np.zeros(n_features)

# Solve for coefficients using back substitution
for i in range(n_features - 1, -1, -1):
    coefs_qr_loop[i] = b[i]
    for j in range(i + 1, n_features):
        coefs_qr_loop[i] -= R[i, j] * coefs_qr_loop[j]
    coefs_qr_loop[i] /= R[i, i]

# Save the coefficients obtained from QR factorization in coefs_qr_loop.txt
np.savetxt("coefs_qr_loop.txt", coefs_qr_loop, delimiter=",")

# solve the normal equations using SVD approach
U, S, Vt = np.linalg.svd(X, full_matrices=False)
S_inv = np.diag(1 / S)
coefs_svd = Vt.T @ S_inv @ U.T @ y

# Save the coefficients obtained from SVD in coefs_svd.txt
np.savetxt("coefs_svd.txt", coefs_svd, delimiter=",")

# Solve the normal equations using eigendecomposition
# X.T @ X @ coefs = X.T @ y
# Let X.T @ X = P @ D @ P.T where P is the matrix of eigenvectors and D is the diagonal matrix of eigenvalues
# Then coefs = P @ D_inv @ P.T @ X.T @ y where D_inv is the inverse of D

# Eigendecomposition of X.T @ X
eigvals, eigvecs = np.linalg.eigh(X.T @ X)

# Inverse of the diagonal matrix of eigenvalues
D_inv = np.diag(1 / eigvals)

# Calculate coefficients using eigendecomposition
coefs_eig = eigvecs @ D_inv @ eigvecs.T @ X.T @ y

# Save the coefficients obtained from eigendecomposition in coefs_eig.txt
np.savetxt("coefs_eig.txt", coefs_eig, delimiter=",")
