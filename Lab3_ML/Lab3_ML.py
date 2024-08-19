import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score

col_names = ['X1', 'X2','Y']
dataset = pd.read_csv("dataset.csv", header=None, names=col_names)
print(dataset.head())

X1=dataset.iloc[:,0]
X2=dataset.iloc[:,1]
X=np.column_stack((X1,X2))
Y=dataset.iloc[:,2]

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter3D(X[:,0],X[:,1],Y)

#add polynommial features
polynomialFeature = PolynomialFeatures(degree=5, include_bias=False)
X_poly_data = polynomialFeature.fit_transform(X)
#split data
X_train, X_test, Y_train, y_test = train_test_split(X_poly_data, Y, test_size=0.2, random_state=42)

print(X2.min())
print(X2.max())

grid_step = 0.1
feature1_range = np.arange(X1.min() - 0.5, X1.max() + 0.5, grid_step)
feature2_range = np.arange(X2.min() - 0.5, X2.max() + 0.5, grid_step)
feature1_grid, feature2_grid = np.meshgrid(feature1_range, feature2_range)
feature_grid = np.c_[feature1_grid.ravel(), feature2_grid.ravel()]



predictions = []
mean_lasso = []
standard_deviation = []
C_Lasso_values = [0.001,0.01, 0.1,0.15,1]

for i in C_Lasso_values:
    #Lasso Regression
    LassoModel = linear_model.Lasso(alpha=1/2*i)
    LassoModel.fit(X_train,Y_train)

    FeatureNames = polynomialFeature.get_feature_names_out(['X1','X2'])
    Lasso_model_coefs = LassoModel.coef_

    print(f"For C={i}")

    for feature,coefficients in zip(FeatureNames,Lasso_model_coefs):
        print(f"{feature}:{coefficients}")
    print("Accuracy with test data:",LassoModel.score(X_test,y_test))

  # Predict on the extended grid
    grid_poly = polynomialFeature.transform(feature_grid)
    grid_predictions = LassoModel.predict(grid_poly)
    grid_predictions = grid_predictions.reshape(feature1_grid.shape)
    predictions.append(grid_predictions)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the predictions as a 3D surface
    ax.plot_surface(feature1_grid, feature2_grid, grid_predictions, alpha=0.7, cmap='viridis', label=f'C={i}')

    # Plot the training data
    ax.scatter(X1, X2, Y, c='red', marker='o', label='Training Data')
    
    # Add labels and a legend
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Target')
    ax.set_title('Lasso Regression Predictions for Different C Values')
    ax.legend()

    #cross-validation
    cross_val = cross_val_score(LassoModel,X_poly_data,Y,cv = 5)
    meanErr=  -np.mean(cross_val)
    stdDevia = np.std(cross_val)
    
    mean_lasso.append(meanErr)
    standard_deviation.append(stdDevia)
    # Show the plot
    plt.show()


plt.errorbar(C_Lasso_values,mean_lasso,yerr=standard_deviation,fmt='-o', capsize=4)
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Mean Prediction Error (RMSE)')
plt.title('Mean Prediction Error vs. C')
plt.grid()
plt.show()

mean_Ridge = []
standard_deviation_ridge = []
C_Ridge_values = [0.1,1,100,300,1500]

for i in C_Ridge_values:
    #Ridge Regression
    RidgeRegModel = linear_model.Ridge(alpha=1/2*i)
    RidgeRegModel.fit(X_train,Y_train)
    
    FeatureNames = polynomialFeature.get_feature_names_out(['X1','X2'])
    Ridge_model_coefs = RidgeRegModel.coef_

    print(f"For C={i}")

    for feature,coefficients in zip(FeatureNames,Ridge_model_coefs):
        print(f"{feature}:{coefficients}")
    print("Accuracy with test data:",RidgeRegModel.score(X_test,y_test))

  # Predict on the extended grid
    grid_poly = polynomialFeature.transform(feature_grid)
    grid_predictions = RidgeRegModel.predict(grid_poly)
    grid_predictions = grid_predictions.reshape(feature1_grid.shape)
    predictions.append(grid_predictions)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the predictions as a 3D surface
    ax.plot_surface(feature1_grid, feature2_grid, grid_predictions, alpha=0.7, cmap='viridis', label=f'C={i}')

    # Plot the training data
    ax.scatter(X1, X2, Y, c='red', marker='o', label='Training Data')
    
    # Add labels and a legend
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Target')
    ax.set_title('Ridge Regression Predictions for Different C Values')
    ax.legend()

     #cross-validation
    cross_val = cross_val_score(RidgeRegModel,X_poly_data,Y,cv = 5)
    meanErr=  -np.mean(cross_val)
    stdDevia = np.std(cross_val)


    mean_Ridge.append(meanErr)
    standard_deviation_ridge.append(stdDevia)
    
    # Show the plot
    plt.show()

plt.errorbar(C_Ridge_values,mean_Ridge,yerr=standard_deviation_ridge,fmt='-o', capsize=4)
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Mean Prediction Error (RMSE)')
plt.title('Mean Prediction Error vs. C')
plt.grid()
plt.show()
