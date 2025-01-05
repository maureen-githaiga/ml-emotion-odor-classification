
'''
Author: Maureen Githaiga

Description:
This script is part of a Master's Thesis and is designed to train and evaluate machine learning models using multiple cross-validation techniques:
- Leave-One-Out Cross-Validation (LOOCV)
- Leave-One-Participant-Out Cross-Validation (LOPOCV)
- K-Fold Cross-Validation

Key functionalities:
- Hyperparameter tuning: Performed using GridSearchCV.
- Performance evaluation: Models are assessed based on accuracy, F1 score, precision, and recall.
- Visualization:
  - ROC and Precision-Recall curves for various cross-validation methods.
  - Learning curves to analyze model performance over time.
  - Distribution of scores and comparison of model accuracies.

Algorithms implemented and evaluated:
- Support Vector Machines (SVM)
- Random Forest
- K-Nearest Neighbors (K-NN)
- Linear Discriminant Analysis (LDA)
- Decision Tree

'''
import sys
import warnings
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict,GridSearchCV, LeaveOneOut,KFold,learning_curve,GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay,roc_curve, auc, precision_recall_curve, average_precision_score,precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


sys.path.append(r'C:\Users\githa\Documents\thesis\Analysis\scripts')
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

param_grid_svm_all = {
    'C': [0.01, 0.1, 1, 5, 10],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1],  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    'degree': [2, 3, 4],  # Degree for 'poly'
    'kernel': ['linear', 'rbf', 'poly']  # Kernel type to be used in the algorithm

}

param_grid_svc_linear = {
    'C': [0.01, 0.1, 1, 5, 10],  # Regularization parameter controls the trade off between smooth decision boundary and classifying the training points correctly
    'kernel': ['linear'],  # Kernel type to be used in the algorithm
}
param_grid_svc_rbf = {

    'C': [0.01, 0.1, 1, 5, 10],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1],#parameter for non linear hyperplanes the higher the gamma value it tries to exactly fit the training data set
    'kernel': ['rbf'],  # Kernel type to be used in the algorithm
}
param_grid_svc_poly = {

    'C': [0.01, 0.1, 1, 5, 10],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1],#parameter for non linear hyperplanes the higher the gamma value it tries to exactly fit the training data set
    'degree': [2, 3, 4],
    'kernel': ['poly'],
}
param_grid_svc_sig = {

    'C': [0.01, 0.1, 1, 5, 10],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1],#parameter for non linear hyperplanes the higher the gamma value it tries to exactly fit the training data set
    'kernel': ['sigmoid'],  # Kernel type to be used in the algorithm
}

param_grid_knn = {
    'n_neighbors': np.arange(2, 10, 1),
    'weights': ['uniform', 'distance'],
    'metric': [ 'euclidean','manhattan','minkowski','cosine','hamming'],
    'p': [1, 2] # Only applicable for Minkowski metric
}

param_grid_lda = [
    
    {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 'auto'],  # shrinkage values for 'lsqr' and 'eigen' shrinkage might help in small datasets

    },
    {
        'solver': ['svd'],#recommended for data with large number of features
        'shrinkage': [None],  # shrinkage not supported, so keep it as None
    }

]

"""param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}"""
#sampling features results in more predictor diversity ,trading more bias for a lower variance
param_grid_rf = {
    'n_estimators': [50,75,100],  # Focus on higher numbers for stability; 25 might be too low
    'max_depth': [3,10,15,None],  # Keep these options for controlling depth
    'max_leaf_nodes': [5, 10,15],  # Simplify to focus on practical, smaller ranges
    'max_features': ['sqrt', 'log2']  # Remove 'auto' and None; focus on sqrt and log2 for more regularization
}


param_grid_dt = {
    'criterion': ['gini'],  # Focus on 'gini' for simplicity
    'max_depth': [5, 10, 15],  # Limit max_depth to control overfitting
    'min_samples_split': [5, 10]  # Start with higher values to ensure splits are more meaningful
}


def combine_data(filtered_df):
    combined_data = []
    labels = []
    for index, row in filtered_df.iterrows():

        #smoothed_binary_map = np.array(ast.literal_eval(row['smoothed_Intensity'])).flatten()#on smoothed intensity
       
        #smoothed_binary_map = np.fromstring(row['smoothed_binary_map'].replace("\n", "").replace("[", "").replace("]", "").replace(".", ""),sep=" ").flatten()
        #smoothed_binary_map = np.fromstring(row['non_smoothed_binary_map'].replace("\n", "").replace("[", "").replace("]", "").replace(".", ""),sep=" ").flatten()
        
        #smoothed_binary_map = np.array(ast.literal_eval(row['smoothed_binary_map'])).flatten()
        smoothed_binary_map = np.array(ast.literal_eval(row['normalized_Intensity'])).flatten()#on normalised intensity

        combined_data.append(smoothed_binary_map)
        labels.append(row['Label'])
    combined_data = np.vstack(combined_data)
    labels = np.array(labels)
    return combined_data, labels

def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plot_roc_curve(fpr, tpr)
        plt.show()


def retain_99_variance(X_train,X_test):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    n_components_99 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1
    #print(f'Number of components to retain 99% variance: {n_components_99}')

    pca_99 = PCA(n_components=n_components_99)
    X_train_reduced = pca_99.fit_transform(X_train)
    X_test_reduced = pca_99.transform(X_test)

    print(f"Number of components to retain 99% variance: {X_train_reduced.shape[1]}")
    return X_train_reduced,X_test_reduced

from sklearn.base import clone

def train_evaluate_model_loocv_kfold(model,param_grid,X,y,model_name,n_components,cross_val,k):
    if n_components:
        X_, _ = retain_99_variance(X,X)
    else:
        X_, _ = X, X

    X_train,y_train = X_,y


    if cross_val == 'loocv':

        loo = LeaveOneOut()
        grid_search_loo = GridSearchCV(model, param_grid, cv=loo, scoring='accuracy', verbose=0, n_jobs=-1)

        grid_search_loo.fit(X_train, y_train)
        best_model_clone = clone(grid_search_loo.best_estimator_)

        print(f"Best Parameters (loocv): {grid_search_loo.best_params_}")
        accuracies = []
        true_labels = []
        predicted_probs = []
        predicted_labels=[]

        for train_index, test_index in loo.split(X_train):
            
            X_train_loo, X_test_loo = X_train[train_index], X_train[test_index]
            y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]
            
            best_model_clone.fit(X_train_loo, y_train_loo)

            y_pred_prob = best_model_clone.predict_proba(X_test_loo)[:, 1]
            true_labels.extend(y_test_loo)
            predicted_probs.extend(y_pred_prob)
            predicted_labels.extend(best_model_clone.predict(X_test_loo))

            y_pred = best_model_clone.predict(X_test_loo)
            accuracy = accuracy_score(y_test_loo, y_pred)
            accuracies.append(accuracy)

        # Calculate overall cross-validation metrics based on collected predictions
        cv_accuracy = accuracy_score(true_labels, predicted_labels)
        cv_f1 = f1_score(true_labels, predicted_labels)
        cv_precision = precision_score(true_labels, predicted_labels)
        cv_recall = recall_score(true_labels, predicted_labels)
        mse = mean_squared_error(true_labels, predicted_probs)
        #standard deviation: {np.std(accuracies)*100:.2f}
            
        print(f"{model_name}: LOO Cross-validated Accuracy: {cv_accuracy * 100:.1f}% ,standard deviation: {np.std(accuracies)*100:.1f}%")
        print(f"Cross-validated F1 Score: {cv_f1:.2f}, Precision: {cv_precision:.2f}, Recall: {cv_recall:.2f}")

        train_sizes=np.linspace(0.1, 1.0, 10)
        return accuracies, true_labels, predicted_probs

    elif cross_val == 'kfold':
        if k == 3:
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        else:
             kfold = KFold(n_splits=5, shuffle=True, random_state=42)


        grid_search_kfold = GridSearchCV(model, param_grid, cv=kfold, scoring='accuracy', verbose=0, n_jobs=-1)

        grid_search_kfold.fit(X_train, y_train)
        best_model_clone = clone(grid_search_kfold.best_estimator_)
        print(f"Best Parameters ({k} fold): {grid_search_kfold.best_params_}")

        accuracies_kf = []
        true_labels = []
        predicted_probs = []
        predicted_labels=[]
        fold_sample_counts = []

        for train_index, test_index in kfold.split(X_train):
            X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
            y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

            best_model_clone.fit(X_train_kf, y_train_kf)
            y_pred_prob = best_model_clone.predict_proba(X_test_kf)[:, 1]
            true_labels.extend(y_test_kf)
            predicted_probs.extend(y_pred_prob)


            y_pred = best_model_clone.predict(X_test_kf)
            fold_accuracy = accuracy_score(y_test_kf, y_pred)
            accuracies_kf.append(fold_accuracy)
            predicted_labels.extend(y_pred)
            fold_sample_counts.append(len(y_test_kf))

        # Calculate overall cross-validation metrics based on collected predictions
        cv_accuracy = accuracy_score(true_labels, predicted_labels)
        cv_f1 = f1_score(true_labels, predicted_labels)
        cv_precision = precision_score(true_labels, predicted_labels)
        cv_recall = recall_score(true_labels, predicted_labels)

        print(f"{model_name}: KF{k} Cross-validated Accuracy: {cv_accuracy * 100:.1f}% , standard deviation: {np.std(accuracies_kf)*100:.1f}%")
        print(f"Cross-validated F1 Score: {cv_f1:.2f}, Precision: {cv_precision:.2f}, Recall: {cv_recall:.2f}")
        
        return  accuracies_kf, true_labels, predicted_probs

def lopo_splitter(data_df, all_participants):
    # Generate train-test splits for each participant left out
    for leave_out in all_participants:
        train_df = data_df[data_df['participants'].apply(lambda x: str(leave_out) not in x)]
        test_df = data_df[data_df['participants'].apply(lambda x: str(leave_out) in x)]
        
        train_indices = train_df.index.values
        test_indices = test_df.index.values
        
        yield train_indices, test_indices


def train_evaluate_leave_one_participant_out(model_name,n_components,model,param_grid,preprocessed_df,all_participants):
    # Combine all data for grid search
    X_combined, y_combined = combine_data(preprocessed_df)
    if n_components:
        X_combined, _ = retain_99_variance(X_combined, X_combined)

    lopo = list(lopo_splitter(preprocessed_df, all_participants))

    # Perform GridSearchCV on the combined dataset
    grid_search = GridSearchCV(model, param_grid, cv=lopo, scoring='accuracy', verbose=0, n_jobs=-1)
    grid_search.fit(X_combined, y_combined)

    # Get the best model after grid search
    best_model = grid_search.best_estimator_
    best_model_clone = clone(best_model)
    print(f"Best Parameters(LOPO): {grid_search.best_params_}")


    accuracy_scores = []
    true_labels = []
    predicted_probs = []
    predicted_labels=[]
    fold_sample_counts = []

    for leave_out in all_participants:

        train_df = preprocessed_df[preprocessed_df['participants'].apply(lambda x: str(leave_out) not in x)]
        test_df = preprocessed_df[preprocessed_df['participants'].apply(lambda x: str(leave_out) in x)]


        X_train, y_train = combine_data(train_df)
        X_test, y_test = combine_data(test_df)

        #pca
        if n_components:
            X_train, X_test = retain_99_variance(X_train,X_test)

        best_model_clone.fit(X_train, y_train)
        #predict probabilities
        y_pred_prob = best_model_clone.predict_proba(X_test)[:, 1]
        true_labels.extend(y_test)
        predicted_probs.extend(y_pred_prob)
        predicted_labels.extend(best_model_clone.predict(X_test))

        #calculate accuracy
        y_pred = best_model_clone.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_scores.append(accuracy)
        fold_sample_counts.append(len(y_test))
        #print(f"Participant {leave_out}: Accuracy = {accuracy:.4f}")
    
    # Calculate overall cross-validation metrics based on collected predictions
    cv_accuracy = accuracy_score(true_labels, predicted_labels)
    cv_f1 = f1_score(true_labels, predicted_labels)
    cv_precision = precision_score(true_labels, predicted_labels)
    cv_recall = recall_score(true_labels, predicted_labels)
    
    print(f"{model_name}: LOPO Cross-validated Accuracy: {cv_accuracy * 100:.1f}% , standard deviation: {np.std(accuracy_scores)*100:.1f}%")
    print(f"Cross-validated F1 Score: {cv_f1:.2f}, Precision: {cv_precision:.2f}, Recall: {cv_recall:.2f}")

   
    """train_sizes=np.linspace(0.1, 1.0, 10)
        

    # Plotting the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_combined, y_combined, cv=lopo, scoring='accuracy', n_jobs=-1, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10,8))
    plt.title(f"Learning Curves ({model_name} - LOPOCV)",fontsize=20)
    plt.xlabel("Training Examples",fontsize=20)
    plt.ylabel("Accuracy",fontsize=20)
    plt.xticks(fontsize=18)  # Font size for x-axis tick labels
    plt.yticks(fontsize=18)  # Font size for y-axis tick labels

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best",fontsize=18)
    plt.show()"""

    return accuracy_scores, true_labels, predicted_probs

def plot_roc_pr_curves(true_labels, predicted_probs, model_name):
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 5))

    # ROC Plot
    plt.subplot(1, 1, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')

    """# Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    average_precision = average_precision_score(true_labels, predicted_probs)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'Average Precision = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')"""

    plt.tight_layout()
    plt.show()
 


def plot_model_accuracies(model_names, loocv_accuracies,loocv_final_acc, nested_cv_accuracies, final_model_accuracies):
    # Plotting logic from previous example
    # X positions for the bars
    x = np.arange(len(model_names))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each set of accuracies with adjusted positions
    ax.bar(x - 1.5 * bar_width, loocv_accuracies, width=bar_width, label='LOOCV Best Parameters Accuracy')
    ax.bar(x - 0.5 * bar_width, loocv_final_acc, width=bar_width, label='LOOCV Final Model Accuracy')
    ax.bar(x + 0.5 * bar_width, nested_cv_accuracies, width=bar_width, label='Nested CV Accuracy')
    ax.bar(x + 1.5 * bar_width, final_model_accuracies, width=bar_width, label='Final Model Test Accuracy')

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Comparison of Model Accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_score_distribution(scores,label):
    '''
    params:scores: dictionary of scores
    Plot the distribution of scores'''
    model_names = list(scores.keys())
    plt.figure(figsize=(8, 6))

    model_names = list(scores.keys())
    plt.boxplot([scores[model] for model in model_names], labels=model_names,patch_artist=False,
                flierprops=dict(marker='o',color= 'blue',alpha = 0.5))
    plt.title('Scores')
    plt.ylabel(f'{label} Cross-Validation Accuracy')

    plt.tight_layout()
    plt.show()

def plot_all_roc_pr_curves(model_name,true_labels_all, predicted_probs_all):
    #fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    #plt.figure(figsize=(10, 8))

    for method in true_labels_all.keys():
        fpr, tpr, _ = roc_curve(true_labels_all[method], predicted_probs_all[method])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(true_labels_all[method], predicted_probs_all[method])
        avg_precision = average_precision_score(true_labels_all[method], predicted_probs_all[method])
        
        ax1.plot(fpr, tpr, label=f'{model_name} {method} (AUC = {roc_auc:.2f})')
        #ax2.plot(recall, precision, label=f'{model_name} {method} (AP = {avg_precision:.2f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate',fontsize=16)
    ax1.set_ylabel('True Positive Rate',fontsize=16)
    ax1.set_title(f'{model_name} ROC Curves for Different Cross-Validation Methods',fontsize=16)
    ax1.legend(loc='lower right',fontsize=14)

    ax1.tick_params(axis='both', which='major', labelsize=12)  # Increase the font size for major ticks
    ax1.tick_params(axis='both', which='minor', labelsize=12)

    """ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves for Different Cross-Validation Methods')
    ax2.legend(loc='lower left')"""

    plt.tight_layout()
    plt.show()

   


def main():
    # Load the preprocessed data
    preprocessed_data = pd.read_csv(r"C:\Users\githa\Documents\thesis\Analysis\Data\preprocessed_data_1.csv",index_col = False)
    #randomize the dataframe
    preprocessed_data = preprocessed_data.sample(frac=1,random_state=42).reset_index(drop=True)
    #print(preprocessed_data.shape)
    #without the sample 0 in fear and neutral
    #preprocessed_df = pd.read_csv(r'C:\Users\githa\Documents\thesis\Analysis\Data\preprocessed_data_2.csv')

    #convert the normalised intensity to arrays
    #preprocessed_df.loc[:,'normalized_Intensity'] = preprocessed_df['normalized_Intensity'].apply(np.array)
    #preprocessed_df = preprocessed_data[['normalized_Intensity','smoothed_binary_map','smoothed_Intensity', 'Label']]
    preprocessed_df = preprocessed_data[['normalized_Intensity','smoothed_peaks','non_smoothed_binary_map','smoothed_binary_map',
                                         'smoothed_Intensity','participants', 'Label']]

    
    #combine the data
    X, y = combine_data(preprocessed_df)
    #x.shape is 61,(60, 36)
    
   
    #split data into training and testing sets
    #X = np.vstack(preprocessed_df['normalized_Intensity'].values) # 61, 1547
    #X = np.vstack(preprocessed_df['smoothed_binary_map'].values) #793,119
    #scaling
    scaler = StandardScaler()

    # Initialize LDA

    X = scaler.fit_transform(X)
    print(f'shape before pca {X.shape}')
    
    pca = PCA(n_components=0.99)
    X_reduced = pca.fit_transform(X, y)

    print(X_reduced.shape)
    #print the number of components retained
    print(f"Number of components to retain 99% variance: {X_reduced.shape[1]}")


    """
    covar_matrix= PCA(n_components= min(X.shape[0],X.shape[1]))
    covar_matrix.fit(X)
    
    plt.figure(figsize=(14, 7))
    plt.ylim(0, max(covar_matrix.explained_variance_))
    plt.axhline(y=1, color='r', linestyle='--')
    plt.plot(covar_matrix.explained_variance_)
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalues')
    plt.show() 

    variance = covar_matrix.explained_variance_ratio_
    cum_variance = np.cumsum(np.round(variance, decimals=3)*100)
    plt.figure(figsize=(14, 7))
    plt.plot(cum_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.axhline(y = 80, color='r', linestyle='--')
    plt.show()   
    """
    """# Scree Plot: Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Dimensions')
    plt.ylabel('Explained Variance')
    plt.title('Scree Plot')--
    plt.grid(True)
    #plt.show()
    """
    # Retain 99% variance
    n_components_99 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1

  
    all_participants = []
    for index, row in preprocessed_df.iterrows():
        participant = np.array(ast.literal_eval(row['participants'])).flatten()

        all_participants.append(participant)
    combined_array = np.concatenate(all_participants)
    all_participants = np.unique(combined_array)
    print(all_participants)

    # classifiers
    classifiers_1 = {
        'SVM Linear': (SVC(probability=True,random_state=42), param_grid_svc_linear,n_components_99),
        'SVM RBF': (SVC(probability=True,random_state=42), param_grid_svc_rbf,n_components_99),
        'SVM Poly': (SVC(probability=True,random_state=42), param_grid_svc_poly,n_components_99),
        'SVM Sigmoid ': (SVC(probability=True,random_state=42), param_grid_svc_sig,n_components_99),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), param_grid_dt,None),
        'Random Forest': (RandomForestClassifier(random_state=42), param_grid_rf,None),
        'k-NN': (KNeighborsClassifier(), param_grid_knn,n_components_99),
        'LDA': (LDA(), param_grid_lda,n_components_99)
     
    }

    # Store accuracies for each model and cross-validation method
    true_labels_all = {}
    predicted_probs_all = {}
    accuracy_data = {
        'Model': [],
        'Accuracy': [],
        'Method': []
    }
  
    # Train and evaluate each model
    #print(f"\n--- Training and evaluating using standard classifiers ")
    for model_name, (model,param_grid,n_components) in classifiers_1.items():
      
        #plot_learning_curve_cv(model,param_grid,X,y,model_name,n_components)
        avg_accurac_loocv, true_labels_loocv, predicted_probs_loocv=train_evaluate_model_loocv_kfold(model,param_grid,X,y,model_name,n_components,'loocv',None)
        true_labels_all['LOOCV'] = true_labels_loocv
        predicted_probs_all['LOOCV'] = predicted_probs_loocv
        accuracy_data['Model'].extend([model_name] * len(avg_accurac_loocv))
        accuracy_data['Accuracy'].extend(avg_accurac_loocv)
        accuracy_data['Method'].extend(['LOOCV'] * len(avg_accurac_loocv))
                                       
    
        
        avg_accuracy_kf3, true_labels_kf3, predicted_probs_kf3=train_evaluate_model_loocv_kfold(model,param_grid,X,y,model_name,n_components,'kfold',3)
        """true_labels_all['KF3'] = true_labels_kf3
        predicted_probs_all['KF3'] = predicted_probs_kf3
        accuracy_data['Model'].extend([model_name] * len(avg_accuracy_kf3))
        accuracy_data['Accuracy'].extend(avg_accuracy_kf3)
        accuracy_data['Method'].extend(['KF3'] * len(avg_accuracy_kf3))"""
        
        
        avg_accuracy_kf5, true_labels_kf5, predicted_probs_kf5 =train_evaluate_model_loocv_kfold(model,param_grid,X,y,model_name,n_components,'kfold',5)
        """true_labels_all['KF5'] = true_labels_kf5
        predicted_probs_all['KF5'] = predicted_probs_kf5
        accuracy_data['Model'].extend([model_name] * len(avg_accuracy_kf5))
        accuracy_data['Accuracy'].extend(avg_accuracy_kf5)
        accuracy_data['Method'].extend(['KF5'] * len(avg_accuracy_kf5))"""
    
        avg_accuracy_lopo, true_labels_lopo, predicted_probs_lopo = train_evaluate_leave_one_participant_out(model_name,n_components,model,param_grid,preprocessed_df,all_participants)
        """true_labels_all['LOPOCV'] = true_labels_lopo
        predicted_probs_all['LOPOCV'] = predicted_probs_lopo
        accuracy_data['Model'].extend([model_name] * len(avg_accuracy_lopo))
        accuracy_data['Accuracy'].extend(avg_accuracy_lopo)
        accuracy_data['Method'].extend(['LOPOCV'] * len(avg_accuracy_lopo))"""
        #plot_learning_curve_cv(model,param_grid,X,y,model_name,n_components,preprocessed_df,all_participants)
        #plot_roc_pr_curves(true_labels, predicted_probs, model_name)

        #plot_all_roc_pr_curves(model_name,true_labels_all, predicted_probs_all)

    #plot_score_distribution(kfold_scores, 'KFold')
    #plot_mse(mse_values)
    df_accuracies = pd.DataFrame(accuracy_data)
    #STORE THE ACCURACIES IN A CSV FILE
    #df_accuracies.to_csv(r'C:\Users\githa\Documents\thesis\Analysis\Data\lda_accuracies.csv',index=False)
   
    data = pd.read_csv(r'C:\Users\githa\Documents\thesis\Analysis\Data\lda_accuracies.csv')
    df_accuracies = pd.concat([data,df_accuracies],axis=0)

  
    # Convert 'Model' and 'Method' columns to 'category' type if they are not already
    df_accuracies['Model'] = df_accuracies['Model'].astype('category')
    df_accuracies['Method'] = df_accuracies['Method'].astype('category')

   
 
"""    
    # Create a box plot to show accuracy distributions
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Model', y='Accuracy', hue='Method', data=df_accuracies)
    #plt.title('Distribution of Model Accuracies Across Cross-Validation Methods',fontsize=20)
    plt.title('Distribution of Model Accuracies Across LOPOCV Method',fontsize=20)
    plt.xlabel('Model',fontsize=18)
    plt.ylabel('Cross-Validation Accuracy',fontsize=18)
    plt.xticks(rotation=45,fontsize=16)
    plt.legend(title='Cross-Validation Method',fontsize=16)
    plt.show()"""
if __name__ == "__main__":
    main()
