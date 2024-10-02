import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score,accuracy_score
from sklearn.tree import plot_tree
from sklearn.preprocessing import label_binarize
#from sklearn.metrics import plot_roc_curve
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import openpyxl
# Load your dataset here
# data = pd.read_csv('your_data.csv')

# For demonstration, create a dummy DataFrame
np.random.seed(42)
data=pd.read_excel("Compiled_2017.xlsx",sheet_name='Compiled_2017')

# Preprocess data
X = data[['Hosp_Type','TypeofData','Type_of_Breach', 'CausalAgent','Crime','Literacy', 'PCI', 'Density', 'Policy', 'Connectivity']]
y = data['Type']

# Encode categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Balance the dataset
oversample = SMOTE()
undersample = RandomUnderSampler()
pipeline = Pipeline(steps=[('o', oversample), ('u', undersample)])
X_resampled, y_resampled = pipeline.fit_resample(X, y)
# Define classifiers
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
classifiers = {
    'CART': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Bagging Tree': BaggingClassifier(DecisionTreeClassifier(random_state=0, criterion='entropy'), n_estimators=50, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(silent=True),
    'LightGBM': lgb.LGBMClassifier()
}
# Store results
results = []
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
# Train and evaluate classifiers

results_df = pd.DataFrame(columns=['Classifier', 'Accuracy', 'AUC'])

class_names = ['DISC', 'HACK', 'ID', 'PHYS']
for i, (name, clf) in enumerate(classifiers.items(), start=1):
    print(i,name, clf)
    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # # AUC Score
    # if y_prob is not None:
    #     auc = roc_auc_score(y_test, y_prob, multi_class='ovo')
    # else:
    #     auc = np.nan

    # Store results in dataframe
    results_df = results_df._append({'Classifier': name, 'Accuracy': acc, 'AUC': auc}, ignore_index=True)
    #
    # #Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.subplot(len(classifiers), 2, 2 * i - 1)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title(f'{name} Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # ROC Curve
    # Predict probabilities (needed for ROC)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        y_prob = None
    if y_prob is not None:
        # For multiclass classification (ovr scheme)
        fpr = {}
        tpr = {}
        roc_auc = {}
        # For multiclass, one-vs-rest ROC curve
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plotting the ROC Curve
        plt.figure(figsize=(7, 7))
        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print(f"Classifier {name} does not support predict_proba, skipping ROC curve.")

class_names = ['DISC', 'HACK', 'ID', 'PHYS']
# Iterate over classifiers
roc_values = {class_name: {} for class_name in class_names}
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict probabilities (needed for ROC)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        y_prob = None

    # If the classifier has predict_proba, calculate the ROC curve
    if y_prob is not None:
        for i, class_name in enumerate(class_names):
            print(class_name)
            # Compute ROC curve and AUC for each class
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)

            # Store the ROC values for this class and classifier
            roc_values[class_name][name] = (fpr, tpr, roc_auc)

# Plot ROC curves for each class, comparing classifiers
for i, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 8))
    for clf_name, (fpr, tpr, roc_auc) in roc_values[class_name].items():
        plt.plot(fpr, tpr, lw=2, label=f'{clf_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison for Class: {class_name}')
    plt.legend(loc='lower right')
    plt.show()
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=class_names, rounded=True, max_depth=3, fontsize=10)
plt.title('Decision Tree (Max Depth = 3)')
plt.show()
y=pd.DataFrame(y_resampled,columns=['Type of Breach'])
df_merged = pd.concat([X_resampled, y], axis=1)