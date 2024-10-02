import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular

# Set random seed for reproducibility
np.random.seed(42)
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
# Load dataset
data = pd.read_excel("Compiled_2017.xlsx", sheet_name='Compiled_2017')

# Feature and target separation
X = data[['Hosp_Type', 'TypeofData', 'Type_of_Breach', 'CausalAgent', 'Crime', 'Literacy', 'PCI', 'Density', 'Policy', 'Connectivity']]
y = data['Type']
# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])  # Use .loc to avoid SettingWithCopyWarning
    label_encoders[column] = le

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coerce errors to NaN
X.fillna(0, inplace=True)  # Optionally handle NaNs by filling them with zeros or another strategy

# SMOTE for balancing
oversample = SMOTE()
undersample = RandomUnderSampler()
pipeline = Pipeline(steps=[('o', oversample), ('u', undersample)])
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define classifiers
classifiers = {
    'CART': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Bagging Tree': BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'CatBoost': CatBoostClassifier(silent=True, random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# Initialize results dataframe
results_df = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Cross-validation Accuracy', 'AUC'])
results = []
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
# Define the number of cross-validation folds
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SHAP and LIME variable importance
for name, clf in classifiers.items():
    print(clf)
    clf.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation accuracy
    cross_val_acc = np.mean(cross_val_score(clf, X_resampled, y_resampled, cv=cv_folds))

    # Calculate AUC if supported
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        auc_score = np.nan

    # Append results
    results_df = results_df._append({'Classifier': name, 'Accuracy': acc, 'Cross-validation Accuracy': cross_val_acc, 'AUC': auc_score}, ignore_index=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Check for overfitting and underfitting
for name, clf in classifiers.items():
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"{name} - Training Accuracy: {train_acc:.2f}, Testing Accuracy: {test_acc:.2f}")

# Show final results
print(results_df)
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


classifiers_shap = {
    #'CART': DecisionTreeClassifier(),
    #'Bagging Tree': BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'CatBoost': CatBoostClassifier(silent=True, random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}
import sklearn
import xgboost as xgb
from catboost import CatBoostRegressor
for name, clf in classifiers_shap.items():
    print(name)
    # if name=='CART':
    #     cart_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    #     cart_model.fit(X_train, y_train)
    #     # Tree on Random Forest explainer
    #     explainer = shap.TreeExplainer(cart_model)
    # if name=='Bagging Tree':
    #     bagging_model  = BaggingClassifier(DecisionTreeClassifier(max_depth=3, random_state=42), n_estimators=50, random_state=42)
    #     bagging_model .fit(X_train, y_train)
    #     # Tree on Random Forest explainer
    #     explainer = shap.TreeExplainer(bagging_model )
    if name=='Random Forest':
        rf = sklearn.ensemble.RandomForestRegressor()
        rf.fit(X_train, y_train)
        # Tree on Random Forest explainer
        explainer = shap.TreeExplainer(rf)
    if name == 'XGBoost':
        xgb_model = xgb.train({'objective':'reg:linear'}, xgb.DMatrix(X_train, label=y_train))
        # Tree on Random Forest explainer
        explainer = shap.TreeExplainer(xgb_model)
    if name == 'CatBoost':
        cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, silent=True)
        cat_model.fit(X_train, y_train)
        explainer = shap.Explainer(cat_model,X_train)
    if name == 'LightGBM':
        lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        lgb_model.fit(X_train, y_train)
        explainer = shap.Explainer(lgb_model,X_train)

    shap_values_test = explainer.shap_values(X_test)
    #shap_values_train = explainer.shap_values(X_train)
    print(f"SHAP Summary Plot for {name}")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Variable Importance for {name}')
    plt.show()

    #### lIME
    # Identifying categorical features
    l_clf = lgb.LGBMClassifier(num_leaves=1024, learning_rate=0.01, n_estimators=5000, boosting_type="gbdt",
                               min_child_samples=100,verbosity=0)