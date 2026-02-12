import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor as task
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def runAutoGluon(outcome, df, save_path, quality, type, metric):
    predictor = task(label=outcome, problem_type=type, eval_metric=metric, path=save_path).fit(train_data=df, presets=quality)
    
    return predictor


def split_data_into_array_by_index(target, df, ratio):
    df_train_readmitonly = df[df[target]==1]
    df_train_noreadmitonly = df[df[target]==0]
    
    number_noreadmitonly = df_train_noreadmitonly.shape[0]
    number_each_iteration = np.ceil(number_noreadmitonly/ratio).astype(int)
    
    index_array_noreadmitonly = np.array(df_train_noreadmitonly.index.tolist())
    
    splits_list = np.array_split(index_array_noreadmitonly, ratio)
    return splits_list, df_train_readmitonly, df_train_noreadmitonly


def calculate_scores_normalized_truth(df, target, y_pred_proba_mean, y_pred_mean):
    auc = roc_auc_score(df[target], y_pred_proba_mean[1])
    print("ROC_AUC: %.3f" % auc)
    
    f1 = f1_score(df[target], y_pred_mean)
    print("F1: %.3f" % f1)

    f1_micro = f1_score(df[target], y_pred_mean, average='micro')
    print("F1 micro: %.3f" % f1_micro)

    f1_macro = f1_score(df[target], y_pred_mean, average='macro')
    print("F1 macro: %.3f" % f1_macro)

    print("\nConfusion Matrix normalized by Truth")
    cm = confusion_matrix(df[target], y_pred_mean, labels=[0, 1], normalize='true')
    print(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=[0, 1])
    disp.plot()


def calculate_scores_normalized_predicted(df, target, y_pred_proba_mean, y_pred_mean):
    print("\nConfusion Matrix normalized by Predicted")
    cm2 = confusion_matrix(df[target], y_pred_mean, labels=[0, 1], normalize='pred')
    print(cm2)
    
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                   display_labels=[0, 1])
    disp2.plot()


def calculate_scores_normalized_all(df, target, y_pred_proba_mean, y_pred_mean):
    auc = roc_auc_score(df[target], y_pred_proba_mean[1])
    print("ROC_AUC: %.3f" % auc)
    
    f1 = f1_score(df[target], y_pred_mean)
    print("F1: %.3f" % f1)

    f1_micro = f1_score(df[target], y_pred_mean, average='micro')
    print("F1 micro: %.3f" % f1_micro)

    f1_macro = f1_score(df[target], y_pred_mean, average='macro')
    print("F1 macro: %.3f" % f1_macro)

    print("\nConfusion Matrix normalized by total count")
    cm2 = confusion_matrix(df[target], y_pred_mean, labels=[0, 1], normalize='all')
    print(cm2)
    
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                   display_labels=[0, 1])
    disp2.plot()


def evaluate_ensemble(predictor, y_pred_proba, y_pred, y_true):
    perf_list = list()
    cm_list = list()

    perf = predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred_proba, auxiliary_metrics=True)
    print(perf)
    perf_list.append(perf)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize='all')
    print("\nConfusion Matrix normalized by all counts")
    print(cm)
    cm_list.append(cm)


def ensemble_prediction(quality, metric, df, target, evaluate=False, verbose=False):
    y_pred_proba_list = list()
    # y_pred_proba_shap_list = list()
    
    y_pred_list = list()
    predictor_list =list()

    save_path = quality + '_' + metric
    print("\n")
    print(save_path)
    predictor = task.load(save_path)
    predictor_list.append(predictor)

    if verbose:
        print(predictor.leaderboard(df, silent=True))
        print("\n")
        print(predictor.feature_importance(data=df))
        print("\n")

    df_notarget = df.drop(columns=[target], axis=1)
    y_pred_proba = predictor.predict_proba(df_notarget)
    # y_pred_proba_shap = predictor.predict_proba(df_notarget, as_multiclass=False, as_pandas =False)

    y_pred_proba_list.append(y_pred_proba)
    # y_pred_proba_shap_list.append(y_pred_proba_shap)

    y_pred = predictor.predict(df_notarget)
    y_pred_list.append(y_pred)

    if evaluate:
        evaluate_ensemble(predictor, y_pred_proba, y_pred, df[target])
        
    return y_pred_proba_list, y_pred_list, predictor_list


def ensemble_predictions_from_splits(quality, metric, df, target, number_splits, evaluate=False, verbose=False):
    y_pred_proba_list = list()
    # y_pred_proba_shap_list = list()
    
    y_pred_list = list()
    predictor_list =list()

    for split in range(number_splits):
        save_path = str(number_splits) + '/split_' + str(split) + '_' + quality + '_' + metric
        print("\n")
        print(save_path)
        predictor = task.load(save_path)
        predictor_list.append(predictor)

        if verbose:
            print(predictor.leaderboard(df, silent=True))
            print("\n")
            print(predictor.feature_importance(data=df))
            print("\n")

        df_notarget = df.drop(columns=[target], axis=1)
        y_pred_proba = predictor.predict_proba(df_notarget)
        # y_pred_proba_shap = predictor.predict_proba(df_notarget, as_multiclass=False, as_pandas =False)

        y_pred_proba_list.append(y_pred_proba)
        # y_pred_proba_shap_list.append(y_pred_proba_shap)

        y_pred = predictor.predict(df_notarget)
        y_pred_list.append(y_pred)

        if evaluate:
            evaluate_ensemble(predictor, y_pred_proba, y_pred, df[target])
        
    return y_pred_proba_list, y_pred_list, predictor_list


def ensemble_proba(y_pred_proba_list):
    proba_df = y_pred_proba_list[0]
    for df in y_pred_proba_list[1:]:
        proba_df = proba_df + df

    y_pred_proba_mean = proba_df/len(y_pred_proba_list)

    return y_pred_proba_mean


def ensemble_pred(y_pred_list, target, ratio):
    y_pred = y_pred_list[0].to_frame()
    for df in y_pred_list[1:]:
        y_pred = y_pred + df.to_frame()

    y_pred[target] = y_pred[target].apply(lambda x: 1 if x >= np.ceil(ratio/2)+3 else 0)
    print(y_pred.value_counts())
    return y_pred


def ensemble_pred_from_proba(y_pred_proba_mean, threshold):
    y_pred = y_pred_proba_mean[1].apply(lambda x: 1 if x > threshold else 0)
    return y_pred


def determine_threshold(df, target, y_pred_proba_mean):
    thres_list = []
    for threshold in np.arange(0.5, 0.7, 0.01):
        y_pred = ensemble_pred_from_proba(y_pred_proba_mean, threshold)
        cm = confusion_matrix(df[target], y_pred, labels=[0, 1], normalize='true')
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

        precison = precision_score(df[target], y_pred)
        recall = recall_score(df[target], y_pred)
        
        thres_list.append([threshold, FN + FP])

    df_thres = pd.DataFrame(thres_list, columns=['threshold', 'sum'])
    
    return df_thres['threshold'].iloc[df_thres[['sum']].idxmin()].values[0]


def single_point_explainer(NSHAP_SAMPLES, single_datapoint, explainer):  
    shap_values_single = explainer.shap_values(single_datapoint, nsamples=NSHAP_SAMPLES)
    shap.force_plot(explainer.expected_value, shap_values_single, single_datapoint, matplotlib=True)


class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, as_multiclass=False)

    