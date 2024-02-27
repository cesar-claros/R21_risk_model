import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from optbinning import BinningProcess
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from copy import deepcopy
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
from src.utils import *

def data_preparation(df_raw):
    y = df_raw['Injury']['Post_injury']
    # Timepoints
    X_Baseline = df_raw['Baseline'].add_suffix("_Baseline")
    X_Acute = df_raw['Acute'].add_suffix("_Acute")
    X_Asymptomatic = df_raw['Asymptomatic'].add_suffix("_Asymptomatic")
    X_RTA = df_raw['RTA'].add_suffix("_RTA")
    # Baseline differences
    X_Baseline_Acute = (df_raw['Baseline'].drop(columns=['VOMS_fail']) - 
                        df_raw['Acute'].drop(columns=['VOMS_fail'])).add_suffix("_Baseline_Acute")
    X_Baseline_Asymptomatic = (df_raw['Baseline'].drop(columns=['VOMS_fail']) - 
                            df_raw['Asymptomatic'].drop(columns=['VOMS_fail'])).add_suffix("_Baseline_Asymptomatic")
    X_Baseline_RTA = (df_raw['Baseline'].drop(columns=['VOMS_fail']) - 
                    df_raw['RTA'].drop(columns=['VOMS_fail'])).add_suffix("_Baseline_RTA")
    # Injuries
    X_Injuries = df_raw['Injury'][['Time_lost','Day_of_the_year_concussion','Injury_severity_A_pre_concussion',
                                    'Injury_severity_B_pre_concussion','Pre_concussion_injuries']]
    # One-time tests
    X_OTT = df_raw['One_time_test']
    X_Demographics = df_raw['Demographics']
    X_Disorders = df_raw['Disorders']
    # 
    X_tp = reduce(lambda left,right: pd.merge(left,right,on='ID'),
                    [X_Baseline, X_Acute, X_Asymptomatic, X_RTA])
    tp_columns_sorted = sorted(X_tp.columns)
    X_tp = X_tp[tp_columns_sorted]
    #
    X_tp_diff = reduce(lambda left,right: pd.merge(left,right,on='ID'),
                    [X_Baseline_Acute, X_Baseline_Asymptomatic, X_Baseline_RTA])
    tp_diff_columns_sorted = sorted(X_tp_diff.columns)
    X_tp_diff = X_tp_diff[tp_diff_columns_sorted]
    #
    X_info =  reduce(lambda left,right: pd.merge(left,right,on='ID'),
                    [X_Injuries, X_OTT, X_Demographics, X_Disorders])
    info_columns_sorted = sorted(X_info.columns)
    X_info = X_info[info_columns_sorted]
    X_tp_and_diff = pd.merge(left=X_tp, right=X_tp_diff, on='ID')
    X_full = pd.merge(left=X_tp_and_diff, right=X_info, on='ID')
    return X_full, y

#%%
def preprocessing_WOE(X_train, X_test, y_train):
    categorical_variables = ['VOMS_fail_Acute', 'VOMS_fail_Asymptomatic',
                        'VOMS_fail_Baseline', 'VOMS_fail_RTA',
                        'Injury_severity_A_pre_concussion',
                        'Injury_severity_B_pre_concussion',
                        'LOC', 'Hx_concussion', 'Hx_contact sport',
                        'contact level', 'Anxiety', 'ADD_ADHD', 'Depression',
                        'Psychiatric_Disorder', 'LD','Sports','Sex']
    variable_names = list(X_train.columns)
    binning_fit_params = {
        'VOMS_fail_Acute': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'VOMS_fail_Baseline': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'VOMS_fail_RTA': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'Anxiety': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'LOC': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'LD': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'Depression': {"user_splits":np.array([['no'],['yes']], dtype=object) },
        'Sex': {"user_splits":np.array([['male'],['female']], dtype=object) },
    }
    binning_process = BinningProcess(variable_names,
                                    categorical_variables=categorical_variables,
                                    binning_fit_params=binning_fit_params,
                                    )
    # Fit and transform training set 
    X_train_WOE = binning_process.fit_transform(X_train,y_train)
    X_train_WOE = -1*X_train_WOE
    # Transform test set
    X_test_WOE = binning_process.transform(X_test)
    X_test_WOE = -1*X_test_WOE
    return X_train_WOE, X_test_WOE 

def preprocessing_mean(X_train, X_test):
    imp_mean = DataFrameImputer()
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")
    scaler = StandardScaler().set_output(transform="pandas")
    X_train_mean = imp_mean.fit_transform(X_train)
    X_test_mean = imp_mean.transform(X_test)
    X_train_mean_cat = one_hot_encoder.fit_transform(X_train_mean.select_dtypes(include=['object']).copy())
    X_test_mean_cat = one_hot_encoder.transform(X_test_mean.select_dtypes(include=['object']).copy())
    X_train_mean_num = scaler.fit_transform(X_train_mean.select_dtypes(exclude=['object']).copy())
    X_test_mean_num = scaler.transform(X_test_mean.select_dtypes(exclude=['object']).copy())
    X_train_mean_all = pd.merge(right=X_train_mean_num,left=X_train_mean_cat,on='ID')
    X_test_mean_all = pd.merge(right=X_test_mean_num,left=X_test_mean_cat,on='ID')
    return X_train_mean_all, X_test_mean_all


def feature_selection(X_train_, y_train, reg_path):
    eps = np.finfo(float).eps
    n_train = y_train.shape[0]
    seed = 123
    y_train_ = y_train.astype(int).reset_index(drop=True)
    results = dict()
    cs = reg_path
    clf = LogisticRegression(
        class_weight='balanced',
        penalty="l1",
        solver="saga", #"saga"
        tol=1e-6,
        max_iter=int(1e6),
        warm_start=True,
        random_state=seed,
    )
    n_coefs_ = []
    weights_ = []
    clfs_ = []
    ll_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X_train_, y_train_)
        clf_weights = clf.coef_.ravel().copy()
        k = np.count_nonzero(clf_weights)
        p = np.clip(clf.predict_proba(X_train_),eps,1)
        ll = (y_train_*np.log(p[:,1])+(1-y_train_)*np.log(p[:,0])).sum()
        clfs_.append(deepcopy(clf))
        weights_.append(clf_weights)
        n_coefs_.append(k)
        ll_.append(ll)
    K_ = np.array(n_coefs_)+1
    LL_ = np.array(ll_)
    results.update({
        'n_coefs' : np.array(n_coefs_),
        'weights' : np.array(weights_),
        'clfs' : clfs_, 
        'll' : LL_,
        'aic' : -2*LL_ + 2*K_,
        'aicc' : -2*LL_ + 2*K_ + (2*K_*(K_+1))/(n_train-K_-1),
        'bic' : -2*LL_ + np.log(n_train)*(K_),
        'cs': cs
    })
    return results            

def plot_metrics_fs(results_boostrap, cs_dict, i=0, 
                    color={'WOE':'blue','MEAN':'orange'}, 
                    linestyle={'WOE':'--','MEAN':':'}):
    N_trials = len(results_boostrap)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    # Plot log likelihood vs regularization parameter
    fig_ll,ax_ll = plot_fs(results_boostrap,cs_dict,'ll',reverse=True)
    ax_ll.set_ylabel(r"$-\log(L(\boldsymbol{w}_{\lambda}^{\star}))$")
    ax_ll.set_xlabel(r"$1/\lambda$")
    ax_ll.set_ylabel(r"$-\log(L(\boldsymbol{w}_{\lambda}^{\star}))$")
    # ax.set_xlabel(r"$\frac{1}{\lambda}$")
    ax_ll.set_xlabel(r"$1/\lambda$")
    fig_ll.savefig(f'plots/feature_selection_ll.pdf',bbox_inches='tight')
    
    # Plot cardinality non-zero coefficients vs regularization parameter
    fig_K,ax_K = plot_fs(results_boostrap,cs_dict,'n_coefs')
    ax_K.set_ylabel(r"$K(\boldsymbol{w}^{\star}_{\lambda})$")
    ax_K.set_xlabel(r"$1/\lambda$")
    ax_K.set_ylabel(r"$K(\boldsymbol{w}^{\star}_{\lambda})$")
    ax_K.set_xlabel(r"$1/\lambda$")
    fig_K.savefig(f'plots/feature_selection_card.pdf',bbox_inches='tight')
    
    # Plot cardinality AIC vs regularization parameter
    fig_aic,ax_aic = plot_fs(results_boostrap,cs_dict,'aic')
    ax_aic.set_ylabel(r"$AIC(\lambda)$")
    ax_aic.set_xlabel(r"$1/\lambda$")
    woe_aic_avg = np.array([results_boostrap[str(i)]['WOE']['aic'] for i in range(N_trials)]).mean(axis=0)
    mean_aic_avg = np.array([results_boostrap[str(i)]['MEAN']['aic'] for i in range(N_trials)]).mean(axis=0)
    woe_n_coeff_avg = np.array([results_boostrap[str(i)]['WOE']['n_coefs'] for i in range(N_trials)]).mean(axis=0)
    mean_n_coeff_avg = np.array([results_boostrap[str(i)]['MEAN']['n_coefs'] for i in range(N_trials)]).mean(axis=0)

    results = results_boostrap[str(i)]
    ax_aic.axvline(results['WOE']['cs'][np.argmin(woe_aic_avg)],
                color=color['WOE'], linestyle='-',alpha=0.4)
    ax_aic.axvline(results['MEAN']['cs'][np.argmin(mean_aic_avg)],
                color=color['MEAN'], linestyle='-',alpha=0.4)
    props_woe = dict(boxstyle='round', facecolor=color['WOE'], alpha=0.4, linestyle=linestyle['WOE'], linewidth=2)
    props_mean = dict(boxstyle='round', facecolor=color['MEAN'], alpha=0.4, linestyle=linestyle['MEAN'], linewidth=2)
    k_woe = woe_n_coeff_avg[np.argmin(woe_aic_avg)]
    k_mean = mean_n_coeff_avg[np.argmin(mean_aic_avg)]
    ax_aic.text(30,
            150,r"$K(\boldsymbol{w}^{\star}_{\lambda})=$"+f'{int(k_woe)}', bbox=props_woe)
    ax_aic.text(0.15,
            250,r"$K(\boldsymbol{w}^{\star}_{\lambda})=$"+f'{int(k_mean)}', bbox=props_mean)
    ax_aic.legend(loc=1)
    ax_aic.set_ylabel(r"$AIC(\lambda)$")
    ax_aic.set_xlabel(r"$1/\lambda$")
    fig_aic.savefig(f'plots/feature_selection_aic.pdf',bbox_inches='tight')
    
    # Plot cardinality AICc vs regularization parameter
    fig_aicc,ax_aicc = plot_fs(results_boostrap,cs_dict,'aicc')
    ax_aicc.set_ylim(100,1500)
    ax_aicc.set_ylabel(r"$AICc(\lambda)$")
    ax_aicc.set_xlabel(r"$1/\lambda$")
    woe_aicc_avg = np.array([results_boostrap[str(i)]['WOE']['aicc'] for i in range(N_trials)]).mean(axis=0)
    mean_aicc_avg = np.array([results_boostrap[str(i)]['MEAN']['aicc'] for i in range(N_trials)]).mean(axis=0)
    ax_aicc.axvline(results['WOE']['cs'][np.argmin(woe_aicc_avg)],
                color=color['WOE'], linestyle='-',alpha=0.4)
    ax_aicc.axvline(results['MEAN']['cs'][np.argmin(mean_aicc_avg)],
                color=color['MEAN'], linestyle='-',alpha=0.4)
    props_woe = dict(boxstyle='round', facecolor=color['WOE'], alpha=0.4, linestyle=linestyle['WOE'], linewidth=2)
    props_mean = dict(boxstyle='round', facecolor=color['MEAN'], alpha=0.4, linestyle=linestyle['MEAN'], linewidth=2)
    k_woe = woe_n_coeff_avg[np.argmin(woe_aicc_avg)]
    k_mean = mean_n_coeff_avg[np.argmin(mean_aicc_avg)]
    ax_aicc.text(30,
            400,r"$K(\boldsymbol{w}^{\star}_{\lambda})=$"+f'{int(k_woe)}', bbox=props_woe)
    ax_aicc.text(0.05,
            600,r"$K(\boldsymbol{w}^{\star}_{\lambda})=$"+f'{int(k_mean)}', bbox=props_mean)
    ax_aicc.legend(loc=1)
    ax_aicc.set_ylabel(r"$AICc(\lambda)$")
    ax_aicc.set_xlabel(r"$1/\lambda$")
    fig_aicc.savefig(f'plots/feature_selection_aicc.pdf',bbox_inches='tight')


def fit(X_train_, y_train, fs_model):
    results_clf = dict()
    y_train_ = y_train.astype(int)
    fs = SelectFromModel(estimator=fs_model,prefit=True,threshold=1e-6)
    model = LogisticRegression(class_weight='balanced', solver='saga', 
                                penalty='l2',
                                C=1.0,
                                random_state=123, tol=1e-6, max_iter=int(1e6))
    pipeline_model = Pipeline(steps = [('feature_selection',fs),
                                        ('model', model)]
                                )
    pm = pipeline_model.fit(X_train_,y_train_)
    df_lr_coef = pd.DataFrame({'variable':list(pm[0].get_feature_names_out()),
                                'coefficient':list(pm[-1].coef_.reshape(-1).round(3))}).sort_values(by=['variable'])
    results_clf.update({
                        'coeff':df_lr_coef,
                        'pipeline':pm,
                        })
    return results_clf

def plot_predictions(X_test_, y_test, df_raw, pm, pp_label):
    y_test_ = y_test.astype(int)
    stats_test = stat_test(pm[:-1].transform(X_test_),y_test_.reset_index(drop=True),pm[-1].coef_.T,pm[-1].intercept_)
    box_plot_fig = boxplot_MSK(pm[:-1].transform(X_test_), y_test_.astype(bool).reset_index(drop=True), pm[-1].coef_.T, pm[-1].intercept_, 
                               stats_test['p-val']['T-test'],label=f'Test Set ({pp_label})')
    box_plot_fig.savefig(f'plots/box_plot_test_{pp_label}.pdf',bbox_inches='tight')

    predictions_severity_test = get_severity(df_raw,X_test_,y_test_.astype(bool),pm)
    bucketed_probs_test_fig  = plot_bucketed_probabilities(predictions_severity_test, plot_type='boxplot', alpha=0.5)
    bucketed_probs_test_fig.savefig(f'plots/bucketed_probs_test_{pp_label}_.pdf',bbox_inches='tight')

def plot_eval_metrics(X_test_WOE, X_test_mean_all, y_test, results_clf, 
                      color={'WOE':'blue','MEAN':'orange'}):
    y_test_ = y_test.astype(int)
    
    fig, ax = plt.subplots(figsize=(7,7))
    roc_lr_woe = metrics.RocCurveDisplay.from_estimator(results_clf['WOE']['pipeline'], X_test_WOE, y_test_, name='WoE', drop_intermediate=True, color=color['WOE'], ax=ax)
    roc_lr_mean = metrics.RocCurveDisplay.from_estimator(results_clf['MEAN']['pipeline'], X_test_mean_all, y_test_, name='Mean imputation', drop_intermediate=True, color=color['MEAN'], plot_chance_level=True, ax=roc_lr_woe.ax_)
    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate (Positive: subsequent MSK)")
    ax.set_ylabel("True Positive Rate (Positive: subsequent MSK)")
    ax.set_aspect('equal')
    ax.grid(color='gray', linestyle=':', linewidth=1, alpha=0.7)
    fig.savefig(f'plots/ROC_test.pdf',bbox_inches='tight')
    
    precision_WOE, recall_WOE, ths_WOE = metrics.precision_recall_curve(y_test.astype(int).values, results_clf['WOE']['pipeline'].predict_proba(X_test_WOE)[:,1])
    precision_MEAN, recall_MEAN, ths_MEAN = metrics.precision_recall_curve(y_test.astype(int).values, results_clf['MEAN']['pipeline'].predict_proba(X_test_mean_all)[:,1])
    f1_scores_WOE = 2 * (recall_WOE * precision_WOE) / (recall_WOE + precision_WOE)
    best_f1_WOE = np.max(f1_scores_WOE)
    best_precision_WOE, best_recall_WOE = precision_WOE[np.argmax(f1_scores_WOE)], recall_WOE[np.argmax(f1_scores_WOE)]

    f1_scores_MEAN = 2 * (recall_MEAN * precision_MEAN) / (recall_MEAN + precision_MEAN)
    best_f1_MEAN = np.max(f1_scores_MEAN)
    best_precision_MEAN, best_recall_MEAN = precision_MEAN[np.argmax(f1_scores_MEAN)], recall_MEAN[np.argmax(f1_scores_MEAN)]
    
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10")

    fig, ax = plt.subplots(figsize=(7,7))
    pr_lr_woe = metrics.PrecisionRecallDisplay.from_estimator(results_clf['WOE']['pipeline'], X_test_WOE, y_test, name='WoE', drop_intermediate=True, color=color['WOE'], plot_chance_level=False, ax=ax)
    pr_lr_mean = metrics.PrecisionRecallDisplay.from_estimator(results_clf['MEAN']['pipeline'], X_test_mean_all, y_test, name='Mean imputation', drop_intermediate=True, color=color['MEAN'], plot_chance_level=True, ax=pr_lr_woe.ax_)
    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1, 50)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("F1={0:0.2f}".format(f_score), xy=(0.94, y[45] - 0.01))
    offset_WOE = 20
    ax.annotate(r'$F1^{\star}_{WoE}=$'+f'{best_f1_WOE:.2f}\n' +
                r'$P^{\star}_{WoE}=$'+f'{best_precision_WOE:.2f}\n' +
                r'$R^{\star}_{WoE}=$'+f'{best_recall_WOE:.2f}',(best_recall_WOE, best_precision_WOE), 
                xytext=(offset_WOE, -0.0*offset_WOE), textcoords='offset points',
                bbox=bbox, arrowprops=arrowprops)
    offset_MEAN = 30
    ax.annotate(r'$F1^{\star}_{mean}=$'+f'{best_f1_MEAN:.2f}\n' +
                r'$P^{\star}_{mean}=$'+f'{best_precision_MEAN:.2f}\n' +
                r'$R^{\star}_{mean}=$'+f'{best_recall_MEAN:.2f}',(best_recall_MEAN, best_precision_MEAN), 
                xytext=(-4*offset_MEAN, -2*offset_MEAN), textcoords='offset points',
                bbox=bbox, arrowprops=arrowprops)
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(0.0,1.05)
    ax.set_ylim(0.0,1.05)
    ax.set_title("Precision-Recall curve")
    ax.set_aspect('equal')
    ax.grid(color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel("Recall (Positive: subsequent MSK)")
    ax.set_ylabel("Precision (Positive: subsequent MSK)")
    fig.savefig(f'plots/PR_test.pdf',bbox_inches='tight')