#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from functools import reduce
# from optbinning import BinningProcess
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC, LinearSVC
# from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, cross_val_score, cross_validate
# from sklearn.feature_selection import RFECV
# from sklearn.linear_model import LogisticRegression, Lasso, LassoLarsIC, LassoCV, LogisticRegressionCV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from sklearn.pipeline import Pipeline, make_pipeline
# from celer import GroupLassoCV, LassoCV
# from sklearn.feature_selection import SelectFromModel
# from sklearn import metrics
import pingouin as pg
# from sksurv.compare import compare_survival
# from sklearn.impute import SimpleImputer
# from utils import plot
import matplotlib.ticker as mtick
# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import l1_min_c
# import statsmodels.api as sm
# from sklearn.utils import resample
# from sklearn.base import clone
# from scipy.stats import gaussian_kde

#%%
# import shap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.exceptions import NotFittedError
# from copy import deepcopy
#%%
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
#%%
def _check_is_built(table):
    if not table._is_built:
        raise NotFittedError("This {} instance is not built yet. Call "
                             "'build' with appropriate arguments."
                             .format(table.__class__.__name__))

def _bin_str_label_format(bin_str, max_length=27):
    _bin_str = []
    for bs in bin_str:
        label = str(bs)
        if len(label) > max_length:
            label = label[:max_length] + '...'
        _bin_str.append(label)

    return _bin_str

#%%
def plot_woe(binning_table, raw_variable=None, target=None, metric="woe", add_special=False, add_missing=False,
            reverse_woe=False, style="bin", show_bin_labels=False, log_scale=False, labels_events=None,
            legend_position="upper center", legend_xy=(0,0), y_count_label=[1.5,1.3], y_shift=0.0, figsize=(8,6)):
    """Plot the binning table.

    Visualize the non-event and event count, and the Weight of Evidence or
    the event rate for each bin.

    Parameters
    ----------
    metric : str, optional (default="woe")
        Supported metrics are "woe" to show the Weight of Evidence (WoE)
        measure and "event_rate" to show the event rate.

    add_special : bool (default=True)
        Whether to add the special codes bin.

    add_missing : bool (default=True)
        Whether to add the special values bin.

    style : str, optional (default="bin")
        Plot style. style="bin" shows the standard binning plot. If
        style="actual", show the plot with the actual scale, i.e, actual
        bin widths.

    show_bin_labels : bool (default=False)
        Whether to show the bin label instead of the bin id on the x-axis.
        For long labels (length > 27), labels are truncated.

        .. versionadded:: 0.15.1

    savefig : str or None (default=None)
        Path to save the plot figure.
    """
    _check_is_built(binning_table)

    if metric not in ("event_rate", "woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                            'values are "event_rate" and "woe".')

    if not isinstance(add_special, bool):
        raise TypeError("add_special must be a boolean; got {}."
                        .format(add_special))

    if not isinstance(add_missing, bool):
        raise TypeError("add_missing must be a boolean; got {}."
                        .format(add_missing))

    if style not in ("bin", "actual"):
        raise ValueError('Invalid value for style. Allowed string '
                            'values are "bin" and "actual".')

    if not isinstance(show_bin_labels, bool):
        raise TypeError("show_bin_labels must be a boolean; got {}."
                        .format(show_bin_labels))

    if show_bin_labels and style == "actual":
        raise ValueError('show_bin_labels only supported when '
                            'style="actual".')

    if style == "actual":
        # Hide special and missing bin
        add_special = False
        add_missing = False

        if binning_table.dtype == "categorical":
            raise ValueError('If style="actual", dtype must be numerical.')

        elif binning_table.min_x is None or binning_table.max_x is None:
            raise ValueError('If style="actual", min_x and max_x must be '
                                'provided.')

    if metric == "woe":
        metric_values = binning_table._woe
        metric_label = "WoE"
        if reverse_woe:
            metric_values = -1*metric_values
    elif metric == "event_rate":
        metric_values = binning_table._event_rate
        metric_label = "Event rate"

    fig, ax1 = plt.subplots(figsize=figsize)

    if style == "bin":
        n_bins = len(binning_table._n_records)
        n_metric = n_bins - 1 - binning_table._n_specials

        if len(binning_table.cat_others):
            n_metric -= 1

        _n_event = list(binning_table.n_event)
        _n_nonevent = list(binning_table.n_nonevent)
        _n_events = list(binning_table.n_event+binning_table.n_nonevent)
        if not add_special:
            n_bins -= binning_table._n_specials
            for _ in range(binning_table._n_specials):
                _n_event.pop(-2)
                _n_nonevent.pop(-2)
                _n_events.pop(-2)

        if not add_missing:
            _n_event.pop(-1)
            _n_nonevent.pop(-1)
            _n_events.pop(-1)
            n_bins -= 1

        p2 = ax1.bar(range(n_bins), _n_event, color="tab:red",
                     bottom=_n_nonevent)
        p1 = ax1.bar(range(n_bins), _n_nonevent, color="tab:blue",
                        )
        # total_values = binning_table.n_event+binning_table.n_nonevent
        for i, total in enumerate(_n_events):
            y_added = 0
            if i>=1:
                if np.abs(_n_events[i]-_n_events[i-1]) <= 1:
                    y_added = y_shift

            ax1.text(i+ y_count_label[0], total+1  + y_added, f'$n_{{MSK}}={_n_event[i]}$',
                ha = 'center', weight = 'bold', color = 'black', fontsize=11, rotation=90)
            ax1.text(i+ y_count_label[1], total+1  + y_added, f'$n_{{non-MSK}}={_n_nonevent[i]}$',
                ha = 'center', weight = 'bold', color = 'black', fontsize=11, rotation=90)

        handles = [p1[0], p2[0]]
        if labels_events==None:
            labels = ['Non-event', 'Event']
        else:
            labels = labels_events

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)
        ax1_ylims = ax1.get_ylim()
        ax1.set_ylim([ax1_ylims[0],ax1_ylims[1]*1.05])
        ax2 = ax1.twinx()

        ax2.plot(range(n_metric), metric_values[:n_metric],
                    linestyle="solid", marker="o", color="black")
        # ax2.plot(range(n_metric), 0*metric_values[:n_metric],
        #             linestyle="dashed", marker="", color="black")

        # Positions special and missing bars
        pos_special = 0
        pos_missing = 0

        if add_special:
            pos_special = n_metric
            if add_missing:
                pos_missing = n_metric + binning_table._n_specials
        elif add_missing:
            pos_missing = n_metric 

        # Add points for others (optional), special and missing bin
        if len(binning_table.cat_others):
            pos_others = n_metric
            pos_special += 1
            pos_missing += 1

            p1[pos_others].set_alpha(0.5)
            p2[pos_others].set_alpha(0.5)

            ax2.plot(pos_others, metric_values[pos_others], marker="o",
                        color="black")

        if add_special:
            for i in range(binning_table._n_specials):
                p1[pos_special + i].set_hatch("/")
                p2[pos_special + i].set_hatch("/")

            handle_special = mpatches.Patch(hatch="/", alpha=0.1)
            label_special = "Bin special"

            for s in range(binning_table._n_specials):
                ax2.plot(pos_special+s, metric_values[pos_special+s],
                            marker="o", color="black")

        if add_missing:
            p1[pos_missing].set_hatch("\\")
            p2[pos_missing].set_hatch("\\")
            handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
            label_missing = "Bin missing"

            ax2.plot(pos_missing, metric_values[pos_missing+1], marker="o",
                        color="black")

        if add_special and add_missing:
            handles.extend([handle_special, handle_missing])
            labels.extend([label_special, label_missing])
        elif add_special:
            handles.extend([handle_special])
            labels.extend([label_special])
        elif add_missing:
            handles.extend([handle_missing])
            labels.extend([label_missing])

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        if show_bin_labels:
            # xlabels = ax1.set_xlabel("Bin", fontsize=12)
            # ax1.set_xlabel("Bin", fontsize=12)
            # ax1.set_xticks(np.arange(len(binning_table._bin_str)))
            if not add_special and not add_missing:
                xlabels = binning_table._bin_str[:-2] 
                ax1.set_xticks(np.arange(len(xlabels)))
            elif not add_special and add_missing:
                xlabels = []
                xlabels.extend(binning_table._bin_str[:-2])
                xlabels.append(binning_table._bin_str[-1])
                ax1.set_xticks(np.arange(len(xlabels)))     

            if binning_table.dtype == "categorical":
                bin_str = _bin_str_label_format(xlabels)
            else:
                bin_str = xlabels

            ax1.set_xticklabels(bin_str, rotation=45, ha="right")
        #
        ax2.plot(np.arange(len(xlabels)), 0*np.arange(len(xlabels)),
                    linestyle="dashed", marker="", color="black")

    elif style == "actual":
        _n_nonevent = binning_table.n_nonevent[:-(binning_table._n_specials + 1)]
        _n_event = binning_table.n_event[:-(binning_table._n_specials + 1)]

        n_splits = len(binning_table.splits)

        y_pos = np.empty(n_splits + 2)
        y_pos[0] = binning_table.min_x
        y_pos[1:-1] = binning_table.splits
        y_pos[-1] = binning_table.max_x

        width = y_pos[1:] - y_pos[:-1]
        y_pos2 = y_pos[:-1]

        splits = binning_table.splits
        if log_scale:
            raw_variable = np.log(raw_variable.copy())
            y_pos = np.log(y_pos.copy())
            width = y_pos[1:] - y_pos[:-1]
            y_pos2 = y_pos[:-1]
            splits = np.log(splits.copy())

        if raw_variable is not None and target is not None:
            # ax3 = ax1.twinx()
            keep_idx = ~np.isnan(raw_variable)
            raw_variable = raw_variable[keep_idx]
            target = target[keep_idx]
            counts, bins = np.histogram(raw_variable, bins=20)
            p2  = ax1.hist(raw_variable[target==True], color='tab:red', alpha=0.80, bins=bins)
            p1  = ax1.hist(raw_variable[target==False], color='tab:blue', alpha=0.80, bins=bins)
            handles = [p1[2], p2[2]]
            # labels = ['Non-event', 'Event']
        else:
            p2 = ax1.bar(y_pos2, _n_event, width, color="tab:red",
                            align="edge")
            p1 = ax1.bar(y_pos2, _n_nonevent, width, color="tab:blue",
                                align="edge")
            handles = [p1[0], p2[0]]
            # labels = ['Non-event', 'Event']

        # handles = [p1[0], p2[0]]
        labels = ['Non-event', 'Event']

        ax1.set_xlabel("x" if not log_scale else "log(x)", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)
        
        ax2 = ax1.twinx()

        for i in range(n_splits + 1):
            ax2.plot([y_pos[i], y_pos[i+1]], [metric_values[i]] * 2,
                        linestyle="solid", color="black")

        print('width:',width)
        print('y_pos2:',y_pos2)
        ax2.plot(width / 2 + y_pos2 ,
                    metric_values[:-(binning_table._n_specials + 1)],
                    linewidth=0.75, marker="o", color="black")

        for split in splits:
            ax2.axvline(x=split, color="black", linestyle="--",
                        linewidth=0.9)

        ax2.set_ylabel(metric_label, fontsize=13)
        
        

    ax1.set_title(binning_table.name, fontsize=14)

    if show_bin_labels:
        # legend_high = 0.05
        ax2.legend(handles, labels, loc=legend_position,
                    bbox_to_anchor=legend_xy, ncol=1, fontsize=12)
    else:
        ax2.legend(handles, labels, loc=legend_position,
                    bbox_to_anchor=legend_xy, ncol=1, fontsize=12)

    return fig, ax1, ax2



#%%
def stat_test(x, y, w, w0):
    newx = (x@w).squeeze() + w0
    df_newx = pd.DataFrame({'X':newx,'y':y})
    res = pg.ttest(df_newx[df_newx['y']==False]['X'], df_newx[df_newx['y']==True]['X'], paired=False, correction='auto')
    return res

#%%
def boxplot_MSK(x, y, w, w0, p_value, label=None):
    newx = (x@w).squeeze() + w0
    # yslice = y+1
    newx = pd.DataFrame({'coefficients':newx})
    y = pd.DataFrame({'Post_injury':y})
    df_X = pd.merge(left=newx, right=y, left_index=True, right_index=True).rename({'coefficients':'X','Post_injury':'y'},axis='columns')
    df_X['y'] = df_X['y'].replace({False: "non-MSK", True: "MSK"})
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    sns.boxplot(x="y", y="X", data=df_X, order=['MSK','non-MSK'],
                notch=False, showcaps=False, color=".5",linewidth=2,
                ax=ax)
    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y_max, col = newx.values.max(),  'k'
    ax.plot([x1, x1, x2, x2], [y_max*1.05, y_max*1.1, y_max*1.1, y_max*1.05], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y_max*1.11, r"T-test p-value$<$"+f"{p_value.round(3)}", ha='center', va='bottom', color=col, fontsize=15)
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(newx.values.min() - np.abs(newx.values.min())*0.1, y_max*1.35)
    ax.set_xlabel(label, fontsize=15)
    ax.set_ylabel(r'log$\frac{P(Y=1|\boldsymbol{x})}{P(Y=0|\boldsymbol{x})}$',fontsize=15)
    return fig

#%%
def draw_as_table(df, size, colwidths):
    alternating_colors = [['white'] * len(df.columns), ['lightgray'] * len(df.columns)] * len(df)
    alternating_colors = alternating_colors[:len(df)]
    fig, ax = plt.subplots(figsize=size)
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        colWidths=colwidths,
                        rowColours=['lightblue']*len(df),
                        colColours=['lightblue']*len(df.columns),
                        cellColours=alternating_colors,
                        loc='center')
    fig.tight_layout()
    return fig

# #%%
# def plot_bucketed_probabilities(df_raw,X_test_WOE,y_test,pm,label=None):
#     predictions_severity = pd.merge(left=y_test,right=df_raw['Injury'][['Injury_severity_A_post_concussion','Injury_severity_B_post_concussion','Injury_severity_C_post_concussion']],on='ID')
#     predictions_severity['probability'] = pm.predict_proba(X_test_WOE)[:,1]
#     predictions_severity['No_injury'] = ~predictions_severity['Post_injury']
#     probability_buckets = np.linspace(0,1,11)
#     index_prob = ['(0,10]','(10,20]','(20,30]','(30,40]','(40,50]','(50,60]','(60,70]','(70,80]','(80,90]','(90,100)']
#     counts_injuries_buckets = []
#     for i in range(len(probability_buckets)-1):
#         condition = (predictions_severity['probability']>=probability_buckets[i]) & (predictions_severity['probability']<probability_buckets[i+1])
#         counts_injuries_buckets.append([(predictions_severity['No_injury'][condition]==True).sum(),
#                                         (predictions_severity['Post_injury'][condition]==True).sum(),
#                                         (predictions_severity['Injury_severity_A_post_concussion'][condition]=='yes').sum(),
#                                         (predictions_severity['Injury_severity_B_post_concussion'][condition]=='yes').sum(),
#                                         (predictions_severity['Injury_severity_C_post_concussion'][condition]=='yes').sum()])
#     prob_buckets_df = pd.DataFrame(counts_injuries_buckets,
#                                 columns=['No_injury','Post_injury','Severity_A','Severity_B','Severity_C'],
#                                 index=index_prob)
#     fig, ax = plt.subplots(figsize=(10,5))
#     sns.heatmap(prob_buckets_df.T, annot=True, cmap='cividis', ax=ax)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
#     ax.set_title(f'Counts for bucketed probabilities using {label}')
#     return fig

#%%
def violinplot_MSK(x, y, w, w0, p_value, label=None):
    newx = (x@w).squeeze() + w0
    # yslice = y+1
    newx = pd.DataFrame({'coefficients':newx})
    y = pd.DataFrame({'Post_injury':y})
    df_X = pd.merge(left=newx, right=y, left_index=True, right_index=True).rename({'coefficients':'X','Post_injury':'y'},axis='columns')
    df_X['y'] = df_X['y'].replace({False: "non-MSK", True: "MSK"})
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    sns.violinplot(x="y", y="X", data=df_X, order=['MSK','non-MSK'], split=True, inner="quart").set(xlabel=label, ylabel=r'log$\frac{P(Y=1|x)}{P(Y=1|x)}$')
    return fig

#%%
def get_severity(df_raw,X_test_,y_test_,pm):
    predictions_severity = pd.merge(left=y_test_.astype(bool),right=df_raw['Injury'][['Injury_severity_A_post_concussion','Injury_severity_B_post_concussion','Injury_severity_C_post_concussion']],on='ID').replace({'yes':1,'no':0})
    predictions_severity['probability'] = pm.predict_proba(X_test_)[:,1]
    predictions_severity['No_injury'] = (~predictions_severity['Post_injury']).astype(int)
    predictions_severity['Post_injury'] = predictions_severity['Post_injury'].replace({True:'Injury',False:'Injury'})
    predictions_severity['Severity']='' # to create an empty column
    predictions_severity = predictions_severity.rename(columns={'Injury_severity_A_post_concussion':'A',
                                                                'Injury_severity_B_post_concussion':'B',
                                                                'Injury_severity_C_post_concussion':'C',
                                                                'probability':'P'})
    for col_name in ['A','B','C','No_injury']:
        predictions_severity.loc[predictions_severity[col_name]==1,'Severity'] =predictions_severity['Severity'] + col_name
    
    return predictions_severity

#%%
def plot_bucketed_probabilities(predictions_severity,plot_type='custom',alpha=1):
    # predictions_severity = pd.merge(left=y_test_.astype(bool),right=df_raw['Injury'][['Injury_severity_A_post_concussion','Injury_severity_B_post_concussion','Injury_severity_C_post_concussion']],on='ID').replace({'yes':1,'no':0})
    # predictions_severity['probability'] = pm.predict_proba(X_test_)[:,1]
    # predictions_severity['No_injury'] = (~predictions_severity['Post_injury']).astype(int)
    # predictions_severity['Post_injury'] = predictions_severity['Post_injury'].replace({True:'Injury',False:'Injury'})
    # predictions_severity['Severity']='' # to create an empty column
    # predictions_severity = predictions_severity.rename(columns={'Injury_severity_A_post_concussion':'A',
    #                                                             'Injury_severity_B_post_concussion':'B',
    #                                                             'Injury_severity_C_post_concussion':'C',
    #                                                             'probability':'P'})
    # for col_name in ['A','B','C','No_injury']:
    #     predictions_severity.loc[predictions_severity[col_name]==1,'Severity'] =predictions_severity['Severity'] + col_name
    # print(predictions_severity['Severity'].sort_values().unique())
    fig, ax = plt.subplots(figsize=(10,3))
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    # ax.grid(color='gray', linestyle=':', linewidth=1, alpha=0.7)
    # palette = ['#ea916e', '#e5715e', '#d9535d', '#c14168', '#a3386f', '#863071', '#672a6b', '#1f77b4']
    # hue_order = ['A','B','C','AB','AC','BC','ABC','No_injury']
    hue_sorted = predictions_severity['Severity'].sort_values().unique()
    ordered_injuries = ['A', 'B', 'AB', 'C', 'AC', 'ABC', 'No_injury']
    order = {v:i for i,v in enumerate(ordered_injuries)}
    hue_order = sorted(hue_sorted, key=lambda x: order[x])
    if plot_type=='custom':
        for k, severity in enumerate(hue_order[::-1]):
            prob = predictions_severity.loc[predictions_severity['Severity']==severity,'P']
            if severity=='No_injury':
                facecolor='None'
            else:
                facecolor='k'
            ax.scatter(prob, [k]*len(prob), marker='D', s=230, alpha=alpha, facecolors=facecolor, edgecolors='k')
        yticks_ = np.arange(0,len(hue_order))
        # y_ticks_shifted = (yticks_-yticks_.mean())*.65
        ax.set_yticks(yticks_,labels=hue_order[::-1], fontsize=15)
        ax.set_ylim(0-len(hue_order)*0.08,len(hue_order)-1+len(hue_order)*0.08)
    elif plot_type=='boxplot':
        sns.boxplot(data=predictions_severity, x="P", y="Severity", order=hue_order, showcaps=False, color=".5",linewidth=2, ax=ax)
    elif plot_type=='violinplot':
        sns.violinplot(data=predictions_severity, x="P", y="Severity", order=hue_order, color=".5",linewidth=2, split=True, inner="quart", bw_adjust=.1, ax=ax)

    ax.xaxis.grid(True)
    ax.axvline(0.5, linewidth=2, linestyle='--', color='k')
    
    ax.set_xlabel(r"$P(Y=1|x)$", fontsize=15)
    ax.set_ylabel("Severity", fontsize=15)
    # print(ax.get_yticks(minor=True))
    
    ax.set_xlim(-0.05,1.05)
    
    return fig

#%%
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#%%
def plot_fs(results_boostrap, reg_path, crit_type, reverse=False, color = {'WOE':'blue','MEAN':'orange'}, linestyle = {'WOE':'--','MEAN':':'}):
    fig, ax = plt.subplots(figsize=(5,3))
    n_bootstraps = range(len(results_boostrap))
    woe_avg = np.array([results_boostrap[str(i)]['WOE'][crit_type] for i in n_bootstraps]).mean(axis=0)
    mean_avg = np.array([results_boostrap[str(i)]['MEAN'][crit_type] for i in n_bootstraps]).mean(axis=0)
    if reverse:
        woe_avg = -1*woe_avg
        mean_avg = -1*mean_avg
    ax.semilogx(reg_path['WOE'], woe_avg,
                    label='WoE', color=color['WOE'], linestyle=linestyle['WOE'])
    ax.semilogx(reg_path['MEAN'], mean_avg,
                    label='Mean imputation', color=color['MEAN'], linestyle=linestyle['MEAN'])
    ax.legend()
    return fig,ax
