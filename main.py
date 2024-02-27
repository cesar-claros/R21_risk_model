#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import l1_min_c
from src.funcs import data_preparation, preprocessing_WOE, preprocessing_mean,\
                        feature_selection, plot_metrics_fs, fit, plot_predictions,\
                        plot_eval_metrics

#%%
def main():
    # Data preparation
    sheets_xlsx_raw = ['Dates','Injury','Baseline','Acute','Asymptomatic','RTA','One_time_test','Demographics','Disorders']
    datasheet_file_raw = 'datas/R21_dataset_jul62023.xlsx'
    df_raw = pd.read_excel(f'{datasheet_file_raw}', sheet_name=sheets_xlsx_raw, index_col=0)
    X, y = data_preparation(df_raw) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=20)
    # Data preprocessing
    X_train_WOE, X_test_WOE = preprocessing_WOE(X_train, X_test, y_train)
    X_train_MEAN, X_test_MEAN = preprocessing_mean(X_train, X_test)
    # Feature selection
    cs_dict = {'WOE':l1_min_c(X_train_WOE, y_train.reset_index(drop=True), loss="log") * np.logspace(0, 10, 1000),
            'MEAN':l1_min_c(X_train_MEAN, y_train.reset_index(drop=True), loss="log") * np.logspace(0, 10, 1000)}
    results_fs_boostrap = dict()
    results_fs_WOE = feature_selection(X_train_WOE, y_train, cs_dict['WOE'])
    results_fs_MEAN = feature_selection(X_train_MEAN, y_train, cs_dict['MEAN'])
    results_fs_boostrap.update({'0':{'WOE':results_fs_WOE, 'MEAN':results_fs_MEAN}})
    plot_metrics_fs(results_fs_boostrap, cs_dict)
    # Select best feature selection model based on evaluation metric ('aic', 'aicc')  
    eval_metric ='aicc'
    best_fs_WOE = results_fs_WOE['clfs'][np.argmin(results_fs_WOE[eval_metric])]
    best_fs_MEAN = results_fs_MEAN['clfs'][np.argmin(results_fs_MEAN[eval_metric])]
    # Training models
    results_model_WOE = fit(X_train_WOE, y_train, best_fs_WOE)
    results_model_MEAN = fit(X_train_MEAN, y_train, best_fs_MEAN)
    plot_predictions(X_test_WOE, y_test, df_raw, results_model_WOE['pipeline'], pp_label='WoE')
    plot_predictions(X_test_MEAN, y_test, df_raw, results_model_MEAN['pipeline'], pp_label='Mean imputation')
    # Model evaluations
    results_clfs = dict()
    results_clfs.update({'WOE':results_model_WOE, 'MEAN':results_model_MEAN})
    plot_eval_metrics(X_test_WOE, X_test_MEAN, y_test, results_clfs)

# %%
if __name__ == "__main__":
    main()
