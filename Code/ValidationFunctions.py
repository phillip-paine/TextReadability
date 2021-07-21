import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")

from sklearn.metrics import mean_squared_error

# Validation Functions:
# Functions to calculate the error in target prediction, and produce plots showing the error as target changes

def evaluate_predictions(df_test, method_cols, plot_path):
    fig, axs = plt.subplots(1, len(method_cols))
    fig.suptitle('Target Residual Plot')
    # Evaluate methods and baseline using RMSE:
    i = 0
    for m in method_cols:
        print('Method ' + m + ' : ' + str(mean_squared_error(df_test['target'], df_test[m])))
        df_test['residual_size_' + m] = np.abs(df_test['target'] - df_test[m])
        axs[i].scatter(df_test['target'], df_test['residual_size' + m])
        axs[i].set_title('Residual Target for \n Method ' + m)
        i += 1
         
    fig.savefig(plot_path + 'residual_target_comparison_plot.png', bbox_inches='tight') 

    return df_test

    

