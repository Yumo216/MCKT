from scipy.stats import ttest_rel

# My model's AUC and ACC results
my_model_auc = [0.7778, 0.7719]
my_model_acc = [0.7561, 0.7541]

# Best model's AUC and ACC results
best_model_auc = [0.7635, 0.7616]
best_model_acc = [0.7462, 0.7438]

# Calculate paired t-tests
t_statistic_auc, p_value_auc = ttest_rel(my_model_auc, best_model_auc)
t_statistic_acc, p_value_acc = ttest_rel(my_model_acc, best_model_acc)

print(t_statistic_auc, p_value_auc, t_statistic_acc, p_value_acc)