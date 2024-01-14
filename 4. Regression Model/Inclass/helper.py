import seaborn as sns
import matplotlib.pyplot as plt

def linearity_test(model):
    '''
    Args:
    * model - fitted OLS model from statsmodels
    '''
    
    fitted_vals = model.fittedvalues
    resids = model.resid
    fig, ax = plt.subplots(1)
    
    sns.regplot(x=fitted_vals, y=resids, lowess=True, line_kws={'color': 'red'})
    ax.set_title('Residual vs. Fitted Values', fontsize=16)
    ax.set(xlabel='Fitted', ylabel='Residuals')
