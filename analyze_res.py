import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
    
def stat_analyze(res):
    '''Performs binomial fit on the model results to determine POD properties.
    Segmentation accuracy is transformed into a binary outcome. A segmentation is considered good if F1 score is higher than 50%.
    
    :param res: Array with model results. Should contain FO thickness and average accuracy of segmentation
    :type res: :class:`np.ndarray`
    
    :return: Statsmodels object with fit results
    :rtype: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    '''    
    y = res['Mean'] > 0.5
    x = np.zeros((res['FO_th'].shape[0], 2))
    x[:,0] = res['FO_th']
    x[:,1] = 1

    glm = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    fit = glm.fit()
    #print(fit.summary())
    return fit

def draw_pod(ax, fit, res=None, default_x_range=np.linspace(0., 3., 1000), draw_confidence_interval=True, draw_s90=False, label='POD', colors = ['r', 'b']):
    '''Draws POD curve on ax based on fit parameters
    s_90 and s_90/95 are drawn if they exist in this size range
    
    :param ax: Subplot axes to draw on
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot`
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param res: Array with model results. Only used to get range of FO size. If None, default_x_range will be used instead
    :type res: :class:`np.ndarray`
    :param default_x_range: Default range of defect size to use if res array is not provided
    :type default_x_range: :class:`np.ndarray`
    :param draw_confidence_interval: Set to True to draw confidence interval
    :type draw_confidence_interval: :class:`bool`
    :param draw_s90: Set to True to draw s_90 and s_90/95
    :type draw_s90: :class:`bool`
    :param label: Label to use in legend
    :type label: :class:`str`
    :param colors: Colors to use for the curve and confidence interval
    :type colors: :class:`list`
    
    '''    
    x_range = default_x_range
    if res is not None:
        x = res['FO_th']
        x_range = np.linspace(x.min(), x.max(), 1000)
    
    nu = fit.params[0]*x_range + fit.params[1]
    nu_low = (fit.params[0]-2*fit.bse[0])*x_range + (fit.params[1]-2*fit.bse[1])
    nu_high = (fit.params[0]+2*fit.bse[0])*x_range + (fit.params[1]+2*fit.bse[1])
    p = 1 / (1+np.exp(-nu))
    p_low = 1 / (1+np.exp(-nu_low))
    p_high = 1 / (1+np.exp(-nu_high))
    
    if draw_s90:
        s90_95_exists = np.any(p_low > 0.9)
        s90_exists = np.any(p > 0.9)
        if s90_exists:
            s90 = x_range[np.where(p > 0.9)].min()
            print('s90 = {}'.format(s90))
        else:
            print('s90 does not exist')
        if s90_95_exists:
            s90_95 = x_range[np.where(p_low > 0.9)].min()
            print('s90/95% = {}'.format(s90_95))
        else:
            print('s90/95% does not exist')
    else:
        s90_exists = False
        s90_95_exists = False
    
    if draw_confidence_interval:
        ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2, label = '{} 95% confidence'.format(label))
    ax.plot(x_range, p, c=colors[0], label = label)
    if s90_95_exists:
        ax.vlines([s90, s90_95], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90, s90_95], [0.9, 0.9], color='k', s=20)
        plt.scatter(s90_95, 0.9, color='g', s=30, label = label + ' ' + r"$s_{90/95\%}$")
    if s90_exists:
        ax.vlines([s90], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90], [0.9], color='k', s=20)
        plt.scatter(s90, 0.9, color='k', s=30, label = label + ' ' + r"$s_{90}$")
        
    ax.set_ylim(0., 1.)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax.grid(True)
    ax.legend(loc = 4, fontsize=16)
    ax.set_xlabel('Defect size, mm', fontsize=20)
    ax.set_ylabel("Probability of F1>50%", fontsize=20)
    
def draw_F1(ax, res):
    '''Draws F1 scores on ax as a function of defect size
    
    :param ax: Subplot axes to draw on
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot`
    :param res: Array with model results
    :type res: :class:`np.ndarray`
    '''
    y = res['Mean']
    x = res['FO_th']
    
    ax.scatter(x, y, alpha = 0.5)
    ax.set_xlabel('Defect size, mm', fontsize=16)
    ax.set_ylabel("F1 score", fontsize=16)
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., 2.5)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax.grid(True)
    
def draw_segm_ratio(ax, res):
    '''Draws an approximation of POD on ax
    F1 scores are transformed into binary outcomes by checking if F1 > 50%
    Then the range of defect sizes is binned, and in every bin a fraction of cases with F1 > 50% is computed
    
    :param ax: Subplot axes to draw on
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot`
    :param res: Array with model results
    :type res: :class:`np.ndarray`
    '''
    y = res['Mean']
    x = res['FO_th']

    bins = np.linspace(x.min(), x.max(), 20)
    bin_width = bins[1]-bins[0]
    print(bins)
    prob = np.zeros_like(bins)
    for i in range(bins.shape[0]-1):
        mask = np.logical_and(x >= bins[i], x < bins[i+1])
        total = np.count_nonzero(x[mask])
        success = np.count_nonzero(y[mask] > 0.5)
        if total != 0:
            prob[i] = float(success) / total
        else:
            prob[i] = 1
    prob[-1] = prob[-2]
    ax.bar(bins, prob, bin_width, align='edge')
    ax.set_xlabel('Defect size, mm', fontsize=16)
    ax.set_ylabel("Fraction of cases with F1 > 50%", fontsize=16)
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., 2.5)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.grid(True)
    
def show_pod_pipeline(res, fit):
    '''Creates a figure explaining how raw data can be analyzed via POD curve
    
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param res: Array with model results. Only used to get range of FO size. If None, default_x_range will be used instead
    :type res: :class:`np.ndarray`    
    '''
    fig = plt.figure(figsize = (16,12))
    gs = fig.add_gridspec(2, 2, height_ratios = [0.4,0.6])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    draw_F1(ax1, res)
    ax1.set_title('(a)', y = -0.19, fontsize=16, weight='bold')

    draw_segm_ratio(ax2, res)
    ax2.set_title('(b)', y = -0.19, fontsize=16, weight='bold')
    
    draw_pod(ax3, fit, res=res, draw_s90=True, label='POD', colors = ['r', 'b'])
    ax3.legend(loc=2, fontsize=16)
    ax3.set_title('(c)', y = -0.15, fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('tmp_img/pod_pipeline.pdf', format='pdf')
    plt.show()
    
def comp_models(fits, fit_labels, draw_s90 = False, x_range = np.linspace(0., 3., 1000)):
    '''Compare models on the same dataset (one figure)
    
    :param fits: Fits for models
    :type fits: :class:`list`
    :param fit_labels: Labels decribing the difference between models (what kind of training dataset is used)
    :type fit_labels: :class:`list`
    :param draw_s90: Set to True to draw s_90 and s_90/95
    :type draw_s90: :class:`bool`
    :param x_range: Range of FO size to use
    :type x_range: :class:`np.ndarray`
    '''
    fig, ax = plt.subplots(1, 1, figsize = (12,9))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(fits)):
        draw_pod(ax, fits[i], draw_confidence_interval=False, draw_s90=draw_s90, 
                 label=fit_labels[i], colors = [colors[i], colors[i]], default_x_range=x_range)

    plt.tight_layout()
    plt.savefig('tmp_img/comp_models.pdf', format='pdf')
    plt.show()
    
def comp_mat(fits1, fits2, fit_labels, test_labels, x_range = np.linspace(0., 3., 1000)):
    '''Creates a figure comparing 2 pairs of models applied to 2 test datasets
    
    :param fits1: Fits for two models applied to the first test dataset
    :type fits1: :class:`list`
    :param fits1: Fits for two models applied to the second test dataset
    :type fits1: :class:`list`
    :param fit_labels: Labels describing difference between 2 models (what kind of training dataset is used)
    :type fit_labels: :class:`list`
    :param test_labels: Labels describing test data (material + voltage)
    :type test_labels: :class:`list`
    :param x_range: Range of FO size to use in both subplots
    :type x_range: :class:`np.ndarray`
    '''
    fig = plt.figure(figsize = (18,10))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    draw_pod(ax1, fits1[0], draw_s90=False, label=fit_labels[0], colors = ['#1f77b4', '#1f77b4'], default_x_range=x_range)
    draw_pod(ax1, fits1[1], draw_s90=False, label=fit_labels[1], colors = ['orange', 'orange'], default_x_range=x_range)
    ax1.set_title('(a) {}'.format(test_labels[0]), y = -0.12, fontsize=16, weight='bold')

    draw_pod(ax2, fits2[0], draw_s90=False, label=fit_labels[0], colors = ['#1f77b4', '#1f77b4'], default_x_range=x_range)
    draw_pod(ax2, fits2[1], draw_s90=False, label=fit_labels[1], colors = ['orange', 'orange'], default_x_range=x_range)
    ax2.set_title('(b) {}'.format(test_labels[1]), y = -0.12, fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('tmp_img/comp_mat.pdf', format='pdf')
    plt.show()
    
def compare_bo_fo(res1, res2):
    '''Compares output of two models on the same dataset in BO/FO thickness coordinates
    
    :param res1: Array with results of the fist model
    :type res1: :class:`np.ndarray`
    :param res2: Array with results of the second model
    :type res2: :class:`np.ndarray`
    '''
    
    y = np.zeros_like(res1['Mean'])
    y[np.logical_and(res1['Mean'] < 0.5, res2['Mean'] < 0.5)] = 0
    y[np.logical_and(res1['Mean'] >= 0.5, res2['Mean'] < 0.5)] = 1
    y[np.logical_and(res1['Mean'] < 0.5, res2['Mean'] >= 0.5)] = 2
    y[np.logical_and(res1['Mean'] >= 0.5, res2['Mean'] >= 0.5)] = 3
    
    total_samples = y.shape[0]
    print('Both models fail:      {:4d}/{}'.format(np.count_nonzero(y == 0), total_samples))
    print('M1 succeeds, M2 fails: {:4d}/{}'.format(np.count_nonzero(y == 1), total_samples))
    print('M2 succeeds, M1 fails: {:4d}/{}'.format(np.count_nonzero(y == 2), total_samples))
    print('Both succeed:          {:4d}/{}'.format(np.count_nonzero(y == 3), total_samples))
    
    cmap = colors.ListedColormap(['b', 'k', 'g', 'r'])
    
    plt.figure(figsize=(12,9))
    plt.scatter(res1['BO_th'], res1['FO_th'], c=y, s=36, cmap = cmap)
    plt.xlabel('BO thickness, mm')
    plt.ylabel('FO thickness, mm')
    plt.grid(True)
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('tmp_img/comp_bo_fo.pdf', format='pdf')
    plt.show()
    
def load_res_file(fname):
    '''Loads model test results from file
    
    :param fname: Name of the file with test results
    :type fname: :class:`str`
    
    :return: Array with test results
    :rtype: :class:`np.ndarray`
    '''
    res = np.genfromtxt(fname, delimiter=',', names=True)
    #nonzero_FO_mask = res['FO_th'] != 0
    #res = res[nonzero_FO_mask]
    return res

def single_test(model_name):
    res = load_res_file('./test_res/{}.csv'.format(model_name))
    fit = stat_analyze(res)
    show_pod_pipeline(res, fit)

def batch_test(mat_name = 'pl90', draw_s90=False):
    res_r_r = load_res_file('./test_res/{}_r_r.csv'.format(mat_name))
    res_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(mat_name))
    res_mc_r = load_res_file('./test_res/{}_mc_r.csv'.format(mat_name))
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(mat_name))
    
    fit_r_r = stat_analyze(res_r_r)
    fit_r_mc = stat_analyze(res_r_mc)
    fit_mc_r = stat_analyze(res_mc_r)
    fit_mc_mc = stat_analyze(res_mc_mc)
    
    comp_models([fit_r_r, fit_r_mc, fit_mc_r, fit_mc_mc], ['R/R', 'R/MC', 'MC/R', 'MC/MC'], draw_s90=draw_s90)

def mat_comparison(mat1 = 'pl90', mat2 = 'fe450'):
    res1_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(mat1))
    res1_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(mat1))
    res2_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(mat2))
    res2_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(mat2))
    
    fit1_r_mc = stat_analyze(res1_r_mc)
    fit1_mc_mc = stat_analyze(res1_mc_mc)
    fit2_r_mc = stat_analyze(res2_r_mc)
    fit2_mc_mc = stat_analyze(res2_mc_mc)
    
    comp_mat([fit1_r_mc, fit1_mc_mc], [fit2_r_mc, fit2_mc_mc], ['Train MC', 'Train Radon'], ['PMMA, 90kV', 'Iron, 450kV'], np.linspace(0., 3., 1000))
    
def heatmap_comparison(mat_name = 'pl90'):
    res_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(mat_name))
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(mat_name))
    
    compare_bo_fo(res_r_mc, res_mc_mc)

if __name__ == "__main__":
    #materials are pl90, fe450
    batch_test('biron', draw_s90=False)
    #single_test('fe450biron_mc_mc')
    #heatmap_comparison('fe450biron')
