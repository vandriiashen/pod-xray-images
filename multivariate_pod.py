import numpy as np
from pathlib import Path
import statsmodels.api as sm
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import ticker, colors, cm

def stat_analyze(res, y_val='Mean'):
    y = res[y_val] > 0.5
    x = np.ones((res['FO_th'].shape[0], 2))
    x[:,0] = res['FO_th']

    glm = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    fit = glm.fit()
    return fit

def multivariate_stat_analyze(res, y_val='Mean', second_variable='Scat_fract'):
    y = res[y_val] > 0.5
    x = np.ones((res['FO_th'].shape[0], 3))
    x[:,0] = res['FO_th']
    x[:,1] = res[second_variable]

    glm = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    fit = glm.fit()
    return fit

def draw_pod(ax, fit, res=None, NN_confidence=False, default_x_range=np.linspace(0., 3., 1000), 
             draw_confidence_interval=True, draw_s90=False, label='POD', colors = ['r', 'b'], linestyle='-', linewidth=1.5):
    '''Draws POD curve on ax based on fit parameters
    s_90 and s_90/95 are drawn if they exist in this size range
    
    :param ax: Subplot axes to draw on
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot`
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param res: Array with model results. Only used to get range of FO size. If None, default_x_range will be used instead
    :type res: :class:`np.ndarray`
    :param NN_confidence: Draw confidence intervals based on the variance of different instances of the same model
    :type NN_confidence: :class:`bool`
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
    :param linestyle: Linestyle for POD curve
    :type linestyle: :class:`str`
    
    '''    
    x_range = default_x_range
    if res is not None:
        x = res['FO_th']
        x_range = np.linspace(x.min(), x.max(), 1000)
    
    if not NN_confidence:
        fit_x = np.ones((x_range.shape[0], 2))
        fit_x[:,0] = x_range
        prediction = fit.get_prediction(fit_x)
        fit_y = prediction.summary_frame(alpha=0.05)
        
        p = fit_y['mean']
        p_low = fit_y['mean_ci_lower']
        p_high = fit_y['mean_ci_upper']
    else:
        p, p_low, p_high = compute_confidence_NN(res, x_range, fit)
        
    s90, s90_95 = compute_s90(fit, x_range=x_range)
    s90_exists = True if s90 > -1 else False
    s90_95_exists = True if s90_95 > -1 else False
    
    if draw_confidence_interval:
        #ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2, label = '{} 95% confidence'.format(label))
        ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2)
    ax.plot(x_range, p, c=colors[0], label = label, linestyle=linestyle, linewidth=linewidth)
    if draw_s90 and s90_95_exists:
        ax.vlines([s90, s90_95], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90, s90_95], [0.9, 0.9], color='k', s=20)
        plt.scatter(s90_95, 0.9, color='g', s=30, label = label + ' ' + r"$s_{90/95\%}$")
    if draw_s90 and s90_exists:
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

def compute_s90(fit, x_range=np.linspace(0., 3., 1000)):
    '''Computes s_90 - defect size for 90\% probability of good segmentation and s_90/95 - lower bound of the same value for 95% confidence interval.
    It is possiblee that one value or both do not exist. The function will return in this case.
    
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param x_range: Range of defect size to compute probabilities
    :type x_range: :class:`np.ndarray`

    :return: List containing s_90 and s_90/95. If any does not exist, it is replaced with -1.
    :rtype: :class:`list`
    '''
    fit_x = np.ones((x_range.shape[0], 2))
    fit_x[:,0] = x_range
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
    
    p = fit_y['mean']
    p_low = fit_y['mean_ci_lower']
    p_high = fit_y['mean_ci_upper']
    
    res = [-1., -1.]
    s90_95_exists = np.any(p_low > 0.9)
    s90_exists = np.any(p > 0.9)
    if s90_exists:
        s90 = x_range[np.where(p > 0.9)].min()
        res[0] = s90
    if s90_95_exists:
        s90_95 = x_range[np.where(p_low > 0.9)].min()
        res[1] = s90_95
    return res


def draw_multivariate_pod(ax, fit, res=None, snr_level=0., default_x_range=np.linspace(0., 3., 1000), 
             draw_confidence_interval=False, label='POD', colors = ['r', 'b'], linestyle='-'):
    x_range = default_x_range
    if res is not None:
        x = res['FO_th']
        x_range = np.linspace(x.min(), x.max(), 1000)
    
    fit_x = np.ones((x_range.shape[0], 3))
    fit_x[:,0] = x_range
    fit_x[:,1] = snr_level
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
        
    p = fit_y['mean']
    p_low = fit_y['mean_ci_lower']
    p_high = fit_y['mean_ci_upper']
        
    ax.plot(x_range, p, c=colors[0], label = label, linestyle=linestyle)
    if draw_confidence_interval:
        ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2)
        
    ax.set_ylim(0., 1.)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax.grid(True)
    ax.legend(loc = 3, fontsize=16)
    ax.set_xlabel('Defect size, mm', fontsize=20)
    ax.set_ylabel("Probability of F1>50%", fontsize=20)
    
def pair_check(model_name):
    res_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(model_name))
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    scat_max = res_mc_mc['Scat_fract'].max()
    scat_min = res_mc_mc['Scat_fract'].min()
    att_max = res_mc_mc['FO_att'].max()
    att_min = res_mc_mc['FO_att'].min()
    print('Attenuation range = [{:.3f}, {:.3f}]'.format(att_min, att_max))
    print('Scattering ratio range = [{:.3f}, {:.3f}]'.format(scat_min, scat_max))
    
    #par_range = np.arange(0., 6., 1.)
    par_range = np.linspace(0., 0.2, 5, endpoint=True)
    lstyles = ['-', '--', '-.', ':']
    print(par_range)
    for i in range(par_range.shape[0]-1):
        #mask = np.logical_and(res_mc_mc['FO_att'] > par_range[i], res_mc_mc['FO_att'] < par_range[i+1])
        mask = np.logical_and(res_mc_mc['Scat_fract'] > par_range[i], res_mc_mc['Scat_fract'] < par_range[i+1])
        print('{}/{}'.format(np.count_nonzero(mask), res_mc_mc.shape[0]))
        subset_res_r_mc = res_r_mc[mask]
        subset_res_mc_mc = res_mc_mc[mask]
        fit_r_mc = stat_analyze(subset_res_r_mc)
        fit_mc_mc = stat_analyze(subset_res_mc_mc)
        s90, s90_95 = compute_s90(fit_r_mc)
        print('{:.3f} - {:.3f}:, R/MC:\t s_90 = {:.3f} | s_90/95 = {:.3f}'.format(par_range[i], par_range[i+1], s90, s90_95))
        s90, s90_95 = compute_s90(fit_mc_mc)
        print('{:.3f} - {:.3f}:, MC/MC:\t s_90 = {:.3f} | s_90/95 = {:.3f}'.format(par_range[i], par_range[i+1], s90, s90_95))
        draw_pod(ax, fit_r_mc, subset_res_r_mc, NN_confidence=False, default_x_range=np.linspace(0., 3., 1000), draw_confidence_interval=False, draw_s90=False, 
                 label='{:.0%} < Scat < {:.0%} R/MC'.format(par_range[i], par_range[i+1]), colors = ['#1f77b4', 'b'], linestyle=lstyles[i])
        draw_pod(ax, fit_mc_mc, subset_res_mc_mc, NN_confidence=False, default_x_range=np.linspace(0., 3., 1000), draw_confidence_interval=True, draw_s90=False, 
                 label='{:.0%} < Scat < {:.0%} MC/MC'.format(par_range[i], par_range[i+1]), colors = ['#ff7f0e', '#ff7f0e'], linestyle=lstyles[i])
        
    ax.set_title(model_name, fontsize=24)
    
    plt.tight_layout()
    plt.savefig('tmp_img/comp_scat.pdf', format='pdf')
    plt.savefig('tmp_img/comp_scat_nn_fe450.png')
    plt.show()

def parameter_check(model_name):
    res = load_res_file('./test_res/{}.csv'.format(model_name))
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    scat_max = res['Scat_fract'].max()
    scat_min = res['Scat_fract'].min()
    att_max = res['FO_att'].max()
    att_min = res['FO_att'].min()
    print('Attenuation range = [{:.3f}, {:.3f}]'.format(att_min, att_max))
    print('Scattering ratio range = [{:.3f}, {:.3f}]'.format(scat_min, scat_max))
    
    par_range = np.arange(0., 5., 1.0)
    #par_range = np.arange(0., 0.8, 0.1)
    print(par_range)
    for i in range(par_range.shape[0]-1):
        mask = np.logical_and(res['FO_att'] > par_range[i], res['FO_att'] < par_range[i+1])
        #mask = np.logical_and(res['Scat_fract'] > par_range[i], res['Scat_fract'] < par_range[i+1])
        print('{}/{}'.format(np.count_nonzero(mask), res.shape[0]))
        subset_res = res[mask]
        fit = stat_analyze(subset_res)
        s90, s90_95 = compute_s90(fit)
        print('{:.1f} - {:.1f}:\t s_90 = {:.3f} | s_90/95 = {:.3f}'.format(par_range[i], par_range[i+1], s90, s90_95))
        s_max = subset_res['Scat_fract'].max()
        s_min = subset_res['Scat_fract'].min()
        s_mean = subset_res['Scat_fract'].mean()
        s_std = subset_res['Scat_fract'].std()
        print('Scat range: [{:.2f}, {:.2f}], avg = {:.2f} +- {:.2f}'.format(s_min, s_max, s_mean, s_std))
        draw_pod(ax, fit, default_x_range=np.linspace(0., 3., 1000), draw_confidence_interval=True, draw_s90=False, 
                 label='{:.1f} < Att < {:.1f}'.format(par_range[i], par_range[i+1]), colors = ['r', 'b'])
    
    plt.tight_layout()
    plt.savefig('tmp_img/comp_inst.pdf', format='pdf')
    plt.savefig('tmp_img/comp_inst.png')
    plt.show()
    
def scat_att_plot(model_name_list):
    fig, ax = plt.subplots(1, 1, figsize=(15,9))
    
    #colors = ['#82e0aa', '#229954', '#0b5345', '#d6eaf8', '#2e86c1', '#154360', '#f5b7b1', '#e74c3c', '#641e16']
    colors = ['#e74c3c', '#641e16', '#d6eaf8', '#2e86c1', '#154360', '#82e0aa', '#229954', '#0b5345', '#f5b7b1']

    for i, model_name in enumerate(model_name_list):
        res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
        scat_max = res_mc_mc['Scat_fract'].max()
        scat_min = res_mc_mc['Scat_fract'].min()
        att_max = res_mc_mc['FO_att'].max()
        att_min = res_mc_mc['FO_att'].min()
        print(model_name)
        print('Attenuation range = [{:.3f}, {:.3f}]'.format(att_min, att_max))
        print('Scattering ratio range = [{:.3f}, {:.3f}]'.format(scat_min, scat_max))
    
        ax.scatter(res_mc_mc['FO_att'], res_mc_mc['Scat_fract'], c=colors[i], label=model_name, alpha=0.8)
    
    ax.grid(True)
    ax.legend(loc = 2, fontsize=16)
    ax.set_xlabel('Attenuation rate', fontsize=20)
    ax.set_ylabel("Scattering-to-primary ratio", fontsize=20)
    ax.set_ylim(0., 2.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.set_title(model_name, fontsize=24)
    
    plt.tight_layout()
    plt.savefig('tmp_img/scat_att.pdf', format='pdf')
    plt.savefig('tmp_img/scat_att.png')
    plt.show()
    
def geom_im_plot(model_name, mode):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))

    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    scat_max = res_mc_mc['Scat_fract'].max()
    scat_min = res_mc_mc['Scat_fract'].min()
    att_max = res_mc_mc['FO_att'].max()
    att_min = res_mc_mc['FO_att'].min()
    print(model_name)
    print('Attenuation range = [{:.3f}, {:.3f}]'.format(att_min, att_max))
    print('Scattering ratio range = [{:.3f}, {:.3f}]'.format(scat_min, scat_max))
    
    plt.scatter(res_mc_mc['Cyl_R'], res_mc_mc['Cav_Pos'], c=res_mc_mc[mode], label=model_name)
    
    ax.grid(True)
    ax.legend(loc = 4, fontsize=16)
    ax.set_xlabel('Cylinder Radius', fontsize=20)
    ax.set_ylabel("Void location", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    title_mode = {'FO_att': 'Attenuation rate', 'Scat_fract': 'Scattering-to-primary'}
    ax.set_title(title_mode[mode], fontsize=24)
    
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('tmp_img/geom_im.pdf', format='pdf')
    plt.savefig('tmp_img/geom_im.png')
    plt.show()
    
def snr_to_error(model_name):
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    res_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(model_name))
    
    snrs = np.arange(0., 2., 0.2)
    num_samples = []
    ratios = []
    for i in range(len(snrs)-1):
        mask = np.logical_and(res_mc_mc['Scat_fract'] >= snrs[i], res_mc_mc['Scat_fract'] < snrs[i+1])
        samples = np.count_nonzero(mask)
        if samples < 50:
            ratio = 'N/A'
        else:
            subset_res_r_mc = res_r_mc[mask]
            subset_res_mc_mc = res_mc_mc[mask]
            fit_r_mc = stat_analyze(subset_res_r_mc)
            fit_mc_mc = stat_analyze(subset_res_mc_mc)
            s90_1, s90_95_1 = compute_s90(fit_mc_mc)
            s90_2, s90_95_2 = compute_s90(fit_r_mc)
            ratio = (s90_2 - s90_1) / s90_1
        ratios.append(ratio)
        num_samples.append(samples)
    str_conv = lambda x: '{:.1%}'.format(x) if x != 'N/A' else '{}'.format(x)
    snr_str = ['{:.1f}'.format(x) for x in snrs]
    sample_str = ['{}'.format(x) for x in num_samples]
    ratios_str = [str_conv(x) for x in ratios]
    return snr_str, sample_str, ratios_str

def snr_comparison(model_list):
    for model in model_list:
        snr_str, sample_str, ratios_str = snr_to_error(model)
        print(model)
        print('\t'.join(snr_str))
        print('\t'.join(sample_str))
        print('\t'.join(ratios_str))
        
def snr_fit(model_name):
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    
    fit_mc_mc = stat_analyze(res_mc_mc)
    
    mask = np.logical_and(res_mc_mc['Scat_fract'] >= 0.4, res_mc_mc['Scat_fract'] < 0.6)
    subset_res_mc_mc = res_mc_mc[mask]
    subset_fit = stat_analyze(subset_res_mc_mc)
    
    mult_mc_mc = multivariate_stat_analyze(res_mc_mc)
    
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    draw_pod(ax, subset_fit, subset_res_mc_mc, NN_confidence=False, draw_confidence_interval=False, draw_s90=False, label='POD | SNR < 0.1', colors = ['b', 'b'])
    draw_pod(ax, fit_mc_mc, res_mc_mc, NN_confidence=False, draw_confidence_interval=False, draw_s90=False, label='Average POD', colors = ['g', 'b'])
    draw_multivariate_pod(ax, mult_mc_mc, res_mc_mc, snr_level=0., label='POD | SNR = 0', colors = ['r', 'b'])
    draw_multivariate_pod(ax, mult_mc_mc, res_mc_mc, snr_level=0.5, label='POD | SNR = 0.5', colors = ['r', 'b'], linestyle = '--')
    
    plt.tight_layout()
    plt.savefig('tmp_img/multivariate.pdf', format='pdf')
    plt.savefig('tmp_img/multivariate.png')
    plt.show()
    
def compute_snr_s90(res, fit, snr_level):
    x = res['FO_th']
    x_range = np.linspace(0., 9., 10000)
    fit_x = np.ones((x_range.shape[0], 3))
    fit_x[:,0] = x_range
    fit_x[:,1] = snr_level
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
    p = fit_y['mean']
    p_low = fit_y['mean_ci_lower']
    p_high = fit_y['mean_ci_upper']
    s90 = x_range[np.where(p > 0.9)].min()
    return s90
    
def snr_diff(model_name, ax, snr_levels=[0., 0.5, 1.0]):
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    res_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(model_name))
    
    mult_mc_mc = multivariate_stat_analyze(res_mc_mc)
    mult_r_mc = multivariate_stat_analyze(res_r_mc)
    
    print('Scattering range = [{:.2f},{:.2f}]'.format(
          res_mc_mc['Scat_fract'].min(), res_mc_mc['Scat_fract'].max()))
    
    s90_mc_0 = compute_snr_s90(res_mc_mc, mult_mc_mc, 0.)
    s90_mc_max = compute_snr_s90(res_mc_mc, mult_mc_mc, res_mc_mc['Scat_fract'].max())
    s90_r_0 = compute_snr_s90(res_r_mc, mult_r_mc, 0.)
    s90_r_max = compute_snr_s90(res_r_mc, mult_r_mc, res_r_mc['Scat_fract'].max())
    print('{:.2f} {:.2f} | {:.2f} {:.2f}'.format(
          s90_mc_0, s90_r_0, s90_mc_max, s90_r_max))
    
    plot = True
    if plot:
        linestyles = ['-', '--', ':']
        for i in range(len(snr_levels)):
            draw_multivariate_pod(ax, mult_mc_mc, res_mc_mc, snr_level=snr_levels[i], label='MC | SPR = {}'.format(snr_levels[i]), draw_confidence_interval=True, colors = ['#ff7f0e', '#ff7f0e'], linestyle = linestyles[i])
            draw_multivariate_pod(ax, mult_r_mc, res_mc_mc, snr_level=snr_levels[i], label='Radon | SPR = {}'.format(snr_levels[i]), colors = ['#1f77b4', '#1f77b4'], linestyle = linestyles[i])
            
def snr_diff_single(model_name):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    snr_diff(model_name, ax)
    
    plt.tight_layout()
    plt.savefig('tmp_img/multivariate.pdf', format='pdf')
    plt.savefig('tmp_img/multivariate.png')
    plt.show()
       
def snr_diff_comp(model1, model2, model3):
    fig, ax = plt.subplots(1, 3, figsize=(18,8))
    
    snr_diff(model1, ax[0], [0.])
    ax[0].set_title('(a) PMMA, 90kV', y = -0.15, fontsize=16, weight='bold')
    snr_diff(model2, ax[1], [0., 0.5])
    ax[1].set_title('(b) Al, 150kV', y = -0.15, fontsize=16, weight='bold')
    snr_diff(model3, ax[2], [0., 0.5, 1.0])
    ax[2].set_title('(c) Fe, 450kV', y = -0.15, fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('tmp_img/multivariate.pdf', format='pdf')
    plt.savefig('tmp_img/multivariate.png')
    plt.show()
    
def snr_s90(model_name, snr_levels):
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    res_r_mc = load_res_file('./test_res/{}_r_mc.csv'.format(model_name))
    
    mult_mc_mc = multivariate_stat_analyze(res_mc_mc)
    mult_r_mc = multivariate_stat_analyze(res_r_mc)
    
    x_range = np.linspace(0., 20.0, 100000)
    fit_x = np.ones((x_range.shape[0], 3))
    fit_x[:,0] = x_range
    
    errors = np.zeros_like(snr_levels)
    
    for i in range(snr_levels.shape[0]):
        fit_x[:,1] = snr_levels[i]
        pred_mc_mc = mult_mc_mc.get_prediction(fit_x)
        pred_r_mc = mult_r_mc.get_prediction(fit_x)
        fit_mc_mc = pred_mc_mc.summary_frame(alpha=0.05)
        fit_r_mc = pred_r_mc.summary_frame(alpha=0.05)
        
        s90_mc = x_range[np.where(fit_mc_mc['mean'] > 0.9)].min()
        s90_r = x_range[np.where(fit_r_mc['mean'] > 0.9)].min()
        
        errors[i] = (s90_r - s90_mc) / s90_mc
        
    return errors

def snr_error_plot(model_list):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    #snr_levels = np.arange(0., 0.5, 0.05)
    
    for model in model_list:
        res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model))
        snr_levels = np.linspace(res_mc_mc['Scat_fract'].min(), res_mc_mc['Scat_fract'].max(), 100)
        print(res_mc_mc['Scat_fract'].min(), res_mc_mc['Scat_fract'].max())
        errors = snr_s90(model, snr_levels)
        ax.plot(snr_levels, errors, label=model)
        
    ax.grid(True)
    ax.legend(loc = 4, fontsize=16)
    ax.set_xlabel('SPR value', fontsize=20)
    ax.set_ylabel("MC is better than R by", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0.)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    
    plt.tight_layout()
    plt.savefig('tmp_img/error_snr.pdf', format='pdf')
    plt.savefig('tmp_img/error_snr.png')
    plt.show()
    
def att_s90(model_name, att_levels):
    res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model_name))
    
    mult_mc_mc = multivariate_stat_analyze(res_mc_mc, second_variable='FO_att')
    
    x_range = np.linspace(0., 3.0, 1000)
    fit_x = np.ones((x_range.shape[0], 3))
    fit_x[:,0] = x_range
    
    s90s = np.zeros_like(att_levels)
    
    for i in range(att_levels.shape[0]):
        fit_x[:,1] = att_levels[i]
        pred_mc_mc = mult_mc_mc.get_prediction(fit_x)
        fit_mc_mc = pred_mc_mc.summary_frame(alpha=0.05)
        s90_mc = x_range[np.where(fit_mc_mc['mean'] > 0.9)].min()
        s90s[i] = s90_mc
        
    return s90s
    
def att_s90_plot(model_list):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    #att_levels = np.arange(0., 5.0, 0.05)
    
    for model in model_list:
        print(model)
        res_mc_mc = load_res_file('./test_res/{}_mc_mc.csv'.format(model))
        att_levels = np.linspace(res_mc_mc['FO_att'].min(), res_mc_mc['FO_att'].max(), 100)
        
        s90s = att_s90(model, att_levels)
        ax.plot(att_levels, s90s, label=model)
        
    ax.grid(True)
    ax.legend(loc = 4, fontsize=16)
    ax.set_xlabel('Attenuation rate', fontsize=20)
    ax.set_ylabel("Smallest detectable defect s90, mm", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0.)
    
    plt.tight_layout()
    plt.savefig('tmp_img/att_s90.pdf', format='pdf')
    plt.savefig('tmp_img/att_s90.png')
    plt.show()
    
def load_res_file(fname):
    '''Loads model test results from file
    
    :param fname: Name of the file with test results
    :type fname: :class:`str`
    
    :return: Array with test results
    :rtype: :class:`np.ndarray`
    '''
    res = np.genfromtxt(fname, delimiter=',', names=True)
    res = res[res['Area'] > 0.]
    res = res[np.isfinite(res['Scat_fract'])]
    
    #break standard confidence intervals
    #res = np.repeat(res, 100, axis=0)
    
    return res
    
if __name__ == "__main__":
    #materials are pl90, pl150, al90, al150, al300, fe300, fe450 
    #parameter_check('al90_r_r')
    #pair_check('al300')
    #scat_att_plot(['fe450'])
    #scat_att_plot(['fe450', 'fe300', 'al300', 'al150', 'al90', 'pl150', 'pl90'])
    #geom_im_plot('fe450', 'FO_att')
    #geom_im_plot('fe450', 'Scat_fract')
    #snr_comparison(['al90', 'al300', 'fe300', 'fe450', 'pl90'])
    #snr_diff_single('al150')
    snr_diff_comp('pl90', 'al150', 'fe450')
    #snr_error_plot(['al90', 'al150', 'al300', 'fe300', 'fe450', 'pl90', 'pl150'])
    #att_s90_plot(['al90', 'al150', 'al300', 'fe300', 'fe450', 'pl90', 'pl150'])
