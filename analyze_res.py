import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import ticker
    
def stat_analyze(res):
    '''Perform binomial fit on the model results to determine POD properties.
    Segmentation accuracy is transformed into a binary outcome. A segmentation is considered good if Dice's score is higher than 50%.
    
    :param res: Array with model results. Should contain FO thickness and average accuracy of segmentation
    :type proj: :class:`np.ndarray`
    '''    
    y = res['Iter1'] > 0.5
    x = np.zeros((res['FO_th'].shape[0], 2))
    x[:,0] = res['FO_th']
    x[:,1] = 1

    glm = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    res = glm.fit()
    print(res.summary())
    return res

def compare_pod(fit_list, labels, fit_x = np.linspace(0., 3., 1000)):
    plt.figure(figsize = (12,9))
    for i, fit in enumerate(fit_list):
        nu = fit.params[0]*fit_x + fit.params[1]
        nu_low = (fit.params[0]-2*fit.bse[0])*fit_x + (fit.params[1]-2*fit.bse[1])
        nu_high = (fit.params[0]+2*fit.bse[0])*fit_x + (fit.params[1]+2*fit.bse[1])
        p = 1 / (1+np.exp(-nu))
        p_low = 1 / (1+np.exp(-nu_low))
        p_high = 1 / (1+np.exp(-nu_high))
        
        plt.fill_between(fit_x, p_low, p_high, alpha = 0.2, label = '{}_95%'.format(labels[i]))
        plt.plot(fit_x, p, label = labels[i])
    plt.xlabel('FO thickness, mm', fontsize=20)
    plt.ylabel("Probability of location", fontsize=20)
    
    ax = plt.gca()
    ax.set_ylim(0., 1.)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    
    plt.grid(True)
    plt.legend(loc=2, fontsize=16)
    plt.tight_layout()
    #plt.savefig('tmp_img/img.eps', format='eps', transparent=True)
    plt.savefig('tmp_img/img.png', format='png')
    plt.show()

def pod_glm(res, fit):
    
    y = res['Iter1'] > 0.5
    x = res['FO_th']
    
    fit_x = np.linspace(x.min(), x.max(), 1000)
    nu = fit.params[0]*fit_x + fit.params[1]
    nu_low = (fit.params[0]-2*fit.bse[0])*fit_x + (fit.params[1]-2*fit.bse[1])
    nu_high = (fit.params[0]+2*fit.bse[0])*fit_x + (fit.params[1]+2*fit.bse[1])
    p = 1 / (1+np.exp(-nu))
    p_low = 1 / (1+np.exp(-nu_low))
    p_high = 1 / (1+np.exp(-nu_high))
    
    bins = np.linspace(x.min(), x.max(), 20)
    prob = np.zeros_like(bins)
    for i in range(bins.shape[0]-1):
        mask = np.logical_and(x >= bins[i], x < bins[i+1])
        total = np.count_nonzero(x[mask])
        success = y[mask].sum()
        #print(total, success)
        if total != 0:
            prob[i] = float(success) / total
        else:
            prob[i] = 1
    prob[-1] = prob[-2]
    
    s90_95_exists = np.any(p_low > 0.9)
    s90_exists = np.any(p > 0.9)
    
    if s90_exists:
        s90 = fit_x[np.where(p > 0.9)].min()
        print('s90 = {}'.format(s90))
    else:
        print('s90 does not exist')
    if s90_95_exists:
        s90_95 = fit_x[np.where(p_low > 0.9)].min()
        print('s90/95% = {}'.format(s90_95))
    else:
        print('s90/95% does not exist')
    
    plt.figure(figsize = (12,9))
    #plt.scatter(x, y, alpha = 0.5, label = 'Binary outcomes')
    #plt.plot(bins, prob, c='g', linestyle='--', label = 'Ratio estimate')
    plt.fill_between(fit_x, p_low, p_high, color='b', alpha = 0.2, label = '95% confidence interval')
    plt.plot(fit_x, p, c='r', label = 'POD Curve')
    if s90_95_exists:
        plt.vlines([s90, s90_95], 0., 0.9, linestyles='--', color='k')
        plt.scatter([s90, s90_95], [0.9, 0.9], color='k', s=20)
    elif s90_exists:
        plt.vlines([s90], 0., 0.9, linestyles='--', color='k')
        plt.scatter([s90], [0.9], color='k', s=20)
    plt.xlabel('Defect size, mm', fontsize=20)
    plt.ylabel("Probability of detection", fontsize=20)
    
    ax = plt.gca()
    ax.set_ylim(0., 1.)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    
    plt.grid(True)
    plt.legend(loc=2, fontsize=16)
    plt.tight_layout()
    #plt.savefig('tmp_img/img.eps', format='eps', transparent=True)
    plt.savefig('tmp_img/img.png', format='png')
    plt.show()
    
def show_pod_pipeline(res, fit):
    fig = plt.figure(figsize = (16,12))
    gs = fig.add_gridspec(2, 2, height_ratios = [0.4,0.6])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    y = res['Iter1']
    x = res['FO_th']
    
    ax1.scatter(x, y, alpha = 0.5)
    ax1.set_xlabel('FO size, mm', fontsize=16)
    ax1.set_ylabel("F1 score", fontsize=16)
    ax1.set_title('(a)', y = -0.19, fontsize=16, weight='bold')
    ax1.set_ylim(0., 1.)
    ax1.set_xlim(0., 2.5)
    ax1.yaxis.set_major_locator(plt.LinearLocator(11))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax1.grid(True)
    
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
    ax2.bar(bins, prob, bin_width, align='edge')
    ax2.set_xlabel('FO size, mm', fontsize=16)
    ax2.set_ylabel("Fraction of cases with F1 > 50%", fontsize=16)
    ax2.set_title('(b)', y = -0.19, fontsize=16, weight='bold')
    ax2.set_ylim(0., 1.)
    ax2.set_xlim(0., 2.5)
    ax2.yaxis.set_major_locator(plt.LinearLocator(11))
    ax2.grid(True)
    
    fit_x = np.linspace(x.min(), x.max(), 1000)
    nu = fit.params[0]*fit_x + fit.params[1]
    nu_low = (fit.params[0]-2*fit.bse[0])*fit_x + (fit.params[1]-2*fit.bse[1])
    nu_high = (fit.params[0]+2*fit.bse[0])*fit_x + (fit.params[1]+2*fit.bse[1])
    p = 1 / (1+np.exp(-nu))
    p_low = 1 / (1+np.exp(-nu_low))
    p_high = 1 / (1+np.exp(-nu_high))
    s90 = fit_x[np.where(p > 0.9)].min()
    s90_95 = fit_x[np.where(p_low > 0.9)].min()
    print(s90, s90_95)
    ax3.fill_between(fit_x, p_low, p_high, color='b', alpha = 0.2, label = '95% confidence interval')
    ax3.plot(fit_x, p, c='r')
    plt.vlines([s90, s90_95], 0., 0.9, linestyles='--', color='k')
    plt.scatter(s90, 0.9, color='k', s=30, label = r"$s_{90}$")
    plt.scatter(s90_95, 0.9, color='g', s=30, label = r"$s_{90/95\%}$")
    ax3.set_xlabel("FO size, mm", fontsize=16)
    ax3.set_ylabel('Probability of F1>50%', fontsize=16)
    ax3.set_title('(c)', y = -0.13, fontsize=16, weight='bold')
    ax3.set_ylim(0., 1.)
    ax3.set_xlim(0., 2.5)
    ax3.yaxis.set_major_locator(plt.LinearLocator(11))
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax3.grid(True)
    ax3.legend(fontsize=20)
    
    plt.tight_layout()
    plt.savefig('tmp_img/pod_pipeline.pdf', format='pdf')
    #plt.savefig('tmp_img/img.png', format='png')
    plt.show()
    
def comp_mat(fits1, fits2, labels, fit_x = np.linspace(0., 3., 1000)):
    fig = plt.figure(figsize = (18,10))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    fit_labels = ['Train MC', 'Train Radon']
    for i, fit in enumerate(fits1):
        nu = fit.params[0]*fit_x + fit.params[1]
        nu_low = (fit.params[0]-2*fit.bse[0])*fit_x + (fit.params[1]-2*fit.bse[1])
        nu_high = (fit.params[0]+2*fit.bse[0])*fit_x + (fit.params[1]+2*fit.bse[1])
        p = 1 / (1+np.exp(-nu))
        p_low = 1 / (1+np.exp(-nu_low))
        p_high = 1 / (1+np.exp(-nu_high))
        
        ax1.fill_between(fit_x, p_low, p_high, alpha = 0.2)
        ax1.plot(fit_x, p, label = fit_labels[i])
    ax1.set_xlabel("FO size, mm", fontsize=16)
    ax1.set_ylabel('Probability of F1>50%', fontsize=16)
    ax1.set_title('(a) {}'.format(labels[0]), y = -0.1, fontsize=16, weight='bold')
    ax1.set_ylim(0., 1.)
    ax1.set_xlim(0., 2.5)
    ax1.yaxis.set_major_locator(plt.LinearLocator(11))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax1.grid(True)
    ax1.legend(fontsize=20, loc=4)
    
    for i, fit in enumerate(fits2):
        nu = fit.params[0]*fit_x + fit.params[1]
        nu_low = (fit.params[0]-2*fit.bse[0])*fit_x + (fit.params[1]-2*fit.bse[1])
        nu_high = (fit.params[0]+2*fit.bse[0])*fit_x + (fit.params[1]+2*fit.bse[1])
        p = 1 / (1+np.exp(-nu))
        p_low = 1 / (1+np.exp(-nu_low))
        p_high = 1 / (1+np.exp(-nu_high))
        
        ax2.fill_between(fit_x, p_low, p_high, alpha = 0.2)
        ax2.plot(fit_x, p, label = fit_labels[i])
    ax2.set_xlabel("FO size, mm", fontsize=16)
    ax2.set_ylabel('Probability of F1>50%', fontsize=16)
    ax2.set_title('(b) {}'.format(labels[1]), y = -0.1, fontsize=16, weight='bold')
    ax2.set_ylim(0., 1.)
    ax2.set_xlim(0., 2.5)
    ax2.yaxis.set_major_locator(plt.LinearLocator(11))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax2.grid(True)
    ax2.legend(fontsize=20, loc=4)
    
    plt.tight_layout()
    plt.savefig('tmp_img/comp_mat.pdf', format='pdf')
    plt.show()
    
def load_res_file(fname):
    res = np.genfromtxt(fname, delimiter=',', names=True)
    nonzero_FO_mask = res['FO_th'] != 0
    res = res[nonzero_FO_mask]
    return res

if __name__ == "__main__":
    
    #res_r_r = load_res_file('./test_res/bal_r_r.csv')
    res_r_mc = load_res_file('./test_res/pl90_r_mc.csv')
    #res_mc_r = load_res_file('./test_res/bal_mc_r.csv')
    res_mc_mc = load_res_file('./test_res/pl90_mc_mc.csv')
    
    res_r_mc2 = load_res_file('./test_res/biron_r_mc.csv')
    res_mc_mc2 = load_res_file('./test_res/biron_mc_mc.csv')
    
    fit = stat_analyze(res_mc_mc)
    #pod_glm(res_mc_mc, fit)
    #show_pod_pipeline(res_r_mc, fit)
    fit2 = stat_analyze(res_r_mc)
    #fit3 = stat_analyze(res_r_r)
    #fit4 = stat_analyze(res_mc_r)
    #compare_pod([fit, fit2], ['Train MC/Test MC', 'Train Radon/Test MC'])
    #compare_pod([fit, fit2, fit3, fit4], ['Train MC/Test MC', 'Train Radon/Test MC', 'Train Radon/Test Radon', 'Train MC/Test Radon'])
    
    fit21 = stat_analyze(res_mc_mc2)
    fit22 = stat_analyze(res_r_mc2)
    
    comp_mat([fit, fit2], [fit21, fit22], ['PMMA, 90kV', 'Iron, 450kV'])
    #pod_glm(res_r_mc, fit2)
