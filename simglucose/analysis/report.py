import glob
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PatchCollection
# from pandas.plotting import lag_plot
import logging

logger = logging.getLogger(__name__)


def ensemble_BG(BG, ax=None, plot_var=False, nstd=3):
    mean_curve = BG.transpose().mean()
    std_curve = BG.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve - nstd * std_curve

    # t = BG.index.to_pydatetime()
    t = pd.to_datetime(BG.index)
    if ax is None:
        fig, ax = plt.subplots(1)
    if plot_var and not std_curve.isnull().all():
        ax.fill_between(
            t, up_env, down_env, alpha=0.5, label='+/- {0}*std'.format(nstd))
    for p in BG:
        ax.plot_date(
            t, BG[p], '-', color='grey', alpha=0.5, lw=0.5, label='_nolegend_')
    ax.plot(t, mean_curve, lw=2, label='Mean Curve')
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.axhline(70, c='green', linestyle='--', label='Hypoglycemia', lw=1)
    ax.axhline(180, c='red', linestyle='--', label='Hyperglycemia', lw=1)

    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([BG.min().min() - 10, BG.max().max() + 10])
    ax.legend()
    ax.set_ylabel('Blood Glucose (mg/dl)')
    #     fig.autofmt_xdate()
    return ax


def ensemblePlot(df):
    df_BG = df.unstack(level=0).BG
    df_CGM = df.unstack(level=0).CGM
    df_CHO = df.unstack(level=0).CHO
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1 = ensemble_BG(df_BG, ax=ax1, plot_var=True, nstd=1)
    ax2 = ensemble_BG(df_CGM, ax=ax2, plot_var=True, nstd=1)
    # t = df_CHO.index.to_pydatetime()
    t = pd.to_datetime(df_CHO.index)
    ax3.plot(t, df_CHO)

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    ax3.set_xlim([t[0], t[-1]])
    ax1.set_ylabel('Blood Glucose (mg/dl)')
    ax2.set_ylabel('CGM (mg/dl)')
    ax3.set_ylabel('CHO (g)')
    return fig, ax1, ax2, ax3


def percent_stats(BG, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    p_hyper = (BG > 180).sum() / len(BG) * 100
    p_hyper.name = 'BG>180'
    p_hypo = (BG < 70).sum() / len(BG) * 100
    p_hypo.name = 'BG<70'
    p_normal = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
    p_normal.name = '70<=BG<=180'
    p_250 = (BG > 250).sum() / len(BG) * 100
    p_250.name = 'BG>250'
    p_50 = (BG < 50).sum() / len(BG) * 100
    p_50.name = 'BG<50'
    p_stats = pd.concat([p_normal, p_hyper, p_hypo, p_250, p_50], axis=1)
    p_stats.plot(ax=ax, kind='bar')
    ax.set_ylabel('Percent of time in Range (%)')
    fig.tight_layout()
    #     p_stats.transpose().plot(kind='bar', legend=False)
    return p_stats, fig, ax


def risk_index_trace(df_BG, sample_time=3, window_length=60, visualize=False):
    step_size = int(window_length / sample_time)  # window size set to 1 hour for calculating Risk Index
    chunk_BG = [df_BG.iloc[i:i + step_size, :] for i in range(0, len(df_BG), step_size)]

    if len(chunk_BG[-1]) != step_size:  # Remove the last chunk which is not full
        chunk_BG.pop()

    fBG = [
        1.509 * (np.log(BG[BG > 0]) ** 1.084 - 5.381) for BG in chunk_BG
    ]

    rl = [(10 * (fbg * (fbg < 0)) ** 2).mean() for fbg in fBG]
    rh = [(10 * (fbg * (fbg > 0)) ** 2).mean() for fbg in fBG]

    LBGI = pd.concat(rl, axis=1).transpose()
    HBGI = pd.concat(rh, axis=1).transpose()
    RI = LBGI + HBGI

    ri_per_hour = pd.concat(
        [LBGI.transpose(), HBGI.transpose(),
         RI.transpose()],
        keys=['LBGI', 'HBGI', 'Risk Index'])

    axes = []
    if visualize:
        logger.info('Plotting risk trace plot')
        ri_per_hour_plot = pd.concat(
            [HBGI.transpose(), -LBGI.transpose()], keys=['HBGI', '-LBGI'])
        for i in range(len(ri_per_hour_plot.unstack(level=0))):
            logger.debug(
                ri_per_hour_plot.unstack(level=0).iloc[i].unstack(level=1))
            axtmp = ri_per_hour_plot.unstack(level=0).iloc[i].unstack(
                level=1).plot.bar(stacked=True)
            axes.append(axtmp)
            plt.xlabel('Time (hour)')
            plt.ylabel('Risk Index')

    ri_mean = ri_per_hour.transpose().mean().unstack(level=0)
    fig, ax = plt.subplots(1)
    ri_mean.plot(ax=ax, kind='bar')
    fig.tight_layout()

    axes.append(ax)
    return ri_per_hour, ri_mean, fig, axes


def CVGA_background(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.set_xlim(109, 49)
    ax.set_ylim(105, 405)
    ax.set_xticks([110, 90, 70, 50])
    ax.set_yticks([110, 180, 300, 400])
    ax.set_xticklabels(['110', '90', '70', '<50'])
    ax.set_yticklabels(['110', '180', '300', '>400'])
    #     fig.suptitle('Control Variability Grid Analysis (CVGA)')
    ax.set_title('Control Variability Grid Analysis (CVGA)')
    ax.set_xlabel('Min BG (2.5th percentile)')
    ax.set_ylabel('Max BG (97.5th percentile)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rectangles = {
        'A-Zone': plt.Rectangle((90, 110), 20, 70, color='limegreen'),
        'Lower B': plt.Rectangle((70, 110), 20, 70, color='green'),
        'Upper B': plt.Rectangle((90, 180), 20, 120, color='green'),
        'B-Zone': plt.Rectangle((70, 180), 20, 120, color='green'),
        'Lower C': plt.Rectangle((50, 110), 20, 70, color='yellow'),
        'Upper C': plt.Rectangle((90, 300), 20, 100, color='yellow'),
        'Lower D': plt.Rectangle((50, 180), 20, 120, color='orange'),
        'Upper D': plt.Rectangle((70, 300), 20, 100, color='orange'),
        'E-Zone': plt.Rectangle((50, 300), 20, 100, color='red')
    }
    facecolors = [rectangles[r].get_facecolor() for r in rectangles]
    pc = PatchCollection(
        rectangles.values(),
        facecolor=facecolors,
        edgecolors='w',
        lw=2,
        alpha=1)
    ax.add_collection(pc)
    for r in rectangles:
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width() / 2.0
        cy = ry + rectangles[r].get_height() / 2.0
        if r in ['Lower B', 'Upper B', 'B-Zone']:
            ax.annotate(
                r, (cx, cy),
                weight='bold',
                color='w',
                fontsize=10,
                ha='center',
                va='center')
        else:
            ax.annotate(
                r, (cx, cy),
                weight='bold',
                color='k',
                fontsize=10,
                ha='center',
                va='center')

    return fig, ax


def CVGA_analysis(BG):
    BG_min = np.percentile(BG, 2.5, axis=0)
    BG_max = np.percentile(BG, 97.5, axis=0)
    BG_min[BG_min < 50] = 50
    BG_min[BG_min > 400] = 400
    BG_max[BG_max < 50] = 50
    BG_max[BG_max > 400] = 400

    perA = ((BG_min > 90) & (BG_min <= 110) & (BG_max >= 110)
            & (BG_max < 180)).sum() / float(len(BG_min))
    perB = ((BG_min > 70) & (BG_min <= 110) & (BG_max >= 110)
            & (BG_max < 300)).sum() / float(len(BG_min)) - perA
    perC = (((BG_min > 90) & (BG_min <= 110) & (BG_max >= 300)) |
            ((BG_min <= 70) & (BG_max >= 110) &
             (BG_max < 180))).sum() / float(len(BG_min))
    perD = (((BG_min > 70) & (BG_min <= 90) & (BG_max >= 300)) |
            ((BG_min <= 70) & (BG_max >= 180) &
             (BG_max < 300))).sum() / float(len(BG_min))
    perE = ((BG_min <= 70) & (BG_max >= 300)).sum() / float(len(BG_min))
    return BG_min, BG_max, perA, perB, perC, perD, perE


def CVGA(BG_list, label=None):
    if not isinstance(BG_list, list):
        BG_list = [BG_list]
    if not isinstance(label, list):
        label = [label]
    if label is None:
        label = ['BG%d' % (i + 1) for i in range(len(BG_list))]
    fig, ax = CVGA_background()
    zone_stats = []
    for (BG, l) in zip(BG_list, label):
        BGmin, BGmax, A, B, C, D, E = CVGA_analysis(BG)
        ax.scatter(
            BGmin,
            BGmax,
            edgecolors='k',
            zorder=4,
            label='%s (A: %d%%, B: %d%%, C: %d%%, D: %d%%, E: %d%%)' %
                  (l, 100 * A, 100 * B, 100 * C, 100 * D, 100 * E))
        zone_stats.append((A, B, C, D, E))

    zone_stats = pd.DataFrame(zone_stats, columns=['A', 'B', 'C', 'D', 'E'])
    #     ax.legend(bbox_to_anchor=(1, 1.10), borderaxespad=0.5)
    ax.legend()
    return zone_stats, fig, ax


def report(df, cgm_sensor=None, save_path=None):
    BG = df.unstack(level=0).BG

    fig_ensemble, ax1, ax2, ax3 = ensemblePlot(df)
    pstats, fig_percent, ax4 = percent_stats(BG)
    if cgm_sensor is not None:
        ri_per_hour, ri_mean, fig_ri, ax5 = risk_index_trace(BG, sample_time=cgm_sensor.sample_time, visualize=False)
    else:
        ri_per_hour, ri_mean, fig_ri, ax5 = risk_index_trace(BG, visualize=False)
    zone_stats, fig_cvga, ax6 = CVGA(BG, label='')
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    figs = [fig_ensemble, fig_percent, fig_ri, fig_cvga]
    results = pd.concat([pstats, ri_mean], axis=1)

    if save_path is not None:
        results.to_csv(os.path.join(save_path, 'performance_stats.csv'))
        ri_per_hour.to_csv(os.path.join(save_path, 'risk_trace.csv'))
        zone_stats.to_csv(os.path.join(save_path, 'CVGA_stats.csv'))

        fig_ensemble.savefig(os.path.join(save_path, 'BG_trace.png'))
        fig_percent.savefig(os.path.join(save_path, 'zone_stats.png'))
        fig_ri.savefig(os.path.join(save_path, 'risk_stats.png'))
        fig_cvga.savefig(os.path.join(save_path, 'CVGA.png'))

    plt.show()
    return results, ri_per_hour, zone_stats, figs, axes


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('analysis.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - \n %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    # logger.addHandler(fh)
    logger.addHandler(ch)
    # For test only
    path = os.path.join('..', '..', 'examples', 'results',
                        '2017-12-31_17-46-32')
    os.chdir(path)
    filename = glob.glob('*#*.csv')
    name = [_f[:-4] for _f in filename]
    df = pd.concat([pd.read_csv(f, index_col=0) for f in filename], keys=name)
    # df_BG = df.unstack(level=0).BG
    # df_CGM = df.unstack(level=0).CGM
    # report(df_BG, df_CGM)
    results, ri_per_hour, zone_stats, axes = report(df)
    # print results
    # # print ri_per_hour
    # print zone_stats
