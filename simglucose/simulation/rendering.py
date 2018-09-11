import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class Viewer(object):
    def __init__(self, start_time, patient_name, figsize=None):
        self.start_time = start_time
        self.patient_name = patient_name
        self.fig, self.axes, self.lines = self.initialize()
        self.update()

    def initialize(self):
        plt.ion()
        fig, axes = plt.subplots(4)

        axes[0].set_ylabel('BG (mg/dL)')
        axes[1].set_ylabel('CHO (g/min)')
        axes[2].set_ylabel('Insulin (U/min)')
        axes[3].set_ylabel('Risk Index')

        lineBG, = axes[0].plot([], [], label='BG')
        lineCGM, = axes[0].plot([], [], label='CGM')
        lineCHO, = axes[1].plot([], [], label='CHO')
        lineIns, = axes[2].plot([], [], label='Insulin')
        lineLBGI, = axes[3].plot([], [], label='Hypo Risk')
        lineHBGI, = axes[3].plot([], [], label='Hyper Risk')
        lineRI, = axes[3].plot([], [], label='Risk Index')

        lines = [lineBG, lineCGM, lineCHO, lineIns, lineLBGI, lineHBGI, lineRI]

        axes[0].set_ylim([70, 180])
        axes[1].set_ylim([-5, 30])
        axes[2].set_ylim([-0.5, 1])
        axes[3].set_ylim([0, 5])

        for ax in axes:
            ax.set_xlim(
                [self.start_time, self.start_time + timedelta(hours=3)])
            ax.legend()

        # Plot zone patches
        axes[0].axhspan(70, 180, alpha=0.3, color='limegreen', lw=0)
        axes[0].axhspan(50, 70, alpha=0.3, color='red', lw=0)
        axes[0].axhspan(0, 50, alpha=0.3, color='darkred', lw=0)
        axes[0].axhspan(180, 250, alpha=0.3, color='red', lw=0)
        axes[0].axhspan(250, 1000, alpha=0.3, color='darkred', lw=0)

        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
        axes[2].tick_params(labelbottom=False)
        axes[3].xaxis.set_minor_locator(mdates.AutoDateLocator())
        axes[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        axes[3].xaxis.set_major_locator(mdates.DayLocator())
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

        axes[0].set_title(self.patient_name)

        return fig, axes, lines

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self, data):
        self.lines[0].set_xdata(data.index.values)
        self.lines[0].set_ydata(data['BG'].values)

        self.lines[1].set_xdata(data.index.values)
        self.lines[1].set_ydata(data['CGM'].values)

        self.axes[0].draw_artist(self.axes[0].patch)
        self.axes[0].draw_artist(self.lines[0])
        self.axes[0].draw_artist(self.lines[1])

        adjust_ylim(self.axes[0], min(min(data['BG']), min(data['CGM'])),
                    max(max(data['BG']), max(data['CGM'])))
        adjust_xlim(self.axes[0], data.index[-1])

        self.lines[2].set_xdata(data.index.values)
        self.lines[2].set_ydata(data['CHO'].values)

        self.axes[1].draw_artist(self.axes[1].patch)
        self.axes[1].draw_artist(self.lines[2])

        adjust_ylim(self.axes[1], min(data['CHO']), max(data['CHO']))
        adjust_xlim(self.axes[1], data.index[-1])

        self.lines[3].set_xdata(data.index.values)
        self.lines[3].set_ydata(data['insulin'].values)

        self.axes[2].draw_artist(self.axes[2].patch)
        self.axes[2].draw_artist(self.lines[3])
        adjust_ylim(self.axes[2], min(data['insulin']), max(data['insulin']))
        adjust_xlim(self.axes[2], data.index[-1])

        self.lines[4].set_xdata(data.index.values)
        self.lines[4].set_ydata(data['LBGI'].values)

        self.lines[5].set_xdata(data.index.values)
        self.lines[5].set_ydata(data['HBGI'].values)

        self.lines[6].set_xdata(data.index.values)
        self.lines[6].set_ydata(data['Risk'].values)

        self.axes[3].draw_artist(self.axes[3].patch)
        self.axes[3].draw_artist(self.lines[4])
        self.axes[3].draw_artist(self.lines[5])
        self.axes[3].draw_artist(self.lines[6])
        adjust_ylim(self.axes[3], min(data['Risk']), max(data['Risk']))
        adjust_xlim(self.axes[3], data.index[-1], xlabel=True)

        self.update()

    def close(self):
        plt.close(self.fig)


def adjust_ylim(ax, ymin, ymax):
    ylim = ax.get_ylim()
    update = False

    if ymin < ylim[0]:
        y1 = ymin - 0.1 * abs(ymin)
        update = True
    else:
        y1 = ylim[0]

    if ymax > ylim[1]:
        y2 = ymax + 0.1 * abs(ymax)
        update = True
    else:
        y2 = ylim[1]

    if update:
        ax.set_ylim([y1, y2])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.yaxis)


def adjust_xlim(ax, timemax, xlabel=False):
    xlim = mdates.num2date(ax.get_xlim())
    update = False

    # remove timezone awareness to make them comparable
    timemax = timemax.replace(tzinfo=None)
    xlim[0] = xlim[0].replace(tzinfo=None)
    xlim[1] = xlim[1].replace(tzinfo=None)

    if timemax > xlim[1] - timedelta(minutes=30):
        xmax = xlim[1] + timedelta(hours=6)
        update = True

    if update:
        ax.set_xlim([xlim[0], xmax])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.xaxis)
        if xlabel:
            ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
