import matplotlib.pyplot as plt
import time
from matplotlib.font_manager import FontProperties
import numpy as np

def plotgrid(rankers, plots, initial_steps, save_path=None):
    
    nrows = 1 + max(plots[pt]['row'] for pt in plots)
    ncols = 1 + max(plots[pt]['col'] for pt in plots)
    plt.close("all")
    plt.ioff()
    fontP = FontProperties()
    fontP.set_size('small')

    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,  gridspec_kw={'wspace': 0}, figsize=(10.5,4))

    for i in range(nrows):
        for j in range(ncols):
            if (i,j) not in [(plots[pt]['row'],plots[pt]['col']) for pt in plots]:
                axs[i][j].remove()

    for pt in plots:
        ax = axs[plots[pt]['row'],plots[pt]['col']]
        for i,r in enumerate(rankers):
            plots[pt][r] = ax.plot(plots[pt]['xlim'],(np.nan,np.nan), f"C{i}{plots[pt]['style']}")

        ax.set_title(plots[pt]['title'])
        #ax.label_outer()
        ax.set_yscale(plots[pt]['yscale'])
        if plots[pt]['row'] != nrows - 1:
            ax.xaxis.set_visible(False)
        #if plots[pt]['yscale'] == 'log':
        #    ax.yaxis.set_major_locator(plt.LogLocator(base=10,numticks=10))
        #    ax.yaxis.set_minor_locator(plt.LogLocator(base=10,numticks=100))
        ax.yaxis.set_visible(True)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plots[pt]['label'])
        ax.set_xlim(plots[pt]['xlim'])
        ax.set_ylim(plots[pt]['ylim'])
        ax.yaxis.grid(False, which='major')

    for ax in axs.flat:
        ax.axvline(x=initial_steps, color="grey", linestyle="--")

    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = False
    #plt.subplots_adjust(right=0.83,left=0.05)
    fig.legend(labels=rankers.keys(), loc='lower right')
    def callback(data):
        for pt in plots:
            if pt in data:
                plots[pt][data["algo"]][0].set_data(np.linspace(*plots[pt]['xlim'],len(data[pt])), data[pt])
        #plt.show()
        time.sleep(0.1)
        fig.canvas.draw()
        if save_path:
            fig.savefig(save_path)
    
    return fig, callback