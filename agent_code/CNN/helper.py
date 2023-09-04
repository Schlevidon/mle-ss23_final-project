import matplotlib.pyplot as plt
from IPython import display

plt.ion()
#fig, (score_ax, eps_ax) = plt.subplots(2, sharex=True)
'''def plot(scores, mean_scores, eps):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    score_ax.set_title('Training scores')
    #score_ax.set_xlabel('Number of Games')
    score_ax.set_ylabel('Score')
    score_ax.plot(scores, label='score')
    score_ax.plot(mean_scores, label='average score')
    score_ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    score_ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    eps_ax.set_title('Training eps')
    eps_ax.set_xlabel('Number of ')
    eps_ax.set_ylabel('Epsilon')
    eps_ax.plot(eps)
    eps_ax.text(len(eps)-1, eps[-1], str(eps[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)'''

def plot(scores, mean_scores, eps):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    ax1 = plt.subplot(211)
    plt.title('Training...')
    #plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='average score')
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(eps)
    plt.ylabel('Epsilon')
    plt.xlabel('Number of Games')
    plt.text(len(eps)-1, eps[-1], str(eps[-1]))

    #plt.legend()
    plt.show(block=False)
    plt.pause(.1)

'''
def plot(scores, mean_scores, eps):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='average score')
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
    '''


