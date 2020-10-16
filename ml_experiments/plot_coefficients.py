import seaborn as sns
sns.set(rc={'figure.figsize':(10,10)})
import matplotlib.pyplot as plt

import pandas as pd


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    print(imp[::-1][0:top])
    print(names[::-1][0:top])

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()


def f_importances2(coef, names, top):
    df = pd.DataFrame()
    df['coefficients'] = coef
    df['names'] = names

    df = df.sort_values(by='coefficients', ascending=False)
    df = df.head(top)
    sns.set_style("darkgrid")

    myplot = sns.barplot(x='names', y='coefficients', data=df, color='dimgray') # palette='mako')
    myplot.set_xlabel('')
    myplot.yaxis.tick_right()
    myplot.yaxis.set_label_position("right")
    myplot.set_ylabel("coefficient", fontsize=30)
    myplot.tick_params(labelsize=30)
    plt.xticks(rotation=90)
    plt.yticks(rotation=180)
    #plt.show()

