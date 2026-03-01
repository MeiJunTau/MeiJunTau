import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

df=pd.read_csv("scores.csv", index_col='Day')

def prepare_data(df, steps=5):
    df=df.reset_index()
    df.index=df.index*5
    last_idx = df.index[-1]+1
    df2 = df.reindex(range(last_idx))
    df2['Day'] = df2['Day'].ffill()
    df2=df2.set_index('Day')
    df_rank=df2.rank(axis=1,method='first')
    df2=df2.interpolate()
    df_rank=df_rank.interpolate()
    return df2, df_rank

df2, df_rank, = prepare_data(df)

print(df.head())
print(df2.head())
print(df_rank.head())

##fig, ax_array = plt.subplots(nrows=1,ncols=6,figsize=(12,2),dpi=144, tight_layout=True)

labels=df2.columns

colors = plt.cm.Dark2(range(21  ))

def axes(ax):
    ax.set_facecolor('.8')
    ax.tick_params(labelsize=8, length=0)
    ax.grid(True, axis='x', color='white')
    ax.set_axisbelow(True)
    [spine.set_visible(False) for spine in ax.spines.values()]    

def init():
    ax.clear()
    axes(ax)

def update(i):
    for bar in ax.containers:
        bar.remove()
    y=df_rank.iloc[i]
    width=df2.iloc[i]
    ax.barh(y=y,width=width,color=colors,tick_label=labels)
    ax.set_title(f'Predictions by Day', fontsize='smaller')

fig=plt.Figure(figsize=(8,4), dpi=144)
ax=fig.add_subplot()
anim=FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df2), interval=150, repeat=False)
anim.save('scores.mp4')
