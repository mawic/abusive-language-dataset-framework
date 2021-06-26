import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import operator
import pandas as pd
import processing.basic_stats
import processing.user_stats
import math
import numpy as np

from time import gmtime, strftime
from collections import Counter 
from datetime import datetime
from calendar import monthrange
from matplotlib.ticker import PercentFormatter

import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.")

def last_day_of_month(date_value):
    return date_value.replace(day = monthrange(date_value.year, date_value.month)[1]).date()

def plotTimeDistSingle(timestamps,title):
    quantile = 0.05
    
    first_tweet = last_day_of_month(min(timestamps))
    last_tweet = last_day_of_month(max(timestamps))

    # get quantiles
    df_quantile_1 = pd.DataFrame(timestamps, columns=['date'])
    df_quantile_1 = df_quantile_1.sort_values(by=['date'])
    df_quantile_2 = pd.DataFrame(df_quantile_1['date'].to_list(),columns=["date"])
    lower_q_pos = math.floor(len(df_quantile_2)*(quantile/2))
    upper_q_pos = math.ceil(len(df_quantile_2)*(1-quantile/2))
    lower_q_value = last_day_of_month(df_quantile_2.iloc[lower_q_pos]['date'])
    upper_q_value = last_day_of_month(df_quantile_2.iloc[upper_q_pos]['date'])
    quantile_border = [lower_q_value,upper_q_value]
        
    #if min_timestamp is not None:
    #    timestamps.append(min_timestamp)
        
    #if max_timestamp is not None:
    #    timestamps.append(max_timestamp)
    
    # prepare data 
    df = pd.DataFrame(timestamps, columns=['date'])
    df = df.set_index('date')
    df['count'] = 1
    g = df.groupby(pd.Grouper(freq="M")).sum() 
    x = g.index
    y = g['count']

    fig = plt.figure()
    ax1 = fig.add_axes((0, 0, 1, 1))
    # Make the same graph
    #plt.fill_between( x, y, color="skyblue", alpha=0)
    plt.fill_between([lower_q_value,upper_q_value], 0, max(y),facecolor='lightgrey', alpha=0.5)
    plt.fill_between( x, y, color="skyblue", alpha=0.3)#
    plt.plot(x, y, color="skyblue")
    plt.plot((first_tweet, first_tweet), (0, max(y)),'b--',linewidth=1)
    plt.plot((last_tweet, last_tweet), (0, max(y)),'b--',linewidth=1)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter("%m/%y"))
        
    
    # Add titles
    plt.title(title)
    plt.ylabel("Tweets")
    
def plotTimeDistMultiple(timestamps_list,labels,title,width=8,height=6):
    quantile = 0.05
    path_fig = "./results/"+strftime("%y%m%d", gmtime())+ "-" + "-".join(labels).replace(" ","_")
    fig, axs = plt.subplots(len(timestamps_list),figsize=(width,height))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.4)
 
    min_timestamp = min(np.concatenate(timestamps_list).ravel().tolist())
    max_timestamp = max(np.concatenate(timestamps_list).ravel().tolist())
    
    for i, a in enumerate(axs):
        timestamps = timestamps_list[i]
        first_tweet = last_day_of_month(min(timestamps))
        last_tweet = last_day_of_month(max(timestamps))

        # get quantiles
        df_quantile_1 = pd.DataFrame(timestamps, columns=['date'])
        df_quantile_1 = df_quantile_1.sort_values(by=['date'])
        df_quantile_2 = pd.DataFrame(df_quantile_1['date'].to_list(),columns=["date"])
        lower_q_pos = math.floor(len(df_quantile_2)*(quantile/2))
        upper_q_pos = math.ceil(len(df_quantile_2)*(1-quantile/2))
        lower_q_value = last_day_of_month(df_quantile_2.iloc[lower_q_pos]['date'])
        upper_q_value = last_day_of_month(df_quantile_2.iloc[upper_q_pos]['date'])
        quantile_border = [lower_q_value,upper_q_value]

        if min_timestamp is not None:
            timestamps.append(min_timestamp)

        if max_timestamp is not None:
            timestamps.append(max_timestamp)

        # prepare data 
        df = pd.DataFrame(timestamps, columns=['date'])
        df = df.set_index('date')
        df['count'] = 1
        g = df.groupby(pd.Grouper(freq="M")).sum() 
        x = g.index
        y = g['count']

        # Make the same graph
        #plt.fill_between( x, y, color="skyblue", alpha=0)
        a.fill_between([lower_q_value,upper_q_value], 0, max(y),facecolor='lightgrey', alpha=0.7)
        a.fill_between( x, y, color="darkblue", alpha=0.3)
        a.plot(x, y, color="darkblue",linewidth=1)
        a.plot((first_tweet, first_tweet), (0, max(y)),'b--',linewidth=1)
        a.plot((last_tweet, last_tweet), (0, max(y)),'b--',linewidth=1)

        a.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        a.xaxis.set_minor_formatter(mdates.DateFormatter("%m/%y"))
        a.set_title(labels[i])

        sns.despine(ax=a, top=True, bottom=False, right=True, left=True)
        if i != len(timestamps_list)-1:
            a.get_xaxis().set_visible(False)
            
        if len(timestamps_list) > 1:
            a.get_yaxis().set_visible(False)
    fig.savefig(path_fig + "-meta_time_distribution.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-meta_time_distribution.png", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-meta_time_distribution.eps", bbox_inches='tight', dpi=600)
            
def getLabelDistribution(data):
    cnt = Counter()
    for item in data:
        cnt[item['label']] += 1
    return cnt

def getLabels(data):
    labels = []
    for item in getLabelDistribution(data):
        labels.append(item)
    return labels

def getDataFrameForUserPareto(data,n=10):
    # get labels from data
    labels = getLabels(data)
    
    user_hitlist, user_ids, user_posts = processing.user_stats.get_posts_per_user(data)
    user_stats = processing.user_stats.get_user_stats(user_hitlist, user_posts, labels)
    
    # create data frame from data for easier handling and processing
    lst_of_lsts = []
    for item in data:
        row = []
        row.append(item['user']['id'])
        row.append(item['label'])
        lst_of_lsts.append(row)
    df=pd.DataFrame(lst_of_lsts,columns=['user','label'])
    
    # pivot data frame and aggregate on user level
    df_n = pd.crosstab([df.user],df.label)
    df_n['total'] = 0
    for label in labels:
        df_n['total'] += df_n[label]
    df_n = df_n.sort_values(by='total', ascending=False)

    # calculate others
    df_f = df_n.sort_values(by='total', ascending=False)[:n].copy()
    df_f = df_f.reset_index()
    row = {'user':'#' +str(n)+ ' - \n #' + str(len(df_n))}
    for label in labels:
        total_label = sum(df_n[label].to_list())
        total_select = sum(df_n.sort_values(by='total', ascending=False)[:n][label].to_list())
        row[label] = total_label-total_select
    df_f = df_f.append(row, ignore_index=True)

    # update total
    df_f['total'] = 0
    for label in labels:
        df_f['total'] += df_f[label]
        
    #calculate cumpercentage
    df_f["cumpercentage"] = df_f["total"].cumsum()/df_f["total"].sum()*100

    # rename users
    user = df_f['user'].to_list()
    for i,name in enumerate(user):
        if i < n:
            user[i] = "#"+str(i+1)
    df_f['user'] = user

    return df_f

def plotUserDistSingle(data,title,n=10):

    fig, ax = plt.subplots()
    fig.suptitle(title,fontsize=16)

    labels = getLabels(data)

    palette = "colorblind"

    colors = sns.color_palette(palette, len(labels)+1)
    sns.set_palette(palette, len(labels)+1)


    df_f = getDataFrameForUserPareto(data,n=n)

    prev_data = []
    for i,label in enumerate(labels):
        if i == 0:
            ax.bar(df_f.user, df_f[label], label=label)
            prev_data = df_f[label]
        else:
            ax.bar(df_f.user, df_f[label], bottom=prev_data, label=label)
            prev_data += df_f[label]

    ax.legend()        


    ax2 = ax.twinx()
    ax2.plot(df_f.user, df_f["cumpercentage"], marker="o",color=colors[len(labels)], ms=5)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim([0,105])

    ax.tick_params(axis="y")
    ax2.tick_params(axis="y")
    plt.show()
    
def plotUserDistMultiple(data,title,subtitles,n=10,rows=2,cols=1,palette = "colorblind",width=12,height=6):
    path_fig = "./results/"+strftime("%y%m%d", gmtime())+ "-" + "-".join(subtitles).replace(" ","_")
    fig2 = plt.figure(constrained_layout=True,figsize=(width, height))
    fig2.suptitle(title,y=1.05,fontsize=16)
    spec2 = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig2)
    
    m = 0
    for k in range(rows):
        for l in range(cols):
            if m < len(data):
                # get data
                labels = getLabels(data[m])
                df_f = getDataFrameForUserPareto(data[m],n=n)

                # define colors and style
                colors = sns.color_palette(palette, len(labels)+1)
                sns.set_palette(palette, len(labels)+1)                 
                
                # define figure
                ax = fig2.add_subplot(spec2[k, l])
                ax.set_title(subtitles[m])

                # bar chart
                prev_data = []
                for i,label in enumerate(labels):
                    if i == 0:
                        ax.bar(df_f.user, df_f[label], label=label)
                        prev_data = df_f[label]
                    else:
                        ax.bar(df_f.user, df_f[label], bottom=prev_data, label=label)
                        prev_data += df_f[label]

                ax.legend()  

                # line chart
                ax2 = ax.twinx()
                ax2.plot(df_f.user, df_f["cumpercentage"], marker="o",color=colors[len(labels)], ms=5)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax2.set_ylim([0,105])

                ax.get_yaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                
                sns.despine(ax=ax, top=True, bottom=False, right=True, left=True)
                sns.despine(ax=ax2, top=True, bottom=False, right=True, left=True)
                m += 1
    
    
    fig2.savefig(path_fig + "-meta_user_distribution.pdf", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-meta_user_distribution.png", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-meta_user_distribution.eps", bbox_inches='tight', dpi=600)

               
def plotClassesMultiple(title,subtitles,data_full,data_available,rows=2,cols=1,sync_scaling=False,palette = "Blues",width=12,height=6):
    path_fig = "./results/"+strftime("%y%m%d", gmtime())+ "-" + "-".join(subtitles).replace(" ","_")
    fig2 = plt.figure(constrained_layout=True,figsize=(width, height))
    fig2.suptitle(title,y=1.05,fontsize=16)
    spec2 = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig2)
    
    m = 0
    for k in range(rows):
        for l in range(cols):
            if m < len(data_full):
                # prepare data
                labels = []
                available = []
                not_available = []
                total = []
                percentage = []
                
                labels_full = []
                labels_avail = []
                
                # collect labels
                for elem in data_full[m]:
                    labels_full.append(elem['label'])
                c_full = Counter(labels_full)
                
                for elem in data_available[m]:
                    labels_avail.append(elem['label'])
                c_avail =  Counter(labels_avail)

                # calculate bars
                for c in  sorted(c_full, key=c_full.get, reverse=True):
                    labels.append(c)
                    available.append(c_avail[c])
                    not_available.append(c_full[c] - c_avail[c])
                    total.append(c_full[c])
                # percentage of total
                for elem in total:
                    percentage.append(elem/sum(total))   
                
                ax = fig2.add_subplot(spec2[k, l])
                # define colors and style
                
                sns.set_palette(palette, 5)
                colors = sns.color_palette(palette, 5)

                # bar chart
                label_available = "available ({:.0%})".format(sum(available)/(sum(available)+sum(not_available)))
                lable_na =  "n/a ({:.0%})".format(sum(not_available)/(sum(available)+sum(not_available)))

                ax.bar(labels, available, label=label_available,color=colors[4])
                ax.bar(labels, not_available, bottom=available, label=lable_na,color=colors[1])

                #print(subtitles[m])
                #print("Not available")
                #for i,c in enumerate(available):
                    #print(not_available[i]/(not_available[i]+available[i]))
                
                # total labels for bars
                for i, v in enumerate(total):
                    offset = max(total)*0.02
                    label_total = "{:,}".format(v) + " ({:.0%})".format(percentage[i])
                    ax.text(i, v + offset,label_total , color='black', horizontalalignment='center', fontsize=10)
                
                ax.legend()  

                ax.set_ylim([0,max(total)*1.20])
                ax.tick_params(axis="y")
                ax.get_xaxis().set_tick_params(width=1)
                #print("##",ax.get_xaxis().get_label_text())
                ax.get_yaxis().set_visible(False)
                ax.set_title(subtitles[m] +' (n = {:,})'.format(sum(total)))
                sns.despine(ax=ax, top=True, bottom=False, right=True, left=True)
                m += 1
    fig2.savefig(path_fig + "-meta_class_distribution.pdf", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-meta_class_distribution.png", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-meta_class_distribution.eps", bbox_inches='tight', dpi=600)

def plotClassesSingle(title,subtitle,data_full,data_available,sync_scaling=False,palette = "colorblind"):
    plotClassesMultiple(title,subtitle,[data_full],[data_available],rows=1,cols=1,sync_scaling=sync_scaling,palette = palette)  