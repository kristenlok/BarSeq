import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import statsmodels.stats.multitest as multi

df_og = pd.read_csv(sys.argv[1]) #counts
r_tot = pd.read_csv(sys.argv[2]) #rtot
outliers = pd.read_csv(sys.argv[3]) #outliers
s = pd.read_csv(sys.argv[4]) #fitness
meanfit = pd.read_csv(sys.argv[5]) #mean fitness
title = sys.argv[6] #name of csv file you want it to be

#drop outliers
RD_threshold=6 # defined in SI (sec S3.5) of Ascensao et al 2022
cond1=outliers['RD'] > RD_threshold
cond2=outliers['RD'] == -1
outliers=outliers[cond1 | cond2]
df = df_og[~df_og.barcode.isin(outliers.barcode)]

#drop NaN
df = df.dropna(subset=['gene_ID'])

#drop barcodes with average count less than 30
df['avg_count'] = df[['0_1','0_2','1','2','3','4']].mean(axis=1)
df = df[df['avg_count'] >= 30]
df = df.drop(columns=['avg_count'])

#frequency of each day using rtot: f(t)
for day in ['0_1','0_2','1','2','3','4']:
    df["%s_freq" % day] = df[day]/r_tot.iloc[0][day]
    
#VST of frequency: x(f(t))
df['x(0_freq)'] = np.sqrt((df[['0_1_freq', '0_2_freq']].mean(axis=1)))
for freq in ['1_freq','2_freq','3_freq','4_freq']:
    df["x(%s)" % freq] = np.sqrt((df[freq]))

#incorporate fitness
df = pd.merge(df, s[['gene_ID','s']], on=['gene_ID'], how='left')
df = df.dropna(subset=['s'])

#correct for fitness effect: x_corr(x(t))
for day in ['1','2','3','4']:
    df['x_corr(%s)' % day] = df['x(%s_freq)' % day] - (df['x(0_freq)']*np.exp((df['s'].astype('float') - meanfit.iloc[int(day)-1]['xbar'])*(meanfit.iloc[int(day)-1]['gens']/2)))

#variance over time for each barcode
df['var(x_corr)'] = df[['x_corr(1)', 'x_corr(2)', 'x_corr(3)', 'x_corr(4)']].var(axis=1) #sample variance

#avg variance 
df_mv = df.groupby('gene_ID')
df_mv = df_mv['var(x_corr)'].describe().drop(columns=['min','25%','50%','75%','max']).sort_values(by='mean',ascending=False)
df_mv = df_mv[df_mv['count'] > 3.0].reset_index()


##get p-value:
#calculate variance of all intergenic barcodes
df_null = df_og.dropna(subset=['pos'])
df_null = df_null[df_null['gene_ID'].isnull()]

#drop barcodes with average count less than 30
df_null['avg_count'] = df_null[['0_1','0_2','1','2','3','4']].mean(axis=1)
df_null = df_null[df_null['avg_count'] >= 30]
df_null = df_null.drop(columns=['avg_count'])

#frequency of each day using rtot: f(t)
for day in ['0_1','0_2','1','2','3','4']:
    df_null["%s_freq" % day] = df_null[day]/r_tot.iloc[0][day]
    
#VST of frequency: x(f(t))
df_null['x(0_freq)'] = np.sqrt((df_null[['0_1_freq', '0_2_freq']].mean(axis=1)))
for freq in ['1_freq','2_freq','3_freq','4_freq']:
    df_null["x(%s)" % freq] = np.sqrt((df_null[freq]))
    
##no gene ID to collect fitness correction --> take variance of vst of frequency: var(x(f(t)))
#variance over time for each barcode
df_null['var(x(f(t)))'] = df_null[['x(1_freq)', 'x(2_freq)', 'x(3_freq)', 'x(4_freq)']].var(axis=1) #sample variance

def sampling(gene_ID): #n is sample size (barcode count for a gene in df_mv)
    num_times = 0
    for _ in range(500):
        n = int(df_mv.loc[df_mv.gene_ID == gene_ID, 'count'].values[0])
        barc_var = df_mv.loc[df_mv.gene_ID == gene_ID, 'mean'].values[0]
        null_var = df_null.sample(n, replace=False)['var(x(f(t)))'].mean()
        if barc_var < null_var:
            num_times += 1
    return num_times/500

df_mv['p_val'] = df_mv['gene_ID'].apply(sampling)
df_mv['p_corr'] = multi.multipletests(df_mv['p_val'].tolist(), method='fdr_bh')[1]
#df_mv['p_val'] = 1 - df_mv['p_val']
#df_mv['p_corr'] = 1 - df_mv['p_corr']

df_mv.to_csv('%s.csv' % title)


