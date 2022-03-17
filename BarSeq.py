import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

df = pd.read_csv(sys.argv[1]) #counts
r_tot = pd.read_csv(sys.argv[2]) #rtot
outliers = pd.read_csv(sys.argv[3]) #outliers
df_s = pd.read_csv(sys.argv[4]) #fitness
df_t_xbar = pd.read_csv(sys.argv[5]) #mean fitness
title = sys.argv[6] #name of csv file you want it to be

###preprocess DFs:
RD_threshold=6 # defined in SI (sec S3.5) of Ascensao et al 2022
cond1=outliers['RD'] > RD_threshold
cond2=outliers['RD'] == -1
outliers=outliers[cond1 | cond2]
df = df[~df.barcode.isin(outliers.barcode)]
gene_index = []
for gene in df["gene_ID"]:
    gene_index.append(gene)
df.dropna(subset = ["gene_ID"], inplace = True)
df = df.reset_index() 
all_genes = [] #list of all gene_IDs
for gene in gene_index:
    if gene != "NaN" and gene != "nan" and gene not in all_genes:
        all_genes.append(gene)
all_genes.pop(0)
r_tot = r_tot.iloc[: , 1:]

#dictionary of gene_ID to corresponding dataframe
all_dfs = {} 
for gene in all_genes:            
    all_dfs[gene] = df[df.gene_ID == gene]  



###discard of counts under an average of 30 and genes with less than 4 barcodes:
def discard(df):
    to_discard = []
    for i in range(len(df)):
        sum = 0
        for day in ["0_1","0_2","1","2","3","4"]:
            sum += df.loc[i,day] 
        if sum/6 < 30: #average count across days is < 30 --> discard
            to_discard.append(i)
    return to_discard

df_dropped = df.copy()
df_dropped.drop(labels=discard(df), axis=0, inplace=True)
df_dropped =  df_dropped.groupby('gene_ID').filter(lambda x : len(x)>3)
df_dropped = df_dropped.reset_index() 




###frequency of all days:
rtot_dict = dict(zip(r_tot.columns.values, r_tot.iloc[0].values)) #dictionary with days and corresponding rtot
df_freqs = df_dropped.copy() #df with freqs of all days
def frequency(df_freqs, rtot):
    for day in ["0_1","0_2","1","2","3","4"]:
        freq_vals = [] #all frequency values
        for count in df_freqs[day]:
            freq = count/rtot[day]
            freq_vals.append(freq)
        df_freqs["%s_freq" % day] = freq_vals
    df_freqs.insert(15, "0_freq", df_freqs[["0_1_freq", "0_2_freq"]].mean(axis=1))
frequency(df_freqs, rtot_dict)


###VST of frequencies: x(t) := sqrt(f(t))
df_x = df_freqs.copy() #df with freqs of VST (variance stabilizing transformation)
def x_freq(df_x):
    for freq in ["0_freq","1_freq","2_freq","3_freq","4_freq"]:
        x_vals = [] #all frequency values
        for count in df_x[freq]:
            count = np.sqrt(count)
            x_vals.append(count)
        df_x["x(%s)" % freq] = x_vals
x_freq(df_x)  
df_x = df_x.drop(df_x.columns[[0, 1, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21]], axis=1)


###x_corr:
t_dict = dict(zip(df_t_xbar["Day label"], df_t_xbar["gens"]))
xbar_dict = dict(zip(df_t_xbar["Day label"], df_t_xbar["xbar"]))
s_dict = dict(zip(df_s["gene_ID"], df_s["s"]))
df_x_corr = df_x.copy() #df with corrected freqs of VST (variance stabilizing transformation)
#dropping rows/gene_IDs with no "s" value:
no_s = []
for i in range(len(df_x_corr)):
    index_gene = df_x_corr.columns.get_loc("gene_ID")
    if df_x_corr.iloc[i,index_gene] not in s_dict.keys():
        no_s.append(i)
df_x_corr.drop(no_s, axis=0, inplace=True)
df_x_corr = df_x_corr.reset_index()
df_x_corr = df_x_corr.drop(df_x_corr.columns[[0]], axis=1)
#calculating x_corrected:
days = [1, 2, 3, 4]
x_freqs = ["x(1_freq)","x(2_freq)","x(3_freq)","x(4_freq)"]
def x_corr_form(day, x, s_dict, t_dict, xbar_dict, df_x_corr):
    x_corr_vals = []
    for i in range(len(df_x_corr)):
        #index = day-1
        index_0 = df_x_corr.columns.get_loc("x(0_freq)")
        index_t = df_x_corr.columns.get_loc(x)
        x_0 = df_x_corr.iloc[i, index_0]
        x_t = df_x_corr.iloc[i, index_t]
        index_gene = df_x_corr.columns.get_loc("gene_ID")
        gene_ID = df_x_corr.iloc[i, index_gene]
        if gene_ID in s_dict.keys():
            s = s_dict[gene_ID]
        else:
            continue;
        t = t_dict[day]
        xbar = xbar_dict[day]
        x_corr = x_corr_ele(x_0, x_t, s, xbar, t)
        x_corr_vals.append(x_corr)
    return x_corr_vals
def x_corr_ele(x_0, x_t, s, xbar, t):
    x_corr = x_t - (x_0*math.exp((s-xbar)* (t/2)))
    return x_corr
#x_corr
for day, x in zip(days, x_freqs):
    x_corr = x_corr_form(day, x, s_dict, t_dict, xbar_dict, df_x_corr)
    df_x_corr["x_corr(%s)" % day] = x_corr


###x_var for each barcode:
x_corrs = df_x_corr[["x_corr(1)","x_corr(2)","x_corr(3)","x_corr(4)"]].values.tolist() #list of all x_corr(t) for each barcode:
def x_var(x_corrs, df_x_corr):
    x_c_var = []
    for x_c in x_corrs:
        x_var = np.var(x_c)
        x_c_var.append(x_var)
    df_x_corr["var(x_corr)"] = x_c_var
x_var(x_corrs, df_x_corr)

###create df with variances and corresponding geneIDs:
def list_genes(df_x_corr):
    updated_genes = []
    for gene in df_x_corr["gene_ID"].tolist():
        if gene not in updated_genes:
            updated_genes.append(gene)
    return updated_genes
#getting average variance of barcodes in each gene:
def avg_var(df_x_corr):
    avg_vars = []
    std_vars = []
    for gene in list_genes(df_x_corr):
        df_gene = df_x_corr.loc[df_x_corr["gene_ID"] == gene]
        x_vars = df_gene["var(x_corr)"].mean()
        avg_vars.append(x_vars)
        x_std = np.sqrt(x_vars)
        std_vars.append(x_std)
    return [avg_vars, std_vars]
bc_data = {"gene_ID": list_genes(df_x_corr), "avg_var(x_corr)": avg_var(df_x_corr)[0], "std(x_corr)": avg_var(df_x_corr)[1]}
df_bc = pd.DataFrame(bc_data)
df_bc = df_bc.sort_values(by = 'avg_var(x_corr)', ascending=[False])
df_bc = df_bc.reset_index() 
df_bc.to_csv('%s.csv' % title)

### plot first n genes with highest variance and last n genes with lowest variance:
def plot_x(df_gene, gene_ID):
    for i in range(len(df_gene)):
        barcode = df_gene.iloc[i,0]
        x_array = df_gene.iloc[i,8:12].tolist()
        days = [1, 2, 3, 4]
        plt.plot(days, x_array)#, label = barcode)
        plt.xticks(np.arange(1,len(days)+1,1))
        plt.xlabel("Days")
        plt.ylabel("x(t)")
        plt.ylim([-0.02,0.02])
        #plt.legend()
        plt.title(gene_ID)
    plt.show()
def plot_x_corr(df_gene, gene_ID):
    for i in range(len(df_gene)):
        barcode = df_gene.iloc[i,0]
        x_array = df_gene.iloc[i,12:16].tolist()
        days = [1, 2, 3, 4]
        plt.plot(days, x_array)#, label = barcode)
        plt.xticks(np.arange(1,len(days)+1,1))
        plt.xlabel("Days")
        plt.ylabel("x_corr(t)")
        plt.ylim([-0.02,0.02])
        #plt.legend()
        plt.title(gene_ID)
    plt.show()
def plot_gene_x(gene_ID, df_x_corr):
    #gene_ID = df_bc.iloc[i, 1] #1 is index of gene_ID
    df_gene = df_x_corr.loc[df_x_corr["gene_ID"] == gene_ID]
    plot_x(df_gene, gene_ID)
def plot_gene_x_corr(gene_ID, df_x_corr):
    #gene_ID = df_bc.iloc[i, 1] #1 is index of gene_ID
    df_gene = df_x_corr.loc[df_x_corr["gene_ID"] == gene_ID]
    plot_x_corr(df_gene, gene_ID)
def plot_first(df_bc, df_x_corr, n): #number of first nth genes to plot for df_bc
    gene_order = df_bc["gene_ID"].tolist()
    for gene in gene_order[0:n]:
        plot_gene_x(gene, df_x_corr)
        plot_gene_x_corr(gene, df_x_corr)
def plot_last(df_bc, df_x_corr, n): #number of first nth genes to plot for df_bc
    gene_order = df_bc["gene_ID"].tolist()
    for gene in gene_order[-n:]:
        plot_gene_x(gene, df_x_corr)
        plot_gene_x_corr(gene, df_x_corr)

plot_first(df_bc, df_x_corr, 5)
plot_last(df_bc, df_x_corr, 3)




