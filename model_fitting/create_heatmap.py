import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import sys
import os



name_test=['COBRA','naive','sichuan','swiss','texas']
dic={'COBRA_col':['sample_name','CGPHSSLFSC', 'CGHSSLFSCLAC','CLHSSLFSSPCP','CLSYSSLFSC','CPHTSTIFSGPC','CPHHSSLFSC'],
'naive_col':['sample_name','CLPQLPALSQCC', 'CGGPEWREKC','CGSNESREKC','CETDVTGLLINC'],
'sichuan_col':['sample_name','SLLRLEPSS','CQIRAKDQGLVC','AASMIDKPT','LLLSLSAPCC','PYHMLHIWGVEP','SWYVETPIAK'],
'swiss_col':['sample_name','CVTAVDPIHLTLMC','PPSLYARFD','TTPLGLFERF','PPSLYDRFSP'],
'texas_col':['sample_name','CKLSSLFSSC','CKLSSLFSRC','CRDVSLFSSC','RKESSLFSVC','CKTINLFSRC']}

def getData(path_output,path_model_fitting):
    #get the sample names form one of the csv file
    sample_name=pd.read_csv(path_model_fitting+'/ferret_COBRA/ferret_COBRA_hits.csv')['sample_name']
    
    #get the data and create list of motifs for axis x
    data=pd.DataFrame({'sample_name':sample_name})
    for name in name_test:
        table=pd.read_csv(path_model_fitting+'/ferret_'+name+'/ferret_'+name+'_hits_model/perfect_feature_names.csv')
        table_new= table[dic[f'{name}_col']]
        data=pd.merge(data,table_new)
    motif_name=data.columns
    create_heatmap(data,sample_name,motif_name,path_output)
    
def create_heatmap(df,sample_name,motif_name,path_output):
    df= df.drop(columns=['sample_name'])
    #normalization the data, the data is mix of many runs. 
    train_data=((df-df.min())/(df.max()-df.min()))
    train_data.insert(loc=0, column='sample_name', value=sample_name)
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)
    
    train_data.to_csv(path_output+'/data.csv',index=False)
    df_new=pd.read_csv(path_output+'/data.csv',index_col=0)
    cm = sb.clustermap(df_new, cmap="Blues",col_cluster=False)
    cm.ax_heatmap.set_title(f"A heat-map of the significance of the top motifs")
    cm.savefig(f"{path_output}.svg", format='svg', bbox_inches="tight")
    plt.close()
    
    
     
    
    

if __name__ == '__main__':
    print(f'Starting {sys.argv[0]}. Executed command is:\n{" ".join(sys.argv)}')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_results', type=str, help='A output folder to put the run results')
    parser.add_argument('model_fitting_results', type=str, help='output folder')
    
    args = parser.parse_args()

    getData(args.output_results,args.model_fitting_results)
