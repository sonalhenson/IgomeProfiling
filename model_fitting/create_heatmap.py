import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import sys
import os


def getData(path_output,path_model_fitting,names_motif):    
    file_motif=pd.read_csv(names_motif)
    #transpose the data frame to a dictionary
    dic_motif=file_motif.set_index('name_bio').T.to_dict('list')
    
    #get the sample names form one of the csv file
    name_bio_one= list(dic_motif.keys())[0] 
    sample_name=pd.read_csv(path_model_fitting+'/'+name_bio_one+'/'+name_bio_one+'_hits.csv')['sample_name']

    #get the hits/value from every table of other biological condition
    data=pd.DataFrame({'sample_name':sample_name})
    for key in dic_motif:
        table=pd.read_csv(path_model_fitting+'/'+key+'/'+key+'_hits_model/perfect_feature_names.csv')
        values=dic_motif[key]
        clean_values=[x for x in values if str(x) != 'nan']
        table_new= table[clean_values]
        data=pd.merge(data,table_new)
    motif_name=data.columns
    create_heatmap(data,sample_name,motif_name,path_output)
    
def create_heatmap(df,sample_name,motif_name,path_output):
    #create map of all motif form every biological condiation
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
    parser.add_argument('output_heatmap', type=str, help='A output folder to put the heatmap result')
    parser.add_argument('model_fitting_results', type=str, help='folder of the last run to get the data')
    parser.add_argument('file_of_select_motif', type=str, help='csv file with the all the selected motif to connect together')

    args = parser.parse_args()

    getData(args.output_heatmap,args.model_fitting_results,args.file_of_select_motif)
