from pytablewriter import MarkdownTableWriter
from pytablewriter import LatexTableWriter
from pytablewriter import ExcelXlsxTableWriter
from pytablewriter import CsvTableWriter
from pytablewriter.style import Style
import numpy as np
import sys
import os
import glob
import pickle

def args_to_excel(model,header,value_matrix):
    writer = ExcelXlsxTableWriter()
    writer.table_name = model
    writer.headers = header
    writer.value_matrix = value_matrix
    writer.dump("results.xlsx")

def args_to_csv(model,header,value_matrix):
    writer = CsvTableWriter()
    writer.table_name = model
    writer.headers = header
    writer.value_matrix = value_matrix
    writer.dump("results_val_test_ppl_in.csv")

def args_to_latex(model,header,value_matrix):
    writer = LatexTableWriter()
    writer.table_name = model
    writer.headers = header
    writer.value_matrix = value_matrix
    writer.write_table()


# data_loc="/backup_chain_saved_configs/*.pkl"
# data_loc="/backup_chain_saved_configs/*.pkl"
data_loc='/scratch/mehdi/chain_saved_configs/*.pkl'
all_loc=data_loc
# dir_path = os.path.dirname(os.path.realpath(__file__))
# all_loc=dir_path+data_loc
# print("all_loc: ",all_loc)
all_files=glob.glob(all_loc)
for idx,file in enumerate(all_files):
    print(idx,": ",file)
for idx,file in enumerate(all_files):
    first_file=True
    with open(file, 'rb') as f:
        args_dict, args_info = pickle.load(f)
        print('args_dict: ',args_dict.get('obsv_prob','-'),' val: ',args_dict.get("min_ppl",'-'),' test:',args_dict.get("WikiTestPPL",'-'))

chain_files =[file for file in all_files if file.split('/')[-1][0]!='h' and file.split('/')[-1][0]!='L' and file.split('/')[-1][0]!='R']
haqae_file = [file for file in all_files if file.split('/')[-1][0]=='h']
lstmlm_file = [file for file in all_files if file.split('/')[-1][0]=='L' or file.split('/')[-1][0]=='R']
lstmlm_file =[lstmlm_file[1],lstmlm_file[0]]


probs = np.array([float(item.split("prob_")[1].split('_')[0]) for item in chain_files])
# print('haqae_file: ',haqae_file)
# print("probs: ",probs)
probs_values = np.sort(probs)
probs_index = np.argsort(probs)[:-1]
# probs_index = np.argsort(probs)
# print("probs: ",probs)
# print("probs_values: ",probs_values)
# print("probs_index: ",probs_index)

files = lstmlm_file
files += haqae_file
chain_files = [chain_files[k] for k in probs_index]
files += chain_files
all_values=[]
writer = MarkdownTableWriter()
first=True
# headers=["model","obsv_prob","min_ppl","invNarClz","emb_size","nlayers",
#          "batch_size","num_clauses","num_latent_values",
#          "latent_dim","dropout","bidir","use_pretrained","template","frame_max"]
headers=["model","obsv_prob","WikiValPPL","WikiTestPPL","WikiValInvNarClz","WikiTestInvNarClz","NytVal2k","NytTest2k"]
ppls=[]
for idx,file in enumerate(files):
    first_file=True
    with open(file, 'rb') as f:
        args_dict, args_info = pickle.load(f)
        # print(args_dict.get('obsv_prob','-'))
        keys=[key for key in args_dict.keys() if 'NYT' in key]
        # print(keys)
        try:
            ppls.append(args_dict["WikiTestPPL"])
        except:
            continue
min_idx=np.argmin(ppls)

for idx,file in enumerate(files):
    first_file=True
    with open(file, 'rb') as f:
        args_dict, args_info = pickle.load(f)
        if "invNarClz" in args_dict.keys(): args_dict["invNarClz"]= round(args_dict["invNarClz"],2)
        try:
            args_dict["WikiTestPPL"]= "{:4.2f}".format(float(args_dict["WikiTestPPL"]))
            args_dict["WikiValPPL"]= "{:4.2f}".format(float(args_dict["WikiValPPL"]))
        except:
            continue
        # print(args_dict)
        # args_dict["WikiTestPPL"]= "{:4.2f}".format(float(args_dict["WikiTestPPL"])) 
        if idx==min_idx:
            args_dict["WikiTestPPL"]= "{:4.2f}".format(float(args_dict["WikiTestPPL"]))
            # args_dict["min_ppl"]= "{:4.2f}".format(float(args_dict["WikiTestPPL"]))
            values=["**"+str(args_dict.get(key,'-'))+"**" for key in headers]
        else:
            values=[str(args_dict.get(key,'-')) for key in headers]

        # if float(args_dict["tau"])==0.1 and int(args_dict["pretrained"])==1 and int(args_dict["nlayers"])==1:
        if True:
            all_values.append(values)
        writer.table_name ="SemiSupChain"
        writer.headers=headers
        writer.value_matrix=all_values
        writer.column_styles = [Style(align="center") for _ in range(len(writer.headers))]
writer.write_table()
# args_to_excel("chain",headers,all_values)
# args_to_latex("chain",headers,all_values)
args_to_csv("chain",headers,all_values)





def args_to_md(model,args_dict):
    writer = MarkdownTableWriter()
    writer.table_name = model
    writer.headers=list(args_dict.keys())
    # print('headers: ',writer.headers)
    writer.value_matrix=[list(args_dict.values())]
    # print('value_matrix: ',writer.value_matrix)
    writer.column_styles = [Style(align="center") for _ in range(len(writer.headers))]
    print(writer.write_table())

