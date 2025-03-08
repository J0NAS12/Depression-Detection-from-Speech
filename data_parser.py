import pandas as pd

def parse_data(data_folder, excel_path):
    dfs = pd.read_excel(excel_path, sheet_name='Munka1')
    excel_filtered = dfs.loc[:, ["File name", "Type", "Sex", "Age", "BDI"]]

    ecapa1 = pd.read_csv(f"{data_folder}/dep_merged_output_ecapa.txt", sep="\t", header=None, decimal=",")
    ecapa2 = pd.read_csv(f"{data_folder}/hc_merged_output_ecapa.txt", sep="\t", header=None, decimal=",")
    ecapa = pd.concat([ecapa1, ecapa2])
    ecapa['File name'] = ecapa[0].str.split('.', expand=True)[0]
    ecapa.drop(columns=[0], inplace=True)
    df_ecapa = pd.merge(excel_filtered, ecapa, on='File name', how='inner')

    resnet1 = pd.read_csv(f"{data_folder}/dep_merged_output_resnet.txt", sep="\t", header=None, decimal=",")
    resnet2 = pd.read_csv(f"{data_folder}/hc_merged_output_resnet.txt", sep="\t", header=None, decimal=",")
    resnet = pd.concat([resnet1, resnet2])
    resnet['File name'] = resnet[0].str.split('.', expand=True)[0]
    resnet.drop(columns=[0], inplace=True)
    df_resnet = pd.merge(excel_filtered, resnet, on='File name', how='inner')

    xvector = pd.DataFrame()
    xvectorlist = []
    for name in excel_filtered["File name"]:
        a = pd.DataFrame()
        if "DE" in name:
            a = pd.read_csv(f"{data_folder}/raw_dep/{name}.wav_embeddings.txt", sep="\t", header=None, decimal=",")
        else:
            a = pd.read_csv(f"{data_folder}/raw_hc/{name}.wav_embeddings.txt", sep="\t", header=None, decimal=",")
        a['File name'] = name
        xvectorlist.append(a)
    xvector = pd.concat(xvectorlist)
    xvector.reset_index(drop=True)
    df_xvector = pd.merge(excel_filtered, xvector, on='File name', how='inner')

    return df_resnet, df_ecapa, df_xvector