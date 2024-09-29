from itertools import *
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_data(train_path, test_path):
    df_tr = pd.read_csv(train_path).set_index("PassengerId", drop=True)
    df_ts = pd.read_csv(test_path).set_index("PassengerId", drop=True)
    df = pd.concat([df_tr, df_ts], axis=0)

    return df

def clean_cabin_deck(df):

    df["Deck"] = df["Cabin"].str[:1]
    df["Deck"] = df["Deck"].replace(np.nan,"N/A")

    df.loc[df["Deck"]=='T',"Deck"] = 'N/A'

    replaces = {'B51 B53 B55': 'B55', 'B52 B54 B56': 'B56', 'B57 B59 B63 B66': 'B66', 'B58 B60': 'B60', 
            'B82 B84': 'B84', 'B96 B98': 'B98', 'C22 C26': 'C26', 'C23 C25 C27': 'C27', 'C55 C57': 'C57',
            'C62 C64': 'C64', 'D10 D12': 'D12', 'E39 E41': 'E41', 'F E46': 'E46', 'F E57': 'E57',
            'F E69': 'E69', 'F G63': 'G63', 'F G73': 'G73', 'F': None, 'D': None, ' ': None, 'T': None, np.nan: None}
    df["Cabin"] = df["Cabin"].replace(replaces)
    df["Cabin"] = df.fillna(np.nan)["Cabin"].str[1:].astype(float)
    df["Side"] = df["Cabin"]
    df.loc[df["Side"]!=0,"Side"] = (df["Cabin"][df["Cabin"]!=0]%2-0.5)*2

    for i in set(df["Deck"].values):
        v = df[df["Deck"]==i]["Cabin"]//2
        df.loc[df["Deck"]==i, "Cabin"]= v
        df.loc[(df["Deck"]==i) & (df["Cabin"]==0),"Cabin"] = np.median(v)
    
    df.loc[df["Cabin"].isna(),"Cabin"]=-1
    df["Cabin"] = df["Cabin"].astype(int)

    df["Side"] = df["Side"].fillna(0)

    return df


def replace_line(df):
    lin_rep = lambda x: x.replace({'LINE':"370160"})
    df = lin_rep(df)
    return df

def ppr_tickets(df):
    prefixes = []
    nums, prefs = [],[]
    for i in df["Ticket"].values:   
        if not i.isdigit():
            nums.append(int(re.search('.* {1}([0-9]+)', i).groups()[0]))
            prefix = re.search('(.*)( {1})[0-9]+', i).groups()[0]
            prefs.append(prefix.replace(".","").replace(" ", "").replace("/","")) # Needed to put in one group such prefixes as "A/5", "A/5.", "A.5" etc.
        else:
            nums.append(int(i))
            prefs.append("")
    df["Ticket"] = nums
    df["Ticket_p"] = prefs
    return df

def do_replaces(df):
    drop = ["SP", "SOP", "Fa", "SCOW", "PPP", "AS", "CASOTON", "SWPP", "SCAHBasle", "SCA3", "STONOQ", "AQ4", "A2", "LP", "AQ3", ""]
    df = df.replace(drop, 'N/A')
    return df


def clean_names(df):
    df[["Surname","Name"]] = [i.split(",") for i in df["Name"].values]

    a = df.groupby("Surname")["Surname"].count()
    fam_list = a[a>1].index.values
    df.loc[~df["Surname"].isin(fam_list),"Surname"] = "Other"

    df["Namesakes"] = 1
    df.loc[df["Surname"]=="Other","Namesakes"] = 0

    not_imp_s = ["Braund","Allen","Moran","Meyer","Holverson","Turpin","Arnold-Franchi","Panula","Harris","Skoog","Kantor","Petroff","Gustafsson","Zabour",
                "Jussila","Attalah","Baxter","Hickman","Nasser","Futrelle","Navratil","Calic","Bourke","Strom","Backstrom","Ali","Jacobsohn","Larsson",
                "Carter","Lobb","Taussig","Johnson","Abelson","Hart","Graham","Pears","Barbara","O'Brien","Hakkarainen","Van Impe","Flynn","Silvey","Hagland",
                "Morley","Renouf","Stanley","Penasco y Castellana","Webber","Coleff","Yasbeck","Collyer","Thorneycroft","Jensen","Newell","Saad","Thayer","Hoyt",
                "Andrews","Lam","Harper","Nicola-Yarred","Doling","Hamalainen","Beckwith","Mellinger","Bishop","Hippach","Richards","Baclini","Goldenberg",
                "Beane","Duff Gordon","Tylor","Dick","Chambers","Moor","Snyder", "Howard", "Jefferys", "Franklin","Abelseth","Straus","Khalil","Dyker","Stengel",
                "Foley","Buckley","Zakarian","Peacock","Mahon","Clark","Pokrnic","Ware","Gibson","Taylor"]
    df = df.replace(not_imp_s,'Other')

    df[(df["Surname"]=="Other") & (df["Namesakes"]==True)].head(10).style.background_gradient(cmap="Blues")

    drop = ["Abbott","Keane","Minahan","Crosby","Hocking","Dean","Mallet",""]
    df = df.replace(drop,'Other')
    return df

def clean_titles(df):
    df["Title"] = pd.DataFrame(df["Name"].str.strip().str.split(".").tolist()).set_index(df.index).iloc[:,0]
    df["Title"] = df["Title"].fillna("Others")

    rename = {"Miss":"Ms",
            "Mrs": "Mme",
            "Others": ["Don","Rev","Dr","Lady","Sir","Mlle","Col","the Countess","Mme","Major","Capt","Jonkheer","Dona"]}
    for k in rename:
        df["Title"] = df["Title"].replace(rename[k],k)
    
    return df

def add_kid_col(df):
    df["Kid"]=0
    df.loc[(df["Age"]<18),"Kid"]=1
    return df

def add_old_col(df):
    df["Old"]=0
    df.loc[(df["Age"]>60),"Old"]=1

    return df

def add_alone_col(df):
    df["Alone"] = 0
    df.loc[(df["Parch"]==0) & (df["SibSp"]==0),"Alone"]=1

    return df

def ppr_sex(df):
    l1, l2 = [1,2,3], ["female","male"]
    for c,s in product(l1,l2):
        msk = (df["Pclass"]==c) & (df["Sex"]==s)
        df.loc[msk,"Age"] = df[msk]["Age"].fillna(df[msk]["Age"].median())

    return df

def ppr_fare(df):
    df.loc[1044,"Fare"] = df[df["Pclass"]==3]["Fare"].mean()

    df["Fare"] = df["Fare"].rank(method='max')

    return df

def ppr_embarked(df):
    df.loc[df["Embarked"].isna(),"Embarked"] = "S"

    return df

def onehot(df):
    onehot_df = pd.DataFrame(index=df.index)

    for c in ["Pclass","Sex","Embarked","Deck","Ticket_p","Surname","Title"]:
        encoded = OneHotEncoder().fit_transform(df[c].to_numpy().reshape(-1,1)).toarray()
        columns = [f"{c}_{i}" for i in range(encoded.shape[1])]
        _df =pd.DataFrame(data=encoded, columns=columns, index=df.index)
        onehot_df = pd.concat([_df,onehot_df], axis=1)
        
    onehot_df = pd.concat([onehot_df,df[["Survived","Age","SibSp","Parch","Fare","Cabin","Namesakes","Kid","Alone","Side"]]], axis=1)

    for c in ["Age","Fare","Cabin","SibSp","Parch"]:
        onehot_df[c] = MinMaxScaler().fit_transform(onehot_df[c].to_numpy().reshape(-1,1))

    return df

def split_and_save(df, train_out_path, test_out_path):
    df_train = df.copy(deep=True)
    mask = df_train["Survived"].isna()
    train, test = df_train[~mask], df_train[mask]
    df_test = test.drop("Survived", axis=1)

    df_train.to_csv(train_out_path)
    df_test.to_csv(test_out_path)

    return df_train, df_test

    
