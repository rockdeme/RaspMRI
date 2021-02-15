import scipy
import scipy.io
import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def read_matlab_file(path):
    mat_file = scipy.io.loadmat(path)
    data = pd.Series(np.array(mat_file['PROPS']).reshape(-1))
    return data


paths = glob.glob("G:/Gym-lab/*.mat")
paths.sort()
df = pd.DataFrame()
animal_ids = []
treatment_grps = []
days = []

for path in paths:
    strsplit = path.split('\\')[1]
    if 'control' in strsplit:
        treatment_grp = 'Control'
    elif 'MCAO' in strsplit:
        treatment_grp = 'MCAO'
    else:
        raise ValueError

    id = strsplit.split('_')[0] + '_' + strsplit.split('_')[1]
    day = path.split('_day')
    props = read_matlab_file(path)
    treatment_grps.append(treatment_grp)
    animal_ids.append(id)
    days.append(day[-1][:-4])
    df = df.append(props, ignore_index=True)

df['Animal ID'] = animal_ids
df['Treatment Group'] = treatment_grps
df['Day'] = days
df = df.fillna(0)

df.to_csv('G:/parameters.csv', sep = ';')

df = pd.read_csv('G:/parameters.csv', sep = ';')
pca = PCA()
scaler = StandardScaler()

scaled_df = pd.DataFrame()
for animal in df['Treatment ID'].unique():
    animal_df = df[df['Treatment ID'] == animal]
    animal_df = animal_df.sort_values(by = ['Day'])
    animal_df = animal_df.iloc[-8:, :]
    scaled_index = [id + '_' + date for id, date in zip(animal_df['Animal ID'], animal_df['Day'])]
    treatment_grp = animal_df['Treatment Group'].values
    id = animal_df['Animal ID'].values
    treatment_id = animal_df['Treatment ID'].values
    X = animal_df.iloc[:, 4:]
    X_scaled = scaler.fit_transform(X)
    X_transformed = pca.fit_transform(X_scaled)
    X_df = pd.DataFrame(X_transformed, index=scaled_index)
    X_df['Treatment Group'] = treatment_grp
    X_df['Animal ID'] = id
    X_df['Treatment ID'] = treatment_id
    scaled_df = pd.concat([scaled_df, X_df], axis=0)


scaled_df.to_csv('G:/self_normed_last.csv', sep = ';')


rat4 = pd.read_csv('G:/rat4.csv', sep = ';', header = None)

rat4_scaled = scaler.fit_transform(rat4.iloc[:,4:])
X_transformed = pca.fit_transform(rat4_scaled)
loadings = pd.DataFrame(pca.components_.T)


X_transformed = pd.DataFrame(X_transformed)
X_transformed['Treatment Group'] = scaled_df['Treatment Group'].values
X_transformed['Animal ID'] = scaled_df['Animal ID'].values

X_transformed.to_csv('G:/rat_4_pcs_scaled.csv', sep = ';')
pd.DataFrame(pca.explained_variance_ratio_).to_csv('G:/evar.csv', sep = ';')
pd.DataFrame(X_transformed).to_csv('G:/xasd.csv', sep = ';')

def get_principal_component(pc):

    def calc_mean_std(df):
        df_copy = df.copy()
        df_copy['Mean'] = df_copy.mean(axis=1)
        df_copy['Std'] = df_copy.std(axis=1)
        return df_copy

    control = pd.DataFrame()
    mcao = pd.DataFrame()
    for animal in X_transformed['Animal ID'].unique():
        animal_df = X_transformed[X_transformed['Animal ID'] == animal].reset_index(drop=True)
        if 'Control' in animal_df['Treatment Group'].values:
            control = pd.concat([control, animal_df.iloc[:, pc]], axis=1)
        elif 'MCAO' in animal_df['Treatment Group'].values:
            mcao = pd.concat([mcao, animal_df.iloc[:, pc]], axis=1)
        else:
            raise TypeError('ASD')

    control = calc_mean_std(control)
    mcao = calc_mean_std(mcao)

    plt.errorbar(control.index,control['Mean'], yerr=control['Std'])
    plt.errorbar(mcao.index,mcao['Mean'], yerr=mcao['Std'])
    plt.show()

get_principal_component(2)


X_scaled = scaler.fit_transform(X)
X_transformed = pca.fit_transform(X_scaled)
X_transformed_df = pd.DataFrame(X_transformed)
X_transformed_df['Treatment Group'] = df['Treatment Group'].values
control = X_transformed_df[X_transformed_df['Treatment Group'] == 'Control']
mcao = X_transformed_df[X_transformed_df['Treatment Group'] == 'MCAO']
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(control.iloc[:,0], control.iloc[:,1], control.iloc[:,2])
ax.scatter(mcao.iloc[:,0], mcao.iloc[:,1], mcao.iloc[:,2])
plt.show()