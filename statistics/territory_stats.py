import numpy as np
import pandas as pd
from scipy import stats


def select_treatment_group(groups_df, data, name: str):
    selected = groups_df[groups_df['Group'] == name]

    def select_columns(selected, data):
        rats_in_group = [data.split('_')[:4] for data in selected['Rat']]
        columns = [data.split('_')[:4] for data in [*data.columns][7:]]
        in_group = [True for i in range(7)]

        for rat in columns:
            if rat in rats_in_group:
                in_group.append(True)
            else:
                in_group.append(False)
        new_cols = [rat for (rat, in_group) in zip(data.columns, in_group) if in_group]
        return new_cols

    return select_columns(selected, data)


def calculate_left_right_metrics(df):
    df_copy = df.copy()
    means_df = df_copy.drop(columns=list(df_copy.filter(regex='Area')))

    left_df = means_df.drop(columns=list(means_df.filter(regex='Right')))
    l_means = left_df.iloc[:, 7:].mean(axis=1)
    l_stdevs = left_df.iloc[:, 7:].std(axis=1)
    df_copy['Left Mean'] = l_means
    df_copy['Left Std'] = l_stdevs

    right_df = means_df.drop(columns=list(means_df.filter(regex='Left')))
    r_means = right_df.iloc[:, 7:].mean(axis=1)
    r_stdevs = right_df.iloc[:, 7:].std(axis=1)
    df_copy['Right Mean'] = r_means
    df_copy['Right Std'] = r_stdevs
    df_copy = df_copy.reset_index(drop=True)
    return df_copy


def filter_area(df):
    df_copy = df.copy()
    intensity_df = df_copy.drop(columns=list(df_copy.filter(regex='Area')))
    return intensity_df


def filter_data(df):
    df_copy = df.copy()
    intensity_df = df_copy.drop(columns=list(df_copy.filter(regex='Area')))
    left_df = intensity_df.drop(columns=list(intensity_df.filter(regex='Right')))
    right_df = intensity_df.drop(columns=list(intensity_df.filter(regex='Left')))
    left_df = left_df.reset_index(drop=True)
    right_df = right_df.reset_index(drop=True)
    return left_df, right_df


groups = pd.read_csv("G:/mri-results/exp_groups.csv", sep=';')

data03 = pd.read_csv("G:/mri-results/t2_data_day03_territories_normalized.csv", index_col=0, sep=';')
data03 = data03.set_index(np.arange(0, data03.shape[0]))

ctrl03 = data03[select_treatment_group(groups, data03, 'Ctrl')]
ctrl03_l, ctrl03_r = filter_data(ctrl03)
cyclo03 = data03[select_treatment_group(groups, data03, 'Cyclo')]
cyclo03_l, cyclo03_r = filter_data(cyclo03)
mph03 = data03[select_treatment_group(groups, data03, 'MPH')]
mph03_l, mph03_r = filter_data(mph03)

ctr_comp03 = calculate_left_right_metrics(ctrl03)
cyclo_comp03 = calculate_left_right_metrics(cyclo03)
mph_comp03 = calculate_left_right_metrics(mph03)


data07 = pd.read_csv("G:/mri-results/t2_data_day07_territories_normalized.csv", index_col=0, sep=';')
data07 = data07.set_index(np.arange(0, data07.shape[0]))

ctrl07 = data07[select_treatment_group(groups, data07, 'Ctrl')]
ctrl07_l, ctrl07_r = filter_data(ctrl07)
cyclo07 = data07[select_treatment_group(groups, data07, 'Cyclo')]
cyclo07_l, cyclo07_r = filter_data(cyclo07)
mph07 = data07[select_treatment_group(groups, data07, 'MPH')]
mph07_l, mph07_r = filter_data(mph07)

ctr_comp07 = calculate_left_right_metrics(ctrl07)
cyclo_comp07 = calculate_left_right_metrics(cyclo07)
mph_comp07 = calculate_left_right_metrics(mph07)


data21 = pd.read_csv("G:/mri-results/t2_data_day21_territories_normalized.csv", index_col=0, sep=';')
data21 = data21.set_index(np.arange(0, data21.shape[0]))

ctrl21 = data21[select_treatment_group(groups, data21, 'Ctrl')]
ctrl21_l, ctrl21_r = filter_data(ctrl21)
cyclo21 = data21[select_treatment_group(groups, data21, 'Cyclo')]
cyclo21_l, cyclo21_r = filter_data(cyclo21)
mph21 = data21[select_treatment_group(groups, data21, 'MPH')]
mph21_l, mph21_r = filter_data(mph21)

ctr_comp21 = calculate_left_right_metrics(ctrl21)
cyclo_comp21 = calculate_left_right_metrics(cyclo21)
mph_comp21 = calculate_left_right_metrics(mph21)

stats_df = pd.DataFrame()


for i in range(data03.shape[0]):
    stats_df = stats_df.append({
        'day 03 Ctrl v. MPH': stats.ttest_ind(filter_area(ctrl03).iloc[i, 7:], filter_area(mph03).iloc[i, 7:]).pvalue,
        'day 03 Ctrl v. Cyclo': stats.ttest_ind(filter_area(ctrl03).iloc[i, 7:], filter_area(cyclo03).iloc[i, 7:]).pvalue,
        'day 03 Cyclo v. MPH': stats.ttest_ind(filter_area(cyclo03).iloc[i, 7:], filter_area(mph03).iloc[i, 7:]).pvalue,
        'day 03 Ctrl I v. C': stats.ttest_ind(ctrl03_l.iloc[i, 7:], ctrl03_r.iloc[i, 7:]).pvalue,
        'day 03 Cyclo I v. C': stats.ttest_ind(cyclo03_l.iloc[i, 7:], cyclo03_r.iloc[i, 7:]).pvalue,
        'day 03 MPH I v. C': stats.ttest_ind(mph03_l.iloc[i, 7:], mph03_r.iloc[i, 7:]).pvalue,
        'day 07 Ctrl v. MPH': stats.ttest_ind(filter_area(ctrl07).iloc[i, 7:], filter_area(mph07).iloc[i, 7:]).pvalue,
        'day 07 Ctrl v. Cyclo': stats.ttest_ind(filter_area(ctrl07).iloc[i, 7:], filter_area(cyclo07).iloc[i, 7:]).pvalue,
        'day 07 Cyclo v. MPH': stats.ttest_ind(filter_area(cyclo07).iloc[i, 7:], filter_area(mph07).iloc[i, 7:]).pvalue,
        'day 07 Ctrl I v. C': stats.ttest_ind(ctrl07_l.iloc[i, 7:], ctrl07_r.iloc[i, 7:]).pvalue,
        'day 07 Cyclo I v. C': stats.ttest_ind(cyclo07_l.iloc[i, 7:], cyclo07_r.iloc[i, 7:]).pvalue,
        'day 07 MPH I v. C': stats.ttest_ind(mph07_l.iloc[i, 7:], mph07_r.iloc[i, 7:]).pvalue,
        'day 21 Ctrl v. MPH': stats.ttest_ind(filter_area(ctrl21).iloc[i, 7:], filter_area(mph21).iloc[i, 7:]).pvalue,
        'day 21 Ctrl v. Cyclo': stats.ttest_ind(filter_area(ctrl21).iloc[i, 7:], filter_area(cyclo21).iloc[i, 7:]).pvalue,
        'day 21 Cyclo v. MPH': stats.ttest_ind(filter_area(cyclo21).iloc[i, 7:], filter_area(mph21).iloc[i, 7:]).pvalue,
        'day 21 Ctrl I v. C': stats.ttest_ind(ctrl21_l.iloc[i, 7:], ctrl21_r.iloc[i, 7:]).pvalue,
        'day 21 Cyclo I v. C': stats.ttest_ind(cyclo21_l.iloc[i, 7:], cyclo21_r.iloc[i, 7:]).pvalue,
        'day 21 MPH I v. C': stats.ttest_ind(mph21_l.iloc[i, 7:], mph21_r.iloc[i, 7:]).pvalue,
    }, ignore_index=True)


stats_df = stats_df[['day 03 Ctrl v. MPH', 'day 03 Ctrl v. Cyclo', 'day 03 Cyclo v. MPH',
                     'day 03 Ctrl I v. C','day 03 Cyclo I v. C', 'day 03 MPH I v. C',
                     'day 07 Ctrl v. MPH', 'day 07 Ctrl v. Cyclo', 'day 07 Cyclo v. MPH',
                     'day 07 Ctrl I v. C', 'day 07 Cyclo I v. C', 'day 07 MPH I v. C',
                     'day 21 Ctrl v. MPH', 'day 21 Ctrl v. Cyclo', 'day 21 Cyclo v. MPH',
                     'day 21 Ctrl I v. C', 'day 21 Cyclo I v. C', 'day 21 MPH I v. C']]

regions_df = ctrl03.iloc[:, :3]

final_df = pd.concat([regions_df, stats_df], axis=1)
final_df.to_csv("G:/mri-results/territory_statistics.csv", sep=';')
