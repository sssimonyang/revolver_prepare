import os
import re

import numpy as np
import pandas as pd


def batch_split(func):
    import time

    def wrapper(*args, **kwargs):
        print(
            '-------------------------------------------------------------------'
        )
        print(f'start at {time.strftime("%Y %m %d %H:%M:%S")}')
        func(*args, **kwargs)
        print(f'end at {time.strftime("%Y %m %d %H:%M:%S")}')
        print(
            '-------------------------------------------------------------------'
        )
        for i in range(10):
            print()

    return wrapper


def get_info_from_gtf(
        get_type,
        gtf_file="/public/home/wangzj/yangjk/Homo_sapiens_GRCh37_87_geneonly.gtf"
):
    gtf = pd.read_csv(gtf_file, header=None, sep='\t')
    gtf['gene_id'] = gtf[8].apply(
        lambda x: re.search(r'gene_id \"(.*?)\"', x).group(1))
    gtf['gene_name'] = gtf[8].apply(
        lambda x: re.search(r'gene_name \"(.*?)\"', x).group(1))

    def fetch_gene_id(chrom, start):
        start = int(start)
        value = gtf[(gtf[0] == chrom) & (gtf[3] <= start) &
                    (gtf[4] >= start)]['gene_id'].tolist()
        if value:
            return value[0]
        else:
            return None

    def fetch_length(gene_id):
        value = (gtf[gtf['gene_id'] == gene_id][4] -
                 gtf[gtf['gene_id'] == gene_id][3]).tolist()
        if value:
            return value[0]
        else:
            return None

    def fetch_gene_name(chrom, start):
        start = int(start)
        value = gtf[(gtf[0] == chrom) & (gtf[3] <= start) &
                    (gtf[4] >= start)]['gene_name'].tolist()
        if value:
            return value[0]
        else:
            return None

    def fetch_gene_name_from_gene_id(gene_id):
        value = gtf[gtf['gene_id'] == gene_id]['gene_name'].tolist()
        if value:
            return value[0]
        else:
            return None

    def fetch_gene_id_from_gene_name(gene_name):
        value = gtf[gtf['gene_name'] == gene_name]['gene_id'].tolist()
        if value:
            return value[0]
        else:
            return None

    return locals().get(get_type)


fetch_gene = get_info_from_gtf('fetch_gene_name')
transform_name_to_id = get_info_from_gtf('fetch_gene_id_from_gene_name')


def get_meaningful_mutations(path_to_meaningful_mutation):
    df = pd.read_csv(path_to_meaningful_mutation, sep='\t')
    return df['gene_id_ref_base_var_base'].tolist()


def value_to_str(values):
    return ','.join([str(round(i, 2)) for i in values])


def process_single_patient(patient_id='GWZY026',
                           meaningful_mutations=None,
                           min_cluster_size=10,
                           min_mutation_ccf=0.1,
                           driver_genes=None):
    cluster = pd.read_csv(f"./{patient_id}/tables/cluster.tsv", sep='\t')
    loci = pd.read_csv(f"./{patient_id}/tables/loci.tsv", sep='\t')

    loci.drop(['variant_allele_frequency', 'cellular_prevalence_std'],
              axis=1,
              inplace=True)

    samples = loci["sample_id"].unique()
    samples_map = {f'R{i+1:02}': sample for i, sample in enumerate(samples)}

    def concat_ccf_value(x):
        # lambda x: ';'.join([i + ':' + str(
        #     0 if x[x.sample_id == samples_map[i]].empty else
        #     x[x.sample_id == samples_map[i]].cellular_prevalence.tolist()[
        #         0])
        #                     for i in samples_map]
        out = []
        for i in samples_map:
            value = x[x["sample_id"] ==
                      samples_map[i]]["cellular_prevalence"].tolist()
            if value:
                out.append(i + ':' + str(round(value[0], 2)))
            else:
                out.append(i + ':' + str(0))

        return ';'.join(out)

    cluster_id = loci.groupby('mutation_id')['cluster_id'].first()
    ccf_value = loci.groupby('mutation_id').apply(concat_ccf_value)
    result = pd.concat([cluster_id, ccf_value], axis=1, join='inner')
    result.columns = ['cluster', 'CCF']
    result.index.name = 'Misc'
    result.reset_index(inplace=True)
    result['patientID'] = patient_id
    if meaningful_mutations:
        result = result[result['Misc'].isin(meaningful_mutations)]

    # 筛选至少有一个sample CCF大于MUTATION_MIN_CCF的突变
    remove_mutation = loci.groupby('mutation_id').apply(lambda x: x[x[
        'cellular_prevalence'] >= min_mutation_ccf].shape[0] == 0)
    remove_mutation = remove_mutation[remove_mutation].index
    result = result[~result['Misc'].isin(remove_mutation)]

    # 筛选cluster内部突变数大于等于min_cluster_size
    cluster_count = result.groupby('cluster')["Misc"].count()
    filter_cluster = cluster_count[cluster_count >= min_cluster_size].index
    result = result[result.cluster.isin(filter_cluster)]

    # 找clonal_cluster剩的cluster里mean_CCF最高的
    cluster_mean = cluster.groupby('cluster_id')['mean'].mean()
    cluster_mean = cluster_mean.sort_values(ascending=False)

    cluster_mean_max_before = value_to_str(cluster_mean.index.to_list()[:10])
    cluster_mean_value_before = value_to_str(cluster_mean.to_list()[:10])
    cluster_count_before = value_to_str(
        cluster_count[cluster_mean.index.to_list()[:10]].to_list())

    clonal_cluster = cluster_mean[result['cluster'].unique()].idxmax()

    cluster_mean = cluster_mean[result['cluster'].unique()].sort_values(
        ascending=False)
    cluster_mean_max_after = value_to_str(cluster_mean.index.to_list()[:10])
    cluster_mean_value_after = value_to_str(cluster_mean.to_list()[:10])
    cluster_count_after = value_to_str(
        cluster_count[cluster_mean.index.to_list()[:10]].to_list())

    cluster_mean_max = [
        cluster_mean_max_before, cluster_mean_value_before,
        cluster_count_before, cluster_mean_max_after, cluster_mean_value_after,
        cluster_count_after
    ]

    result['is.clonal'] = False
    result.loc[result.cluster == clonal_cluster, 'is.clonal'] = True

    result['gene'] = result['Misc'].apply(
        lambda x: fetch_gene(*x.split('_')[:2]))

    result['mutation_type'] = result['Misc'].apply(
        lambda x: '_'.join(x.split('_')[3:]))
    result['mutation_type'] = result['mutation_type'].replace(
        ['G_T', 'G_C', 'G_A', 'A_T', 'A_G', 'A_C'],
        ['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G'])

    result['variantID'] = None
    if driver_genes:
        # result.loc[result['gene'].isin(driver_genes), 'variantID'] = result.loc[result['gene'].isin(
        #     driver_genes), 'gene'] + '_' + result.loc[result['gene'].isin(driver_genes), 'mutation_type']
        result.loc[result['gene'].isin(driver_genes),
                   'variantID'] = result.loc[result['gene'].isin(driver_genes),
                                             'gene']
        result.loc[~result['gene'].isin(driver_genes),
                   'variantID'] = result.loc[
                       ~result['gene'].isin(driver_genes), 'gene']

        overlap_variant_ids_counts = result.loc[
            result['gene'].isin(driver_genes), 'variantID'].value_counts()
        overlap_variant_ids = overlap_variant_ids_counts[
            overlap_variant_ids_counts > 1].index

    else:
        result['variantID'] = result['gene']
        overlap_variant_ids_counts = result['gene'].value_counts()
        overlap_variant_ids = overlap_variant_ids_counts[
            overlap_variant_ids_counts > 1].index

    if not overlap_variant_ids.empty:
        for variant_id in overlap_variant_ids:
            variant = result.loc[result['variantID'] == variant_id,
                                 'CCF'].apply(
                                     lambda x: np.mean(parse_concat_ccf(x)))
            remain_variants = variant.idxmax()
            modify_variants = variant.index.tolist()
            modify_variants.remove(remain_variants)
            result.loc[modify_variants,
                       'variantID'] = result.loc[modify_variants,
                                                 'variantID'].str.cat([
                                                     f'{patient_id}-{i}'
                                                     for i in modify_variants
                                                 ],
                                                                      sep='-')
    return result, clonal_cluster, samples_map, cluster_mean_max


def parse_concat_ccf(ccf):
    return [float(i[4:]) for i in ccf.split(';')]


def analyse_cluster(patient_id, result, min_cluster_ccf):
    cluster_infos = {}
    removed_cluster = []
    for i in np.unique(result.cluster):
        cluster_ccf_value = result.loc[result.cluster == i,
                                       'CCF'].apply(parse_concat_ccf)
        cluster_ccf_value = np.array([i for i in cluster_ccf_value])
        median_cluster_ccf = np.median(cluster_ccf_value, axis=0)

        if np.max(median_cluster_ccf) <= min_cluster_ccf:
            print(
                f'{patient_id} cluster {i} removed with max value {np.max(median_cluster_ccf)}'
            )
            result.drop(result[result.cluster == i].index,
                        axis=0,
                        inplace=True)
            removed_cluster.append(i)

        mutation_sig = result.loc[result.cluster == i,
                                  'mutation_type'].value_counts().to_frame()
        mutation_sig.index.name = 'mutation_type'
        mutation_sig.columns = ['counts']
        mutation_sig = mutation_sig.reindex(
            index=['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G'], fill_value=0)
        mutation_sig['prop'] = mutation_sig.counts / np.sum(
            mutation_sig.counts)
        mutation_sig['prop'] = mutation_sig['prop'].round(decimals=2)
        mutation_sig = [
            i for j in zip(mutation_sig['counts'].to_list(),
                           mutation_sig['prop'].to_list()) for i in j
        ]

        cluster_infos[i] = [*mutation_sig, *median_cluster_ccf]
    removed_cluster = value_to_str(removed_cluster)
    return result, cluster_infos, removed_cluster


@batch_split
def process_all_patients(paths,
                         min_cluster_size=10,
                         min_mutation_ccf=0.1,
                         min_cluster_ccf=0.1,
                         driver_genes=None,
                         problem_remove=True):
    curdir = os.path.abspath(os.curdir)
    print(
        f'This run parameter min_cluster_size = {min_cluster_size} min_mutation_ccf = {min_mutation_ccf} min_cluster_ccf = {min_cluster_ccf}'
    )

    dfs = []
    max_cluster_infos = []
    all_patients = []
    samples_infos = []
    cluster_infos = []
    # meaningful_mutations = get_meaningful_mutations(
    #     '/public/home/wupin/project/1-liver-cancer/3-pyclone/mut-in-exonic-non-syn/1-MET-cohort/all_mutation_and_gene_from_exonic.txt')
    meaningful_mutations = None
    if problem_remove:
        problem_patients = [
            'GWZY072', 'GWZY048_PT_S2010-27295', 'GWZY100_PT_S2013-08183',
            'GWZY121_PT_S2013-35637'
        ]
    else:
        problem_patients = []

    for path in paths:
        os.chdir(path)
        patients = os.listdir()

        for patient in patients:
            if 'tables' not in os.listdir(os.path.join(path, patient)):
                continue
            if len(os.listdir(os.path.join(path, patient, 'tables'))) != 2:
                continue
            if patient in problem_patients:
                continue
            all_patients.append(patient)
            df, clonal_cluster, samples_map, cluster_mean_max = process_single_patient(
                patient_id=patient,
                meaningful_mutations=meaningful_mutations,
                min_cluster_size=min_cluster_size,
                min_mutation_ccf=min_mutation_ccf,
                driver_genes=driver_genes)
            df, cluster_info, removed_cluster = analyse_cluster(
                patient_id=patient, result=df, min_cluster_ccf=min_cluster_ccf)

            dfs.append(df)

            max_cluster_info = [[
                patient, clonal_cluster, *cluster_mean_max, removed_cluster,
                *cluster_info[clonal_cluster][12:]
            ]]
            max_cluster_infos.append(
                pd.DataFrame(max_cluster_info,
                             columns=[
                                 'patientID', 'clonal_cluster',
                                 'max_cluster_before', 'mean_value_before',
                                 'count_before', 'max_cluster_after',
                                 'mean_value_after', 'count_after',
                                 'removed_cluster', *samples_map.keys()
                             ]))

            samples_info = [[
                patient, region, "_".join(sample.split("_")[1:4:2])
            ] for region, sample in samples_map.items()]
            samples_infos.append(
                pd.DataFrame(samples_info,
                             columns=['patientID', 'region', 'sample']))

            cluster_info = [[patient, cluster, *info]
                            for cluster, info in cluster_info.items()]

            sig_columns = [(i + '_count', i + '_prop')
                           for i in ['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G']]
            sig_columns = [i for j in sig_columns for i in j]
            cluster_infos.append(
                pd.DataFrame(cluster_info,
                             columns=[
                                 'patientID', 'cluster', *sig_columns,
                                 *samples_map.keys()
                             ]))

    os.chdir(curdir)

    dfs = pd.concat(dfs, join='outer')
    dfs['is.driver'] = False
    dfs['variantID'] = dfs['variantID'].fillna('non_coding_region')

    max_cluster_infos = pd.concat(max_cluster_infos, join='outer', sort=False)
    max_cluster_infos['min_clonal_ccf'] = max_cluster_infos.iloc[:, 5:].min(
        axis=1)
    max_cluster_infos['max_clonal_ccf'] = max_cluster_infos.iloc[:, 5:].max(
        axis=1)

    samples_infos = pd.concat(samples_infos, join='outer', sort=False)
    samples_infos.set_index(['patientID', 'region'], inplace=True)
    samples_infos = samples_infos.unstack()

    cluster_infos = pd.concat(cluster_infos, join='outer', sort=False)
    cluster_infos.sort_values(['patientID', 'cluster'], inplace=True)

    cluster_infos.to_csv(
        f'cluster_info_{min_cluster_size}_{min_mutation_ccf}_{min_cluster_ccf}.csv',
        index=False)
    samples_infos.to_csv(
        f'samples_info_{min_cluster_size}_{min_mutation_ccf}_{min_cluster_ccf}.csv'
    )
    max_cluster_infos.to_csv(
        f'max_cluster_info_{min_cluster_size}_{min_mutation_ccf}_{min_cluster_ccf}.csv',
        index=False)

    remove_patients = max_cluster_infos.loc[
        max_cluster_infos['min_clonal_ccf'] <= min_cluster_ccf,
        'patientID'].to_list()
    print(
        f'we jump {problem_patients} have n_patients = {len(all_patients)} remove {remove_patients} and {list(set(all_patients) - set(remove_patients))} are left\n'
    )
    dfs = dfs[~dfs['patientID'].isin(remove_patients)]

    dfs['cluster'] = dfs['cluster'].astype(str)
    assert dfs.groupby('patientID')['is.clonal'].any().all()

    def to_ccf_csv(dfs, driver_cut):
        dfs['is.driver'] = False
        if isinstance(driver_cut, int):
            overlap_genes = dfs['variantID'].value_counts()[
                dfs['variantID'].value_counts() >= driver_cut].index.tolist()
            overlap_genes.remove('non_coding_region')
            driver_cut = str(driver_cut)
        elif isinstance(driver_cut, list):
            driver_gene_part = dfs[dfs['gene'].isin(driver_cut)]
            overlap_genes = driver_gene_part['variantID'].value_counts(
            )[driver_gene_part['variantID'].value_counts() >= 2].index.tolist(
            )
            driver_cut = 'list'
        else:
            print(f'driver_cut = {driver_cut} failed')
            return

        drivers_num = len(overlap_genes)
        dfs.loc[dfs['variantID'].isin(overlap_genes), 'is.driver'] = True

        patients_has_no_drivers = dfs.groupby('patientID')['is.driver'].any()
        patients_has_no_drivers = patients_has_no_drivers[
            ~patients_has_no_drivers].index.tolist()
        if patients_has_no_drivers:
            print(
                f'under driver_cut = {driver_cut} and patients with no drivers {patients_has_no_drivers}'
            )
            out = dfs.loc[~dfs.patientID.isin(patients_has_no_drivers), :]
        else:
            out = dfs
        out.to_csv(
            f'python_{min_cluster_size}_{min_mutation_ccf}_{min_cluster_ccf}_{driver_cut}_{drivers_num}_driver_genes.csv',
            index=False)
        out.loc[:,
                'is.clonal'] = out.loc[:,
                                       'is.clonal'].replace([True, False],
                                                            ['TRUE', 'FALSE'])
        out.loc[:,
                'is.driver'] = out.loc[:,
                                       'is.driver'].replace([True, False],
                                                            ['TRUE', 'FALSE'])
        out.loc[:, 'patientID'] = out.loc[:, 'patientID'].str.replace("-", "_")
        out = out[[
            'Misc', 'patientID', 'variantID', 'cluster', 'is.driver',
            'is.clonal', 'CCF'
        ]]
        out.to_csv(
            f'R_{min_cluster_size}_{min_mutation_ccf}_{min_cluster_ccf}_{driver_cut}_{drivers_num}_driver_genes.csv',
            index=False)

    for driver_cut in [driver_genes]:
        if driver_cut:
            to_ccf_csv(dfs, driver_cut)

    os.chdir(curdir)


def main():
    patient_level = True
    tumor_level = True
    only_driver_genes = True
    problem_remove = True

    if only_driver_genes:
        driver_genes = [
            'TP53', 'ARID1A', 'RB1', 'PTEN', 'CTNNB1', 'ALB', 'AXIN1'
        ]
        # driver_genes = [transform_name_to_id(i) for i in driver_genes]
        # driver_genes = ['ENSG00000141510',
        #                 'ENSG00000117713',
        #                 'ENSG00000139687',
        #                 'ENSG00000171862',
        #                 'ENSG00000168036',
        #                 'ENSG00000163631',
        #                 'ENSG00000103126']
    else:
        driver_genes = None
    curdir = os.path.abspath(os.curdir)
    if patient_level:
        workdir = f"patient_level_{str(len(driver_genes))+'_drivers' if isinstance(driver_genes,list) else 'no_driver_specfied'}_{'problem_remove' if problem_remove else 'whole'}"
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        os.chdir(workdir)
        paths = [
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/for-jky/4-pyclone-results-based-on-patient"
        ]
        process_all_patients(paths,
                             min_cluster_size=10,
                             min_mutation_ccf=0.2,
                             min_cluster_ccf=0.2,
                             driver_genes=driver_genes,
                             problem_remove=problem_remove)
        os.chdir(curdir)
    if tumor_level:
        workdir = f"tumor_level_{str(len(driver_genes))+'_drivers' if isinstance(driver_genes,list) else 'no_driver_specfied'}_{'problem_remove' if problem_remove else 'whole'}"
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        os.chdir(workdir)
        paths = [
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/RT-node/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/GWZY020-met-node/4-pyclone-results"
        ]
        process_all_patients(paths,
                             min_cluster_size=10,
                             min_mutation_ccf=0.2,
                             min_cluster_ccf=0.2,
                             driver_genes=driver_genes,
                             problem_remove=problem_remove)
        os.chdir(curdir)
    # from itertools import product
    # for i, j, k in product([5, 10, 15, 20], [0.1, 0.2], [0.1, 0.2]):
    #     process_all_patients(paths,
    #                          min_cluster_size=i,
    #                          min_mutation_ccf=j,
    #                          min_cluster_ccf=k,
    #                          driver_genes=driver_genes,
    #                          problem_remove=False)


if __name__ == '__main__':
    main()
