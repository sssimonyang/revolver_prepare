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
        for _ in range(10):
            print()

    return wrapper


def main():
    patient_level = False
    tumor_level = True
    sample_split = False
    final_type_split = True
    problem_remove = True
    remove_unuseful_cluster = False

    driver = ['TP53', 'ARID1A', 'RB1', 'PTEN', 'CTNNB1', 'ALB', 'AXIN1']
    # driver = 2
    cnv_arms = ['+1q', '+8q', '-4q', '-8p', '-16q']
    # cnv_arms = [
    #     '+1q', '+8q', '-4q', '-8p', '-16q', '+2p', '+2q', '-11p', '+5q', '+7p',
    #     '+6p', '-15q', '+5p', '-21p'
    # ]

    curdir = os.path.abspath(os.curdir)

    if patient_level:
        workdir = f"patient_level_{str(len(driver)+len(cnv_arms)) +'_drivers' if isinstance(driver,list) else str(driver) +'_cut'}_{'problem_remove' if problem_remove else 'problem_remain'}_{'sample_split' if sample_split else 'sample_not_split'}"
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        os.chdir(workdir)
        paths = [
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/for-jky/4-pyclone-results-based-on-patient",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/5-within-liver-pyclone/pyclone-patient-level/3-pyclone-results"
        ]
        ccf = CCF(paths,
                  level='patient_level',
                  driver=driver,
                  cnv_arms=cnv_arms,
                  problem_remove=problem_remove,
                  sample_split=sample_split,
                  final_type_split=final_type_split,
                  remove_unuseful_cluster=remove_unuseful_cluster)
        ccf.main()
        os.chdir(curdir)

    if tumor_level:
        workdir = f"tumor_level_{str(len(driver)+len(cnv_arms)) +'_drivers' if isinstance(driver,list) else str(driver) +'_cut'}_{'problem_remove' if problem_remove else 'problem_remain'}_{'sample_split' if sample_split else 'sample_not_split'}"
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        os.chdir(workdir)
        paths = [
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/RT-node/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/GWZY020-met-node/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/5-within-liver-pyclone/3-pyclone-results"
        ]
        ccf = CCF(paths,
                  level='tumor_level',
                  driver=driver,
                  cnv_arms=cnv_arms,
                  problem_remove=problem_remove,
                  sample_split=sample_split,
                  final_type_split=final_type_split,
                  remove_unuseful_cluster=remove_unuseful_cluster)
        ccf.main()
        os.chdir(curdir)


class CCF:
    gtf_file = "/public/home/wangzj/yangjk/Homo_sapiens_GRCh37_87_geneonly.gtf"
    # cnv_file = "/public/home/wupin/Gistic-analysis/2-MET-cohort-all-samples-new_label-float_ploidy/4-GISTIC-results-center/2-based-on-sample/2-all-samples-in-MET-cohort/broad_values_by_arm.txt"
    cnv_file = "/public/home/wupin/Gistic-analysis/2-MET-cohort-all-samples-new_label-float_ploidy/4-GISTIC-results-center/2-based-on-sample/1-all-sample-in-three-cohort/broad_values_by_arm.txt"
    sig_columns = [(i + '_count', i + '_prop')
                   for i in ['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G']]
    sig_columns = [i for j in sig_columns for i in j]

    def __init__(self,
                 paths,
                 level,
                 driver=None,
                 cnv_arms=None,
                 problem_remove=True,
                 sample_split=False,
                 final_type_split=False,
                 remove_unuseful_cluster=True,
                 min_cluster_size=10,
                 min_mutation_ccf=0.1,
                 min_cluster_ccf=0.1,
                 cnv_cut_value=0.5):
        self.paths = paths
        self.level = level
        self.driver = driver
        self.cnv_arms = cnv_arms
        if self.cnv_arms:
            self.cnv_data = pd.read_csv(self.__class__.cnv_file, sep='\t')
            self.cnv_data.set_index("Chromosome Arm", inplace=True)
            amp_cnv = self.cnv_data >= cnv_cut_value
            amp_cnv.index = '+' + amp_cnv.index
            del_cnv = self.cnv_data <= -cnv_cut_value
            del_cnv.index = '-' + del_cnv.index
            self.cnv_data = pd.concat([amp_cnv, del_cnv])

        if isinstance(self.driver, list):
            self.sepcified_driver = True
            self.driver_genes = self.driver + self.cnv_arms
        elif isinstance(self.driver, int):
            self.sepcified_driver = False
            self.driver_cut = self.driver

        self.problem_remove = problem_remove
        if self.problem_remove:
            self.problem_patients = [
                'GWZY072',
                'GWZY858729',
                'GWZY048_PT',
                'GWZY100_PT',
                'GWZY121_PT',
                'GWZY913925_PT1'  # , 'GWZY054_PT1'
            ]
        else:
            self.problem_patients = []

        self.sample_split = sample_split
        self.final_type_split = final_type_split
        self.remove_unuseful_cluster = remove_unuseful_cluster

        if self.level == 'patient_level' and (not self.sample_split):
            self.final_type_split = False

        self.min_cluster_size = min_cluster_size
        self.min_mutation_ccf = min_mutation_ccf
        self.min_cluster_ccf = min_cluster_ccf

        self.dfs = []
        self.processed_patients = []
        self.samples_info_dfs = []
        self.cluster_info_dfs = []
        self.patient_info_dfs = []
        self.fetch_gene = self.__class__.get_info_from_gtf('fetch_gene_name')

        self.current_patient = None

    @batch_split
    def main(self):
        curdir = os.path.abspath(os.curdir)
        print(
            f'This run parameter ',
            f'level = {self.level} sample split = {self.sample_split} problem remove = {self.problem_remove} problem patients = {self.problem_patients}',
            f'min_cluster_size = {self.min_cluster_size} min_mutation_ccf = {self.min_mutation_ccf} min_cluster_ccf = {self.min_cluster_ccf}',
            f'specified driver {self.driver_genes if self.sepcified_driver else self.driver_cut}',
            sep='\n')
        for path in self.paths:
            os.chdir(path)
            patients = os.listdir()

            for patient in patients:
                if 'tables' not in os.listdir(os.path.join(path, patient)):
                    continue
                if len(os.listdir(os.path.join(path, patient, 'tables'))) != 2:
                    continue
                loci = pd.read_csv(f"./{patient}/tables/loci.tsv", sep='\t')
                patient = self.__class__.name_transform(patient)
                if patient in self.problem_patients:
                    continue
                loci.drop(
                    ['variant_allele_frequency', 'cellular_prevalence_std'],
                    axis=1,
                    inplace=True)

                if self.sample_split:
                    for name, index in loci.groupby(
                            'sample_id').groups.items():
                        self.current_patient = self.__class__.name_transform(
                            name)
                        current_loci = loci.loc[index, :]
                        samples_map = {f'R{1:02}': name}
                        self.single_patient(current_loci, samples_map)
                        self.processed_patients.append(self.current_patient)
                else:
                    self.current_patient = patient
                    samples = loci["sample_id"].unique()
                    samples_map = {
                        f'R{i+1:02}': sample
                        for i, sample in enumerate(samples)
                    }
                    self.single_patient(loci, samples_map)
                    self.processed_patients.append(self.current_patient)

        os.chdir(curdir)
        print(f'Processed {self.processed_patients}')
        self.combine()

    def single_patient(self, loci, samples_map):
        df = self.single(loci=loci, samples_map=samples_map)
        if df.empty:
            return
        df, cluster_info, removed_cluster = self.analyse_cluster(df)
        cluster_info_df = [[self.current_patient, cluster, *info]
                           for cluster, info in cluster_info.items()]
        cluster_info_df = pd.DataFrame(cluster_info_df,
                                       columns=[
                                           'patientID', 'cluster',
                                           *self.__class__.sig_columns,
                                           *samples_map.keys()
                                       ])
        cluster_info_df['cluster_mean_ccf'] = cluster_info_df.loc[:,
                                                                  samples_map.
                                                                  keys()].mean(
                                                                      axis=1)
        clonal_cluster = cluster_info_df.loc[
            cluster_info_df['cluster_mean_ccf'].idxmax(),
            ['cluster']].values[0]

        clonal_cluster_ccf = cluster_info_df.loc[
            cluster_info_df['cluster_mean_ccf'].idxmax(),
            samples_map.keys()]
        remove_regions = clonal_cluster_ccf[
            clonal_cluster_ccf <= self.min_cluster_ccf].index.tolist()
        if remove_regions:
            print(
                f"patient {self.current_patient} clonal cluster CCF {clonal_cluster_ccf.tolist()} remove {remove_regions} rerun"
            )
            remove_samples = [samples_map[i] for i in remove_regions]
            loci = loci[~loci["sample_id"].isin(remove_samples)]
            self.current_patient = f"{self.current_patient}_remove_{'_'.join(remove_regions)}"
            for i in remove_regions:
                del samples_map[i]
            if samples_map:
                self.single_patient(loci, samples_map)
            return
        if removed_cluster:
            print(f'{self.current_patient} remove clusters {removed_cluster}')

        self.samples_info_dfs.append(
            pd.DataFrame([[
                self.current_patient, region, "_".join(
                    sample_name.split("_")[1:4:2])
            ] for region, sample_name in samples_map.items()],
                         columns=['patientID', 'region', 'sample']))

        self.cluster_info_dfs.append(cluster_info_df)

        patient_info_df = pd.DataFrame([[
            self.current_patient, clonal_cluster, removed_cluster,
            *cluster_info[clonal_cluster][12:]
        ]],
                                       columns=[
                                           'patientID', 'clonal_cluster',
                                           'removed_cluster',
                                           *samples_map.keys()
                                       ])
        self.patient_info_dfs.append(patient_info_df)

        samples_map = {
            i: '_'.join(j.split('_')[:4])
            for i, j in samples_map.items()
        }

        if self.cnv_arms and not (set(samples_map.values()) -
                                  set(self.cnv_data.columns)):
            patient_cnv = self.cnv_data.loc[self.cnv_arms,
                                            samples_map.values()].copy()
            patient_cnv['count'] = np.sum(patient_cnv[samples_map.values()],
                                          axis=1)
            for arm, value in patient_cnv.iterrows():
                if value['count'] == len(samples_map):
                    value = ";".join([
                        f'{sample}:{round(ccf,2)}' for sample, ccf in zip(
                            samples_map.keys(), clonal_cluster_ccf)
                    ])
                    df = df.append(
                        {
                            "Misc": arm,
                            'variantID': arm,
                            'gene': arm,
                            'cluster': clonal_cluster,
                            'CCF': value
                        },
                        ignore_index=True)
                else:
                    value = value[list(samples_map.values())]
                    temp_map = {j: i for i, j in samples_map.items()}
                    value.index = [temp_map[i] for i in value.index]
                    value = value.astype(bool)

                    amp_samples = value[value].index.tolist()
                    if not amp_samples:
                        continue

                    not_amp_samples = value[~value].index.tolist()
                    amp_ccf = cluster_info_df.loc[:, [
                        'cluster', *samples_map.keys()
                    ]].copy()
                    amp_ccf['not_amp_sample'] = (
                        amp_ccf.loc[:, not_amp_samples] <
                        self.min_cluster_ccf).all(axis=1)
                    amp_ccf['amp_sample'] = (amp_ccf.loc[:, amp_samples] >
                                             self.min_cluster_ccf).all(axis=1)
                    amp_ccf = amp_ccf[(amp_ccf['not_amp_sample'])
                                      & (amp_ccf['amp_sample'])]

                    if amp_ccf.empty:
                        print(
                            f'{self.current_patient} {arm} amp_samples {amp_samples} don\'t have proper cluster'
                        )
                        continue
                    amp_ccf[
                        'amp_sample_mean_ccf'] = amp_ccf.loc[:,
                                                             amp_samples].mean(
                                                                 axis=1)
                    amp_cluster = amp_ccf.loc[
                        amp_ccf['amp_sample_mean_ccf'].idxmax(),
                        ['cluster']].values[0]
                    amp_cluster_ccf = amp_ccf.loc[
                        amp_ccf['amp_sample_mean_ccf'].idxmax(),
                        samples_map.keys()]
                    value = ";".join([
                        f'{sample}:{round(ccf,2)}' for sample, ccf in zip(
                            samples_map.keys(), amp_cluster_ccf)
                    ])
                    df = df.append(
                        {
                            "Misc": arm,
                            'variantID': arm,
                            'gene': arm,
                            'cluster': amp_cluster,
                            'CCF': value
                        },
                        ignore_index=True)
        elif self.cnv_arms:
            print(
                f'{self.current_patient} samples {list(samples_map.values())} not all in cnv data'
            )
        else:
            pass

        df['patientID'] = self.current_patient
        df['is.clonal'] = False
        df.loc[df.cluster == clonal_cluster, 'is.clonal'] = True

        self.dfs.append(df)
        return

    def single(self, loci, samples_map):
        cluster_id = loci.groupby('mutation_id')['cluster_id'].first()
        ccf_value = loci.groupby('mutation_id').apply(
            self.concat_ccf_value(samples_map))
        result = pd.concat([cluster_id, ccf_value], axis=1, join='inner')
        result.columns = ['cluster', 'CCF']
        result.index.name = 'Misc'
        result.reset_index(inplace=True)

        # 筛选至少有一个sample CCF大于MUTATION_MIN_CCF的突变
        remove_mutation = loci.groupby('mutation_id').apply(lambda x: x[x[
            'cellular_prevalence'] >= self.min_mutation_ccf].shape[0] == 0)
        remove_mutation = remove_mutation[remove_mutation].index
        result = result[~result['Misc'].isin(remove_mutation)]

        # 筛选cluster内部突变数大于等于min_cluster_size
        cluster_count = result.groupby('cluster')["Misc"].count()
        filter_cluster = cluster_count[
            cluster_count >= self.min_cluster_size].index
        result = result[result.cluster.isin(filter_cluster)]

        result['gene'] = result['Misc'].apply(
            lambda x: self.fetch_gene(*x.split('_')[:2]))

        result['mutation_type'] = result['Misc'].apply(
            lambda x: '_'.join(x.split('_')[3:]))
        result['mutation_type'] = result['mutation_type'].replace(
            ['G_T', 'G_C', 'G_A', 'A_T', 'A_G', 'A_C'],
            ['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G'])

        # 处理单个patient一个基因的多个mutation
        result['variantID'] = result['gene']
        if self.sepcified_driver:
            overlap_variant_ids_counts = result.loc[
                result['gene'].isin(self.driver_genes),
                'variantID'].value_counts()
            overlap_variant_ids = overlap_variant_ids_counts[
                overlap_variant_ids_counts > 1].index

        else:
            overlap_variant_ids_counts = result['gene'].value_counts()
            overlap_variant_ids = overlap_variant_ids_counts[
                overlap_variant_ids_counts > 1].index

        if not overlap_variant_ids.empty:
            for variant_id in overlap_variant_ids:
                variant = result.loc[
                    result['variantID'] == variant_id,
                    'CCF'].apply(lambda x: np.mean(self.parse_concat_ccf(x)))
                remain_variants = variant.idxmax()
                modify_variants = variant.index.tolist()
                modify_variants.remove(remain_variants)
                result.loc[modify_variants, 'variantID'] = result.loc[
                    modify_variants, 'variantID'].str.cat([
                        f'{self.current_patient}-{i}' for i in modify_variants
                    ],
                                                          sep='-')
        return result

    def analyse_cluster(self, result):
        cluster_infos = {}
        removed_cluster = []
        for i in result.cluster.unique():
            cluster_ccf_value = result.loc[result.cluster == i,
                                           'CCF'].apply(self.parse_concat_ccf)
            cluster_ccf_value = np.array([i for i in cluster_ccf_value])
            median_cluster_ccf = np.median(cluster_ccf_value, axis=0)

            if np.max(median_cluster_ccf) <= self.min_cluster_ccf:
                result.drop(result[result.cluster == i].index,
                            axis=0,
                            inplace=True)
                removed_cluster.append(i)

            mutation_sig = result.loc[
                result.cluster == i,
                'mutation_type'].value_counts().to_frame()
            mutation_sig.index.name = 'mutation_type'
            mutation_sig.columns = ['counts']
            mutation_sig = mutation_sig.reindex(
                index=['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G'], fill_value=0)
            mutation_sig['prop'] = mutation_sig.counts / np.sum(
                mutation_sig.counts)
            mutation_sig['prop'] = mutation_sig['prop'].round(decimals=2)
            mutation_sig = [
                i for j in zip(mutation_sig['counts'].tolist(),
                               mutation_sig['prop'].tolist()) for i in j
            ]
            cluster_infos[i] = [*mutation_sig, *median_cluster_ccf]
        removed_cluster = self.value_to_str(removed_cluster)

        return result, cluster_infos, removed_cluster

    def combine(self):
        if not self.dfs:
            print('no content write out')
            return
        df = pd.concat(self.dfs, join='outer')
        df['variantID'] = df['variantID'].fillna('non_coding_region')
        df['is.driver'] = False

        patient_info_df = pd.concat(self.patient_info_dfs,
                                    join='outer',
                                    sort=False)
        patient_info_df['min_clonal_ccf'] = patient_info_df.loc[:, [
            i for i in patient_info_df.columns if i.startswith('R')
        ]].min(axis=1)
        patient_info_df['max_clonal_ccf'] = patient_info_df.loc[:, [
            i for i in patient_info_df.columns if i.startswith('R')
        ]].max(axis=1)

        samples_info_df = pd.concat(self.samples_info_dfs,
                                    join='outer',
                                    sort=False)
        samples_info_df.set_index(['patientID', 'region'], inplace=True)
        samples_info_df = samples_info_df.unstack()

        cluster_info_df = pd.concat(self.cluster_info_dfs,
                                    join='outer',
                                    sort=False)
        cluster_info_df.sort_values(['patientID', 'cluster'], inplace=True)

        cluster_info_df.to_csv(
            f'cluster_info_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}.csv',
            index=False)
        samples_info_df.to_csv(
            f'samples_info_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}.csv'
        )
        patient_info_df.to_csv(
            f'patient_info_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}.csv',
            index=False)

        df['cluster'] = df['cluster'].astype(str)
        assert df.groupby('patientID')['is.clonal'].any().all()

        df.reset_index(drop=True, inplace=True)

        if self.final_type_split:
            df['category'] = df['patientID'].apply(
                lambda x: re.search(r'_(RT|PT|.*M)\d?', x).group(1))
            df['category'] = df['category'].str.replace('.*M', "M")
            for name, index in df.groupby('category').groups.items():
                self.to_ccf_csv(df.loc[index, :].copy(), category=name)
        self.to_ccf_csv(df)

    def to_ccf_csv(self, df, category='None'):
        df['is.driver'] = False
        if self.sepcified_driver:
            driver_gene_part = df[df['gene'].isin(self.driver_genes)]
            overlap_genes = driver_gene_part['variantID'].value_counts(
            )[driver_gene_part['variantID'].value_counts() >= 2].index.tolist(
            )
        else:
            overlap_genes = df['variantID'].value_counts(
            )[df['variantID'].value_counts() >= self.driver_cut].index.tolist(
            )
            overlap_genes.remove('non_coding_region')

        drivers_num = len(overlap_genes)
        df.loc[df['variantID'].isin(overlap_genes), 'is.driver'] = True

        patients_has_no_drivers = df.groupby('patientID')['is.driver'].any()
        patients_has_no_drivers = patients_has_no_drivers[
            ~patients_has_no_drivers].index.tolist()
        if patients_has_no_drivers:
            print(
                f"Under category = {category} driver_cut = {'list' if self.sepcified_driver else str(self.driver_cut)} and patients with no drivers {patients_has_no_drivers}"
            )
            out = df.loc[~df.patientID.isin(patients_has_no_drivers), :]
        else:
            out = df
        if self.remove_unuseful_cluster:
            reserve = out.groupby([
                'patientID', 'cluster'
            ]).apply(lambda x: x['is.driver'].any() or x['is.clonal'].all())
            out.set_index(['patientID', 'cluster'], inplace=True)
            out = out.loc[reserve[reserve].index, :]
            out.reset_index(inplace=True)

        out = out.copy()

        out.to_csv(
            f"python_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}_{category}_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.csv",
            index=False)

        df = out[out['is.driver']]
        max_column = df.groupby('patientID')['variantID'].nunique().max()
        results = []
        for patient, index in df.groupby('patientID').groups.items():
            result = [patient]
            temp = df.loc[index, 'variantID'].unique().tolist()
            temp.sort()
            result.extend(temp)
            result.extend(
                [np.nan for _ in range(max_column - len(result) + 1)])
            results.append(result)

        results = pd.DataFrame(results)
        results.to_csv(
            f"adjacency_list_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.tsv",
            sep='\t',
            index=False,
            header=False)
        with open(
                f"adjacency_list_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.tsv",
                'r') as f:
            lines = f.readlines()
            end = lines[0][-1]
            lines = [line.strip() + end for line in lines]
        with open(
                f"adjacency_list_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.tsv",
                'w') as f:
            f.writelines(lines)

        out.loc[:,
                'is.clonal'] = out.loc[:,
                                       'is.clonal'].replace([True, False],
                                                            ['TRUE', 'FALSE'])
        out.loc[:,
                'is.driver'] = out.loc[:,
                                       'is.driver'].replace([True, False],
                                                            ['TRUE', 'FALSE'])

        out = out[[
            'Misc', 'patientID', 'variantID', 'cluster', 'is.driver',
            'is.clonal', 'CCF'
        ]]
        out.to_csv(
            f"R_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}_{category}_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.csv",
            index=False)

    @staticmethod
    def name_transform(name):
        names = np.array(name.split('_'))
        if names.shape[0] >= 4:
            return '_'.join(names[[0, 1, 3]].tolist())
        if names.shape[0] >= 2:
            return '_'.join(names[[0, 1]].tolist())
        if names.shape[0] == 1:
            return name

    @staticmethod
    def concat_ccf_value(samples_map):
        def wrapper(x):
            out = []
            for i in samples_map:
                value = x[x["sample_id"] ==
                          samples_map[i]]["cellular_prevalence"].tolist()
                if value:
                    out.append(i + ':' + str(round(value[0], 2)))
                else:
                    out.append(i + ':' + str(0))

            return ';'.join(out)

        return wrapper

    @classmethod
    def get_info_from_gtf(cls, get_type):
        gtf = pd.read_csv(cls.gtf_file, header=None, sep='\t')
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

    @staticmethod
    def value_to_str(values):
        return ','.join([str(round(i, 2)) for i in values])

    @staticmethod
    def parse_concat_ccf(ccf):
        return [float(i[4:]) for i in ccf.split(';')]


if __name__ == '__main__':
    main()
