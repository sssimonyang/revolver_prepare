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
    patient_level = True
    tumor_level = True

    problem_remove = True
    driver = ['TP53', 'ARID1A', 'RB1', 'PTEN', 'CTNNB1', 'ALB', 'AXIN1']

    curdir = os.path.abspath(os.curdir)

    if patient_level:
        workdir = f"patient_level_{str(len(driver)) +'_drivers' if isinstance(driver,list) else str(driver) +'_cut'}_{'problem_remove' if problem_remove else 'whole'}"
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        os.chdir(workdir)
        paths = [
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/for-jky/4-pyclone-results-based-on-patient"
        ]
        ccf = CCF(paths,
                  level='patient_level',
                  driver=driver,
                  problem_remove=problem_remove)
        ccf.main()
        os.chdir(curdir)
    if tumor_level:
        workdir = f"tumor_level_{str(len(driver)) +'_drivers' if isinstance(driver,list) else str(driver) +'_cut'}_{'problem_remove' if problem_remove else 'whole'}"
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        os.chdir(workdir)
        paths = [
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/RT-node/4-pyclone-results",
            "/public/home/wupin/project/1-liver-cancer/landscape-figure/2-somatic-new-label/GWZY020-met-node/4-pyclone-results"
        ]
        ccf = CCF(paths,
                  level='tumor_level',
                  driver=driver,
                  problem_remove=problem_remove)
        ccf.main()
        os.chdir(curdir)


class CCF:
    gtf_file = "/public/home/wangzj/yangjk/Homo_sapiens_GRCh37_87_geneonly.gtf"
    sig_columns = [(i + '_count', i + '_prop')
                   for i in ['C_A', 'C_G', 'C_T', 'T_A', 'T_C', 'T_G']]
    sig_columns = [i for j in sig_columns for i in j]

    def __init__(self,
                 paths,
                 level,
                 driver=None,
                 problem_remove=True,
                 min_cluster_size=10,
                 min_mutation_ccf=0.1,
                 min_cluster_ccf=0.1):
        self.paths = paths
        self.level = level
        self.driver = driver
        if isinstance(self.driver, list):
            self.sepcified_driver = True
            self.driver_genes = self.driver
        elif isinstance(self.driver, int):
            self.sepcified_driver = False
            self.driver_cut = self.driver

        self.problem_remove = problem_remove
        if self.problem_remove:
            self.problem_patients = [
                'GWZY072', 'GWZY048_PT_S2010-27295', 'GWZY100_PT_S2013-08183',
                'GWZY121_PT_S2013-35637'
            ]
        else:
            self.problem_patients = []

        self.min_cluster_size = min_cluster_size
        self.min_mutation_ccf = min_mutation_ccf
        self.min_cluster_ccf = min_cluster_ccf

        self.dfs = []
        self.processed_patients = []
        self.samples_info_dfs = []
        self.cluster_info_dfs = []
        self.patient_info_dfs = []
        self.fetch_gene = CCF.get_info_from_gtf('fetch_gene_name')

        self.current_patient = None

    @batch_split
    def main(self):
        curdir = os.path.abspath(os.curdir)
        print(
            f'This run parameter ',
            f'level = {self.level} problem_remove = {self.problem_remove} problem patients = {self.problem_patients}',
            f'min_cluster_size = {self.min_cluster_size} min_mutation_ccf = {self.min_mutation_ccf} min_cluster_ccf = {self.min_cluster_ccf}',
            f'driver genes = {self.driver_genes}',
            end='\n')
        for path in self.paths:
            os.chdir(path)
            patients = os.listdir()

            for patient in patients:
                if 'tables' not in os.listdir(os.path.join(path, patient)):
                    continue
                if len(os.listdir(os.path.join(path, patient, 'tables'))) != 2:
                    continue
                if patient in self.problem_patients:
                    continue
                loci = pd.read_csv(f"./{patient}/tables/loci.tsv", sep='\t')
                loci.drop(
                    ['variant_allele_frequency', 'cellular_prevalence_std'],
                    axis=1,
                    inplace=True)
                self.current_patient = patient
                self.single_patient(loci)
                self.processed_patients.append(patient)

        os.chdir(curdir)
        self.combine()

    def single_patient(self, loci):
        df, samples_map = self.single(loci=loci)
        df, cluster_info, removed_cluster = self.analyse_cluster(df)
        cluster_info_df = [[self.current_patient, cluster, *info]
                           for cluster, info in cluster_info.items()]
        cluster_info_df = pd.DataFrame(cluster_info_df,
                                       columns=[
                                           'patientID', 'cluster',
                                           *CCF.sig_columns,
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
            clonal_cluster_ccf <= self.min_cluster_ccf].index.to_list()
        if remove_regions:
            print(
                f"patient {self.current_patient} clonal cluster CCF {clonal_cluster_ccf.to_list()} remove {remove_regions} rerun"
            )
            remove_samples = [samples_map[i] for i in remove_regions]
            loci = loci[~loci["sample_id"].isin(remove_samples)]
            self.current_patient = f"{self.current_patient}_remove_{'_'.join(remove_regions)}"
            self.single_patient(loci)
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

        df['patientID'] = self.current_patient
        df['is.clonal'] = False
        df.loc[df.cluster == clonal_cluster, 'is.clonal'] = True

        self.dfs.append(df)
        return

    def single(self, loci):
        samples = loci["sample_id"].unique()
        samples_map = {
            f'R{i+1:02}': sample
            for i, sample in enumerate(samples)
        }

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
        return result, samples_map

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
                i for j in zip(mutation_sig['counts'].to_list(),
                               mutation_sig['prop'].to_list()) for i in j
            ]
            cluster_infos[i] = [*mutation_sig, *median_cluster_ccf]
        removed_cluster = self.value_to_str(removed_cluster)

        return result, cluster_infos, removed_cluster

    def combine(self):
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
        self.to_ccf_csv(df)

    def to_ccf_csv(self, df):
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
                f"Under driver_cut = {'list' if self.sepcified_driver else str(self.driver_cut)} and patients with no drivers {patients_has_no_drivers}"
            )
            out = df.loc[~df.patientID.isin(patients_has_no_drivers), :]
        else:
            out = df
        out = out.copy()
        out.to_csv(
            f"python_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.csv",
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
            f"R_{self.min_cluster_size}_{self.min_mutation_ccf}_{self.min_cluster_ccf}_{'list' if self.sepcified_driver else str(self.driver_cut)}_{drivers_num}_driver_genes.csv",
            index=False)

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
