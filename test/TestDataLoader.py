import unittest
import os
import numpy as np
import pandas as pd

import src.DataLoader as dl
from datahandler import reader


class TestDataLoader(unittest.TestCase):
    def test_transform_path(self):
        path_tilde = '~/test/to/file'
        path_absolute = '/home/user/test/to/file'
        path_relative = 'data/test/to/file'
        path_back = '/home/user/test/wrong/../to/file'

        self.assertEqual(
            os.path.expanduser(path_tilde),
            dl.transform_path(path_tilde)
        )
        self.assertEqual(
            path_absolute,
            dl.transform_path(path_absolute)
        )
        self.assertEqual(
            '%s/%s' % (os.getcwd(), path_relative),
            dl.transform_path(path_relative)
        )
        self.assertEqual(
            path_absolute,
            dl.transform_path(path_back)
        )

    def test_load_centromeres(self):
        data_path = '../data/ref/centromeres.bed'
        centromeres = dl.load_centromeres(data_path)
        self.assertListEqual(
            list(centromeres.columns),
            ['chr', 'pos']
        )
        self.assertGreater(
            len(centromeres.index),
            0
        )

    def test_load_transcriptome(self):
        high_t_path = '../data/ref/trans_high.txt'
        medium_t_path = '../data/ref/trans_medium.txt'
        low_t_path = '../data/ref/trans_low.txt'

        ht, mt, lt = dl.load_transcriptome(high_t_path, medium_t_path, low_t_path)

        self.assertListEqual(
            list(ht.columns),
            ['chr', 'ORF', 'start', 'end']
        )
        self.assertFalse(np.any(ht.duplicated()))
        self.assertGreater(
            len(ht.index),
            0
        )
        self.assertListEqual(
            list(mt.columns),
            ['chr', 'ORF', 'start', 'end']
        )
        self.assertFalse(np.any(mt.duplicated()))
        self.assertGreater(
            len(mt.index),
            0
        )
        self.assertListEqual(
            list(lt.columns),
            ['chr', 'ORF', 'start', 'end']
        )
        self.assertFalse(np.any(lt.duplicated()))
        self.assertGreater(
            len(lt.index),
            0
        )

    def test_num_pydim_pos(self):
        chrom = 'chrI'
        genome_path = '../data/ref/SacCer3.fa'
        bins = [0, 15, 30, 50]
        exp_pydim_pairs_pos = [3, 2, 4]
        exp_pydim_pairs_neg = [0, 0, 0]
        ref_genome = reader.load_fast(os.path.abspath(genome_path), is_abs_path=True, is_fastq=False)

        plus, plus_ovlap = dl.num_pydim_pos(chrom, bins, '+', ref_genome)
        minus, minus_ovlap = dl.num_pydim_pos(chrom, bins, '-', ref_genome)

        self.assertListEqual(plus.tolist(), exp_pydim_pairs_pos)
        self.assertListEqual(plus_ovlap.tolist(), [0, 0, 0])
        self.assertListEqual(minus.tolist(), exp_pydim_pairs_neg)
        self.assertListEqual(minus_ovlap.tolist(), [0, 0, 0])

    def test_normalise_data(self):
        # Data expectation
        # '0h_A1_minus.bw',
        # '0h_A1_plus.bw',
        # '1h_A1_minus.bw',
        # '1h_A1_plus.bw',
        # '0h_A2_minus.bw',
        # '0h_A2_plus.bw',
        # '20m_A2_minus.bw',
        # '20m_A2_plus.bw',
        # '2h_A2_minus.bw',
        # '2h_A2_plus.bw'

        np.random.seed(0)
        chrom = 'chrI'
        genome_path = '../data/ref/SacCer3.fa'
        bins = [0, 15, 30, 45]
        exp_pydim_pos = np.asarray([6, 4, 6])
        exp_pydim_igr = np.asarray([4])
        mock_transcriptome = pd.DataFrame(
            [[chrom, 'Mock Gene', 46, 1]],
            columns=['chr', 'ORF', 'start', 'end']
        )

        ref_genome = reader.load_fast(os.path.abspath(genome_path), is_abs_path=True, is_fastq=False)
        test_data = np.random.random((10, 55))
        test_data[0] *= 5
        test_data[1] *= 5

        test_data[4] *= 5
        test_data[5] *= 5
        test_data[6] *= 2
        test_data[7] *= 2

        ((
            t_20, t_60, t_120,
            nt_20, nt_60, nt_120,
            igr_20, igr_60, igr_120
        ), (start_igr, end_igr)) = dl.normalise_data(mock_transcriptome, chrom, test_data, ref_genome, num_bins=3)

        # Fulfill criterion for progressing repair
        self.assertTrue(np.all(t_20[t_20 != -1] >= 0))
        self.assertTrue(np.all(t_60[t_60 != -1] >= t_20[t_20 != -1]))
        self.assertTrue(np.all(t_120[t_120 != -1] >= t_60[t_60 != -1]))

        self.assertTrue(np.all(nt_20[nt_20 != -1] >= 0))
        self.assertTrue(np.all(nt_60[nt_60 != -1] >= nt_20[nt_20 != -1]))
        self.assertTrue(np.all(nt_120[nt_120 != -1] >= nt_60[nt_60 != -1]))

        self.assertTrue(np.all(igr_20[igr_20 != -1] >= 0))
        self.assertTrue(np.all(igr_60[igr_60 != -1] >= igr_20[igr_20 != -1]))
        self.assertTrue(np.all(igr_120[igr_120 != -1] >= igr_60[igr_60 != -1]))

        # Transcript data
        expect_0 = np.flip(np.nan_to_num(np.add.reduceat(test_data[5], bins)[:-1] / exp_pydim_pos))
        expect_1 = np.flip(np.nan_to_num(np.add.reduceat(test_data[7], bins)[:-1] / exp_pydim_pos))
        t_20_exp = ((expect_0 - expect_1) / expect_0)

        expect_0 = np.flip(np.nan_to_num(np.add.reduceat(test_data[1], bins)[:-1] / exp_pydim_pos))
        expect_1 = np.flip(np.nan_to_num(np.add.reduceat(test_data[3], bins)[:-1] / exp_pydim_pos))
        t_60_exp = ((expect_0 - expect_1) / expect_0)

        expect_0 = np.flip(np.nan_to_num(np.add.reduceat(test_data[5], bins)[:-1] / exp_pydim_pos))
        expect_1 = np.flip(np.nan_to_num(np.add.reduceat(test_data[9], bins)[:-1] / exp_pydim_pos))
        t_120_exp = ((expect_0 - expect_1) / expect_0)

        t_20_exp = np.maximum(0, t_20_exp)
        t_60_exp = np.maximum(t_20_exp, t_60_exp)
        t_120_exp = np.maximum(t_60_exp, t_120_exp)

        self.assertListEqual(t_20_exp.tolist(), t_20[0].tolist())
        self.assertListEqual(t_60_exp.tolist(), t_60[0].tolist())
        self.assertListEqual(t_120_exp.tolist(), t_120[0].tolist())

        # NTS data
        self.assertListEqual([-1, -1, -1], nt_60[0].tolist())
        self.assertListEqual([-1, -1, -1], nt_20[0].tolist())
        self.assertListEqual([-1, -1, -1], nt_120[0].tolist())

        # IGR data
        # order is + -
        # No need to flip as there is only one value
        expect_0 = np.nan_to_num(np.sum(test_data[5][45:]) / exp_pydim_igr)
        expect_1 = np.nan_to_num(np.sum(test_data[7][45:]) / exp_pydim_igr)
        igr_20_exp = np.asarray([((expect_0 - expect_1) / expect_0)[0], -1.])

        expect_0 = np.nan_to_num(np.sum(test_data[1][45:]) / exp_pydim_igr)
        expect_1 = np.nan_to_num(np.sum(test_data[3][45:]) / exp_pydim_igr)
        igr_60_exp = np.asarray([((expect_0 - expect_1) / expect_0)[0], -1.])

        expect_0 = np.nan_to_num(np.sum(test_data[5][45:]) / exp_pydim_igr)
        expect_1 = np.nan_to_num(np.sum(test_data[9][45:]) / exp_pydim_igr)
        igr_120_exp = np.asarray([((expect_0 - expect_1) / expect_0)[0], -1.])

        igr_20_exp[igr_20_exp != -1] = np.maximum(igr_20_exp[igr_20_exp != -1], 0)
        igr_60_exp[igr_60_exp != -1] = np.maximum(igr_60_exp[igr_60_exp != -1], igr_20_exp[igr_20_exp != -1])
        igr_120_exp[igr_120_exp != -1] = np.maximum(igr_120_exp[igr_120_exp != -1], igr_60_exp[igr_60_exp != -1])
        np.testing.assert_almost_equal(igr_20_exp, igr_20.reshape(-1), 5)
        np.testing.assert_almost_equal(igr_60_exp, igr_60.reshape(-1), 5)
        np.testing.assert_almost_equal(igr_120_exp, igr_120.reshape(-1), 5)

        # IGR indices
        self.assertListEqual([45], start_igr.tolist())
        self.assertListEqual([55], end_igr.tolist())

    def test_load_chrom_data(self):
        train_chrom_list = ['chrI']
        test_chrom_list = ['chrII']
        bw_list = [
            '../data/seq/0h_A1_minus.bw',
            '../data/seq/0h_A1_plus.bw',
            '../data/seq/1h_A1_minus.bw',
            '../data/seq/1h_A1_plus.bw',
            '../data/seq/0h_A2_minus.bw',
            '../data/seq/0h_A2_plus.bw',
            '../data/seq/20m_A2_minus.bw',
            '../data/seq/20m_A2_plus.bw',
            '../data/seq/2h_A2_minus.bw',
            '../data/seq/2h_A2_plus.bw'
        ]
        transcriptome_path_list = [
            '../data/ref/trans_high.txt',
            '../data/ref/trans_medium.txt',
            '../data/ref/trans_low.txt',
        ]
        ref_genome_path = '../data/ref/SacCer3.fa'

        used_transcriptomes = [True, False, False]
        seed = 0
        train_data, test_data = dl.load_chrom_data(
            train_chrom_list=train_chrom_list,
            test_chrom_list=test_chrom_list,
            bw_list=bw_list,
            transcriptome_path_list=transcriptome_path_list,
            ref_genome_path=ref_genome_path,
            used_transcriptomes=used_transcriptomes,
            seed=seed
        )

        self.assertEqual(len(train_data), 6)
        self.assertEqual(len(test_data), 6)

        self.assertListEqual(list(train_data[0].shape), [42, 3, 3])
        self.assertListEqual(list(train_data[1].shape), [42, 3, 3])
        self.assertListEqual(list(train_data[2].shape), [41, 3, 2])
        self.assertEqual(len(train_data[-1].index), 42)

        self.assertListEqual(list(test_data[0].shape), [196, 3, 3])
        self.assertListEqual(list(test_data[1].shape), [196, 3, 3])
        self.assertListEqual(list(test_data[2].shape), [177, 3, 2])
        self.assertEqual(len(test_data[-1].index), 196)

        # Test shuffing
        shuffle_idx = np.arange(42)
        np.random.shuffle(shuffle_idx)
        self.assertListEqual(
            train_data[0][shuffle_idx[0]].reshape(-1).tolist(),
            train_data[0][shuffle_idx][0].reshape(-1).tolist()
        )

    def test_chrom_split(self):
        path = '../chrom.data'
        exp_param_chrom = ['chrI']
        param_chrom = dl.load_chrom_split(path=path, data_type='parameter')
        self.assertListEqual(param_chrom, exp_param_chrom)

        exp_ignore_chrom = ['chrM']
        ignrore_chrom = dl.load_chrom_split(path=path, data_type='ignore')
        self.assertListEqual(ignrore_chrom, exp_ignore_chrom)

        exp_train_chrom = [
            'chrII',
            'chrIII',
            'chrIX',
            'chrV',
            'chrVII',
            'chrVIII',
            'chrXI',
            'chrXII',
            'chrXIV',
            'chrXV'
        ]
        train_chrom = dl.load_chrom_split(path=path, data_type='train')
        self.assertListEqual(train_chrom, exp_train_chrom)

        exp_test_chrom = [
            'chrIV',
            'chrVI',
            'chrX',
            'chrXIII',
            'chrXVI'
        ]
        test_chrom = dl.load_chrom_split(path=path, data_type='test')
        self.assertListEqual(test_chrom, exp_test_chrom)


if __name__ == '__main__':
    unittest.main()
