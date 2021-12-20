import os
import multiprocessing


def main():
    bio_types = ['netseq', 'nucl', 'abf1', 'h2a', 'size', 'meres', 'rel_meres']
    num_classes = 2
    num_trials = 10
    num_cpus = 4
    with multiprocessing.Pool(processes=num_cpus) as parallel:
        for i in range(2):
            if i == 0:
                do_each = True
                no_tcr = False
            else:
                do_each = False
                no_tcr = True
        for bt in bio_types:
            if bt == 'abf1':
                bi = 'uv'
            elif bt == 'nucl':
                bi = '0min'
            else:
                bi = ''

            for i in range(1, num_trials + 1):
                # # Lin
                # parallel.apply_async(
                #     os.system, ('python3.8 predict.py '
                #           '--bio_type=%s '
                #           '--bio_index=%s '
                #           '--ml_type=lin '
                #           '--num_classes=%s '
                #           '%s '
                #           '%s '
                #           '--save_prefix=%s%s_lin_%s%s '
                #           '--save_fig '
                #           '--to_pickle '
                #           '--num_cpus=1 '
                #           '--verbosity=4 '
                #           % (
                #               bt,
                #               bi,
                #               num_classes,
                #               '--do_each' if do_each else '',
                #               '--no_tcr' if no_tcr else '',
                #               'notcr_' if no_tcr else '',
                #               bt,
                #               'each' if do_each else 'total',
                #               i
                #           ), ))
                # parallel.apply_async(
                #     os.system, ('python3.8 predict.py '
                #           '--bio_type=%s '
                #           '--bio_index=%s '
                #           '--ml_type=lin '
                #           '--num_classes=%s '
                #           '%s '
                #           '%s '
                #           '--neg_random '
                #           '--save_prefix=%s%s_lin_%s%s_random '
                #           '--save_fig '
                #           '--to_pickle '
                #           '--num_cpus=1 '
                #           '--verbosity=4 '
                #           % (
                #               bt,
                #               bi,
                #               num_classes,
                #               '--do_each' if do_each else '',
                #               '--no_tcr' if no_tcr else '',
                #               'notcr_' if no_tcr else '',
                #               bt,
                #               'each' if do_each else 'total',
                #               i
                #           ), ))
                # kNN
                for kneighbour in [5, 10, 20, 50, 100]:
                    parallel.apply_async(
                        os.system, ('python3.8 predict.py '
                              '--bio_type=%s '
                              '--bio_index=%s '
                              '--ml_type=knn '
                              '--num_classes=%s '
                              '--kneighbour=%s '
                              '%s '
                              '%s '
                              '--save_prefix=%s%s_knn%s_%s%s '
                              '--save_fig '
                              '--to_pickle '
                              '--num_cpus=1 '
                              '--verbosity=4 '
                              % (
                                  bt,
                                  bi,
                                  num_classes,
                                  kneighbour,
                                  '--do_each' if do_each else '',
                                  '--no_tcr' if no_tcr else '',
                                  'notcr_' if no_tcr else '',
                                  bt,
                                  kneighbour,
                                  'each' if do_each else 'total',
                                  i
                              ), ))
                    parallel.apply_async(
                        os.system, ('python3.8 predict.py '
                              '--bio_type=%s '
                              '--bio_index=%s '
                              '--ml_type=knn '
                              '--num_classes=%s '
                              '--kneighbour=%s '
                              '%s '
                              '%s '
                              '--neg_random '
                              '--save_prefix=%s%s_knn%s_%s%s_random '
                              '--save_fig '
                              '--to_pickle '
                              '--num_cpus=1 '
                              '--verbosity=4 '
                              % (
                                  bt,
                                  bi,
                                  num_classes,
                                  kneighbour,
                                  '--do_each' if do_each else '',
                                  '--no_tcr' if no_tcr else '',
                                  'notcr_' if no_tcr else '',
                                  bt,
                                  kneighbour,
                                  'each' if do_each else 'total',
                                  i
                              ), ))

        parallel.close()
        parallel.join()


if __name__ == '__main__':
    main()


