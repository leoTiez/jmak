import os
import multiprocessing


def main():
    bio_types = ['slam', 'nucl', 'size', 'meres']
    num_classes = 2
    num_trials = 10
    num_cpus = 4
    do_each = True
    with multiprocessing.Pool(processes=num_cpus) as parallel:
        for bt in bio_types:
            if bt == 'slam':
                bi = '20min'
            elif bt == 'nucl':
                bi = '0min'
            else:
                bi = ''

            for i in range(1, num_trials + 1):
                # GP
                parallel.apply_async(
                    os.system,
                    ('python3.8 predict.py '
                        '--bio_type=%s '
                        '--bio_index=%s '
                        '--ml_type=gp '
                        '%s '
                        '--save_prefix=%s_gp_%s%s '
                        '--save_fig '
                        '--to_pickle '
                        '--num_cpus=1 '
                        '--verbosity=4 '
                        '--load_if_exist'
                        % (
                          bt,
                          bi,
                          '--do_each' if do_each else '',
                          bt,
                          'each' if do_each else 'total',
                          i
                        ), ))
                parallel.apply_async(
                    os.system,
                    ('python3.8 predict.py '
                     '--bio_type=%s '
                          '--bio_index=%s '
                          '--ml_type=gp '
                          '%s '
                          '--neg_random '
                          '--save_prefix=%s_gp_%s%s_random '
                          '--save_fig '
                          '--to_pickle '
                          '--num_cpus=1 '
                          '--verbosity=4 '
                          '--load_if_exist'
                          % (
                              bt,
                              bi,
                              '--do_each' if do_each else '',
                              bt,
                              'each' if do_each else 'total',
                              i
                              ), ))

                # Lin
                parallel.apply_async(
                    os.system, ('python3.8 predict.py '
                          '--bio_type=%s '
                          '--bio_index=%s '
                          '--ml_type=lin '
                          '--num_classes=%s '
                          '%s '
                          '--save_prefix=%s_lin_%s%s '
                          '--save_fig '
                          '--to_pickle '
                          '--num_cpus=1 '
                          '--verbosity=4 '
                          '--load_if_exist'
                          % (
                              bt,
                              bi,
                              num_classes,
                              '--do_each' if do_each else '',
                              bt,
                              'each' if do_each else 'total',
                              i
                          ), ))
                parallel.apply_async(
                    os.system, ('python3.8 predict.py '
                          '--bio_type=%s '
                          '--bio_index=%s '
                          '--ml_type=lin '
                          '--num_classes=%s '
                          '%s '
                          '--neg_random '
                          '--save_prefix=%s_lin_%s%s_random '
                          '--save_fig '
                          '--to_pickle '
                          '--num_cpus=5 '
                          '--verbosity=4 '
                          '--load_if_exist'
                          % (
                              bt,
                              bi,
                              num_classes,
                              '--do_each' if do_each else '',
                              bt,
                              'each' if do_each else 'total',
                              i
                          ), ))
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
                              '--save_prefix=%s_lin_%s%s '
                              '--save_fig '
                              '--to_pickle '
                              '--num_cpus=1 '
                              '--verbosity=4 '
                              '--load_if_exist'
                              % (
                                  bt,
                                  bi,
                                  num_classes,
                                  kneighbour,
                                  '--do_each' if do_each else '',
                                  bt,
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
                              '--neg_random '
                              '--save_prefix=%s_lin_%s%s_random '
                              '--save_fig '
                              '--to_pickle '
                              '--num_cpus=5 '
                              '--verbosity=4 '
                              '--load_if_exist'
                              % (
                                  bt,
                                  bi,
                                  num_classes,
                                  kneighbour,
                                  '--do_each' if do_each else '',
                                  bt,
                                  'each' if do_each else 'total',
                                  i
                              ), ))

        parallel.close()
        parallel.join()


if __name__ == '__main__':
    main()


