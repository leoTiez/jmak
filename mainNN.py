import os


def main():
    for i in range(1, 6):
        os.system('python3.8 predict.py '
                  '--bio_type=nucl '
                  '--ml_type=nn '
                  '--do_each '
                  '--save_prefix=nucl_nn_each%s '
                  '--save_fig '
                  '--to_pickle '
                  '--min_mf=.5 '
                  '--num_param_values=100 '
                  '--num_cpus=5 '
                  '--verbosity=4 '
                  '--load_if_exist'
                  % i)

        os.system('python3.8 predict.py '
                  '--bio_type=nucl '
                  '--ml_type=nn '
                  '--do_each '
                  '--save_prefix=nucl_nn_each_random%s '
                  '--save_fig '
                  '--to_pickle '
                  '--min_mf=.5 '
                  '--num_param_values=100 '
                  '--num_cpus=5 '
                  '--verbosity=4 '
                  '--load_if_exist '
                  '--neg_random'
                  % i)


if __name__ == '__main__':
    main()


