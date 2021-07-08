import itertools
import os
import sys


total_rounds = 300
hyperparameter_list = {
    'clients-per-round': [2, 4],
    'learning-rate': [.01, .05, .001],
    'epochs': [1, 3]
}
fed_type = sys.argv[1]
available_gpu = sys.argv[2]
data_location = '../../dsets/opportunity/augmented/icmla'


if __name__ == '__main__':
    keys, values = zip(*hyperparameter_list.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, hparams in enumerate(permutations_dicts):
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{available_gpu}'
        process_string = f'python train_federated_{fed_type}.py {data_location} logs/{fed_type}_{i}' + \
                         f' --rounds {int(total_rounds / hparams["epochs"])}' + \
                         ' --checkpoint-rate 25'
        for hp_name, hp_value in hparams.items():
            process_string += f' --{hp_name} {hp_value}'
        print(f'running {process_string}')
        os.system(process_string)
