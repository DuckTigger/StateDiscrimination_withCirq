import os
from argparse import ArgumentParser
from cirq_backend.train_model_cirq import TrainModelCirq
from tensorflow_backend.train_model_tf import TrainModelTF
import pickle


def main():
    parser = ArgumentParser(description='Runs training defined by arguments passed here.')
    parser.add_argument('--use_tf', action='store_true',
                        help='Use the tensorflow quantum simulator, otherwise use Cirq.')
    parser.add_argument('--cost_error', metavar='-ce', type=float, nargs='?', default=40.,
                        help='Weight of incorrect results in the loss function.')
    parser.add_argument('--cost_incon', metavar='-ci', type=float, nargs='?', default=40.,
                        help='Weight of inconclusive results in the loss function.')
    parser.add_argument('--file_loc', metavar='-f', type=str, nargs='?', default=None,
                        help='File name of pre-generated data file. Must be saved in data folder. '
                             'Overrides other data generation arguments')
    parser.add_argument('--restore_loc', type=str, nargs='?', default=None,
                        help='The location of the checkpoints to restore')
    parser.add_argument('--create_outputs', action='store_true',
                        help='Should be used in conjunction with the restore_loc arg. '
                             'If True will create outputs. If not used training will continue.')
    parser.add_argument('--output_loc', type=str, nargs='?', default=None,
                        help='If the outputs are to be saved to a new location, pass it to this argument.')
    parser.add_argument('--batch_size', metavar='-bs', type=int, nargs='?', default=20,
                        help='Number of states in a single batch.')
    parser.add_argument('--max_epoch', metavar='-me', type=int, nargs='?', default=2500,
                        help='Number of epochs to run.')
    parser.add_argument('--no_qubits', metavar='-nq', type=int, nargs='?', default=4,
                        help='Number of qubits in the simulation.')
    parser.add_argument('--noise_on', nargs='?', default=False,
                        help='Whether to run the simulations with noise on.')
    parser.add_argument('--noise_prob', metavar='-p', type=float, nargs='?', default=0.05,
                        help='Probability of noise.')
    parser.add_argument('--learning_rate', metavar='-lr', type=float, nargs='?', default=0.001,
                        help='Learning rate of the optimizer')
    parser.add_argument('--beta1', metavar='-b1', type=float, nargs='?', default=0.9,
                        help='beta1 of the optimizer.')
    parser.add_argument('--beta2', metavar='-b2', type=float, nargs='?', default=0.999,
                        help='beta2 of the optimizer.')
    parser.add_argument('--job_name', metavar='-j', type=str, nargs='?', default=None,
                        help='The job name, if given creates a subdirectory for this job.')
    parser.add_argument('--dicts', nargs='?', default=None,
                        help='A Tuple of the three gate dictionaries defining the circuit.')
    parser.add_argument('--n_output_states', nargs='?', default=250,
                        help='The number of states to use when creating the output')
    parser.add_argument('--full_dicts', nargs='?', default=False,
                        help='If True, will use the full dictionaries')

    parser.add_argument('--prop_a', type=float, nargs='?', default=1/3,
                        help='Proportion of a type states.')
    parser.add_argument('--b_const', nargs='?', default=True,
                        help='Whether to use constant b values.')
    parser.add_argument('--a_const', nargs='?', default=False,
                        help='Whether to use constant a states, default False.')
    parser.add_argument('--mu_a', type=float, nargs='?', default=0.5,
                        help='Mean of a distribution.')
    parser.add_argument('--sigma_a', type=float, nargs='?', default=0.15,
                        help='Std dev of a distribution.')
    parser.add_argument('--mu_b', type=float, nargs='?', default=0.75,
                        help='Mean of b distribution.')
    parser.add_argument('--sigma_b', type=float, nargs='?', default=0.125,
                        help='Std dev of b distribution.')
    parser.add_argument('--random_states', nargs='?', default=False,
                        help='Whether to use Harr random staets instead of a and b distributions.'
                             'Passing 2 will fix a and b from the uniform distribution and train on copies of that.')

    args = parser.parse_args()

    if type(args.dicts) == str:
        with open(args.dicts, 'rb') as f:
            args.dicts = pickle.load(f)

    use_tf = args.use_tf
    create_outputs = args.create_outputs
    output_loc = args.output_loc
    n_output_states = args.n_output_states
    del args.output_loc
    del args.use_tf
    del args.create_outputs
    del args.n_output_states

    if args.noise_prob == 0:
        args.noise_on = False

    if use_tf:
        trainer = TrainModelTF(**vars(args))
    else:
        trainer = TrainModelCirq(**vars(args))

    if create_outputs:
        if output_loc is None:
            trainer.create_outputs(os.path.join(trainer.restore_loc, 'outputs'), n_states=n_output_states)
        else:
            trainer.create_outputs(output_loc, n_states=n_output_states)
    else:
        trainer.save_inputs(args)
        trainer.train()


if __name__ == '__main__':
    main()
