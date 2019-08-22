import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import itertools
import fnmatch
import re
import tensorflow as tf
import cirq
import seaborn as sns
import sys
import pickle as pkl
from PIL import Image, ImageDraw, ImageFont
if sys.platform.startswith('win'):
    code_path = "C:\\Users\\Andrew Patterson\\Google Drive\\PhD\\First Year\\Untitled Folder\\cirq_state_discrimination"
else:
    code_path = "/home/zcapga1/Scratch/state_discrimination/code/"
sys.path.append(code_path)
print(tf.__version__)
print(cirq.__version__)


def create_args(path: str) -> str:
    args = " --create_outputs --restore_loc=\"{}\" ".format(path)

    if re.search('tf', path):
        args = args + " --use_tf"

    with open(os.path.join(path, 'saved_params.json')) as f:
        params = json.load(f)

    args = args + ' --max_epoch=20 --batch_size=20 --n_output_states=5'
    data_args = ['prop_a', 'b_const', 'a_const', 'mu_a', 'mu_b', 'sigma_a', 'sigma_b']
    for a in data_args:
        args = args + ' --{}={}'.format(a, str(params[a]))

    dicts = (params['gate_dict'], params['gate_dict_0'], params['gate_dict_1'])
    dict_path = os.path.join(path, 'temp_dicts.pkl')
    with open(dict_path, 'wb') as f:
        pkl.dump(dicts, f, pkl.HIGHEST_PROTOCOL)

    args = args + " --dicts=\"{}\"".format(dict_path)

    other_args = ['cost_error', 'cost_incon', 'no_qubits', 'noise_on',
                  'noise_prob', 'learning_rate', 'beta1', 'beta2']
    for a in other_args:
        args = args + ' --{}={}'.format(a, str(params[a]))
    return args


def generate_output_file(directory: str) -> None:
    folder_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for restore_path in folder_list:
        full_path = os.path.join(directory, restore_path)
        if os.path.exists(os.path.join(full_path, 'saved_params.json')):
            args = create_args(full_path)

            run_file = os.path.join(code_path, 'run_training.py')
            os.system("python \"" + run_file + "\"" + args)


def label_plot(directory: str) -> None:
    folder_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for path in folder_list:
        plot_path = os.path.join(directory, path, 'outputs')
        if os.path.exists(os.path.join(directory, path, 'saved_params.json')):
            with open(os.path.join(directory, path, 'saved_params.json')) as f:
                param_dict = json.load(f)

            param_list = ['cost_error', 'cost_incon', 'noise_on', 'noise_prob', 'mu_a', 'sigma_a', 'job_name']
            if not param_dict['b_const']:
                param_list.extend(['mu_b', 'sigma_b'])

            msg = ""
            for i, param in enumerate(param_list):
                msg = msg + '{}: {} '.format(param, param_dict[param])
                if i % 3 == 0:
                    msg = msg + "\n"

            font_path = os.path.join("C:\\Users\\Andrew Patterson\\Documents\\PhD\\fonts\\fonts\\ofl\\sourcecodepro",
                                     'SourceCodePro-Light.ttf')
            font = ImageFont.truetype(font_path, size=10)
            plot = Image.open(os.path.join(plot_path, 'bar_graph.png'))

            cropped = plot.crop((100, 35, 570, 435))
            draw = ImageDraw.Draw(cropped)
            (x, y) = (5, 5)
            colour = 'rgb(0, 0, 0)'
            draw.text((x, y), msg, fill=colour, font=font)
            save_path = os.path.join(plot_path, 'plot_labeled.png')
            cropped.save(save_path)
            os.system("convert \"{}\" -fuzz 2% -transparent white \"{}\"".format(save_path, save_path))


def nwise(iterable, n):
    ts = itertools.tee(iterable, n)
    for c, t in enumerate(ts):
        next(itertools.islice(t, c, c), None)
    return zip(*ts)


def moving_average(iterable, n):
    yield from (sum(x) / n for x in nwise(iterable, n))


def plot_loss_fn(folder, n=6, cutoff=100000):
    """
    Input: Takes a list of folders where models are strored and returns a graph of convergences plotted together
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss, Moving average = {}'.format(n))
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    event_file = fnmatch.filter(files, 'events*')
    steps = []
    loss = []
    for e in tf.compat.v1.train.summary_iterator(os.path.join(folder, event_file[0])):
        for v in e.summary.value:
            if v.tag == 'Training loss':
                if e.step <= cutoff:
                    steps.append(e.step)
                    val = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)[0]
                    loss.append(val)

    param_file = os.path.join(folder, 'saved_params.json')
    with open(param_file, 'r') as f:
        params = json.load(f)
    if params['noise_on'] == "False":
        noise_level = 0
    else:
        noise_level = params['noise_prob']
    tot_cost = float(params['cost_error']) + float(params['cost_incon'])
    loss = np.array(loss)
    loss = loss / tot_cost
    np.save(os.path.join(folder, 'outputs', 'loss_fn.npy'), (steps, loss))
    loss_ma = moving_average(loss, n)
    step_ma = moving_average(steps, n)
    ax.plot(list(step_ma), list(loss_ma), label=noise_level)
    ax.legend()
    plt.savefig(os.path.join(folder, 'outputs', 'loss_fn.png'))


def save_loss_fns(directory: str):
    folder_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for path in folder_list:
        if os.path.exists(os.path.join(directory, path, 'saved_params.json')):
            plot_loss_fn(os.path.join(directory, path))


def create_data_frame(directory: str):
    df = pd.DataFrame()
    folder_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for f in folder_list:
        path = os.path.join(directory, f)
        probs_path = os.path.join(path, 'outputs', 'probs.npy')
        if os.path.isfile(probs_path):
            with open(os.path.join(path, 'saved_params.json')) as js:
                params = json.load(js)
            probs = np.load(probs_path)

            p_error = np.average([probs[0][1], probs[1][0]])
            p_inc = np.average([probs[0][2], probs[1][2]])
            p_suc = np.average([probs[0][0], probs[1][1]])

            loss = p_error + p_inc

            df = df.append({'P_err': p_error, 'P_inc': p_inc, 'P_suc': p_suc, 'cost_err': params['cost_error'],
                            'cost_inc': params['cost_incon'], 'noise_on': params['noise_on'],
                            'noise_prob': params['noise_prob'], 'learning_rate': params['learning_rate'],
                            'loss': loss, 'folder': str(f)},
                           ignore_index=True)

    numeric = ['cost_err', 'cost_inc', 'noise_prob', 'learning_rate']
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce', axis=1)
    if not os.path.exists(os.path.join(directory, 'output')):
        os.mkdir(os.path.join(directory, 'output'))
    df.to_pickle(os.path.join(directory, 'output', 'dataframe.pkl'))
    f, ax = plt.subplots(figsize=(7,7))
    ax.set(xscale='log', title='Loss with log noise')
    df.plot(x='noise_prob', y='loss', ax=ax, style='k--')
    plt.savefig(os.path.join(directory, 'output', 'loss_noise.png'))
    return df


def pre_trained_new_noise_level(restore_directory: str, noise_levels: list):
    noise_levels_path = os.path.join(restore_directory, 'outputs_with_noise_levels')
    if not os.path.exists(noise_levels_path):
        os.mkdir(noise_levels_path)

    for noise in noise_levels:
        args_str = create_args(restore_directory)
        args_str = re.sub('(--noise_prob=\d*.\d*)', "--noise_prob={}".format(noise), args_str)
        args_str = re.sub('(--noise_on=\w*|\"\w*\")', "--noise_on=True", args_str)

        noise_dir = os.path.join(noise_levels_path, re.sub('\.', '_', str(noise)))
        args_str = args_str + ' --output_loc=\"{}\"'.format(noise_dir)
        run_file = os.path.join(code_path, 'run_training.py')
        os.system("python \"" + run_file + "\"" + args_str)


def pre_trained_multiple(directory: str, noise_levels: list):
    folder_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for path in folder_list:
        # if (not os.path.exists(os.path.join(directory, path, 'outputs_with_noise_levels')))
        # and os.path.exists(os.path.join(directory, path, 'saved_params.json')):
        if os.path.exists(os.path.join(directory, path, 'saved_params.json')):
            pre_trained_new_noise_level(os.path.join(directory, path), noise_levels)


def create_noise_levels_df(directory: str) -> pd.DataFrame:
    df = pd.DataFrame()
    params_path = os.path.join(directory, 'saved_params.json')
    with open(params_path) as js:
        params = json.load(js)
    noise_folder = os.path.join(directory, 'outputs_with_noise_levels')
    noise_paths = [f for f in os.listdir(noise_folder) if os.path.isdir(os.path.join(noise_folder, f))]
    for p in noise_paths:
        path = os.path.join(noise_folder, p)
        probs = np.load(os.path.join(path, 'probs.npy'))

        p_error = np.average([probs[0][1], probs[1][0]])
        p_inc = np.average([probs[0][2], probs[1][2]])
        p_suc = np.average([probs[0][0], probs[1][1]])

        loss = p_error + p_inc

        if params['noise_on'] == 'True':
            training_noise = params['noise_prob']
        else:
            training_noise = 0

        df = df.append({'P_err': p_error, 'P_inc': p_inc, 'cost_err': params['cost_error'],
                        'cost_inc': params['cost_incon'], 'training_noise': training_noise,
                        'noise_prob': re.sub('_', '.', p), 'loss': loss},
                       ignore_index=True)
    return df


def log_on_one_plot(directory: str):
    df = pd.DataFrame()
    folder_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    for f in folder_list:
        if os.path.exists(os.path.join(directory, f, 'saved_params.json')):
            df = df.append(create_noise_levels_df(os.path.join(directory, f)))
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set(xscale='log', title='Loss with log noise')

    numeric = ['cost_err', 'cost_inc', 'noise_prob', 'loss']
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce', axis=1)
    df['training_noise'] = df['training_noise'].astype('category')
    sns.scatterplot(x='noise_prob', y='loss', style='training_noise', data=df)

    if not os.path.exists(os.path.join(directory, 'output')):
        os.mkdir(os.path.join(directory, 'output'))
    ax.legend()
    plt.savefig(os.path.join(directory, 'output', 'log_noise_plot.png'))
    df.to_pickle(os.path.join(directory, 'output', 'noise_df.pkl'))


def run_all_on_folders(folders: list, noise_levels: list):
    for folder in folders:
        generate_output_file(folder)
        if sys.platform.startswith('win'):
            label_plot(folder)

        save_loss_fns(folder)
        create_data_frame(folder)
        pre_trained_multiple(folder, noise_levels)
        log_on_one_plot(folder)


if __name__ == '__main__':
    #
    if sys.platform.startswith('win'):
        run_folder = "C:\\Users\\Andrew Patterson\\Documents\\PhD\\cirq_state_discrimination\\checkpoints\\myriad_data\\"
    else:
        run_folder = "/home/zcapga1/Scratch/state_discrimination/training_out/"

    if len(sys.argv) == 1:
        runs = ['tf_noise_off', 'tf_old_dicts_noise_off', 'cirq_training_noise_off', 'tf_noise_array',
                'tf_old_dicts_noise_array', 'tf_old_dicts_noise_off']
    else:
        runs = sys.argv[1:]
    folders_to_run = [os.path.join(run_folder, f) for f in runs]
    noise_levels = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25]
    run_all_on_folders(folders_to_run, noise_levels)
