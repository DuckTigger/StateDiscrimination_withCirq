def plot_loss_fn(folder, n=6, no_noise=None, cutoff=7500) -> plt.figure:
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
    for e in tf.train.summary_iterator(os.path.join(path, folder, event_file[0])):
        for v in e.summary.value:
            if v.tag == 'Summaries/training_loss':
                if e.step <= cutoff:
                    steps.append(e.step)
                    loss.append(v.simple_value)

    param_file = os.path.join(folder, 'saved_params.json')
    with open(param_file, 'r') as f:
        params = json.load(f)
    if params['noise_on'] == "False":
        noise_level = 0
    else:
        noise_level = params['noise_prob']
    tot_cost = float(params['cost_error']) + float(params['cost_inconclusive'])
    loss = np.array(loss)
    loss /= tot_cost
    loss_ma = moving_average(loss, n)
    step_ma = moving_average(steps, n)
    ax.plot(list(step_ma), list(loss_ma), label=noise_level)
    ax.legend()
    plt.show()
