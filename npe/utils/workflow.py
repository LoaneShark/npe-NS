def save_state(path, model, optimizer, scheduler):
    import torch
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state_dict, path)
    
def load_state(path, model, optimizer=None, scheduler=None):
    import torch
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None: optimizer.load_state_dict(state_dict['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(state_dict['scheduler'])
    return
    
def initialize(model_type, optimizer_type, scheduler_type, 
               model_kwargs={}, optimizer_kwargs={}, scheduler_kwargs={}, device='cpu'):
    model = model_type(**model_kwargs).to(device)
    optimizer = optimizer_type(model.parameters(), **optimizer_kwargs)
    scheduler = scheduler_type(optimizer, **scheduler_kwargs)
    return model, optimizer, scheduler

def get_current_epoch_value(v, i_epoch):
    if type(v) not in {list, tuple}:
        return v
    for ep, cv in v:
        if ep is None or i_epoch < ep:
            return cv

def train(args):

    import os
    import sys
    import warnings
    from glob import glob
    from datetime import datetime
    from matplotlib import pyplot as plt
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from . import loss
    from .data import DatasetManager
    from .train import VAEEvaluationAgent, get_graph_arguments
    
    # FOLLOWING ARE GLOBAL VARIABLES
    
    run_title = args.run_title
    resume_title = args.resume_title
    resume_epochs = args.resume_epochs
    add_epochs = args.add_epochs
    epochs_per_latent_plot = args.epochs_per_latent_plot
    epochs_per_checkpoint = args.epochs_per_checkpoint
    optimizer_override = args.optimizer_override
    scheduler_override = args.scheduler_override

    batch_size_train = args.batch_size_train
    batch_size_val = args.batch_size_val
    batches_per_summary = args.batches_per_summary

    lr = args.lr
    wd = args.wd
    gamma = args.gamma
    loss_kwargs = args.loss_kwargs
    diagnosis_kwargs = args.diagnosis_kwargs

    model_type = args.model_type
    model_kwargs = args.model_kwargs
    optimizer_type = args.optimizer_type
    scheduler_type = args.scheduler_type
    loss_fn = args.loss_fn
    diagnosis_fn = args.diagnosis_fn
    training_device = args.training_device
    training_seed = args.training_seed

    npoints_for_latent_plot = args.npoints_for_latent_plot
    npoints_for_generation = args.npoints_for_generation
    show_plot = args.show_plot

    dataset_recipe_from_file = args.dataset_recipe_from_file
    dataset_recipe_save_file = args.dataset_recipe_save_file
    dataset_rootdir = args.dataset_rootdir
    dataset_filenames = args.dataset_filenames
    dataset_type = args.dataset_type
    dataset_n_ppe = getattr(args, 'dataset_n_ppe', None)
    dataset_norm_fac = args.dataset_norm_fac
    dataset_sample_size = args.dataset_sample_size
    dataset_subset_split = args.dataset_subset_split
    dataset_seed = args.dataset_seed

    output_rootdir = args.output_rootdir
    
    
    if type(dataset_type) == str: dataset_type = eval(dataset_type)
    if type(model_type) == str: model_type = eval(model_type)
    if type(loss_fn) == str: loss_fn = eval(loss_fn)


    log_str = "{} Training program started with title '{}'".format(datetime.now().strftime('%H:%M:%S'), run_title)
    print(log_str); sys.stdout.flush()


    tbdir = os.path.join(output_rootdir, "tensorboard/{}/".format(run_title))
    cpdir = os.path.join(output_rootdir, "checkpoints/{}/".format(run_title))
    if not os.path.exists(tbdir): os.makedirs(tbdir)
    if not os.path.exists(cpdir): os.makedirs(cpdir)
    log_str = "Tensorboard directory: {}".format(os.path.abspath(tbdir))
    log_str += "\nCheckpoint directory: {}".format(os.path.abspath(cpdir))
    print(log_str); print(); sys.stdout.flush()


    save_recipe = False
    if dataset_recipe_from_file is not None:
        log_str = "{} Reproducing dataset and split, using recipe from {}...".format(
                                                datetime.now().strftime('%H:%M:%S'), dataset_recipe_from_file)
        print(log_str); print(); sys.stdout.flush()
        dataset_and_split = DatasetManager.from_recipe(dataset_recipe_from_file, root_dir=dataset_rootdir)
    else:
        if dataset_seed is None:
            dataset_seed = np.random.randint(np.iinfo(np.uint32).max)
        log_str = "{} Preparing dataset and split from scratch, using seed = {}...".format(
                                                datetime.now().strftime('%H:%M:%S'), dataset_seed)
        print(log_str); print(); sys.stdout.flush()
        dataset_and_split = DatasetManager(
                    dataset_filenames, root_dir=dataset_rootdir, 
                    sample_size=dataset_sample_size, subset_split=dataset_subset_split, random_state=dataset_seed, 
                    dataset_type=dataset_type, dataset_kwargs=dict(n_ppe=dataset_n_ppe, norm_fac=dataset_norm_fac))
        if dataset_recipe_save_file is not None:
            save_recipe = True
    log_str = "{} Loaded dataset, with".format(datetime.now().strftime('%H:%M:%S'))
    log_str += "\n{}".format(dataset_and_split.brief_recipe)
    print(log_str); print(); sys.stdout.flush()
    if save_recipe:
        assert not os.path.exist(dataset_recipe_save_file)
        dataset_and_split.save_recipe(dataset_recipe_save_file)
        log_str = "{} Saved dataset recipe to {}".format(
            datetime.now().strftime('%H:%M:%S'), dataset_recipe_save_file)
        print(log_str); print(); sys.stdout.flush()

    if training_device is None:
        training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if training_seed is None:
        training_seed = torch.seed()
    else:
        torch.manual_seed(training_seed)
    log_str = "{} Initializing... ".format(datetime.now().strftime('%H:%M:%S'))
    log_str += "Training will use device = {}, seed = {}...".format(str(training_device), training_seed)
    print(log_str); print(); sys.stdout.flush()
    torch.manual_seed(training_seed)
    dataset_train, dataset_val, dataset_test = dataset_and_split.subsets
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False)
    model, optimizer, scheduler = initialize(
                                    model_type, optimizer_type, scheduler_type,
                                    model_kwargs=model_kwargs,
                                    optimizer_kwargs={'lr': lr, 'weight_decay': wd},
                                    scheduler_kwargs={'gamma': gamma}, device=training_device)
    resume_cpdir = os.path.join(output_rootdir, "checkpoints/{}/".format(resume_title))
    if resume_epochs < 0:
        filenames = glob(os.path.join(resume_cpdir, "{}*-epochs.pt".format(run_title)))
        if len(filenames) > 0:
            eps = [int(fn.strip("-epochs.pt").rpartition("_")[-1]) for fn in filenames]
            resume_epochs = sorted(eps)[resume_epochs]
        else:
            resume_epochs = 0
    if resume_title != run_title or resume_epochs != 0:
        resume_filename = "{}_{}-epochs.pt".format(resume_title, resume_epochs)
        resume_filepath = os.path.join(resume_cpdir, resume_filename)
        load_state(resume_filepath, model, 
                   optimizer=optimizer if not optimizer_override else None, 
                   scheduler=scheduler if not scheduler_override else None, )
        log_str = "{} Loaded previous run titiled '{}' after {} epochs, from {}".format(
                                                                datetime.now().strftime('%H:%M:%S'), 
                                                                resume_title, resume_epochs, 
                                                                os.path.abspath(resume_filepath))
    else:
        log_str = "{} Initialized as a new run".format(
                        datetime.now().strftime('%H:%M:%S'))
    print(log_str); sys.stdout.flush()

    log_str = "Model architecture: "
    log_str += "\n" + str(model)
    log_str += "\nModel kwargs: {}".format(model_kwargs)
    log_str += "\nOptimizer type: {}".format(optimizer.__class__.__name__)
    log_str += "\nScheduler type: {}".format(scheduler.__class__.__name__)
    print(log_str); print(); sys.stdout.flush()

    tb_writer = SummaryWriter(tbdir, purge_step=resume_epochs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.jit.TracerWarning)
        model.eval()
        tb_writer.add_graph(*get_graph_arguments(model, next(iter(data_loader_train))))
        tb_writer.flush()


    log_str = "{} Entering training loop...".format(datetime.now().strftime('%H:%M:%S'))
    print(log_str); print(); sys.stdout.flush()

    for i_epoch in range(resume_epochs, resume_epochs+add_epochs):

        current_loss_kwargs = dict()
        for k,v in loss_kwargs.items():
            current_loss_kwargs[k] = get_current_epoch_value(v, i_epoch)
            
        current_epochs_per_latent_plot = get_current_epoch_value(epochs_per_latent_plot, i_epoch)
        current_epochs_per_checkpoint = get_current_epoch_value(epochs_per_checkpoint, i_epoch)

        agent = VAEEvaluationAgent(
                            model, loss_fn, diagnosis_fn, current_loss_kwargs, diagnosis_kwargs,
                            distrib_thin_fac=npoints_for_latent_plot/len(data_loader_val.dataset),
                            sample_size=npoints_for_generation)

        hparams = dict(
            lr=scheduler.get_last_lr()[0], 
            weight_decay=optimizer.defaults['weight_decay'],
            batch_size=data_loader_train.batch_size,
        )
        # hparams.update(agent.loss_fn_kwargs)
        for k, v in agent.loss_fn_kwargs.items():
            try: float(v)
            except TypeError: pass
            else: hparams[k] = v
        log_str = "{} Epoch {}".format(datetime.now().strftime('%H:%M:%S'), i_epoch+1)
        log_str += "\nHparams = {}".format(hparams)
        print(log_str); sys.stdout.flush()

        for k, v in hparams.items():
            tb_writer.add_scalar(f'hparams/{k}', v, i_epoch)

        if type(batches_per_summary) == float:
            batches_per_summary = int(batches_per_summary * len(data_loader_train))

        agent.train()
        for i_batch, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            loss = agent.add_batch(data_batch)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            if batches_per_summary > 0 and (i_batch+1) % batches_per_summary == 0:
                loss_recent = agent.pop_recent_loss()
                log_str = "  {:0.0f}%, interval loss = {:0.3e}".format(
                                    100. * (i_batch+1) / len(data_loader_train), loss_recent)
                print(log_str); sys.stdout.flush()
        scheduler.step()

        loss_train = agent.report_loss()
        tb_writer.add_scalar('loss/train', loss_train, i_epoch+1)
        tb_writer.flush()

        agent.eval()
        for i_batch, data_batch in enumerate(data_loader_val):
            agent.add_batch(data_batch)
        loss_val = agent.report_loss()
        metrics = agent.report_metrics()

        log_str = "Train / Val loss = {:0.3e} / {:0.3e}".format(loss_train, loss_val)
        log_str += "\nVal" + ",".join(" {} = {:0.3e}".format(k, v) for k, v in metrics.items())
        print(log_str); print(); sys.stdout.flush()
        tb_writer.add_scalar('loss/val', loss_val, i_epoch+1)
        for k, v in metrics.items():
            tb_writer.add_scalar(f'metrics/{k}', v, i_epoch+1)
        tb_writer.flush()

        if (current_epochs_per_checkpoint > 0 and (i_epoch+1) % current_epochs_per_checkpoint == 0) \
                or i_epoch + 1 == resume_epochs + add_epochs:
            cp_filename = "{}_{}-epochs.pt".format(run_title, i_epoch+1)
            cp_filepath = os.path.join(cpdir, cp_filename)
            save_state(cp_filepath, model, optimizer, scheduler)
            log_str = "{} Saved checkpoint".format(datetime.now().strftime('%H:%M:%S'))
            print(log_str); print(); sys.stdout.flush()

        if current_epochs_per_latent_plot > 0 and (i_epoch+1) % current_epochs_per_latent_plot == 0:
            log_str = "{} Generating plots...".format(datetime.now().strftime('%H:%M:%S'))
            print(log_str); print(); sys.stdout.flush()
            fig = agent.report_latent_distrib()
            if show_plot: plt.show()
            tb_writer.add_figure('latent/distrib', fig, i_epoch+1)
            fig = agent.report_latent_sample()
            if show_plot: plt.show()
            tb_writer.add_figure('latent/sample', fig, i_epoch+1)
            tb_writer.flush()

    tb_writer.flush()
    tb_writer.close()
    log_str = "{} All epochs completed".format(datetime.now().strftime('%H:%M:%S'), i_epoch + 1)
    print(log_str); sys.stdout.flush()