import os
from pathlib import Path
import logging
from scripts import losses
from scripts import sampling
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets
import torch
from torch.utils import tensorboard
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("blur_type", None, "Type of blurring method(dctblur or fftblur)")
flags.mark_flags_as_required(["workdir", "config","blur_type"])
#flags.DEFINE_string("initialization", "prior", "How to initialize sampling")


def main(argv):
    sample(FLAGS.config, FLAGS.workdir,FLAGS.blur_type)

def sample(config, workdir,blur_type):
    if config.device == torch.device('cpu'):
        logging.info("RUNNING ON CPU")


    # Create checkpoints directory
    #checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")

    #checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
    #Initialize model
    #model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)

    model = mutils.create_model(config)
    loaded_state = torch.load(checkpoint_meta_dir, map_location=config.device)
    model.load_state_dict(loaded_state['model'], strict=False)
    initial_step = int(loaded_state['step'])
    print('initial_step:',initial_step)

    model_evaluation_fn = mutils.get_model_fn(model, train=False)

    # Get the forward process definition
    if blur_type == 'dctblur' :
         scales = config.model.dctblur_schedule
    elif blur_type == 'fftblur':
         scales = config.model.fftblur_schedule
    else :
        raise ValueError("the blur_type is invalid")
        
    heat_forward_module = mutils.create_forward_process_from_sigmas(
        config, scales, config.device,blur_type)

    # Create directory for saving intermediate samples
    sample_dir = os.path.join(workdir, "batch samples")
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    #this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    #Path(this_sample_dir).mkdir(parents=True, exist_ok=True)

    for i in range(0,99+1):
        # Building sampling functions
        delta = config.model.sigma*1.25
        initial_sample, original_images = sampling.get_initial_sample(
            config, heat_forward_module, delta)
        
        #print('initial_sample.shape',initial_sample.shape)
        #print('initial_sample[0,0]',initial_sample[0,0])
        #initial_sample = torch.zeros((128,3,32,32))
        #initial_sample =initial_sample + torch.rand((128,3))[:,:,None,None] 
        sampling_fn = sampling.get_sampling_fn_inverse_heat(config,
                                                            initial_sample, intermediate_sample_indices=list(
                                                                range(config.model.K+1)),
                                                            delta=config.model.sigma*1.25, device=config.device)

        logging.info("create sample batch %d." % (i,))
        sample, n, intermediate_samples = sampling_fn(model_evaluation_fn)
        utils.save_png(sample_dir, sample, f"final_{i}.png")
        utils.save_png(sample_dir, initial_sample, f"init_{i}.png")
        utils.save_png(sample_dir, original_images, f"original_{i}.png")
        utils.save_tensor(sample_dir, sample, f"final{i}.np")
        logging.info("Done. ")



    return
    # Building sampling functions
    delta = config.model.sigma*1.25
    initial_sample, _ = sampling.get_initial_sample(
        config, heat_forward_module, delta)
    
    #print('initial_sample.shape',initial_sample.shape)
    #print('initial_sample[0,0]',initial_sample[0,0])
    #initial_sample = torch.zeros((128,3,32,32))
    #initial_sample =initial_sample + torch.rand((128,3))[:,:,None,None] 
    
    sampling_fn = sampling.get_sampling_fn_inverse_heat(config,
                                                        initial_sample, intermediate_sample_indices=list(
                                                            range(config.model.K+1)),
                                                        delta=config.model.sigma*1.25, device=config.device)

    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Running on {}".format(config.device))


    ### ****************************
    step = 100000
    logging.info("Sampling...")
    # ema.store(model.parameters())
    # ema.copy_to(model.parameters())
    sample, n, intermediate_samples = sampling_fn(model_evaluation_fn)


    # ema.restore(model.parameters())
    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
    #utils.save_tensor(this_sample_dir, sample, "final.np") ###
    utils.save_png(this_sample_dir, sample, "final1.png")
    if initial_sample != None:
        utils.save_png(this_sample_dir, initial_sample, "init1.png")
    #utils.save_gif(this_sample_dir, intermediate_samples) ###
    #utils.save_video(this_sample_dir, intermediate_samples) ###

    for i in range(3):
      sample, n, intermediate_samples = sampling_fn(model_evaluation_fn)
      utils.save_png(this_sample_dir, sample, f"for_final{i}.png")
      utils.save_png(this_sample_dir, initial_sample, "for_init{i}.png")


    ### ****************************
    return ###


if __name__ == "__main__":
    app.run(main)
