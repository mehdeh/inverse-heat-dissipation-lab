"""All functions and modules related to model definition."""
import torch
import torch.nn as nn
import logging
import numpy as np
from model_code.unet import UNetModel
from model_code import torch_dct
from torch.fft import fft2,ifft2

class DCTBlur(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas**2/2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')

class FFTBlur(nn.Module):
    def __init__(self,blur_sigmas, image_size, device):#batch_size,channel,image_width,image_height
        super(FFTBlur,self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        self.image_width = image_size
        self.image_height = image_size
        self.device = device
        self.H_bar = self.create_kernel(self.image_width,self.image_height)
        self.H_tilde = self.matrix_flip(self.H_bar)
        self.fft2_H_tilde = fft2(self.H_tilde)
        #self.fft2_H_tilde_n = torch.zeros((batch_size,channel,image_width,image_height),dtype=torch.complex64)
        #self.fft2_H_tilde_n[:,:] = self.fft2_H_tilde

    def destination_center(self,width,height):
        (cw1,ch1) = (width//2 +1 , height//2 +1) # center of destination
        return (cw1,ch1)

    def source_center(self,width,height):
        (cw1,ch1) = self.destination_center(width,height)
        (cw2,ch2) = (width-cw1 , height-ch1) # center of source
        return (cw2,ch2)

    def create_kernel(self,width,height):
        H =1/8*torch.tensor([[0,1,0],
                             [1,4,1],
                             [0,1,0]]).to(self.device)

        # H =1/1*np.array([[1,1,1],
        #                 [1,1,1],
        #                 [1,1,1]]).to(self.device)
        (cw2,ch2) = self.source_center(width,height)
        H_bar = torch.zeros((width,height)).to(self.device)
        H_bar[cw2-1:cw2+2,ch2-1:ch2+2] = H
        return H_bar

    def matrix_flip(self,H_bar):
        (w,h) = H_bar.shape
        (cw1,ch1) = self.destination_center(w,h)
        (cw2,ch2) = self.source_center(w,h)

        H_tilde = torch.zeros_like(H_bar)
        H_tilde[0:cw1,0:ch1]= H_bar[cw2:w,ch2:h]
        H_tilde[cw1:w,0:ch1]= H_bar[0:cw2,ch2:h]
        H_tilde[0:cw1,ch1:h]= H_bar[cw2:w,0:ch2]
        H_tilde[cw1:w,ch1:h]= H_bar[0:cw2,0:ch2]
        return H_tilde

    def forward(self,x,fwd_steps):
        #imgs_tmp = min_max_norm_batch_0_1(imgs)
        #t1 = 2*torch.exp(fwd_steps/15)-1.99
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas  
        fft2_H_tilde_t = self.fft2_H_tilde[None,None,:,:]**t
        return ifft2(fft2_H_tilde_t*fft2(x)).real#.to(torch.uint8)     
           
def create_forward_process_from_sigmas(config, sigmas, device,blur_type):
    if blur_type == 'dctblur' :
        forward_process_module = DCTBlur(sigmas, config.data.image_size, device)
    elif blur_type == 'fftblur':
        forward_process_module = FFTBlur(sigmas, config.data.image_size, device)
    else :
        raise ValueError("the blur_type is invalid")
    return forward_process_module

"""Utilities related to log-likelihood evaluation"""


def KL(dists, sigma0, sigma1, dim):
    # Calculates a matrix of KL divergences between spherical Gaussian distributions
    # with distances dists between their centers, where dists is a matrix
    return 0.5 * ((sigma0**2/sigma1**2)*dim + (dists)**2/sigma1**2 - dim + 2*dim*np.log(sigma1/sigma0))


def L_K_upperbound(K, trainloader, testloader, blur_module, sigma_inf, sigma_prior, dim,
                   train_size, test_size, device='cpu'):

    # Calculates the upper bound for the term E_q[KL[q(x_K|x_0)|p(x_K)]]
    # in a memory-efficient way, that is, calculates the distances between
    # test and training data points in batches, and uses those distances to calculate
    # the upper bound

    KL_div_upper_bound = torch.zeros(test_size, device=device)
    testdata_count = 0
    count = 0
    for testbatch in testloader:
        logging.info("Batch {}".format(count))
        count += 1
        blur_fwd_steps_test = [K] * len(testbatch[0])
        testbatch = blur_module(testbatch[0].to(
            device), blur_fwd_steps_test).reshape(len(testbatch[0]), -1)
        dists = torch.zeros(train_size, len(testbatch), device=device)
        traindata_count = 0
    # Get distances between the test batch and training data
        for trainbatch in trainloader:
            blur_fwd_steps_train = [K] * len(trainbatch[0])
            trainbatch = blur_module(trainbatch[0].to(
                device), blur_fwd_steps_train).reshape(len(trainbatch[0]), -1)
            dists[traindata_count:traindata_count +
                  len(trainbatch), :] = torch.cdist(trainbatch, testbatch)
            traindata_count += len(trainbatch)
        # Calculate the upper bounds on the KL divergence for each test batch element
        kl_divs = KL(dists, sigma_inf, sigma_prior, testbatch.shape[-1])
        inference_entropy = dim*0.5 * \
            torch.log(
                2*np.pi*torch.exp(torch.tensor([1]))*sigma_inf**2).to(device)
        cross_entropies = kl_divs + inference_entropy
        # log-sum-exp trick
        log_phi = -kl_divs - torch.logsumexp(-kl_divs, 0)[None, :]
        phi = torch.exp(log_phi)
        KL_div_upper_bound_batch = -inference_entropy + \
            (phi * (cross_entropies + log_phi + np.log(train_size))).sum(0)
        KL_div_upper_bound[testdata_count:testdata_count +
                           len(testbatch)] = KL_div_upper_bound_batch
        testdata_count += len(testbatch)
    return KL_div_upper_bound


def neg_ELBO(config, trainloader, testloader, blur_module, sigma, delta, image_size,
             train_size, test_size, model, device='cpu', num_epochs=10):
    """Estimates the terms in the negative evidence lower bound for the model
    num_epochs: Used for the estimation of terms L_k: How many epochs through these?"""

    logging.info("Calculating the upper bound for L_K...")
    L_K_upbound = L_K_upperbound(config.model.K, trainloader, testloader, blur_module, sigma,
                               delta, image_size**2, train_size, test_size, device)
    logging.info("... done! Value {}, len {}".format(
      L_K_upbound, len(L_K_upbound)))

    model_fn = get_model_fn(model, train=False)
    num_dims = image_size**2 * next(iter(trainloader))[0].shape[1]

    # There are K - 1 intermediate scales
    L_others = torch.zeros(config.model.K, device=device)
    mse_losses = torch.zeros(config.model.K, device=device)

    logging.info("Calculating the other terms...")
    with torch.no_grad():
        # Go through the set a few times for more accuracy, not just once
        for i in range(num_epochs):
            for testbatch in testloader:
                testbatch = testbatch[0].to(device).float()
                batch_size = len(testbatch)
                fwd_steps = torch.randint(
                    1, config.model.K, (batch_size,), device=device)
                blurred_batch = blur_module(testbatch, fwd_steps).float()
                less_blurred_batch = blur_module(testbatch, fwd_steps-1).float()
                noise = torch.randn_like(blurred_batch) * sigma
                perturbed_data = noise + blurred_batch
                diff = model_fn(perturbed_data, fwd_steps)
                prediction = perturbed_data + diff
                mse_loss = ((less_blurred_batch - prediction)
                            ** 2).sum((1, 2, 3))
                loss = mse_loss / delta**2
                loss += 2*num_dims*np.log(delta/sigma)
                loss += sigma**2/delta**2*num_dims
                loss -= num_dims
                loss /= 2
                # Normalize so that the significance of these terms matches with L_K and L_0
                # This way, we only go through once for each data point
                loss *= (config.model.K-1)
                mse_loss *= (config.model.K-1)
                L_others.scatter_add_(0, fwd_steps, loss)
                mse_losses.scatter_add_(0, fwd_steps, mse_loss)

        L_others = L_others / (test_size*num_epochs)
        mse_losses = mse_losses / (test_size*num_epochs)

        # Calculate L_0
        for testbatch in testloader:
            testbatch = testbatch[0].to(device).float()
            batch_size = len(testbatch)
            blurred_batch = blur_module(testbatch, [1]).float()
            non_blurred_batch = testbatch
            fwd_steps = torch.ones(batch_size, device=device)
            noise = torch.randn_like(blurred_batch) * sigma
            perturbed_data = noise + blurred_batch
            diff = model_fn(perturbed_data, fwd_steps)
            prediction = perturbed_data + diff
            mse_loss = ((non_blurred_batch - prediction)**2).sum((1, 2, 3))
            loss = 0.5*mse_loss/delta**2
            # Normalization constant
            loss += num_dims*np.log(delta*np.sqrt(2*np.pi))
            L_others[0] += loss.sum()
            mse_losses[0] += mse_loss.sum()
        L_others[0] = L_others[0] / test_size
        mse_losses[0] = mse_losses[0] / test_size

    logging.info("... Done! Values {}".format(L_others))
    return L_K_upbound.detach().cpu(), L_others.detach().cpu(), mse_losses.detach().cpu()


"""The next two functions based on https://github.com/yang-song/score_sde"""


def create_model(config, device_ids=None):
    """Create the model."""
    model = UNetModel(config)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def get_model_fn(model, train=False):
    """A wrapper for using the model in eval or train mode"""
    def model_fn(x, fwd_steps):
        """Args:
                x: A mini-batch of input data.
                fwd_steps: A mini-batch of conditioning variables for different levels.
        """
        if not train:
            model.eval()
            return model(x, fwd_steps)
        else:
            model.train()
            return model(x, fwd_steps)
    return model_fn
