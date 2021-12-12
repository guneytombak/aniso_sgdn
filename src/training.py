
import numpy as np
import torch
from torch import optim
import wandb
import copy
from src.training_utils import select_loss, check_maxiter, random_update_function
from src.model import id_func
from src.utils import LossType

def calculate_integral(h, gh, data_loader):

    integrant_1 = list()
    integrant_2 = list()
    sup_trace = 0.0

    for batch_idx, (data, target) in enumerate(data_loader):

        diff_yh2 = torch.sum((target-h[batch_idx])**2)
        diff_yh = torch.sqrt(diff_yh2)
        M = gh[batch_idx] @ gh[batch_idx].T
        if M.dim() == 0:
            trace = M
        else:
            trace = torch.trace(M)
        if sup_trace < trace:
            sup_trace = trace
        trace_sq = torch.sqrt(trace)

        integrant_1.append(diff_yh2*trace)
        integrant_2.append(diff_yh*trace_sq)

    integral_1 = torch.mean(torch.stack(integrant_1))
    integral_2 = torch.mean(torch.stack(integrant_2))**2

    return integral_1, integral_2, sup_trace
    

def get_n_losses(data_loader, model_actual, loss_func):

    device = torch.device("cpu")
    model = copy.deepcopy(model_actual).to(device)

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)

        if loss.dim() == 0:
            return 1
        else:
            return len(loss.flatten())

def eval_grad(data_loader, model_actual, optimizer_actual, loss_func):

    n_losses = get_n_losses(data_loader, model_actual, loss_func)

    device = torch.device("cpu")
    model = copy.deepcopy(model_actual).to(device)
    optimizer = copy.deepcopy(optimizer_actual) 

    losses = list()
    grads = list()

    for k in range(n_losses):

        if n_losses > 1:
            losses_k = list()
            grads_k = list()

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # REVIEW
            output = model(data)
            loss = loss_func(output, target)

            if n_losses > 1:
                loss = loss.flatten()

            if n_losses == 1:
                loss.backward()
                losses.append(loss)
                grads.append(model.get_grads())
            else:
                loss[k].backward()
                losses_k.append(loss[k])
                grads_k.append(model.get_grads())

        if n_losses > 1:
            grads.append(torch.stack(grads_k, dim=0))
            losses.append(torch.stack(losses_k, dim=0))
    
    grads = torch.stack(grads, dim=0)
    losses = torch.stack(losses, dim=0)

    if n_losses > 1:
        grads = grads.transpose(0, 1)
        losses = losses.transpose(0, 1)

    
    return grads, losses


def train_epoch(model, cfg, batch_loader, single_loader, optimizer):
    
    loss_func = select_loss(cfg.loss_type)
    maxiter = check_maxiter(cfg.maxiter, batch_loader)
    
    model.train()

    per_iter_idx_run = int(np.ceil(maxiter/cfg.per_epoch_test))

    loss_list = list()
    
    for batch_idx, (data, target) in enumerate(batch_loader):

        if batch_idx > (maxiter-1):
            break

        if not (batch_idx % per_iter_idx_run):

            # Gradient Computation
            
            # grad_loss_stack (n_samples, n_params), loss_stack (n_samples)
            grad_loss_stack, loss_stack = eval_grad(single_loader, model, optimizer, loss_func)
            # expected_grad_loss (n_params)
            exp_grad_loss = torch.mean(grad_loss_stack, 0)
            # magnitude_square_grad_loss (n_samples)
            mag_sq_grad_loss = torch.sum(grad_loss_stack**2, 1)
            # expected_magnitude_square_grad_loss ()
            exp_mag_sq_grad_loss = torch.mean(mag_sq_grad_loss)
            # expected_loss ()
            exp_loss = torch.mean(loss_stack)
            # square_deviation_from_expected_grad_loss (n_samples, n_params)
            sq_dev_from_exp_grad_loss = (exp_grad_loss - grad_loss_stack)**2
            # magnitude_square_deviation_from_expected_grad_loss (n_samples)
            mag_sq_dev_from_exp_grad_loss = torch.sum(sq_dev_from_exp_grad_loss, 1)
            # lower_bound ()
            lower_bound = torch.mean(mag_sq_dev_from_exp_grad_loss)
            
            grad_output_stack, output_stack = eval_grad(single_loader, model, optimizer, id_func)

            # REVIEW
            grad_output_stack_flattened = grad_output_stack.reshape(grad_output_stack.size(0), -1) 
            max_magnitude_square_grad_output = torch.max(torch.sum(grad_output_stack_flattened**2, 1))

            if cfg.loss_type == LossType.MSE:
                integral_1, integral_2, sup_trace = calculate_integral(output_stack, grad_output_stack, single_loader)
                assert torch.isclose(sup_trace, max_magnitude_square_grad_output)
                upper_bound = 2*max_magnitude_square_grad_output*exp_loss
            elif cfg.loss_type == LossType.NLL:
                upper_bound = 2*max_magnitude_square_grad_output*min(1.0, exp_loss)
                integral_1, integral_2 = upper_bound, upper_bound

            #theta = wandb.Table(data=[[a] for a in model.get_wb()], columns=["wnb"])

            wandb.log({ 'Lower'  : lower_bound,
                        'EoSq'   : exp_mag_sq_grad_loss,
                        'Loss'   : exp_loss,
                        'Dh2'    : max_magnitude_square_grad_output,
                        'Upper'  : upper_bound,
                        'Int1'   : integral_1,
                        'Int2'   : integral_2,
                        #'Theta'  : wandb.plot.histogram(theta, "wnb", title="WB Histogram")
                        })

        # Step
        
        data, target = data.to(cfg.dev), target.to(cfg.dev)
        optimizer.zero_grad() # REVIEW
        output = model(data)

        loss_batch = loss_func(output, target)
        loss_list.append(np.float64(loss_batch))
        
        if cfg.learn:
            loss_batch.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                for p in model.parameters():
                    new_val = random_update_function(p, 0, loss_batch, cfg)
                    p.copy_(new_val)

        # Data Save
        
    epoch_loss = np.mean(np.array(loss_list))
    
    return epoch_loss

'''

        grad_h2 = batch_max_magnitude(grad_output_global)
        upper_bound = upper_bound_func(grad_h2, loss)
        
        g_sq = np.float64(expected_grad_loss_global)
        gmDL_sq = np.float64(torch.sum(torch.square(expected_grad_loss_global - grad_loss_batch)))
        loss_batch = np.float64(loss)
        loss_list.append(loss)
        # iter_no = batch_idx + epoch*maxiter

        #REVIEW Be sure about mean or sum

        r_term_trace = torch.zeros(grad_output_global.size(2))

        for i in range(grad_output_global.size(0)):
            r_term_trace += torch.trace(grad_output_global[i].T * grad_output_global[i])
        #r_term = torch.einsum('bvd, bdw -> bvw', torch.transpose(grad_output_global,1,2), grad_output_global).to(output.device)
        integral = torch.mean(loss_global*r_term_trace)
        integral = np.float64(integral)



'''