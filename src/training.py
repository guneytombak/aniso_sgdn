
import numpy as np
import torch
from torch import optim
import wandb
import copy
from src.training_utils import select_loss, check_maxiter, random_update_function, id_func
from src.utils import LossType

def calculate_integral(h, gh, data_loader):

    integrant = list()
    grad_stack = list() # gradients as (t-y)@(grad(h))
    sup_trace = 0.0 # supremum of the trace

    for batch_idx, (data, target) in enumerate(data_loader):

        diff_yh2 = torch.sum((target-h[batch_idx])**2)
        M = gh[batch_idx] @ gh[batch_idx].T
        if M.dim() == 0:
            trace = M
            grad_stack.append((target-h[batch_idx])*gh[batch_idx])
        else:
            trace = torch.trace(M)
            grad_stack.append((target-h[batch_idx])@gh[batch_idx])
        if sup_trace < trace:
            sup_trace = trace

        integrant.append(diff_yh2*trace) 

    integral = torch.mean(torch.stack(integrant))
    if M.dim() > 0:
        integral = integral / M.size(0)
    true_integral = torch.mean(torch.sum(torch.stack(grad_stack)**2,2))

    return integral, true_integral, sup_trace, grad_stack
    

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

def eval_grad(data_loader, model_actual, lr, loss_func):

    n_losses = get_n_losses(data_loader, model_actual, loss_func)

    device = torch.device("cpu") # No use of batch, therefore CPU
    model = copy.deepcopy(model_actual).to(device) # Copying model to not interfering the main process
    optimizer = optimizer = optim.SGD(model.parameters(), lr) # Setting an SGD optimizer from scratch

    losses = list()
    grads = list()

    for k in range(n_losses):

        if n_losses > 1:
            losses_k = list()
            grads_k = list()

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # setting gradients to zero
            model.eval() # evaluation mode for the model
            # Note: Since it is a deep copy, it does not interfere with the main process 
            output = model(data)
            loss = loss_func(output, target)

            if n_losses > 1:
                loss = loss.flatten()

            if n_losses == 1: # If there is one output of loss function
                loss.backward()
                losses.append(loss)
                grads.append(model.get_grads())
            else: # If there is more than one output of loss function
                loss[k].backward()
                losses_k.append(loss[k])
                grads_k.append(model.get_grads())

        if n_losses > 1: # stack the different losses
            grads.append(torch.stack(grads_k, dim=0))
            losses.append(torch.stack(losses_k, dim=0))
    
    grads = torch.stack(grads, dim=0)
    losses = torch.stack(losses, dim=0)

    if n_losses > 1:
        grads = grads.transpose(0, 1)
        losses = losses.transpose(0, 1)

    
    return grads, losses


def train_epoch(model, cfg, batch_loader, single_loader, optimizer):
    
    loss_func = select_loss(cfg.loss_type) # Get the loss type
    # Check whether the number of iterations is good to use 
    maxiter = check_maxiter(cfg.maxiter, batch_loader)  
    
    model.train() # enabling the training mode
    per_iter_idx_run = int(np.ceil(maxiter/cfg.per_epoch_test))

    loss_list = list()
    
    for batch_idx, (data, target) in enumerate(batch_loader):

        if batch_idx > (maxiter-1):
            break

        if not (batch_idx % per_iter_idx_run):

            # Gradient Computation
            
            # grad_loss_stack (n_samples, n_params), loss_stack (n_samples)
            grad_loss_stack, loss_stack = eval_grad(single_loader, model, cfg.lr, loss_func)
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
            
            grad_output_stack, output_stack = eval_grad(single_loader, model, cfg.lr, id_func)

            # REVIEW
            grad_output_stack_flattened = grad_output_stack.reshape(grad_output_stack.size(0), -1) 
            max_magnitude_square_grad_output = torch.max(torch.sum(grad_output_stack_flattened**2, 1))

            if cfg.loss_type == LossType.MSE:
                integral, true_integral, sup_trace, int_grad_stack = calculate_integral(output_stack, grad_output_stack, single_loader)
                assert torch.isclose(sup_trace, max_magnitude_square_grad_output)
                upper_bound = 2*max_magnitude_square_grad_output*exp_loss
            elif cfg.loss_type == LossType.NLL:
                upper_bound = 2*max_magnitude_square_grad_output*min(1.0, exp_loss)
                integral, true_integral = upper_bound, upper_bound

            '''
            for k in range(len(int_grad_stack)):
                wandb.log({ 'grad'  : grad_loss_stack[k],
                            'int'   : int_grad_stack[k]})
            '''

            wandb.log({ 'Lower'  : lower_bound,
                        'EoSq'   : exp_mag_sq_grad_loss,
                        'Loss'   : exp_loss,
                        'Dh2'    : max_magnitude_square_grad_output,
                        'Upper'  : upper_bound,
                        'Int'    : true_integral,
                        'Int2'   : integral,
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
                    new_val = random_update_function(p, cfg)
                    p.copy_(new_val)

        # Data Save
        
    epoch_loss = np.mean(np.array(loss_list))
    
    return epoch_loss

