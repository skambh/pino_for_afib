import torch

def AP_loss(data, b_pred):
    """
    Computes the PDE loss for inverse mode in the Aliev-Panfilov model, where the goal is
    to predict parameter `b` given observed dynamics `u_obs` and `v_obs`.
    
    Parameters:
    - u_obs: Observed transmembrane potential [N, 1].
    - v_obs: Observed recovery variable [N, 1].
    - t, x, y: Input coordinates (tensors) [N, 1].
    - b_pred: Predicted parameter `b` (scalar or tensor).
    - D: Diffusion coefficient (scalar or tensor).
    - a, epsilon, k: Model parameters (scalars).
    
    Returns:
    - pde_loss: Scalar tensor representing the inverse-mode physics-informed loss.

    IMPORTANT: only need dw_dt since we are predicting b and dv_dt does not depend on b.
    """
    epsi = 0.002
    mu1 = 0.2
    mu2 = 0.3
    k = 8.0
    # a = 0.01
    # D = 1

    v_obs, w_obs, X, Y, T = data[:, :, :, 0], data[:, :, :, 1], data[:, :, :, 2], data[:, :, :, 3], data[:, :, :, 4]

    # Ensure gradients for observed data
    v_obs.requires_grad_(True)
    w_obs.requires_grad_(True)

    # Compute partial derivatives of observed u
    # v_t_obs = torch.autograd.grad(v_obs, T, grad_outputs=torch.ones_like(v_obs), 
    #                               create_graph=True, retain_graph=True)[0]
    # v_x_obs = torch.autograd.grad(v_obs, X, grad_outputs=torch.ones_like(v_obs), 
    #                               create_graph=True, retain_graph=True)[0]
    # v_y_obs = torch.autograd.grad(v_obs, Y, grad_outputs=torch.ones_like(v_obs), 
    #                               create_graph=True, retain_graph=True)[0]

    # Second-order spatial derivatives for Laplacian of u
    # u_xx_obs = torch.autograd.grad(v_x_obs, X, grad_outputs=torch.ones_like(v_x_obs), 
    #                                create_graph=True, retain_graph=True)[0]
    # u_yy_obs = torch.autograd.grad(v_y_obs, Y, grad_outputs=torch.ones_like(v_y_obs), 
    #                                create_graph=True, retain_graph=True)[0]
    # laplacian_u_obs = u_xx_obs + u_yy_obs

    # Compute observed ∂v/∂t
    dw_dt_obs = torch.autograd.grad(w_obs, T, grad_outputs=torch.ones_like(w_obs), 
                                  create_graph=True, retain_graph=True)[0]

    # Calculate predicted dynamics using b_pred
    # ∂u/∂t (predicted) = ∇ · (D ∇u) + k*u*(1-u)*(u-a) - u*v
    # v_t_pred = D * laplacian_u_obs + k * v_obs * (1 - v_obs) * (v_obs - a) - v_obs * w_obs

    # ∂v/∂t (predicted) = ε(k*u*(u-b-1) - v)
    dw_dt_pred = (epsi + (mu1 * w_obs / v_obs + mu2)) * (-w_obs - k * v_obs * (v_obs - b_pred - 1))

    # Residuals (observed dynamics vs. predicted dynamics)
    # v_residual = v_t_obs - v_t_pred
    w_residual = dw_dt_obs - dw_dt_pred

    # Mean squared error loss
    # loss_v = torch.mean(v_residual**2)
    loss_w = torch.mean(w_residual**2)

    # Total PDE loss
    # pde_loss = loss_v + loss_w
    pde_loss = loss_w

    return pde_loss

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)