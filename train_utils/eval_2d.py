from tqdm import tqdm
import numpy as np
from .losses import LpLoss, AP_loss

def eval_ap(model, dataloader, device, use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []

    for data, b in pbar:
        data, b = data.to(device), b.to(device)
        out = model(data)

        data_loss = myloss(out, b)
        test_err.append(data_loss.item())

        f_loss = AP_loss(out, b)
        f_err.append(f_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')