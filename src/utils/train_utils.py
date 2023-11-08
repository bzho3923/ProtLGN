

def lr_scheduler(s, epoch, opt):
    lr_steps = (opt['lr'] - opt['init_lr']) / opt['warmup']
    decay_factor = opt['lr'] * opt['warmup'] ** .5
    if s < opt['warmup']:
        lr = opt['init_lr'] + s * lr_steps
    else:
        if epoch < opt['step_schedule']:
            lr = opt['lr']
        else:
            lr = opt['lr'] * 0.1
    return lr