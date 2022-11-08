import warnings


class xConfig(object):
    project = 'train'  

    train_ir_dir = './images/Train_ir'
    train_vi_dir = './images/Train_vi'
    checkpoint_dir = "checkpoint"
    
    image_size = 60#60
    label_size = 60#60
    img_size = 60#60
    stride = 60#12#30

    use_gpu = True   
    num_workers = 4 

    batch_size = 4  
    max_epochs = 10  
    # snap_batches = 1  
    lr = 0.0001  
    weight_decay = 1e-4  
    momentum = 0.099 
    step_size = 100 
    #gamma = 0.01  
    #b1 = 0.5
    #b2 = 0.99
    save_point = 1#20

    #lam = 10#1e-7
    #xita = 1.2
    #lam_iten = 120
    #lam_lbp = 1e-7
    #lam_inten = 1#1
    #lam_tv = 1#1.2
    #lam_ssim = 1#1e-7
    #lam_gra= 1#120
    #lam_con= 100 #100、0.5
    channels = 1
    
    
    lam_inten_ir = 2#2
    lam_inten_vis = 1
    lam_gradient = 10#10



def parse(self, kwargs):
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))


xConfig.parse = parse
opt = xConfig()
