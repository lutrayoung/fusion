import warnings


class xConfig(object):
    project = 'train'  # 'train'表示训练，'test'表示测试
    generator_model_name = 'fusion_generator'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    discriminator_model_name = 'fusion_discriminator'

    train_ir_dir = './images/Train_ir'
    train_vi_dir = './images/Train_vi'
    checkpoint_dir = "checkpoint"
    #test_ir_dir = 'E:/project/test/FusionGAN_pytorch/images/Train_ir'
    #test_vi_dir = 'E:/project/test/FusionGAN_pytorch/images/Train_vi'
    image_size = 60#60
    label_size = 60#60
    img_size = 60#60
    stride = 60#12#30

    use_gpu = True   # 是否使用gpu
    num_workers = 4  # 加载数据时使用子进程个数

    batch_size = 4  # 批处理大小 default=4
    max_epochs = 1  # 训练最大次数default=400
    # snap_batches = 1  # 指每执行多少个batches保存一次模型
    lr = 0.0001  # 初始学习速率0.001
    weight_decay = 1e-4  # 1e-4权重衰减，防止过拟合
    momentum = 0.099  # 梯度下降动量系数
    step_size = 100  # 学习速率调整周期
    gamma = 0.01  # 学习速率调整的加权系数0.1
    b1 = 0.5
    b2 = 0.99
    save_point = 1#20

    lam = 10#1e-7
    #xita = 1.2
    #lam_iten = 120
    #lam_lbp = 1e-7
    lam_inten = 1#1
    #lam_tv = 1#1.2
    lam_ssim = 1#1e-7
    #lam_gra= 1#120
    lam_con= 100 #100、0.5
    channels = 1
    n_critic = 5
    lambda_gp_vi = 0.1#0.1
    lambda_gp_ir = 0.1#0.1
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
