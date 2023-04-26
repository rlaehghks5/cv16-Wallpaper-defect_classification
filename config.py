CFG = {
    'IMG_SIZE': 384,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-6,
    'BATCH_SIZE': 32,
    'SEED': 41,
    'MODEL': 'vit_base_patch32_384',
    'OPTIMIZER' : 'AdamW',
    'CRITERION' : 'f1', # 'label_smoothing', # 'focal', # 'cross_entropy',
    'NAME' : 'exp',
    'PATIENCE' : 5
}