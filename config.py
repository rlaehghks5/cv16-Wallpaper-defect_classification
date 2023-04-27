CFG = {
    'IMG_SIZE': 300,
    'EPOCHS': 100,
    'LEARNING_RATE': 5e-4,
    'BATCH_SIZE': 32,
    'SEED': 41,
    'MODEL': 'efficientnet_b3',
    'OPTIMIZER' : 'AdamW',
    'CRITERION' : 'cross_entropy', # 'label_smoothing', # 'focal', # 'cross_entropy',
    'NAME' : 'exp',
    'PATIENCE' : 8
}