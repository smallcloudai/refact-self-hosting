models_tf_mini_db = {
    'codet5/plus_770M_python': {
        'model_class': 'code_contrast:CodeTFModel',
        'diff_scratchpad_class': 'code_contrast:CodeTFScratchpad',
        'model_name': 'codet5',
        'task': 'pretrained',
        'model_type': 'plus-770M-python',
        'is_eval': True,
        'load_in_8bit': True,
        'load_in_4bit': False,
        'weight_sharding': False
    }
}
