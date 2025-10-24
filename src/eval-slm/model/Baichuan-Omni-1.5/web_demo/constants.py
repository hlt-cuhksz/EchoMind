MODEL_PATH = "Baichuan-Omni-1d5"  #your baichuan-inc/Baichuan-Omni-1d5 model path
COSY_VOCODER = "cosy24k_vocoder"  #your vocoder model path
g_cache_dir = "../cache"
sampling_rate = 24000
wave_concat_overlap = int(sampling_rate * 0.01)
role_prefix = {
    'system': '<B_SYS>',
    'user': '<C_Q>',
    'assistant': '<C_A>',
    'audiogen': '<audiotext_start_baichuan>'
}
max_frames = 8