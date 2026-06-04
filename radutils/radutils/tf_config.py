def disable_tf_memory_pre_allocation():
    """Disable the behavior where Tensorflow, by default, allocate all CUDA memory to itself."""
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except ModuleNotFoundError:
        pass
