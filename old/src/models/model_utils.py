def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    from keras.models import load_model
    return load_model(filepath)

def set_gpu_memory_growth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def get_model_summary(model):
    return model.summary()