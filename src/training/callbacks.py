from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_callbacks(checkpoint_path, patience=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    
    return [early_stopping, model_checkpoint]