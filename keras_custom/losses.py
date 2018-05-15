from keras import backend as K


def sin_minimum_angle_loss(y_true, y_pred):
    return K.mean(K.square(K.minimum(K.abs(y_pred - y_true), K.abs(2 - (y_pred - y_true)))))

