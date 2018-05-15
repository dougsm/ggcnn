import os

import tensorflow as tf
import numpy as np
from keras.callbacks import Callback

from evaluate import calculate_iou_matches


def write_log(s):
    with open('validation_output.txt', 'a') as f:
        f.write(s)


class SaveValidationCallback(Callback):
    """
    Run validation and save the output a numpy arrays.
    """
    def __init__(self, output_folder, validation_data, ground_truth_bbs,
                 log_dir, output_postprocessor=None, period=1, save_validation_output=False):
        super().__init__()
        self.output_folder = output_folder
        self.validation_data = validation_data
        self.ground_truth_bbs = ground_truth_bbs
        self.output_postprocessor = output_postprocessor
        self.period = period
        self.save_validation_output = save_validation_output
        self.epochs_since_last_save = 0
        self.best = 0.0
        self.writer = tf.summary.FileWriter(log_dir)

        write_log('\n')
        write_log(log_dir.split('/')[-2])
        write_log('\t')

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            # For some reason validation_data gets turned into a list of arrays.
            pred_out = self.model.predict(self.validation_data[0])

            saved_outputs = {}
            for l, p in zip(self.model.output_layers, pred_out):
                saved_outputs[l.name] = p

            if self.output_postprocessor:
                saved_outputs.update(self.output_postprocessor(pred_out))

            saved_outputs['width_out'] *= 150.0
            succeeded, failed = calculate_iou_matches(saved_outputs['pos_out'], saved_outputs['angle_out'], self.ground_truth_bbs, no_grasps=2, grasp_width_out=saved_outputs['width_out'])

            s = len(succeeded)
            f = len(failed)
            val_accuracy = s/(s+f)*100.0

            write_log('%0.02f\t' % val_accuracy)

            if val_accuracy > self.best:
                self.best = val_accuracy
                self.model.save(os.path.join(self.output_folder, 'best_model.hdf5'), overwrite=True)

            s = tf.Summary()
            sv = s.value.add()
            sv.simple_value = val_accuracy
            sv.tag = 'iou_accuracy'
            self.writer.add_summary(s, epoch)
            self.writer.flush()

            if self.save_validation_output:
                # A lot quicker to save than compressed and not much gain.
                np.savez(os.path.join(self.output_folder, 'epoch_%02d_val_output' % epoch), **saved_outputs)

    def on_train_end(self, _):
        self.writer.close()
