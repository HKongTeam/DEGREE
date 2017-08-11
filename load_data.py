from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base

class DataSet(object):

  def __init__(self, data):
    """Construct a DataSet.

    """

    self._num_examples = data.shape[0]
    self._data = data
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data
    
  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._data = self.data[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      data_rest_part = self._data[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self.data[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      data_new_part = self._data[start:end]
      temp = np.concatenate((data_rest_part, data_new_part), axis=0) 
      return np.concatenate((np.arange(batch_size).reshape(batch_size,1),temp), axis = 1)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return np.concatenate((np.arange(end-start).reshape(end-start,1), self._data[start:end]), axis= 1)
      
def read_data_sets(train_file, test_file, validation_size=5000):

  train_lines = np.loadtxt(train_file)
  test_lines = np.loadtxt(test_file)
   
  validation_lines = train_lines[:validation_size]
  train_lines= train_lines[validation_size:]
    
  train = DataSet(train_lines)
  validation = DataSet(validation_lines)
  test = DataSet(test_lines)
  
  return base.Datasets(train=train, validation=validation, test=test)