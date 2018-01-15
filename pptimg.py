import os,sys,shutil
import bcolz
import keras
from keras.preprocessing import image
import numpy as np

from keras import backend as K

# Explicitly set Theano image ordering.
K.set_image_dim_ordering('th')

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

cwd = os.getcwd()
print('Working directory: %s' % cwd)
contents = os.listdir(cwd)
if 'test' not in contents:
  print('Missing \'test\' directory; cannot proceed.')
  exit(1)

if (('preprocessed' not in contents)
     or ('test' not in os.listdir(os.path.join(cwd, 'preprocessed')))):
  os.makedirs('preprocessed/test')

test_dir = os.path.join(cwd, 'test')
target_dir = os.path.join(cwd, 'preprocessed/test')

# Ten thousand batch size to not run afoul of memory limitations.
BATCH_SIZE=10000

# Create the generator that will load the images.
gen = image.ImageDataGenerator()
testiter = gen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        shuffle=False,
        class_mode=None,
        batch_size=1) # This is done on the CPU.

num_batches = testiter.n // BATCH_SIZE
remainder = testiter.n % BATCH_SIZE

print('Preprocessing: ')
for rotation in range(num_batches):
  print('  Batch %d: %d-%d...' % (rotation,
                             rotation * BATCH_SIZE,
                             (rotation + 1) * BATCH_SIZE - 1))
  data = np.concatenate([testiter.next() for i in range(BATCH_SIZE)])
  filenames = testiter.filenames[rotation * BATCH_SIZE:(rotation+1) * BATCH_SIZE]
  batch_dir = os.path.join(target_dir, 'batch%d' % rotation)
  os.makedirs(batch_dir)
  data_dir = os.path.join(batch_dir, 'bc')
  filenames_dir = os.path.join(batch_dir, 'fn')
  save_array(data_dir, data)
  save_array(filenames_dir, filenames)

if remainder > 0:
  print('  Batch %d: %d-%d...' % (num_batches,
                                  num_batches * BATCH_SIZE,
                                  num_batches * BATCH_SIZE + remainder))
  data = np.concatenate([testiter.next() for i in range(remainder)])
  filenames = testiter.filenames[num_batches * BATCH_SIZE:(num_batches+1) * BATCH_SIZE]
  batch_dir = os.path.join(target_dir, 'batch%d' % num_batches)
  os.makedirs(batch_dir)
  data_dir = os.path.join(batch_dir, 'bc')
  filenames_dir = os.path.join(batch_dir, 'fn')
  save_array(data_dir, data)
  save_array(filenames_dir, filenames)


print('Done!')
