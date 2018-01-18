from __future__ import print_function
import os,sys,shutil
from shutil import copyfile
import os.path
from os.path import basename
import math
import numpy as np
import pandas as pd
import random

def restore(from_dir, to_dir, indent_string=""):
  """Returns files to their original directory under to_dir.
  
     Moves all files in subdirectories of `from_dir` to correspondingly
     named subdirectories under `to_dir` if they exist. If the
     subdirectories don't exist, the method throws an exception.

     For simplicity, this method assumes that `from_dir` contains
     only directories and throw an exception otherwise.

     Args:
       from_dir: the directory whose contents should be returned.
       to_dir: the target directory
       indent_string: string to show before all output newlines."
  """
  # Check that all directories are valid:
  for candidate in os.listdir(from_dir):
    if os.path.isfile(candidate):
      raise ValueError(
        """%s contains file %s\n Cannot return contents of folder if it 
           contains files.""" % from_dir, candidate)
    if not os.path.exists(os.path.join(to_dir, candidate)):
      raise ValueError(
        """%s does not contain folder %s\n Cannot return contents of folder if
           target does not contain folders of the same name.""" % (to_dir, candidate))

  for candidate in os.listdir(from_dir):
    starting_dir = os.path.join(from_dir, candidate)
    target_dir = os.path.join(to_dir, candidate)
    print("%sReturning contents of: %s..." % (indent_string, starting_dir), end="")
    for file_to_move in os.listdir(starting_dir):
      os.rename(os.path.join(starting_dir, file_to_move),
                os.path.join(target_dir, file_to_move))
    print("done")
    
cwd = os.getcwd()
contents = os.listdir(cwd)
if 'train' not in contents:
  print("Missing 'train' directory; cannot proceed.")
  exit()

if 'test' not in contents:
  print("Missing 'test' directory; cannot proceed.")
  exit()

train_dir = os.path.join(cwd, 'train')
valid_dir = os.path.join(cwd, 'valid')
test_dir = os.path.join(cwd, 'test')
sample_dir = os.path.join(cwd, 'sample')
strain_dir = os.path.join(sample_dir, 'train')
svalid_dir = os.path.join(sample_dir, 'valid')
stest_dir = os.path.join(sample_dir, 'test', 'unknown')

if ('valid' in contents) or ('sample' in contents):
  valid_yes = ['y', 'Y', 'yes', 'Yes', 'YES']
  valid_no = ['n', 'N', 'no', 'No', 'NO']
  proceed = raw_input(
    "A 'valid' or 'sample' directory already exists. The setup\n" +
    "will be overwritten, assuming that the data in 'valid\n'" +
    "should be kept and the data in 'sample' should be discarded.\n" +
    "Proceed? (y/N): ")
  while proceed not in ['y', 'n', 'Y', 'N', 'yes', 'Yes', 'no', 'No']:
    proceed = raw_input("Unrecognized choice. Proceed? (y/N): ")
  if proceed in valid_no:
    print("No changes.")
    exit()
  print("  Returning contents of 'valid' to 'train':")
  restore(os.path.join(cwd, 'valid'),
          os.path.join(cwd, 'train'),
          indent_string="    ")
  print("  Deleting 'sample' and 'valid' directories...", end="")
  if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)
  if os.path.exists(valid_dir):
    shutil.rmtree(valid_dir)
  print("done.")

# The fraction of the training set used for validation.
v_frac = 0.1

# The number of sample training elements per class.
sp_count = 16

print("Creating validation and sample sets.")

os.mkdir(valid_dir)
os.mkdir(sample_dir)
os.mkdir(strain_dir)
os.mkdir(svalid_dir)

drv = pd.read_csv('dlist/driver_imgs_list.csv', header=0)
for cls in pd.unique(drv.loc[:,'classname']):
  print('Class %s...' % cls, end="")
  sample_imgs = None
  val_imgs = None
  for dv in pd.unique(drv.loc[:,'subject']):
    subset = drv[(drv.classname == cls) & (drv.subject == dv)].copy()
    rsubset = subset.reindex(np.random.permutation(subset.index))
    val_count = int(math.ceil(len(subset) * v_frac))
    if val_imgs is None:
      val_imgs = rsubset.head(val_count)
    else:
      val_imgs = val_imgs.append(rsubset.head(val_count), ignore_index=False)
    if sample_imgs is None:
      sample_imgs = rsubset.head(1)
    else:
      sample_imgs = sample_imgs.append(rsubset.head(1), ignore_index=False)
  # Copy the samples first, since they also belong to the validation set.
  cls_sample_ttarget = os.path.join(strain_dir, cls)
  cls_sample_vtarget = os.path.join(svalid_dir, cls)
  os.makedirs(cls_sample_ttarget)
  os.makedirs(cls_sample_vtarget)
  for index, sample in sample_imgs.head(len(sample_imgs) - 1).iterrows():
    source = os.path.join(train_dir, cls, sample.img)
    target = os.path.join(cls_sample_ttarget, sample.img)
    copyfile(source, target)
  vsource = os.path.join(train_dir, cls, sample.tail().img)
  vtarget = os.path.join(cls_sample_vtarget, sample.tail().img)
  copyfile(vsource, vtarget)

  # Now move the validation set.
  valid_cls_dir = os.path.join(valid_dir, cls)
  os.makedirs(valid_cls_dir)
  for index, sample in val_imgs.iterrows():
    source = os.path.join(train_dir, cls, sample.img)
    target = os.path.join(valid_cls_dir, sample.img)
    os.rename(source, target)
  print('done.')

os.makedirs(stest_dir)
test_files = np.array(os.listdir(test_dir))
np.random.shuffle(test_files)
print("Creating sample test set...", end="")
for fl in test_files[:sp_count]:
  starting_path = os.path.join(test_dir, fl)
  target_path = os.path.join(stest_dir, fl)
  copyfile(starting_path, target_path)
print("done.")

print("Moving all test items into 'unknown' class...", end = "")
os.mkdir(os.path.join(test_dir, 'unknown'))
for fl in test_files:
  starting_path = os.path.join(test_dir, fl)
  target_path = os.path.join(test_dir, 'unknown', fl)
  os.rename(starting_path, target_path)
print("done.")

print("Done. Happy machine teaching!")
