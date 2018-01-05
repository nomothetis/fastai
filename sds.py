from __future__ import print_function
import os,sys,shutil
from shutil import copyfile
import os.path
import numpy as np
from os.path import basename

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

for cls in os.listdir(train_dir):
  cls_path = os.path.join(train_dir, cls)
  cls_target = os.path.join(valid_dir, cls)
  cls_sample_vtarget = os.path.join(svalid_dir, cls)
  cls_sample_ttarget = os.path.join(strain_dir, cls)
  os.mkdir(cls_target)
  os.mkdir(cls_sample_vtarget)
  os.mkdir(cls_sample_ttarget)
  files = np.array(os.listdir(os.path.join(train_dir, cls)))
  count = len(files)
  valid_count = int(np.round(count * v_frac))
  sp_count = min(sp_count, int(valid_count * 0.9))
  sample_count = int(np.round(sp_count * v_frac))
  print("  Class '%s'..." % cls, end="")
  np.random.shuffle(files)
  copied_count = 0
  for fl in files[:max(valid_count, sample_count+sp_count)]:
    starting_path = os.path.join(cls_path, fl)
    target_path = os.path.join(cls_target, fl)
    if copied_count < sp_count:
      copyfile(starting_path, os.path.join(strain_dir, cls, fl))
    elif copied_count < sp_count+sample_count:
      copyfile(starting_path, os.path.join(svalid_dir, cls, fl))

    if copied_count < valid_count:
      os.rename(starting_path, target_path)
    copied_count += 1
  print("done.")
    
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
