import os,sys
import bcolz
import numpy as np

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
    
def load_array(fname):
    return bcolz.open(fname)[:]


class NotebookData:
    def __init__(self,
                 data_dir,
                 data='statefarm',
                 results_dir='results',
                 sample_mode=True,
                 train=True,
                 preprocess=True):
        
        self.data_root = data_dir + data + '/'
        self.sample_root = self.data_root + 'sample/'
        self.sample_mode = sample_mode
        self.sample_batch_size = 1
        self.rdir = results_dir
        self.train = train
        self.preprocess = preprocess
        self.training_data = None
        self.validation_data = None
        self.training_labels = None
        self.validation_labels = None
        self.cindices = None
        
    def root_dir(self):
        return self.sample_root if self.sample_mode else self.data_root
    
    def test_dir(self):
        return self.root_dir() + 'test/'
    
    def results_dir(self):
        return self.root_dir() + self.rdir
    
    def train_dir(self):
        return self.root_dir() + 'train/'

    def valid_dir(self):
        return self.root_dir() + 'valid/'
    
    def pproc_dir(self):
        return self.root_dir() + 'preprocessed/'
    
    def batch_size(self, requested_size=8):
        return self.sample_batch_size if self.sample_mode else requested_size
    
    def load_data_and_labels(self):
        """Loads the batches and labels to the internal state.
        
           Upon loading, the data is accessible through the
           corresponding methods.
        """
        gen = image.ImageDataGenerator()
        target_size = (224,224)
        if self.preprocess:
            print('Preprocessing data...')
            if not os.path.isdir(self.pproc_dir()):
                os.mkdir(self.pproc_dir())
                
            batch_arr = []
            for ld,segment in [(self.train_dir(), 'train'),
                               (self.valid_dir(), 'valid')]:
                # TODO(ness): segment = os.basename(ld)
                flowgen = gen.flow_from_directory(
                    ld,
                    target_size=target_size,
                    shuffle=False,
                    class_mode=None,
                    batch_size=1)
                # Save the batches using method defined in utils.py
                data = np.concatenate([flowgen.next() for i in range(flowgen.n)])
                batches_dir = self.pproc_dir() + segment + '-bc'
                save_array(batches_dir, data)
                
                # Save the classes.
                cls_dir = self.pproc_dir() + segment + '-cl'
                save_array(cls_dir, flowgen.classes)
                
                batch_arr.append((data, flowgen.classes, flowgen.class_indices))
            
            # Set the data.
            self.training_data = batch_arr[0][0]
            self.validation_data = batch_arr[1][0]
            
            # Classes are zero-indexed and represent a category in
            # numerical form. So if the classes are 'dog' and 'cat',
            # the possible class values will be 0 and 1.
            self.trn_classes = batch_arr[0][1]
            self.val_classes = batch_arr[1][1]
            
            # Labels are the one-hot encoded (i.e. categorical)
            # version of the classes. In other words, if there are
            # 5 classes and an element belongs to class 2,
            # its label will be [0,0,1,0,0] (index 1).
            self.training_labels = to_categorical(batch_arr[0][1])
            self.validation_labels = to_categorical(batch_arr[1][1])
            
            # Class indices are dictionaries of the form
            # {'category_name': 0, 'category_name_2: 1}. They
            # make the mapping between numerical class indices and
            # a human-readable category name. They are (should be...)
            # the same for validation and training, so only load them
            # once, after sanity checking.
            self.cindices = batch_arr[0][2]
            print('Done preprocessing.')
        else:
            print('Loading data...')
            # Load the pre-saved data using methods defined in utils.py. See
            # preprocessing branch for the meaning of the data.
            self.training_data = load_array(self.pproc_dir() + 'train-bc')
            self.validation_data = load_array(self.pproc_dir() + 'valid-bc')
            self.trn_classes = load_array(self.pproc_dir() + 'train-cl')
            self.val_classes = load_array(self.pproc_dir() + 'valid-cl')
            self.training_labels = to_categorical(self.trn_classes)
            self.validation_labels = to_categorical(self.val_classes)
            
            # To get the class indices, we create the generator. It's cheap to
            # run since it doesn't actually load all the data.
            flowgen = gen.flow_from_directory(
                self.train_dir(),
                target_size=target_size,
                shuffle=False,
                class_mode=None,
                batch_size=1)    
            self.cindices = flowgen.class_indices
            print('Done loading.')
        
    def trn_data(self):
        if self.training_data is None:
            self.load_data_and_labels()
        return self.training_data
    
    def val_data(self):
        if self.validation_data is None:
            self.load_data_and_labels()
        return self.validation_data
    
    def trn_labels(self):
        if self.training_labels is None:
            self.load_data_and_labels()
        return self.training_labels
    
    def val_labels(self):
        if self.validation_labels is None:
            self.load_data_and_labels()
        return self.validation_labels
    
    def class_indices(self):
        if self.cindices is None:
            self.load_data_and_labels()
        return self.cindices
        
    def __str__(self):
        return ('Options:\n'
            '  Testing directory: {0}\n'
            '  Training directory: {1}\n'
            '  Validation directory: {2}\n'
            '  Preprocess directory: {3}\n'
            '  Results directory: {4}'
                .format(self.test_dir(),
                        self.train_dir(),
                        self.valid_dir(),
                        self.pproc_dir(),
                        self.results_dir()))


class_names = [
  'safe driving',
  'texting - right',
  'talking on the phone - right',
  'texting - left',
  'talking on the phone - left',
  'operating the radio',
  'drinking',
  'reaching behind',
  'hair and makeup',
  'talking to passenger',
]

def process_model(model,opt,name,sub=False):
    iter_path = opt.results_dir()+ '/' + name
    if not os.path.isdir(iter_path):
        os.makedirs(iter_path)
    model.save_weights(iter_path + '/temp_custom.h5')
    
    if sub: create_submission(model, opt, iter_path)

def create_submission(model, opt, iter_path):
    print(np.__dict__)
    # Save the results to usable files.
    pp_test_dir = opt.pproc_dir() + '/test'
    if not os.path.exists(pp_test_dir):
        print('No preprocessed test data at %s' % pp_test_dir)
        return
    
    pred_list = []
    filename_list = []
    bdirs = os.listdir(pp_test_dir)
    bdirs.sort()
    for bd in bdirs:
        batch_dir = os.path.join(pp_test_dir, bd)
        print('Predicting batch at %s' % batch_dir)
        fns = load_array(os.path.join(batch_dir, 'fn'))
        data = load_array(os.path.join(batch_dir, 'bc'))
        predictions = model.predict(data, batch_size=32, verbose = 1)
        pred_list.append(predictions)
        filename_list.append(fns)
      
    preds = np.concatenate(pred_list)
    filenames = np.concatenate(filename_list)
    predictions_path = iter_path + '/preds.dat'
    save_array(predictions_path, preds)
    print('Saved predictions to: %s' % predictions_path)
    filenames_path = iter_path + '/filenames.dat'
    save_array(filenames_path, filenames)
    print('Saved filenames to: %s' % filenames_path)
    
    # Create the response file.
    file_column = [pth[8:] for pth in filenames]
    clipped_preds = np.clip(preds, 0.05, 0.95)
    preds_col = [','.join(['%.2f' % p for p in pred]) for pred in clipped_preds]
    entries = [','.join([f,p]) for f,p in zip(file_column, preds_col)]
    entries = np.array(entries)

    class_names = ['c%d' % i for i in range(10)]
    title_row = ','.join(['img'] + class_names)

    submission_file_name = os.path.join(iter_path,'submission.csv')
    np.savetxt(submission_file_name,
               entries,
               fmt='%s',
               header=title_row,
               comments='')
    
# Define a fit method to save on time.
def fit_model(model, tbatches, vbatches, opt, batch_size=8, epochs=5):
    bsize = opt.batch_size(batch_size)
    tbatches.batch_size = bsize
    vbatches.batch_size = bsize
    model.fit_generator(tbatches,
                        epochs=epochs,
                        validation_data=vbatches)
    
def load_model(model, opt, iter_name):
    model_dir = opt.results_dir() + '/' + iter_name + '/temp_custom.h5'
    print(model_dir)
    model.load_weights(model_dir)
    print('loaded')
