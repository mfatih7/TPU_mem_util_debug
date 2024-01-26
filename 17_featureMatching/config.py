import os
import shutil
import h5py

class Config:
    def __init__(self):
        
        self.operation = 'train'
        # self.operation = 'test'
        
        self.device = 'cpu'
        self.device = 'cuda'
        self.device = 'tpu'
        
        ### small model, single-core
        # self.tpu_cores = 'single'
        # self.first_experiment = 100        
        # self.model_exp_no = 2  
        
        ### big model, single-core
        # self.tpu_cores = 'single'
        # self.first_experiment = 101        
        # self.model_exp_no = 12
        
        ### small model, multi-core
        self.tpu_cores = 'multi'
        self.first_experiment = 102        
        self.model_exp_no = 2
        
        ### big model, multi-core
        # self.tpu_cores = 'multi'
        # self.first_experiment = 103        
        # self.model_exp_no = 12
        
        self.validation = 'disable'
        self.validation = 'enable'
        
        self.tpu_profiler = 'disable'
        # self.tpu_profiler = 'enable'
        
        if(self.device=='tpu'):
            
            self.XLA_USE_BF16 = 0
            # self.XLA_USE_BF16 = 1
            
            self.home_dir = os.path.expanduser('~')
            
            self.storage_local_or_bucket = 'local'    
            
            self.bucket_name = 'bucket-us-central1-relativeposeestimation'
            self.bucket_name = 'bucket-europe-west4-relativeposeestimation'
            
            self.TPU_DEBUG = 0
            # self.TPU_DEBUG = 1
            self.tpu_debug_path = os.path.join(self.home_dir, 'tpu_debug')                
        else:
            self.storage_local_or_bucket = 'local'
        
        self.model_type = 'model_exp2'
        
        self.input_type = '1_to_1'           
        
        self.use_ratio = 0  # 0-> don't use, 1-> mask xs and ys, 2-> use as side
        self.use_mutual = 0  # 0-> don't use, 1-> mask xs and ys, 2-> use as side
        if(self.use_ratio==0 and self.use_mutual == 0):
            self.model_width = 4
        else:
            self.model_width = 6
        
        self.ess_loss = 'geo_loss'
        # self.ess_loss = 'ess_loss'
        
        # training_params ->   [model_type(n_to_n, 1_to_1), N_images_in_batch, N, batch_size]
        
        if( self.input_type == '1_to_1'):
            # self.training_params = [ [ 1, 512, 64, ],  ]        
            # self.training_params = [ [ 1, 512, 512, ],  ]
            # self.training_params = [ [ 1, 1024, 1024, ],  ]
            self.training_params = [ [ 1, 2048, 2048, ],  ]
            
            # self.n_epochs = [500, 500, 0] # always cls loss
            self.n_epochs = [500, 0, 1] # only first chunk cls loss
            
            self.early_finish_epoch = 20
        else:
            if(self.operation == 'train'):
                # self.training_params = [ [ 32, 512, 32, ],  ]
                self.training_params = [ [ 32, 1024, 32, ],  ]
                # self.training_params = [ [ 32, 2048, 32, ],  ]
            elif(self.operation == 'test'):
                # self.training_params = [ [ 25, 512, 25, ],  ]
                self.training_params = [ [ 25, 1024, 25, ],  ]
                # self.training_params = [ [ 25, 2048, 25, ],  ]
            
            # self.n_epochs = [5000, 5000, 0] # always cls loss
            self.n_epochs = [5000, 0, 1] # only first chunk cls loss
            
            self.early_finish_epoch = 100
            
        if( self.input_type == 'n_to_n'):
            self.input_channel_count = 1
        else:
            self.input_channel_count = 2
        
        self.use_hdf5_or_picle = 'hdf5'
        self.use_hdf5_or_picle = 'pickle'
        
        self.file_name_train = 'yfcc-sift-2000-train.hdf5'
        self.file_name_val = 'yfcc-sift-2000-val.hdf5'
        self.file_name_test = 'yfcc-sift-2000-test.hdf5'
        
        if( self.device == 'tpu' ):
            os.chdir( os.path.join(self.home_dir, 'TPU_mem_util_debug', '17_featureMatching') )
        
        self.input_path_bucket = '01_featureMatchingDatasets' 
        self.input_path_local = os.path.join('..', self.input_path_bucket)                
        
        self.output_path_bucket = '08_featureMatchingOutputs'
        self.output_path_local = os.path.join('..', self.output_path_bucket)        
        
        if(self.use_hdf5_or_picle == 'hdf5'):
                        
            if( self.device != 'tpu' ):
            
                self.num_workers = 3
                
                self.n_image_pairs_train = 541172
                self.n_image_pairs_val = 6694
                self.n_image_pairs_test = 4000
                
                system_ram_mb = 800_000
                # system_ram_mb = 10
                # system_ram_mb = 50
                # system_ram_mb = 100
                s_each_image_pair_apprx_mb = 0.05  
                s_t_image_pair_apprx_mb = self.n_image_pairs_train * s_each_image_pair_apprx_mb * ( self.num_workers + 1 )
                self.n_chunks = int( s_t_image_pair_apprx_mb / system_ram_mb ) + 1
                self.n_chunk_files = 1
            else:
                raise NotImplementedError(f"The feature '{self.device}' is not implemented yet.")
            
        elif(self.use_hdf5_or_picle == 'pickle'):
            
            if( self.device != 'tpu' ):
                if(self.operation == 'train'):
                    self.num_workers = 3                
                    self.pickle_set_no = 1
                    self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                    self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                    self.n_chunks = self.get_n_chunks_from_files()
                    self.n_chunk_files = self.n_chunks
                elif(self.operation == 'test'):
                    self.num_workers = 3                
                    self.pickle_set_no = 0
                    self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                    self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                    self.n_chunks = self.get_n_chunks_from_files() # Must be 1 for tests
                    self.n_chunk_files = self.n_chunks # Must be 1 for tests            
            else:                
                if( self.tpu_cores == 'single' ):
                    if(self.operation == 'train'):
                        if( self.input_type == 'n_to_n'):
                            self.pickle_set_no = 0
                            self.num_workers = 2
                        elif( self.input_type == '1_to_1'):
                            self.pickle_set_no = 1
                            self.num_workers = 1
                        self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                        self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                        self.n_chunk_files = self.get_n_chunks_from_files()
                        self.n_chunks = self.n_chunk_files
                    elif(self.operation == 'test'):
                        if( self.input_type == 'n_to_n'):
                            self.pickle_set_no = 0
                            self.num_workers = 7
                        elif( self.input_type == '1_to_1'):
                            self.pickle_set_no = 0
                            self.num_workers = 7
                        self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                        self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                        self.n_chunk_files = self.get_n_chunks_from_files()
                        self.n_chunks = self.n_chunk_files
                elif( self.tpu_cores == 'multi' ):
                    if( self.input_type == 'n_to_n'):
                        self.pickle_set_no = 1
                        self.num_workers = 0
                    elif( self.input_type == '1_to_1'):
                        # self.pickle_set_no = 2
                        self.pickle_set_no = 1
                        self.num_workers = 0
                    self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                    self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                    self.n_chunk_files = self.get_n_chunks_from_files()
                    if( int(self.n_chunk_files) % int(8) == 0 ):                        
                        self.n_chunks = int( self.n_chunk_files / 8 )
                    else:
                        raise ValueError(f"The provided argument is not valid: {self.n_chunk_files}")
            
        print( f'Number of chunks {self.n_chunks}, Number of chunk files {self.n_chunk_files}' )
        
        if( self.input_type == 'n_to_n'):
            self.learning_rate = 0.0001 # presented in paper
            # self.learning_rate = 0.001
            self.optimizer_types = ['ADAM']
        else:
            self.learning_rate = 0.01
            self.optimizer_types = ['SGD']
            
            # self.learning_rate = 0.0001
            # #### self.learning_rate = 0.001
            # self.optimizer_types = ['ADAM']
            
            # self.learning_rate = 0.0001
            # #### self.learning_rate = 0.001
            # self.optimizer_types = ['ADAMW']
        
        self.geo_loss_ratio = 0.5
        self.ess_loss_ratio = 0.1

        self.ratio_test_th = 0.8
        
        self.obj_geod_th = 1e-4
        
        self.geo_loss_margin = 0.1
        
        self.validation_chunk_or_all = 'chunk'
        self.validation_chunk_or_all = 'all'
        
        self.save_checkpoint_last_or_all = 'last'
        self.save_checkpoint_last_or_all = 'all'
        
        self.en_grad_checkpointing = False
        # self.en_grad_checkpointing = True
    
    def get_n_chunks_from_files(self):
        
        chunk = 0
        if(self.storage_local_or_bucket == 'local'):
            while True:            
                file_name_with_path = os.path.join(self.input_path_pickle_local, 'train' + f'_{chunk:04d}' + '.pkl')            
                if os.path.isfile(file_name_with_path):
                    chunk += 1
                else:
                    return chunk

        elif(self.storage_local_or_bucket == 'bucket'):                
            from google.cloud import storage
            
            storage_client = storage.Client()    
            bucket = storage_client.get_bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=self.input_path_pickle_bucket)

            blob_names = []
            for blob in blobs:
                blob_names.append(blob.name)

            while True:
                file_name = os.path.join(self.input_path_pickle_bucket, 'train' + f'_{chunk:04d}' + '.pkl')
                if(file_name in blob_names):
                    chunk += 1
                else:
                    return chunk
                
    def copy_config_file_to_output_folder(self, experiment):    
        
        folder_name = os.path.join(self.output_path_local, f'{experiment:04d}')        
        if(not os.path.isdir(folder_name)):
            os.makedirs(folder_name)        
        destination = os.path.join(folder_name, 'config.py')
        shutil.copyfile('config.py', destination)
        
        self.update_output_folder(experiment)
    
    def update_output_folder(self, experiment):
    
        self.output_path_local = os.path.join(self.output_path_local, f'{experiment:04d}')
    
    def update_training_params_for_test(self):
        
        for training_params in self.training_params:
            training_params[2] = training_params[1]

def get_config():
    return Config()
        
if __name__ == '__main__':
    
    config = Config()
