import os
import sys
import keras
import json
import csv
import numpy as np
import keras.backend as K
from utils.mean_squared_error import metric_cor
from utils.Adam_mult import AdamWarmup, calc_train_steps
from model.checkpoint_loader import load_model
from keras.callbacks import ModelCheckpoint



class training:

        def __init__(self, 
                     BERT_CONFIG_PATH=None,
                     CHECKPOINT_PATH=None, 
                     NUM_CLASSES=2, 
                     SEQ_LEN=128,
                     EPOCHS=5,
                     BATCH_SIZE=64,
                     TASK=None,
                     OUTPUT_DIR=None,
                     train_data=None,
                     dev_data=None,
                     LOGFILE_PATH=None):
        
            """
            Makes an instance of training.
            Args:
                dev_data: The entire validation data.
                train_data: The entire training data in the format as taken by fit()
            Rest of the args are self explanatory.
            Returns:
                A training instance that can be used for finetuning, config search & retraining.
            """

            self.BERT_CONFIG_PATH = BERT_CONFIG_PATH
            self.CHECKPOINT_PATH = CHECKPOINT_PATH
            self.NUM_CLASSES = NUM_CLASSES
            self.SEQ_LEN = SEQ_LEN
            self.EPOCHS = EPOCHS
            self.BATCH_SIZE = BATCH_SIZE
            self.TASK = TASK
            self.OUTPUT_DIR = OUTPUT_DIR
            self.train_data = train_data
            self.dev_data = dev_data
            self.LOGFILE_PATH = LOGFILE_PATH
            self.NUM_TRAIN = train_data[1].shape[0]

            self.loss = 'sparse_categorical_crossentropy'
            self.metric = ['sparse_categorical_accuracy']
            self.validation_metric = 'val_sparse_categorical_accuracy'

            if self.TASK == "sts-b":
                self.loss = 'mean_squared_error'
                self.metric = [metric_cor,'mae']
                self.validation_metric = 'val_metric_cor'


        def fine_tuning_step(self,
                             LR_BERT=0.00003):
            
            """
            Carries out simple fine-tuning for the data given in train_data above.
            Use it only if the model at checkpoint path has never seen this data.
            Returns:
                A string representing path to the fine-tuned model checkpoint.
            """

            fine_tuned_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                FLAG_EXTRACT_LAYER=0,
                TASK=self.TASK)
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )
            fine_tuned_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_BERT, 
                            lr_mult=None),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Fine-tuned model summary: ", fine_tuned_model.summary())

            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"finetune.hdf5")
            checkpoint = ModelCheckpoint(SAVE_CP_PATH, 
                                         monitor=self.validation_metric,
                                         verbose=1, 
                                         save_best_only=True,
                                         mode='max')
            history = fine_tuned_model.fit(self.train_data[0], 
                                 self.train_data[1], 
                                 batch_size=self.BATCH_SIZE, 
                                 epochs=self.EPOCHS, 
                                 validation_data=(self.dev_data[0], self.dev_data[1], None),
                                 verbose=1, 
                                 callbacks=[checkpoint])
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Fine-tuned model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))

            keras.backend.clear_session()

            return SAVE_CP_PATH


        def configuration_search_step(self, 
                                      fine_tuned_model_path=None, 
                                      LAMBDA=0.0001, 
                                      LR_BERT=0.00003, 
                                      LR_SOFT_EXTRACT=0.0001):
        
            """
            Searches for a good output reduction configuration on given model and data.
                lambda_hyperparam: See the paper for the meaning of this hyper parameter.
                                   Used for searching the configuration.
                fine_tuned_model_path: Path to a checkpoint model that has been finetuned
                                       on given data. Should have been finetuned using above
                                       funtion only.
            Returns:
                (String, np.array): Path to checkpoint representing config search model
                and the retention configuration for this model.
            """

            ## Define a PoWER-BERT model containing Soft-Extract Layers
            configuration_search_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                LAMBDA=LAMBDA,
                FLAG_EXTRACT_LAYER=1,
                TASK=self.TASK)
                
            configuration_search_model.load_weights(fine_tuned_model_path, by_name=True)
            
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )

            ## Set different Learning rates for original BERT parameters and the retnetion parameters fo the Soft-Extract Layers
            lr_mult = {}
            for layer in configuration_search_model.layers:
                if 'Extract' in layer.name:
                        lr_mult[layer.name] = 1.0
                else:
                        lr_mult[layer.name] = LR_BERT/LR_SOFT_EXTRACT

            configuration_search_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_SOFT_EXTRACT, 
                            lr_mult=lr_mult),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Configuration Search model summary: ", configuration_search_model.summary())
            
            ## Train the model
            configuration_search_model.fit(self.train_data[0],
                                           self.train_data[1],
                                           batch_size=self.BATCH_SIZE, 
                                           epochs=self.EPOCHS, 
                                           validation_data=(self.dev_data[0], self.dev_data[1], None),
                                           verbose=1)
            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,'configuration_search_model.hdf5')
            configuration_search_model.save(os.path.join(SAVE_CP_PATH))

            ## Obtain the retention configuration by calculating the mass of each encoder layer
            retention_configuration = self.get_configuration(configuration_search_model)
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Retention Configuration :" + str(retention_configuration))

            keras.backend.clear_session()
    
            return SAVE_CP_PATH, retention_configuration


        def get_configuration(self, configuration_search_model):

            """
            Computes the retention config given a trained model with soft extract layers.
            Args:
                configuration_search_model: A keras.models.Model that has been given by configuration_search_step().
            Returns:
                np.array that contains the word vector elimination (output reduction) configuration.
            """

            retention_configuration = []
            for layer in configuration_search_model.layers:
                for i in range(1,13):
                    if layer.name == 'Encoder-' + str(i) + '-MultiHeadSelfAttention-Soft-Extract':
                        ww = layer.get_weights()
                        weight_sum = int(np.sum(ww[0]))
                        if weight_sum == 0:
                            weight_sum = 1
                        if len(retention_configuration) == 0 or weight_sum < retention_configuration[-1]:
                            retention_configuration.append(weight_sum)
                        else:
                            retention_configuration.append(retention_configuration[-1])
            print ("Retention Configuration :", retention_configuration, np.sum(retention_configuration))

            return retention_configuration


        def retraining_step(self, 
                            configuration_search_model_path=None, 
                            retention_configuration=[], 
                            LR_BERT=0.00003):   
        
            """
            Switches Soft Extract layer to a Hard Extract layer and trains on the given data.
            Args:
                configuration_search_model_path: Path to a checkpoint as given by
                                                 configuration_search_step().
                retention_configuration: A list of integers representing the number
                                         of word-vectors to retain after each layer.
            Returns:
                A keras.models.Model instance that contains the Hard Extract layer,
                and can be used for prediction with word-vector elimination.
            """

            ## Define a PoWER-BERT model where Soft-Extract Layers have been replaced by Extract Layers that eliminates the word-vectors
            retrained_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                retention_configuration=retention_configuration,
                FLAG_EXTRACT_LAYER=2,
                TASK=self.TASK)
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )
            retrained_model.load_weights(configuration_search_model_path, by_name=True)

            retrained_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_BERT, 
                            lr_mult=None),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Re-trained model summary: ", retrained_model.summary())

            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"retrained.hdf5")
            checkpoint = ModelCheckpoint(SAVE_CP_PATH, 
                                         monitor=self.validation_metric, 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max')
            history = retrained_model.fit(self.train_data[0],
                                self.train_data[1],
                                batch_size=self.BATCH_SIZE, 
                                epochs=self.EPOCHS, 
                                validation_data=(self.dev_data[0], self.dev_data[1], None),
                                verbose=1,
                                callbacks=[checkpoint])
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Re-trained model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))


                    
                    
        def retraining_step_pooling(self, 
                                    configuration_search_model_path=None, 
                                    retention_configuration=[], 
                                    LR_BERT=0.00003):   
        
            """
            Switches Soft Extract layer to a Hard Extract layer and trains on the given data.
            Args:
                configuration_search_model_path: Path to a checkpoint as given by
                                                 configuration_search_step().
                retention_configuration: A list of integers representing the number
                                         of word-vectors to retain after each layer.
            Returns:
                A keras.models.Model instance that contains the Hard Extract layer,
                and can be used for prediction with word-vector elimination.
            """

            ## Define a PoWER-BERT model where Soft-Extract Layers have been replaced by Extract Layers that eliminates the word-vectors
            retrained_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                retention_configuration=retention_configuration,
                FLAG_EXTRACT_LAYER=3,
                TASK=self.TASK)
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )
            retrained_model.load_weights(configuration_search_model_path, by_name=True)

            retrained_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_BERT, 
                            lr_mult=None),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Re-trained pooling model summary: ", retrained_model.summary())

            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"retrained.pooling.hdf5")
            checkpoint = ModelCheckpoint(SAVE_CP_PATH, 
                                         monitor=self.validation_metric, 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max')
            history = retrained_model.fit(self.train_data[0],
                                self.train_data[1],
                                batch_size=self.BATCH_SIZE, 
                                epochs=self.EPOCHS, 
                                validation_data=(self.dev_data[0], self.dev_data[1], None),
                                verbose=1,
                                callbacks=[checkpoint])
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Re-trained pooling model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))

                    
                    
        def retraining_step_weight_sum(self, 
                                       configuration_search_model_path=None, 
                                       retention_configuration=[], 
                                       LR_BERT=0.00003):   
        
            """
            Switches Soft Extract layer to a Hard Extract layer and trains on the given data.
            Args:
                configuration_search_model_path: Path to a checkpoint as given by
                                                 configuration_search_step().
                retention_configuration: A list of integers representing the number
                                         of word-vectors to retain after each layer.
            Returns:
                A keras.models.Model instance that contains the Hard Extract layer,
                and can be used for prediction with word-vector elimination.
            """

            ## Define a PoWER-BERT model where Soft-Extract Layers have been replaced by Extract Layers that eliminates the word-vectors
            retrained_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                retention_configuration=retention_configuration,
                FLAG_EXTRACT_LAYER=4,
                TASK=self.TASK)
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )
            retrained_model.load_weights(configuration_search_model_path, by_name=True)

            retrained_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_BERT, 
                            lr_mult=None),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Re-trained weight_sum model summary: ", retrained_model.summary())

            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"retrained.weight_sum.hdf5")
            checkpoint = ModelCheckpoint(SAVE_CP_PATH, 
                                         monitor=self.validation_metric, 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max')
            history = retrained_model.fit(self.train_data[0],
                                self.train_data[1],
                                batch_size=self.BATCH_SIZE, 
                                epochs=self.EPOCHS, 
                                validation_data=(self.dev_data[0], self.dev_data[1], None),
                                verbose=1,
                                callbacks=[checkpoint])
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Re-trained weight_sum model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))

                    
        def retraining_step_weight_sum_cluster(self, 
                                             configuration_search_model_path=None, 
                                             retention_configuration=[], 
                                             LR_BERT=0.00003):   
        
            """
            Switches Soft Extract layer to a Hard Extract layer and trains on the given data.
            Args:
                configuration_search_model_path: Path to a checkpoint as given by
                                                 configuration_search_step().
                retention_configuration: A list of integers representing the number
                                         of word-vectors to retain after each layer.
            Returns:
                A keras.models.Model instance that contains the Hard Extract layer,
                and can be used for prediction with word-vector elimination.
            """

            ## Define a PoWER-BERT model where Soft-Extract Layers have been replaced by Extract Layers that eliminates the word-vectors
            retrained_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                retention_configuration=retention_configuration,
                FLAG_EXTRACT_LAYER=5,
                TASK=self.TASK)
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )
            retrained_model.load_weights(configuration_search_model_path, by_name=True)

            retrained_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_BERT, 
                            lr_mult=None),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Re-trained cluster5_weight_sum model summary: ", retrained_model.summary())

            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"retrained.cluster5_weight.hdf5")
            checkpoint = ModelCheckpoint(SAVE_CP_PATH, 
                                         monitor=self.validation_metric, 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max')
            history = retrained_model.fit(self.train_data[0],
                                self.train_data[1],
                                batch_size=self.BATCH_SIZE, 
                                epochs=self.EPOCHS, 
                                validation_data=(self.dev_data[0], self.dev_data[1], None),
                                verbose=1,
                                callbacks=[checkpoint])
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Re-trained cluster5_weight_sum model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))

                    
                    
        def retraining_step_pool_cluster(self, 
                                         configuration_search_model_path=None, 
                                         retention_configuration=[], 
                                         LR_BERT=0.00003):   
        
            """
            Switches Soft Extract layer to a Hard Extract layer and trains on the given data.
            Args:
                configuration_search_model_path: Path to a checkpoint as given by
                                                 configuration_search_step().
                retention_configuration: A list of integers representing the number
                                         of word-vectors to retain after each layer.
            Returns:
                A keras.models.Model instance that contains the Hard Extract layer,
                and can be used for prediction with word-vector elimination.
            """

            ## Define a PoWER-BERT model where Soft-Extract Layers have been replaced by Extract Layers that eliminates the word-vectors
            retrained_model = load_model(
                self.BERT_CONFIG_PATH,
                self.CHECKPOINT_PATH,
                FLAG_BERT_PRETRAINED=True,
                output_dim=self.NUM_CLASSES,
                seq_len=self.SEQ_LEN,
                retention_configuration=retention_configuration,
                FLAG_EXTRACT_LAYER=6,
                TASK=self.TASK)
            decay_steps, warmup_steps = calc_train_steps(
                self.NUM_TRAIN,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
            )
            retrained_model.load_weights(configuration_search_model_path, by_name=True)

            retrained_model.compile(
                AdamWarmup(decay_steps=decay_steps, 
                            warmup_steps=warmup_steps, 
                            lr=LR_BERT, 
                            lr_mult=None),
                loss=self.loss,
                metrics=self.metric,
            )
            print ("Re-trained cluster5_pool model summary: ", retrained_model.summary())

            SAVE_CP_PATH = os.path.join(self.OUTPUT_DIR,"retrained.cluster5_pool.hdf5")
            checkpoint = ModelCheckpoint(SAVE_CP_PATH, 
                                         monitor=self.validation_metric, 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max')
            history = retrained_model.fit(self.train_data[0],
                                self.train_data[1],
                                batch_size=self.BATCH_SIZE, 
                                epochs=self.EPOCHS, 
                                validation_data=(self.dev_data[0], self.dev_data[1], None),
                                verbose=1,
                                callbacks=[checkpoint])
            with open(self.LOGFILE_PATH, 'a') as fp:
                    fp.write("\n Re-trained cluster5_pool model accuracies for all epochs on the Dev set:" + str(history.history[self.validation_metric]))