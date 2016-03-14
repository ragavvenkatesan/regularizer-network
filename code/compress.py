#!/usr/bin/python
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import time
import sys
import numpy 
import progressbar

sys.path.insert(0, '/home/ASUAD/rvenka10/Desktop/ragav/Convolutional-Neural-Networks/')

from samosa.core import optimizer
from samosa.cnn import cnn_mlp


def cross_entropy_cost ( a , b ):
    return T.mean(T.nnet.categorical_crossentropy(a.flatten(2),b.flatten(2)))

def l2_error ( a,  b ): 
    return T.sqrt(T.sum((a - b) ** 2 ))
    
def l1_error ( a, b ):
    return T.sum( abs (a - b) )    
            
def rmse ( a,  b ): 
    return T.sqrt(T.mean((a - b) ** 2 ))
                    
class regularizer_net(cnn_mlp):  

    def __init__ (   self, filename_params, 
                         optimization_params,
                         arch_params,   
                         retrain_params,
                         init_params,     
                         parent,       
                         joint_params,          
                         verbose = False):
                         
        super(regularizer_net, self).__init__ ( filename_params = filename_params,
                                                optimization_params = optimization_params,
                                                arch_params = arch_params,
                                                retrain_params = retrain_params,
                                                init_params = init_params,                                                
                                                verbose = verbose,                                        
                                                )
        self.parent = parent
        self.learn_layers            = joint_params [ "learn_layers" ]
        self.probe_coeffs                 = joint_params [ "learn_layers_coeffs" ][0]
        self.soft_output_coeffs           = joint_params [ "learn_layers_coeffs" ][1]
        self.hard_output_coeffs           = joint_params [ "learn_layers_coeffs" ][2]
        self.error                   = joint_params [ "error" ]
        self.print_probe_costs       = joint_params [ "print_probe_costs" ]         
        
    def reset_params (self, init_params, verbose = True):
        self.init_params = init_params
        if self.retrain_params is None:
            self.retrain_params = {
                                "copy_from_old"     : [True] * (len(self.nkerns) + len(self.num_nodes) + 1),
                                "freeze"            : [False] * (len(self.nkerns) + len(self.num_nodes) + 1)
                                } 
        self.build_network(verbose = verbose)
        self.build_probes(verbose = verbose)
        
        
    def build_probes(self, verbose = True):
        
        start_time = time.clock()
        self.build_cost_function(verbose =verbose)                                                                     
        epoch = theano.shared(numpy.asarray(1, dtype = theano.config.floatX))
        
        index = T.lscalar('index')
        probe_weight = ifelse(epoch <= self.probe_coeffs[2],
            self.probe_coeffs[0]*(1.0 - epoch/self.probe_coeffs[2]) + self.probe_coeffs[1]*(epoch/self.probe_coeffs[2]),
            float(self.probe_coeffs[1]))        
        
        soft_weight = ifelse(epoch <= self.soft_output_coeffs[2],
            self.soft_output_coeffs[0]*(1.0 - epoch/self.soft_output_coeffs[2]) + self.soft_output_coeffs[1]*(epoch/self.soft_output_coeffs[2]),
            float(self.soft_output_coeffs[1]))
            
        hard_weight = ifelse(epoch <= self.hard_output_coeffs[2],
            self.hard_output_coeffs[0]*(1.0 - epoch/self.hard_output_coeffs[2]) + self.hard_output_coeffs[1]*(epoch/self.hard_output_coeffs[2]),
            float(self.hard_output_coeffs[1]))   
                    
        print 'building the probes'
        # setup child soft loss function
                

                 
        if not soft_weight == 0:
            if self.parent.svm_flag or self.svm_flag is True:
                soft_output_cost = self.error(self.MLPlayers.layers[-1].ouput, self.parent.MLPlayers.layers[-1].output)
                soft_dropout_output_cost =  self.error(self.MLPlayers.dropout_layers[-1].ouput, self.parent.MLPlayers.dropout_layers[-1].output)    
                        
            else:
                soft_output_cost = self.error(self.MLPlayers.layers[-1].p_y_given_x, self.parent.MLPlayers.layers[-1].p_y_given_x) 
                soft_dropout_output_cost = self.error(self.MLPlayers.dropout_layers[-1].p_y_given_x, self.parent.MLPlayers.dropout_layers[-1].p_y_given_x)             
                
            soft_output =  soft_dropout_output_cost if self.mlp_dropout else soft_output_cost                                           
        else:
            soft_output = 0.    
        count = 0 
        
        if not self.parent.nkerns == []:
            parent_layers = self.parent.ConvLayers.conv_layers + self.parent.MLPlayers.layers
        else:
            parent_layers = self.parent.MLPlayers.layers
            
        if not self.nkerns == []:    
            child_layers = self.ConvLayers.conv_layers + self.MLPlayers.layers
        else:
            child_layers = self.MLPlayers.layers
        
        probe_cost = 0.
        for probe in self.learn_layers:                            
            probe_cost =  probe_cost + self.error ( parent_layers[probe[0]].output, child_layers[probe[1]].output )
            count = count + 1 
            
        remove_last_layer_vote = 2   
        if verbose is True:
            print "adding probe weights"    
        child_cost =  probe_weight * probe_cost
        if not ( self.soft_output_coeffs[0] == 0  and self.soft_output_coeffs[1] ==0) :
            if verbose is True:
                print "adding soft label loss"
            child_cost = child_cost + soft_weight * soft_output
            remove_last_layer_vote = remove_last_layer_vote - 1
        if not ( self.hard_output_coeffs[0] == 0  and self.hard_output_coeffs[1] ==0) :        
            if verbose is True:
                print "adding hard label loss"
            child_cost = child_cost + hard_weight * self.output  
            remove_last_layer_vote = remove_last_layer_vote - 1
            
        if remove_last_layer_vote == 2:
            if verbose is True:
                print "going to unsupevised mentoring mode"
            self.learnable_params = self.learnable_params[:-2]
  
        child_optimizer =  optimizer(     
                                        params = self.learnable_params,
                                        objective = child_cost,
                                        optimization_params = self.optim_params,
                                        verbose = verbose
                                    )              
                                                          
        self.eta = child_optimizer.eta
        self.epoch = child_optimizer.epoch  
        epoch.default_update = self.epoch
        self.updates = child_optimizer.updates
        self.updates[epoch] = self.epoch
        self.mom = child_optimizer.mom
        self.probe_cost = probe_cost
        self.soft_output_cost = soft_output
        self.obj_cost = child_cost        
                                                                            
        assert self.batch_size == self.parent.batch_size
        
        self.test_model = theano.function(
                inputs = [index],
                outputs = self.errors(self.y),
                givens={
                    self.x: self.test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    self.y: self.test_set_y[index * self.batch_size:(index + 1) * self.batch_size]})  
                      
        self.validate_model = theano.function(
                inputs = [index],
                outputs = self.errors(self.y),
                givens={
                    self.x: self.valid_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    self.y: self.valid_set_y[index * self.batch_size:(index + 1) * self.batch_size]}) 
                       
        self.prediction = theano.function(
            inputs = [index],
            outputs = self.predicts,
            givens={
                    self.x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]}) 
                       
        self.nll = theano.function(
            inputs = [index],
            outputs = self.probabilities,
            givens={
                self.x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})    
        # function to return activations of each image
        
        if not self.nkerns == [] :
            activity = self.ConvLayers.returnActivity()
            self.activities = theano.function (
                inputs = [index],
                outputs = self.activity,
                givens = {
                        self.x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
                        })               
                        
        self.decay_learning_rate = theano.function(
               inputs=[],          # Just updates the learning rates. 
               updates={self.eta: self.eta -  self.eta * self.learning_rate_decay }
                )    
                
        self.momentum_value = theano.function ( 
                            inputs =[self.epoch],
                            outputs = self.mom,
                            )                       
        self.training_accuracy = theano.function(
                inputs = [index],
                outputs = self.errors(self.y),
                givens={
                    self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]})
                            
        index = T.lscalar('index')
        if self.print_probe_costs is True:               
            if not probe_cost == 0:
                if self.svm_flag is False:
                    self.probe_cost_fn = theano.function(
                            inputs= [index, self.epoch],
                            outputs = [self.probe_cost, soft_output, self.output],
                            givens={
                                self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                                self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size],
                                self.parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                                self.parent.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},
                            on_unused_input = 'ignore'
                                )
                else:
                    self.probe_cost_fn = theano.function(
                            inputs= [index, self.epoch],
                            outputs = [self.probe_cost, soft_output, self.output],
                            givens={
                                self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                                self.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size],
                                self.parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                                self.parent.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size]},
                            on_unused_input = 'ignore'
                                )
            self.probe_weight = theano.function ( 
                            inputs =[self.epoch],
                            outputs = probe_weight,
                            
                            )
            self.soft_weight = theano.function ( 
                            inputs =[self.epoch],
                            outputs = soft_weight,
                            )             
            self.hard_weight = theano.function ( 
                            inputs =[self.epoch],
                            outputs = hard_weight,
                            )  
                                                             
        if self.svm_flag is True:
            self.train_model = theano.function(
                    inputs= [index, self.epoch],
                    outputs = self.obj_cost,
                    updates = self.updates,
                    givens={
                        self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size],
                        self.parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.parent.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size]},                                                            
                    on_unused_input = 'ignore'                    
                        )
        else: 
            self.train_model = theano.function(
                    inputs = [index, self.epoch],
                    outputs = self.obj_cost,
                    updates = self.updates,
                    givens={
                        self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size],
                        self.parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.parent.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},                    
                    on_unused_input='ignore'                    
                        )      
                        
        self.decay_learning_rate = theano.function(
                inputs=[],          # Just updates the learning rates. 
                updates={self.eta: self.eta -  self.eta * self.optim_params["learning_rate"][2] }
                )    
                
        end_time = time.clock()  
                                                                                                                                           
        print "time taken is " +str(end_time - start_time) + " seconds"         
        
    def validate(self, epoch, verbose = True):
        
        super(regularizer_net, self).validate(epoch, verbose = verbose)
        if self.print_probe_costs is True:       
            if self.multi_load is True:   
                bar = progressbar.ProgressBar(maxval=self.batches2train, \
                        widgets=[progressbar.AnimatedMarker(), \
                        ' probe cost estimation ', 
                        ' ', progressbar.Percentage(), \
                        ' ',progressbar.ETA(), \
                        ]).start()          
                probe_cost = []
                soft_label_cost = []
                label_cost = []                          
                for batch in xrange (self.batches2train):
                    self.set_data ( batch = batch , type_set = 'train' , verbose = verbose)
                    for i in xrange(self.n_train_batches):
                        _probe, _soft_label, _label = self.probe_cost_fn(i, epoch) 
                        probe_cost = probe_cost + [_probe]
                        soft_label_cost = soft_label_cost + [_soft_label]
                        label_cost = label_cost + [_label]           
                    bar.update(batch + 1)                                        
                bar.finish()
            else:
                bar = progressbar.ProgressBar(maxval=self.n_train_batches, \
                        widgets=[progressbar.AnimatedMarker(), \
                        ' probe cost estimation ', 
                        ' ', progressbar.Percentage(), \
                        ' ',progressbar.ETA(), \
                        ]).start()    
                probe_cost = []
                soft_label_cost = []
                label_cost = []                                  
                for i in xrange(self.n_train_batches):
                    _probe, _soft_label, _label = self.probe_cost_fn(i, epoch) 
                    probe_cost = probe_cost + [_probe]
                    soft_label_cost = soft_label_cost + [_soft_label]
                    label_cost = label_cost + [_label]           
                    bar.update(i + 1)                                        
                bar.finish()                           
            f = open('dump.txt', 'a')                                  
            print "   cost of probes      : " + str(numpy.mean(probe_cost)) + ", weight    : " + str(self.probe_weight(epoch))
            print "   cost of soft labels : " + str(numpy.mean(soft_label_cost))  + ", weight    : " + str(self.soft_weight(epoch))
            print "   cost of hard labels : " + str(numpy.mean(label_cost)) + ", weight    : " + str(self.hard_weight(epoch))
            f.write( "   cost of probes      : " + str(numpy.mean(probe_cost)) + ", weight    : " + str(self.probe_weight(epoch)) + "\n")
            f.write( "   cost of soft labels : " + str(numpy.mean(soft_label_cost))  + ", weight    : " + str(self.soft_weight(epoch)) + "\n")
            f.write( "   cost of hard labels : " + str(numpy.mean(label_cost)) + ", weight    : " + str(self.hard_weight(epoch)) + "\n")
            f.close()
            
            f = open(self.error_file_name,'a')
            f.write("\t" + str(numpy.sum(probe_cost))  + "\t" + str(numpy.sum(soft_label_cost)) + "\t" + str(numpy.sum(label_cost)))
            f.close()
                    
    def test(self, verbose = True):
        
        super(regularizer_net, self).test( verbose = verbose )
        if self.print_probe_costs is True:       
            if self.multi_load is True:   
                bar = progressbar.ProgressBar(maxval=self.batches2train, \
                        widgets=[progressbar.AnimatedMarker(), \
                        ' probe cost estimation ', 
                        ' ', progressbar.Percentage(), \
                        ' ',progressbar.ETA(), \
                        ]).start()          
                probe_cost = []
                soft_label_cost = []
                label_cost = []                          
                for batch in xrange (self.batches2train):
                    self.set_data ( batch = batch , type_set = 'train' , verbose = verbose)
                    for i in xrange(self.n_train_batches):
                        _probe, _soft_label, _label = self.probe_cost_fn(i,0) 
                        probe_cost = probe_cost + [_probe]
                        soft_label_cost = soft_label_cost + [_soft_label]
                        label_cost = label_cost + [_label]           
                    bar.update(batch + 1)                                        
                bar.finish()
            else:
                probe_cost = []
                soft_label_cost = []
                label_cost = []             
                bar = progressbar.ProgressBar(maxval=self.n_train_batches, \
                        widgets=[progressbar.AnimatedMarker(), \
                        ' probe cost estimation ', 
                        ' ', progressbar.Percentage(), \
                        ' ',progressbar.ETA(), \
                        ]).start()             
                for i in xrange(self.n_train_batches):
                    _probe, _soft_label, _label = self.probe_cost_fn(i, 0) 
                    probe_cost = probe_cost + [_probe]
                    soft_label_cost = soft_label_cost + [_soft_label]
                    label_cost = label_cost + [_label]           
                    bar.update(i + 1)                                        
                bar.finish()         
            f = open('dump.txt', 'a')                                  
            print "   cost of probes      : " + str(numpy.mean(probe_cost)) 
            print "   cost of soft labels : " + str(numpy.mean(soft_label_cost)) 
            print "   cost of hard labels : " + str(numpy.mean(label_cost))
            f.write( "   cost of probes      : " + str(numpy.mean(probe_cost)) + "\n")
            f.write( "   cost of soft labels : " + str(numpy.mean(soft_label_cost))  + "\n")
            f.write( "   cost of hard labels : " + str(numpy.mean(label_cost)) + "\n")
            f.close()