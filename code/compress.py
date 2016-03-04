#!/usr/bin/python
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import time
import sys
import numpy 

sys.path.insert(0, '/home/ASUAD/rvenka10/Desktop/ragav/Convolutional-Neural-Networks/')

from samosa.core import optimizer
from samosa.cnn import cnn_mlp


def cross_entropy_cost ( a , b ):
    L = - T.sum(a * T.log(b) + (1 - a) * T.log(1 - b), axis=1)
    return T.mean(L)
    
def l2_error ( a,  b ): 
    return T.sqrt(T.mean((a - b) ** 2 ))
    
def l1_error ( a, b ):
    return T.mean( abs (a - b) )    
            
class regularizer_net(cnn_mlp):        

    def build_probes(self, parent, joint_params, verbose = True):

        self.learn_layers            = joint_params [ "learn_layers" ]
        probe_coeffs                 = joint_params [ "learn_layers_coeffs" ][0]
        soft_output_coeffs           = joint_params [ "learn_layers_coeffs" ][1]
        hard_output_coeffs           = joint_params [ "learn_layers_coeffs" ][2]
        self.error                   = joint_params [ "error" ]
        self.print_probe_costs       = joint_params [ "print_probe_costs" ] 
        
        epoch = theano.shared(numpy.asarray(0., dtype = theano.config.floatX))
        
        probe_weight = ifelse(epoch <= probe_coeffs[2],
            probe_coeffs[0]*(1.0 - epoch/probe_coeffs[2]) + probe_coeffs[1]*(epoch/probe_coeffs[2]),
            float(probe_coeffs[1]))        
        
        soft_weight = ifelse(epoch <= soft_output_coeffs[2],
            soft_output_coeffs[0]*(1.0 - epoch/soft_output_coeffs[2]) + soft_output_coeffs[1]*(epoch/soft_output_coeffs[2]),
            float(soft_output_coeffs[1]))
            
        hard_weight = ifelse(epoch <= hard_output_coeffs[2],
            hard_output_coeffs[0]*(1.0 - epoch/hard_output_coeffs[2]) + hard_output_coeffs[1]*(epoch/hard_output_coeffs[2]),
            float(hard_output_coeffs[1]))   
                    
        start_time = time.clock()
        print 'building the probes'
        # setup child soft loss function
                
        child_cost = 0.
                 
        if not soft_weight == 0:
            if parent.svm_flag or self.svm_flag is True:
                soft_output_cost = self.error(self.MLPlayers.layers[-1].ouput, parent.MLPlayers.layers[-1].output)
                soft_dropout_output_cost =  self.error(self.MLPlayers.dropout_layers[-1].ouput, parent.MLPlayers.dropout_layers[-1].output)    
                        
            else:
                soft_output_cost = self.error(self.MLPlayers.layers[-1].p_y_given_x, parent.MLPlayers.layers[-1].p_y_given_x) 
                soft_dropout_output_cost = self.error(self.MLPlayers.dropout_layers[-1].p_y_given_x, parent.MLPlayers.dropout_layers[-1].p_y_given_x)             
                
            soft_output =  soft_dropout_output_cost if self.mlp_dropout else soft_output_cost                                
            child_cost = child_cost + soft_weight * soft_output 
            
        count = 0 
        
        if not parent.nkerns == []:
            parent_layers = parent.ConvLayers.conv_layers + parent.MLPlayers.layers
        else:
            parent_layers = parent.MLPlayers.layers
            
        if not self.nkerns == []:    
            child_layers = self.ConvLayers.conv_layers + self.MLPlayers.layers
        else:
            child_layers = self.MLPlayers.layers
        
        probe_cost = 0.
        for probe in self.learn_layers:                            
            probe_cost =  probe_cost + self.error ( parent_layers[probe[0]].output, child_layers[probe[1]].output )
            count = count + 1 
            
        child_cost =  child_cost + probe_weight * probe_cost + hard_weight * self.output                 
        child_optimizer =  optimizer(     
                                        params = self.params,
                                        objective = child_cost,
                                        optimization_params = self.optim_params,
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
                                            
        assert self.batch_size == parent.batch_size         
        index = T.lscalar('index')        
        if not probe_cost == 0:
            self.probe_cost_fn = theano.function(
                        inputs= [index, self.epoch],
                        outputs = self.probe_cost,
                        givens={
                            self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                            self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size],
                            parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                            parent.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},
                        on_unused_input = 'ignore'                  
                            )
                            
            self.soft_output_fn = theano.function(
                        inputs= [index, self.epoch],
                        outputs = self.soft_output_cost,
                        givens={
                            self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                            self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size],
                            parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                            parent.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},
                        on_unused_input = 'ignore'                    
                            )
            self.label_cost = theano.function(
                        inputs= [index, self.epoch],
                        outputs = self.output,
                        givens={
                            self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                            self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]}, 
                        on_unused_input = 'ignore'                    
                            )           
            # recreate the training functions with the new updates.... 
            
        if self.svm_flag is True:
            self.train_model = theano.function(
                    inputs= [index, self.epoch],
                    outputs = self.obj_cost,
                    updates = self.updates,
                    givens={
                        self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size],
                        parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        parent.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size]},                                                            
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
                        parent.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        parent.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},                    
                    on_unused_input='ignore'                    
                        )      
                        
        self.decay_learning_rate = theano.function(
                inputs=[],          # Just updates the learning rates. 
                updates={self.eta: self.eta -  self.eta * self.optim_params["learning_rate"][2] }
                )    
                
        self.momentum_value = theano.function ( 
                            inputs =[self.epoch],
                            outputs = self.mom                          
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
                            
        end_time = time.clock()                                                                                                                             
        print "time taken is " +str(end_time - start_time) + " seconds"         
        
    def validate(self, epoch, verbose = True):
        
        super(regularizer_net, self).validate(epoch, verbose = verbose)
        if self.print_probe_costs is True:
            probe_cost = 0.          
            soft_label_cost = 0.
            label_cost = 0.
            if self.multi_load is True:                    
                for batch in xrange (self.batches2train):
                    self.set_data ( batch = batch , type_set = 'train' , verbose = verbose)            
                    probe_cost = probe_cost + numpy.sum([[self.probe_cost_fn(i) for i in xrange(self.n_train_batches)]])            
                    soft_label_cost = soft_label_cost + numpy.sum([[self.soft_output_fn(i) for i in xrange(self.n_train_batches)]])
                    label_cost = label_cost + numpy.sum([[self.label_cost(i) for i in xrange(self.n_train_batches)]])
            else:
                probe_cost = [self.probe_cost_fn(i, epoch) for i in xrange(self.batches2train)]
                soft_label_cost = [self.soft_output_fn(i, epoch) for i in xrange(self.batches2train)]
                label_cost = [self.label_cost(i, epoch) for i in xrange(self.batches2train)]
                                            
            print "   cost of probes      : " + str(numpy.mean(probe_cost)) + ", weight    : " + str(self.probe_weight(epoch))
            print "   cost of soft labels : " + str(numpy.mean(soft_label_cost))  + ", weight    : " + str(self.soft_weight(epoch))
            print "   cost of hard labels : " + str(numpy.mean(label_cost)) + ", weight    : " + str(self.hard_weight(epoch))
            
            
    def test(self, verbose = True):
        
        super(regularizer_net, self).test( verbose = verbose )
        if self.print_probe_costs is True:
            probe_cost = 0.          
            soft_label_cost = 0.
            label_cost = 0.
            if self.multi_load is True:                    
                for batch in xrange (self.batches2train):
                    self.set_data ( batch = batch , type_set = 'train' , verbose = verbose)            
                    probe_cost = probe_cost + numpy.sum([[self.probe_cost_fn(i) for i in xrange(self.n_train_batches)]])            
                    soft_label_cost = soft_label_cost + numpy.sum([[self.soft_output_fn(i) for i in xrange(self.n_train_batches)]])
                    label_cost = label_cost + numpy.sum([[self.label_cost(i) for i in xrange(self.n_train_batches)]])
            else:
                probe_cost = [self.probe_cost_fn(i, 0) for i in xrange(self.batches2train)]
                soft_label_cost = [self.soft_output_fn(i, 0) for i in xrange(self.batches2train)]
                label_cost = [self.label_cost(i, 0) for i in xrange(self.batches2train)]
                                            
            print "   cost of probes      : " + str(numpy.mean(probe_cost)) 
            print "   cost of soft labels : " + str(numpy.mean(soft_label_cost)) 
            print "   cost of hard labels : " + str(numpy.mean(label_cost))
        