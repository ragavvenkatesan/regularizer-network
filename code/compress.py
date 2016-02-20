#!/usr/bin/python
import theano
import theano.tensor as T
import time

import sys
sys.path.insert(0, '/Users/ragav/GitHub/Convolutional-Neural-Networks/')

from samosa.core import optimizer

def cross_entropy_cost ( a , b ):
    L = - T.sum(a * T.log(b) + (1 - a) * T.log(1 - b), axis=1)
    return T.mean(L)
    
def l2_error ( a,  b ): 
    return T.sqrt(T.mean((a - b) ** 2 ))
    
def l1_error ( a, b ):
    return T.mean( abs (a - b) )    
    
def build_probes(child, parent, joint_params, verbose = True):

    learn_layers            = joint_params [ "learn_layers" ]
    learn_layers_coeffs     = joint_params [ "learn_layers_coeffs" ]
    error                   = joint_params [ "error" ]
    learn_style             = joint_params [ "learn_style" ]
    p                       = joint_params [ "p" ]
    
    start_time = time.clock()
    print '... building the probes'
    # setup child soft loss function 
    if parent.svm_flag or child.svm_flag is True:
        soft_output_cost = error(child.MLPlayers.layers[-1].ouput, parent.MLPlayers.layers[-1].output)
        soft_dropout_output_cost =  error(child.MLPlayers.dropout_layers[-1].ouput, parent.MLPlayers.dropout_layers[-1].output)    
                
    else:
        soft_output_cost = error(child.MLPlayers.layers[-1].p_y_given_x, parent.MLPlayers.layers[-1].p_y_given_x) 
        soft_dropout_output_cost = error(child.MLPlayers.dropout_layers[-1].p_y_given_x, parent.MLPlayers.dropout_layers[-1].p_y_given_x)             
        
    soft_output =  soft_dropout_output_cost if child.mlp_dropout else soft_output_cost
        
    child_cost = 0                     
    if not child.svm_flag is True:
        child_cost = p * soft_output + (1-p) * child.output

    else:
        print "... code doesn't work for non SVM stuff."          
    
    count = 0 
    
    if not parent.nkerns == []:
        parent_layers = parent.ConvLayers.conv_layers + parent.MLPlayers.layers
    else:
        parent_layers = parent.MLPlayers.layers
        
    if not child.nkerns == []:    
        child_layers = child.ConvLayers.conv_layers + child.MLPlayers.layers
    else:
        child_layers = child.MLPlayers.layers
    
    for probe in learn_layers:                            
        child_cost = child_cost + learn_layers_coeffs[count] * error ( parent_layers[probe[0]].output, child_layers[probe[1]].output )
        count = count + 1 
        
    child_optimizer =  optimizer(     
                                    params = child.params,
                                    objective = child_cost,
                                    optimization_params = child.optim_params,
                                )                                    
    child.eta = child_optimizer.eta
    child.epoch = child_optimizer.epoch
    child.updates = child_optimizer.updates
    child.mom = child_optimizer.mom
    end_time = time.clock()
    print "...         time taken is " +str(end_time - start_time) + " seconds"         