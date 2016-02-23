#!/usr/bin/python
import sys
sys.path.insert(0, '/home/ASUAD/rvenka10/Desktop/ragav/Convolutional-Neural-Networks/')

from samosa.core import ReLU, Abs
from samosa.util import load_network
from samosa.cnn import cnn_mlp
from compress import regularizer_net, l1_error, l2_error, cross_entropy_cost

import os
import pdb

def regularized_train( 
                    parent_params,
                    child_params,
                    joint_params,
                    visual_params,
                    dataset,
                    verbose = False, 
                    ):  
                              
    print "... parent network"
    """
    parent_filename_params          = parent_params["parent_filename_params"]
    parent_arch_params              = parent_params["parent_arch_params"]
    parent_optimization_params      = parent_params["parent_optimization_params"]
    parent_retrain_params           = None
    parent_init_params              = None    
    parent_net = cnn_mlp(   
                        filename_params = parent_filename_params,
                        arch_params = parent_arch_params,
                        optimization_params = parent_optimization_params,
                        retrain_params = parent_retrain_params,
                        init_params = parent_init_params,
                        verbose =verbose ) 
    parent_net.init_data ( dataset = dataset[0] , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    parent_net.build_network(verbose = verbose)                               
    
    parent_net.train(   n_epochs = parent_params["parent_n_epochs"], 
                        ft_epochs = parent_params ["parent_ft_epochs"],
                        validate_after_epochs = parent_params ["parent_validate_after_epochs"],
                        verbose = verbose )          
    parent_net.test( verbose = verbose )                                     
    parent_net.save_network ()                                   
    """ 
    # Use this part of code, if the parent was trained elsewhere using samosa and saved.    
    params_loaded, arch_params_loaded = load_network (  parent_filename_params ["network_save_name"] ,
                                                        data_params = False, 
                                                        optimization_params = False)   
 
    # This is dummy. Its only needed in cases where parent is also going to be retrained
    # considering we aren't retraining make all copy_from_old true and we are good.
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, True ],
                        "freeze"            : [ True, True, True, True, True, True, False ]
                    }  
    # retrain is used to do the dataset some wierd experiments.     
    parent_net = cnn_mlp(   filename_params = parent_filename_params,
                            arch_params = arch_params_loaded,
                            optimization_params = parent_optimization_params,
                            retrain_params = retrain_params,
                            init_params = params_loaded,
                            verbose =verbose   )  
                                                
    parent_net.init_data ( dataset = dataset[0] , outs = arch_params_loaded ["outs"], visual_params = visual_params, verbose = verbose )      
    parent_net.build_network ( verbose = verbose )       
    parent_net.test( verbose = verbose )                                     
           
    print "... child network"        
    child_filename_params          = child_params["child_filename_params"]
    child_arch_params              = child_params["child_arch_params"]
    child_optimization_params      = child_params["child_optimization_params"]
    child_retrain_params           = None
    child_init_params              = None                 
    child_net = regularizer_net(   
                        filename_params = child_filename_params,
                        arch_params = child_arch_params,
                        optimization_params = child_optimization_params,
                        retrain_params = child_retrain_params,
                        init_params = child_init_params,
                        verbose =verbose ) 
    child_net.init_data ( dataset = dataset[1] , outs = child_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    child_net.build_network(verbose = verbose)    
    # I don't know why I need a return, isn't python always pass by reference ?                        
    child_net.build_probes(parent = parent_net, joint_params = joint_params, verbose = verbose)                                           
    child_net.train(    n_epochs = child_params ["child_n_epochs"], 
                        ft_epochs = child_params ["child_ft_epochs"] ,
                        validate_after_epochs = child_params ["child_validate_after_epochs"],
                        verbose = verbose )          
    child_net.test( verbose = verbose )                                     
    child_net.save_network ()     
      
    pdb.set_trace()                                                          
## Boiler Plate ## 
if __name__ == '__main__':

    if os.path.isfile('dump.txt'):
        os.remove('dump.txt')
        
    f = open('dump.txt', 'w')        

    # run the base CNN as usual.              
    parent_filename_params = { 
                        "results_file_name"     : "../results/parent_results.txt",      
                        "error_file_name"       : "../results/parent_error.txt",
                        "cost_file_name"        : "../results/parent_cost.txt",
                        "confusion_file_name"   : "../results/parent_confusion.txt",
                        "network_save_name"     : "../results/parent_network.pkl.gz "
                    }

    child_filename_params = { 
                        "results_file_name"     : "../results/child_results_small.txt",      
                        "error_file_name"       : "../results/child_error_small.txt",
                        "cost_file_name"        : "../results/child_cost_small.txt",
                        "confusion_file_name"   : "../results/child_confusion_small.txt",
                        "network_save_name"     : "../results/child_network.pkl.gz "
                    }
                                      
    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 81,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                                                      

    parent_optimization_params = {
                            "mom"                               :( 0.5, 0.99, 100 ),                      
                            "mom_type"                          : 2,                         
                            "initial_learning_rate"             : 0.001,
                            "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,    
                            }        

    parent_arch_params = {
                    
                            "mlp_activations"                   : [  ReLU, ReLU, ReLU ],
                            "cnn_dropout"                       : True,
                            "mlp_dropout"                       : True,
                            "mlp_dropout_rates"                 : [ 0.5, 0.5, 0.5, 0.5],
                            "num_nodes"                         : [ 800, 800, 400 ],                                     
                            "outs"                              : 10,                                                                                                                               
                            "svm_flag"                          : False,                                       
                            "cnn_activations"                   : [  ],             
                            "cnn_batch_norm"                    : [ True,   True,   True ],
                            "mlp_batch_norm"                    : True,
                            "nkerns"                            : [  ],              
                            "filter_size"                       : [ (5,5),  (3,3),  (3,3) ],
                            "pooling_size"                      : [ (2,2),  (2,2),  (1,1) ],
                            "conv_stride_size"                  : [ (1,1),  (1,1),  (1,1) ],
                            "cnn_maxout"                        : [ 1,      1,      1   ],                    
                            "mlp_maxout"                        : [ 1,      1,      1   ],
                            "cnn_dropout_rates"                 : [ 0.5,    0.5,    0.5   ],
                            "random_seed"                       : 23455, 
                            "mean_subtract"                     : False,
                            "use_bias"                          : True,
                            "max_out"                           : 0 
        
                 }                          
    
    child_optimization_params = {
                            "mom"                               : ( 0.5, 0.85, 100 ),
                            "mom_type"                          : 2,                         
                            "initial_learning_rate"             : 0.1,
                            "ft_learning_rate"                  : 0.001,    
                            "learning_rate_decay"               : 0.05,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,      
                            }  
                            
    child_arch_params = {
                    
                    "mlp_activations"                   : [ ReLU ],
                    "cnn_dropout"                       : True,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5 , 0.5 ],
                    "num_nodes"                         : [ 400  ],                                     
                    "outs"                              : 10,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ],             
                    "cnn_batch_norm"                    : [ ],
                    "mlp_batch_norm"                    : True,
                    "nkerns"                            : [ ],              
                    "filter_size"                       : [ ],
                    "pooling_size"                      : [ ],
                    "conv_stride_size"                  : [ ],
                    "cnn_maxout"                        : [ ],                    
                    "mlp_maxout"                        : [ 1 ],
                    "cnn_dropout_rates"                 : [ ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "use_bias"                          : False,                    
                    "max_out"                           : 0 
                    
                 } 
                 
    joint_params  =  {                  
                            "learn_layers"                      : [ (2, 0) ],       # 0 is the first layer
                            "learn_layers_coeffs"               : [ 
                                                                    (0.95,0.2,100),    # This is probes
                                                                    (0.75,0.75,100),   # This is soft outputs
                                                                    (0.5,1,100)        # this is for hard labels
                                                                     ],
                                                                # weights for each probes and soft outputs and labels
                                                                # first term is begining weight,
                                                                # second term is ending weight,
                                                                # third term is until which epoch.    
                            "error"                             : l2_error,         # erros of the probes.
                            "print_probe_costs"                 : True 
                     }
                      
                      
    # other loose parameters.     
    parent_n_epochs = 100
    parent_validate_after_epochs = 1
    parent_ft_epochs = 100
    child_n_epochs = parent_n_epochs = parent_n_epochs
    child_validate_after_epochs = parent_validate_after_epochs
    child_ft_epochs = parent_ft_epochs
        
    verbose = False  
    parent_dataset = "_datasets/_dataset_24340"
    child_dataset = "_datasets/_dataset_96494"




    # Don't edit            
    parent_params  =  {
                            "parent_arch_params"                : parent_arch_params,
                            "parent_filename_params"            : parent_filename_params,
                            "parent_ft_epochs"                  : parent_ft_epochs,
                            "parent_n_epochs"                   : parent_n_epochs,
                            "parent_optimization_params"        : parent_optimization_params,
                            "parent_validate_after_epochs"      : parent_validate_after_epochs
                      }
    
    child_params  =  {
                            "child_arch_params"                : child_arch_params,
                            "child_filename_params"            : child_filename_params,
                            "child_ft_epochs"                  : child_ft_epochs,
                            "child_n_epochs"                   : child_n_epochs,
                            "child_optimization_params"        : child_optimization_params,
                            "child_validate_after_epochs"      : child_validate_after_epochs
                      }                                       
                                    
    regularized_train (   
                            parent_params = parent_params,
                            child_params = child_params, 
                            joint_params = joint_params,
                            visual_params = visual_params,
                            dataset = (parent_dataset, child_dataset),
                            verbose = verbose
                    ) 
    pdb.set_trace()                                