#!/usr/bin/python
import sys
sys.path.insert(0, '/Users/ragav/GitHub/Convolutional-Neural-Networks/')

from samosa.core import ReLU, Abs
from samosa.util import load_network
from samosa.cnn import cnn_mlp
from compress import build_probes, l1_error, l2_error, cross_entropy_cost

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
    parent_net.init_data ( dataset = dataset , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    parent_net.build_network(verbose = verbose)                               
    
    parent_net.train(   n_epochs = parent_params["parent_n_epochs"], 
                        ft_epochs = parent_params ["parent_ft_epochs"],
                        validate_after_epochs = parent_params ["parent_validate_after_epochs"],
                        verbose = verbose )          
    parent_net.test( verbose = verbose )                                     
    parent_net.save_network ()                                   
    

    print "... child network"        
    child_filename_params          = child_params["child_filename_params"]
    child_arch_params              = child_params["child_arch_params"]
    child_optimization_params      = child_params["child_optimization_params"]
    child_retrain_params           = None
    child_init_params              = None                 
    child_net = cnn_mlp(   
                        filename_params = child_filename_params,
                        arch_params = child_arch_params,
                        optimization_params = child_optimization_params,
                        retrain_params = child_retrain_params,
                        init_params = child_init_params,
                        verbose =verbose ) 
    child_net.init_data ( dataset = dataset , outs = child_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    child_net.build_network(verbose = verbose)                            
    build_probes(child_net, parent_net, joint_params, verbose = verbose)                                                
    child_net.train(    n_epochs = child_params ["child_n_epochs"], 
                        ft_epochs = child_params ["child_ft_epochs"] ,
                        validate_after_epochs = child_params ["child_validate_after_epochs"],
                        verbose = verbose )          
    child_net.test( verbose = verbose )                                     
    child_net.save_network ()     
                          
## Boiler Plate ## 
if __name__ == '__main__':
             
    if os.path.isfile('dump.txt'):
        f = open('dump.txt', 'a')
    else:
        f = open('dump.txt', 'w')
        f.close()
        f.open ('dump.txt','a')
        
    f.write("... main net")
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
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.99,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 1,                         
                            "initial_learning_rate"             : 0.01,
                            "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 0,    
                            }        

    parent_arch_params = {
                    
                            "squared_filter_length_limit"       : 15,   
                            "mlp_activations"                   : [ ReLU ],
                            "cnn_dropout"                       : False,
                            "mlp_dropout"                       : False,
                            "mlp_dropout_rates"                 : [ 0.5, 0.5 ],
                            "num_nodes"                         : [ 800  ],                                     
                            "outs"                              : 10,                                                                                                                               
                            "svm_flag"                          : False,                                       
                            "cnn_activations"                   : [ ReLU,  ReLU  ],             
                            "cnn_batch_norm"                    : [ False, False ],
                            "mlp_batch_norm"                    : False,
                            "nkerns"                            : [ 20,   50    ],              
                            "filter_size"                       : [ (5,5),(5,5) ],
                            "pooling_size"                      : [ (2,2),(2,2) ],
                            "conv_stride_size"                  : [ (1,1),(1,1) ],
                            "cnn_maxout"                        : [ 1,    1     ],                    
                            "mlp_maxout"                        : [ 1  ],
                            "cnn_dropout_rates"                 : [ 0.5,  0.5   ],
                            "random_seed"                       : 23455, 
                            "mean_subtract"                     : False,
                            "use_bias"                          : True,
                            "max_out"                           : 0 
        
                 }                          
    """
    child_optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.99,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 0,                         
                            "initial_learning_rate"             : 0.01,
                            "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : False,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 0,   
                            }  
    """
    child_optimization_params = parent_optimization_params
                            
    child_arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU ],
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : False,
                    "mlp_dropout_rates"                 : [ 0.5 , 0.5 ],
                    "num_nodes"                         : [ 800  ],                                     
                    "outs"                              : 10,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ],             
                    "cnn_batch_norm"                    : [ ],
                    "mlp_batch_norm"                    : False,
                    "nkerns"                            : [ ],              
                    "filter_size"                       : [ ],
                    "pooling_size"                      : [ ],
                    "conv_stride_size"                  : [ ],
                    "cnn_maxout"                        : [ ],                    
                    "mlp_maxout"                        : [ 1 ],
                    "cnn_dropout_rates"                 : [ ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "use_bias"                          : True,                    
                    "max_out"                           : 0 
                    
                 } 
                 
    joint_params  =  {                  
                            "learn_layers"                      : [ (2 , 0) ],          # 0 is the first layer
                            "learn_layers_coeffs"               : [ 1 ],                # this is just the coefficients    
                            "error"                             : cross_entropy_cost,   # erros of the probes. 
                            "learn_style"                       : 1,                    # learn_style 0  : parent and child learn together not operational yet.
                                                                                        # learn_style 1  : parent is learnt already in theano, child just learns
                                                                                        # learn_style 2  : parent is learnt in caffe, child just learns            
                            "p"                                 : 1    # combination probability of hard and soft label
                                                                       # 2  p * soft  + (1-p) * hard                                
                      }
                      
                      
    # other loose parameters.     
    parent_n_epochs = 5
    parent_validate_after_epochs = 1
    parent_ft_epochs = 5
    verbose = False  
    dataset = "_datasets/_dataset_57689"
    
    if joint_params["learn_style"] > 0:
            child_n_epochs = parent_n_epochs = parent_n_epochs
            child_validate_after_epochs = parent_validate_after_epochs
            child_ft_epochs = parent_ft_epochs
    else:
            child_n_epochs = 5
            child_validate_after_epochs = 1
            child_ft_epochs = 5


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
                            dataset = dataset,
                            verbose = verbose
                    ) 
    pdb.set_trace()                                