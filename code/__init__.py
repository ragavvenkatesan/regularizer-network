#!/usr/bin/python
import sys
sys.path.insert(0, '/home/ASUAD/rvenka10/Desktop/ragav/Convolutional-Neural-Networks/')

from samosa.core import ReLU, Abs
from samosa.util import load_network
from samosa.cnn import cnn_mlp
from compress import regularizer_net, l1_error, l2_error, cross_entropy_cost
from vgg2samosa import load_vgg

import os
import pdb

def solo( 
                    arch_params,
                    optimization_params ,
                    dataset, 
                    filename_params,
                    visual_params,
                    n_epochs = 200,
                    ft_epochs = 100, 
                    validate_after_epochs = 1,
                    verbose = False, 
           ):            
               
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = None,
                     init_params = None,
                     verbose =verbose ) 
    net.init_data ( dataset = dataset , outs = arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    net.build_network(verbose = verbose)                             
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )   
    net.test( verbose = verbose )                                     
    net.save_network ()   
             
                           
                           
def regularized_train( 
                    parent_params,
                    child_params,
                    joint_params,
                    visual_params,
                    dataset,
                    verbose = False, 
                    ):  
                              
    print "... parent network"
    
    model = './dataset/vgg/vgg19.pkl'  # if vgg being loaded 
    freeze_layer_params = True           
                           
    
    parent_filename_params          = parent_params["parent_filename_params"]
    parent_arch_params              = parent_params["parent_arch_params"]
    parent_optimization_params      = parent_params["parent_optimization_params"]
    parent_retrain_params           = None
    parent_init_params              = None
    """    
    # Use this part if creating parent in samosa also.
    parent_net = cnn_mlp(   
                        filename_params = parent_filename_params,
                        arch_params = parent_arch_params,
                        optimization_params = parent_optimization_params,
                        retrain_params = parent_retrain_params,
                        init_params = parent_init_params,
                        verbose =verbose ) 
    """
    parent_net = load_vgg(   
                            model = model,
                            dataset = dataset[0],
                            filename_params = parent_filename_params,
                            optimization_params = parent_optimization_params,
                            freeze_layer_params = freeze_layer_params,
                            visual_params = visual_params,
                            outs = parent_arch_params["outs"],
                            verbose = verbose
                           )     
    """                
    # use this if fine tuning VGG network.          
    parent_net.init_data ( dataset = dataset[0] , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                                
    parent_net.build_network(verbose = verbose)                                   
    parent_net.train(   n_epochs = parent_params["parent_n_epochs"], 
                        ft_epochs = parent_params ["parent_ft_epochs"],
                        validate_after_epochs = parent_params ["parent_validate_after_epochs"],
                        verbose = verbose )  
    parent_net.save_network ()                        
    parent_net.test( verbose = verbose )                                                                                                                 
    """
    # Use this part of code, if the parent was trained elsewhere using samosa and saved.        
    """  
    params_loaded, arch_params_loaded = load_network (  parent_filename_params ["network_save_name"] ,
                                                        data_params = False, 
                                                        optimization_params = False)
    # This is dummy. Its only needed in cases where parent is also going to be retrained
    # considering we aren't retraining make all copy_from_old true and we are good.
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, True, True, True ],
                        "freeze"            : [ False, False, False, False, False, False, False ]
                    }  
    # retrain is used to do the dataset some wierd experiments.     
    parent_net = cnn_mlp(   filename_params = parent_filename_params,
                            arch_params = arch_params_loaded,
                            optimization_params = parent_optimization_params,
                            retrain_params = retrain_params,
                            init_params = params_loaded,
                            verbose =verbose   )  
    """                                            
    parent_net.init_data ( dataset = dataset[1] , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )          
    parent_net.build_network ( verbose = verbose )          
       
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
    child_net.build_probes(parent = parent_net, joint_params = joint_params, verbose = verbose)
    child_net.train(    n_epochs = child_params ["child_n_epochs"], 
                        ft_epochs = child_params ["child_ft_epochs"] ,
                        validate_after_epochs = child_params ["child_validate_after_epochs"],
                        verbose = verbose )
    child_net.save_network ()                                       
    child_net.test( verbose = verbose )                                     
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
                        "network_save_name"     : "../results/parent_network.pkl "
                    }

    child_filename_params = { 
                        "results_file_name"     : "../results/child_results.txt",      
                        "error_file_name"       : "../results/child_error.txt",
                        "cost_file_name"        : "../results/child_cost.txt",
                        "confusion_file_name"   : "../results/child_confusion.txt",
                        "network_save_name"     : "../results/child_network.pkl "
                    }
                                      
    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 36,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                                                      

    parent_optimization_params = {        
                            "mom"                         	    : (0.5, 0.85, 100),    # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 1,                   # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.1,0.001, 0.05 ),  # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.000),       # l1_coeff, l2_coeff                                
                            "optim_type"                        : 2,                   # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                   # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                            }        

    parent_arch_params = {
                    
                    "mlp_activations"                   : [ ReLU, ReLU, ReLU ],                     
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5 , 0.5, 0.5 ],
                    "num_nodes"                         : [ 4096, 4096  ],  
                    "mlp_maxout"                        : [ 1,    1 ],     
                    "mlp_batch_norm"                    : False,                                                                      
                    "outs"                              : 102,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU    ],             
                    "cnn_batch_norm"                    : [ False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False   ],
                    "nkerns"                            : [ 64,     64,     128,    128,    256,    256,    256,    256,    512,    512,    512,    512,    512,    512,    512,    512     ],              
                    "filter_size"                       : [ (3,3), (3,3),   (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3)   ],
                    "pooling_size"                      : [ (1,1), (2,2),   (1,1),  (2,2),  (1,1),  (1,1),  (1,1),  (2,2),  (1,1),  (1,1),  (1,1),  (2,2),  (1,1),  (1,1),  (1,1),  (2,2)   ],
                    "conv_pad"                          : [ 2,     2,       2,      2,      2,      2,      2,      2,       2,      2,      2,      2,     2,      2,      2,      2,      ],                                       
                    "pooling_type"                      : [ 1,     1,       1,      1,      1,      1,      1,      1,       1,      1,      1,      1,     1,      1,      1,      1,      ],
                    "maxrandpool_p"                     : [ 1,     1,       1,      1,      1,      1,      1,      1,       1,      1,      1,      1,     1,      1,      1,      1,      ],                                                                                                    
                    "conv_stride_size"                  : [ (1,1), (1,1),   (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1)   ],
                    "cnn_maxout"                        : [ 1,     1,       1,      1,      1,      1,      1,      1,       1,      1,      1,      1,     1,      1,      1,      1,      ],                    
                    "cnn_dropout_rates"                 : [ 0,     0,       0,      0,      0,      0,      0,      0,       0,      0,      0,      0,     0,      0,      0,      0,      ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "use_bias"                          : True,                    
                    "max_out"                           : 0 
        
        
                 }                          
    # assuming maxpoolrand_p
    
    child_optimization_params = {
        
                            "mom"                         	    : (0.5, 0.85, 100),     # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 2,                    # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.001,0.0001, 0.05 ),          # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.000),       # l1_coeff, l2_coeff                                
                            "optim_type"                        : 0,                   # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 0,                   # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                              
                                }  
                            
    child_arch_params = {
                    
                            "mlp_activations"                   : [  ReLU, ReLU ],
                            "cnn_dropout"                       : False,
                            "mlp_dropout"                       : True,
                            "mlp_dropout_rates"                 : [ 0.5,  0.5, 0.5],
                            "num_nodes"                         : [ 4096, 4096 ],                                     
                            "outs"                              : 102,                                                                                                                               
                            "svm_flag"                          : False,                                       
                            "cnn_activations"                   : [ ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ],             
                            "cnn_batch_norm"                    : [ False,  True,   True,   True,   True,   ],
                            "mlp_batch_norm"                    : True,
                            "nkerns"                            : [  64,    64,     128,    128,    256,    ],              
                            "filter_size"                       : [ (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  ],
                            "pooling_size"                      : [ (1,1),  (3,3),  (2,2),  (3,3),  (2,2),  ],
                            "conv_pad"                          : [ 2,      0,      0,      0,      2,      ],                            
                            "pooling_type"                      : [ 1,      1,      1,      1,      1,      ],
                            "maxrandpool_p"                     : [ 1,      1,      1,      1,      1,      ],                           
                            "conv_stride_size"                  : [ (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  ],
                            "cnn_maxout"                        : [ 1,      1,      1,      1,      1,      ],                    
                            "mlp_maxout"                        : [ 1,      1,      1,      1,      1,      ],
                            "cnn_dropout_rates"                 : [ 0,    0.5,    0.5,    0.5,      0.5,    ],
                            "random_seed"                       : 23455,
                            "mean_subtract"                     : False,
                            "use_bias"                          : True,
                            "max_out"                           : 0 
                 } 
                 
    joint_params  =  {                  
                            "learn_layers"                      : [ (0, 0), (16,5), (17,6) ],       # 0 is the first layer
                            "learn_layers_coeffs"               : [ 
                                                                    (1,0,75),     # This is probes
                                                                    (0,0,100),    # This is soft outputs
                                                                    (1,1,100)     # this is for hard labels
                                                                  ],
                                                                # weights for each probes and soft outputs and labels
                                                                # first term is begining weight,
                                                                # second term is ending weight,
                                                                # third term is until which epoch.    
                            "error"                             : l2_error,         # erros of the probes.
                            "print_probe_costs"                 : False 
                     }
                      
                      
    # other loose parameters.     
    parent_n_epochs = 2
    parent_validate_after_epochs = 1
    parent_ft_epochs = 2
    child_n_epochs = parent_n_epochs
    child_validate_after_epochs = parent_validate_after_epochs
    child_ft_epochs = parent_ft_epochs
        
    verbose = True  
    parent_dataset = "_datasets/_dataset_76223"
    child_dataset = "_datasets/_dataset_76223"

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
                                       
    solo_filename_params = { 
                        "results_file_name"     : "../results/solo_results_small.txt",      
                        "error_file_name"       : "../results/solo_error_small.txt",
                        "cost_file_name"        : "../results/solo_cost_small.txt",
                        "confusion_file_name"   : "../results/solo_confusion_small.txt",
                        "network_save_name"     : "../results/solo_network.pkl "
                    }   
    
    solo(
                    arch_params             = child_arch_params,
                    optimization_params     = child_optimization_params,
                    dataset                 = child_dataset, 
                    filename_params         = solo_filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = child_validate_after_epochs,
                    n_epochs                = child_n_epochs,
                    ft_epochs               = child_ft_epochs, 
                    verbose                 = verbose                                                
                )
                                    
    pdb.set_trace()                                