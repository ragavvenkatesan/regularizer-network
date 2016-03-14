#!/usr/bin/python
import sys
sys.path.insert(0, '/Users/ragav/GitHub/Convolutional-Neural-Networks/')

from samosa.core import ReLU, Softmax
from samosa.util import load_network
from samosa.cnn import cnn_mlp
from compress import regularizer_net, rmse  ## these are the errors available
from samosa.vgg2samosa import load_vgg

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
           
    print "solo network"
    """
    # use this if solo was trained and you want to fine tune..
    params_loaded, arch_params_loaded = load_network (  filename_params ["network_save_name"] ,
                                                        data_params = False, 
                                                        optimization_params = False)               
    retrain_params = {
                        "copy_from_old"     : [ True ] * (len(arch_params_loaded["nkerns"]) + len(arch_params_loaded["num_nodes"])+1),
                        "freeze"            : [ False ] * (len(arch_params_loaded["nkerns"]) + len(arch_params_loaded["num_nodes"])+1)
                    }
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params_loaded,
                     optimization_params = optimization_params,
                     retrain_params = retrain_params,
                     init_params = params_loaded,
                     verbose =verbose ) 
                          
    """       
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = None,
                     init_params = None,
                     verbose =verbose ) 
                     
    net.init_data ( dataset = dataset , outs = arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    net.build_network(verbose = verbose)  
    net.build_learner(verbose =verbose)                                                                                       
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 patience_epochs = 4,                 
                 verbose = verbose )   
    net.test( verbose = verbose )                                     
    net.save_network ()   





    pdb.set_trace()
             
# Main function.                           
def regularized_train( 
                    parent_params,
                    child_params,
                    joint_params,
                    visual_params,
                    dataset,
                    verbose = False, 
                    ):  
                              
    print "obi-wan network"                      
    
    parent_filename_params          = parent_params["parent_filename_params"]
    parent_arch_params              = parent_params["parent_arch_params"]
    parent_optimization_params      = parent_params["parent_optimization_params"]
    parent_retrain_params           = None
    parent_init_params              = None
    
    
    
    
    
    
    
    
    """     
    ## use this if Parent is VGG ... 
    model = './dataset/vgg/vgg19.pkl'  # if vgg being loaded 
    freeze_layer_params = True                                 
    obi_wan = load_vgg(   
                            model = model,
                            dataset = dataset[0],
                            filename_params = parent_filename_params,
                            optimization_params = parent_optimization_params,
                            freeze_layer_params = freeze_layer_params,
                            visual_params = visual_params,
                            outs = parent_arch_params["outs"],
                            verbose = verbose
                           )   
    obi_wan.build_network(verbose = verbose)                               
    """             
    
    
    
    
    
    
    
    
    
    
    # Use these to load a pre-trained parent in samosa.
    parent_init_params, parent_arch_params = load_network (  parent_filename_params ["network_save_name"] ,
                                                        data_params = False, 
                                                        optimization_params = False)   
    parent_arch_params["outs"] = child_params["child_arch_params"]["outs"]
    parent_retrain_params = {
                        "copy_from_old"     : [ True ] * (len(parent_arch_params["nkerns"]) + len(parent_arch_params["num_nodes"])) + [True], 
                        "freeze"            : [ False ] * (len(parent_arch_params["nkerns"]) + len(parent_arch_params["num_nodes"])) + [False]
                    } 
    obi_wan = cnn_mlp(   
                        filename_params = parent_filename_params,
                        arch_params = parent_arch_params,
                        optimization_params = parent_optimization_params,
                        retrain_params = parent_retrain_params,
                        init_params = parent_init_params,
                        verbose =verbose )                            
    obi_wan.init_data ( dataset = dataset[1] , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                                
    obi_wan.build_network(verbose = verbose)  
    # The next three lines might be skipped if not retraining.     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """   
    ## Create a new parent from scratch in samosa                     
    obi_wan = cnn_mlp(   
                        filename_params = parent_filename_params,
                        arch_params = parent_arch_params,
                        optimization_params = parent_optimization_params,
                        retrain_params = parent_retrain_params,
                        init_params = parent_init_params,
                        verbose =verbose )                            
    obi_wan.init_data ( dataset = dataset[0] , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                                
    obi_wan.build_network(verbose = verbose)  
    obi_wan.build_learner(verbose =verbose)  
                                                                                                
    obi_wan.train(   n_epochs = parent_params["parent_n_epochs"], 
                        ft_epochs = parent_params ["parent_ft_epochs"],
                        validate_after_epochs = parent_params ["parent_validate_after_epochs"],
                        patience_epochs = 4,
                        verbose = verbose )
                         
    obi_wan.save_network ()                              
    obi_wan.test( verbose = verbose )  
    """
    
    
    
    
    
    
    
    
    
    
    """
    # use this to fine tune a pre-trained obi-wan on the skywalker dataset.
    print "obi-wan fine tuning on target dataset"
    #setting the network up for child dataset and fine tunining with the child dataset also..       
    parent_arch_params["outs"] = child_params["child_arch_params"]["outs"]
    parent_init_params, parent_arch_params = load_network (  parent_filename_params ["network_save_name"] ,
                                                        data_params = False, 
                                                        optimization_params = False)
    parent_retrain_params = {
                        "copy_from_old"     : [ True ] * (len(parent_arch_params["nkerns"]) + len(parent_arch_params["num_nodes"])) + [False] , 
                        "freeze"            : [ False] * (len(parent_arch_params["nkerns"]) + len(parent_arch_params["num_nodes"]) + 1)
                    }                                                                                                                  
    obi_wan = cnn_mlp(   
                        filename_params = parent_filename_params,
                        arch_params = parent_arch_params,
                        optimization_params = parent_optimization_params,
                        retrain_params = parent_retrain_params,
                        init_params = parent_init_params,
                        verbose =verbose 
                        )   
    obi_wan.init_data ( dataset = dataset[1] , outs = parent_arch_params ["outs"], visual_params = visual_params, verbose = verbose )
    obi_wan.build_network(verbose = verbose) 
    
    obi_wan.build_learner(verbose =verbose)                                                                                                
    obi_wan.train(   n_epochs = parent_params["parent_n_epochs"], 
                        ft_epochs = parent_params ["parent_ft_epochs"],
                        validate_after_epochs = parent_params ["parent_validate_after_epochs"],
                        patience_epochs = 4,                        
                        verbose = verbose )      
    obi_wan.save_network ()                            
    obi_wan.test( verbose = verbose )                          
    """                        
    
    
    
    
    
    
    
    # Make true for unsupervised 
    un_supervised = True
    if un_supervised is True:
        print "skywalker network"                    
                            
                            
        # use this for unsupervised mentoring ... 
        print "unsupervised mentoring"
        child_filename_params          = child_params["child_filename_params"]
        child_arch_params              = child_params["child_arch_params"]
        child_optimization_params      = child_params["child_optimization_params"]
        child_retrain_params           = None
        child_init_params              = None   
        # make soft label and hard label OFF
        unsupervised_filename_params = child_filename_params    
        unsupervised_filename_params ["network_save_name"] =   "../results/skywalker_network_unsupervised.pkl"    
        
        joint_params_unsupervised  =  {                  
            
                        "learn_layers"                      : [ (0,0), (2,1), (3,2) ],       # 0 is the first layer
                        "learn_layers_coeffs"               : [ 
                                                                (1,1,75),        # This is probes
                                                                (0,0,0),          # This is soft outputs
                                                                (0,0,1)             # this is for hard labels
                                                            ],
                                                            # weights for each probes and soft outputs and labels
                                                            # first term is begining weight,
                                                            # second term is ending weight,
                                                            # third term is until which epoch.    
                        "error"                             : rmse,         # erros of the probes.
                        "print_probe_costs"                 : True 
                        }    
        skywalker = regularizer_net(   
                            filename_params = child_filename_params,
                            arch_params = child_arch_params,
                            optimization_params = child_optimization_params,
                            retrain_params = child_retrain_params,
                            init_params = child_init_params,
                            parent = obi_wan,
                            joint_params = joint_params_unsupervised,                         
                            verbose =verbose )                      
        skywalker.init_data ( dataset = dataset[0] , outs = child_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
        skywalker.build_network(verbose = verbose)  
        
        # build probes also builds the learer Don't bother calling a build learner. 
        # Uncomment for learning the unsupervised network also. 
        skywalker.build_probes(verbose = verbose)    
        skywalker.train(    n_epochs = 75, 
                            ft_epochs = 0 ,
                            validate_after_epochs = child_params ["child_validate_after_epochs"],
                            patience_epochs = 75,  # this will avoid early stopping.
                            verbose = verbose )
        skywalker.save_network ()                                       
        skywalker.print_net(epoch = 'final')
    








      
    pdb.set_trace()
    # regular ...  
    print "supervised learning"      
    child_filename_params          = child_params["child_filename_params"]
    child_arch_params              = child_params["child_arch_params"]
    child_optimization_params      = child_params["child_optimization_params"]
    child_retrain_params           = None
    child_init_params              = None 
    
    if un_supervised is True:        
        # use the next few lines if an unsupervised mentor is already ready or to pre-load a skywalker and fine tune .. 
        child_init_params, child_arch_params = load_network (  unsupervised_filename_params["network_save_name"] ,
                                                            data_params = False, 
                                                            optimization_params = False)   
        child_retrain_params = {
                            "copy_from_old"     : [ True ] * (len(parent_arch_params["nkerns"]) + len(parent_arch_params["num_nodes"])) + [True], 
                            "freeze"            : [ False ] * (len(parent_arch_params["nkerns"]) + len(parent_arch_params["num_nodes"])) + [False]
                        }     
                        
     #####                                                    
    skywalker = regularizer_net(   
                        filename_params = child_filename_params,
                        arch_params = child_arch_params,
                        optimization_params = child_optimization_params,
                        retrain_params = child_retrain_params,
                        init_params = child_init_params,
                        parent = obi_wan,
                        joint_params = joint_params,                         
                        verbose =verbose )                      
    skywalker.init_data ( dataset = dataset[1] , outs = child_arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    skywalker.build_network(verbose = verbose)  
    # build probes also builds the leayer Don't bother calling a build_learner.     
    skywalker.build_probes(verbose = verbose)    
    skywalker.train(    n_epochs = child_params ["child_n_epochs"], 
                        ft_epochs = child_params ["child_ft_epochs"] ,
                        validate_after_epochs = child_params ["child_validate_after_epochs"],
                        patience_epochs = 50,
                        verbose = verbose )
    skywalker.save_network ()                                       
    skywalker.test( verbose = verbose )   
    





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
                        "network_save_name"     : "../results/obi_wan_network.pkl"
                    }

    child_filename_params = { 
                        "results_file_name"     : "../results/child_results.txt",      
                        "error_file_name"       : "../results/child_error.txt",
                        "cost_file_name"        : "../results/child_cost.txt",
                        "confusion_file_name"   : "../results/child_confusion.txt",
                        "network_save_name"     : "../results/skywalker_network.pkl"
                    }
                                      
    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 36,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                                                      

    parent_optimization_params = {  
                            "mom"                         	    : (0.5, 0.9, 50),    # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 1,                   # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.01,0.0001, 0.01 ),  # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.0000),       # l1_coeff, l2_coeff                                
                            "optim_type"                        : 2,                   # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                   # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                         }        

    parent_arch_params = {                    
                    "mlp_activations"                   : [ ReLU,   ReLU,   Softmax ],
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5,    0.5,    0.5],
                    "mlp_maxout"                        : [ 1,      1,           ],                    
                    "num_nodes"                         : [ 800,   800           ],                                     
                    "outs"                              : 10,  
                    "mlp_batch_norm"                    : [ True ],                                                                                                                                                 
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU,   ReLU,        ],             
                    "cnn_batch_norm"                    : [ True ],
                    "nkerns"                            : [  20,    50,          ],              
                    "filter_size"                       : [ (5,5),  (3,3),       ],
                    "pooling_size"                      : [ (2,2),  (2,2),       ],
                    "conv_pad"                          : [ 2,      0,           ],                            
                    "pooling_type"                      : [ 1,      1,           ],
                    "maxrandpool_p"                     : [ 1,      1,           ],                           
                    "conv_stride_size"                  : [ (1,1),  (1,1),       ],
                    "cnn_maxout"                        : [ 1,      1,           ],                    
                    "cnn_dropout_rates"                 : [ 0,      0,           ],
                    "mean_subtract"                     : True,
                    "use_bias"                          : True,
                    "max_out"                           : 0 
                         }                          
    
    child_optimization_params = {
                            "mom"                         	    : (0.5, 0.9, 50),    # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 1,                   # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.001, 0.0001, 0.01 ),  # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.0000),       # l1_coeff, l2_coeff                                
                            "optim_type"                        : 2,                   # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                   # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                         }
          
    solo_optimization_params = {
                            "mom"                         	    : (0.5, 0.9, 50),    # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 1,                   # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.001,0.0001, 0.01 ),  # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.0000),       # l1_coeff, l2_coeff                                
                            "optim_type"                        : 2,                   # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                   # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                         }                                                     
                            
    child_arch_params = {
                    
                    "mlp_activations"                   : [ ReLU,   ReLU,   Softmax ],
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5,    0.5,    0.5],
                    "mlp_maxout"                        : [ 1,      1      ],                    
                    "num_nodes"                         : [ 800,   800 ],                                     
                    "outs"                              : 10,  
                    "mlp_batch_norm"                    : [ True ],                                                                                                                                                 
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [   ReLU ],             
                    "cnn_batch_norm"                    : [   True ],
                    "nkerns"                            : [   20   ],              
                    "filter_size"                       : [  (5,5) ],
                    "pooling_size"                      : [  (2,2) ],
                    "conv_pad"                          : [   2    ],                            
                    "pooling_type"                      : [   1    ],
                    "maxrandpool_p"                     : [   1    ],                           
                    "conv_stride_size"                  : [  (1,1) ],
                    "cnn_maxout"                        : [   1    ],                    
                    "cnn_dropout_rates"                 : [   0    ],
                    "mean_subtract"                     : True,
                    "use_bias"                          : True,
                    "max_out"                           : 0 
                 } 
                 
    joint_params  =  {                  
        
                    "learn_layers"                      : [ (0,0), (2,1), (3,2) ],       # 0 is the first layer
                    "learn_layers_coeffs"               : [ 
                                                            (2,0.1,135),        # This is probes
                                                            (2,0,135),          # This is soft outputs
                                                            (1,1,1)             # this is for hard labels
                                                          ],
                                                        # weights for each probes and soft outputs and labels
                                                        # first term is begining weight,
                                                        # second term is ending weight,
                                                        # third term is until which epoch.    
                    "error"                             : rmse,         # erros of the probes.
                    "print_probe_costs"                 : True 
                     }
                      
                      
    # other loose parameters.     
    parent_n_epochs = 250
    parent_validate_after_epochs = 1
    parent_ft_epochs = 50
    child_n_epochs = parent_n_epochs
    child_validate_after_epochs = parent_validate_after_epochs
    child_ft_epochs = parent_ft_epochs
        
    verbose = False  
    parent_dataset = "_datasets/_dataset_96736"
    child_dataset = "_datasets/_dataset_96736"

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
                                                           
    solo_filename_params = { 
                        "results_file_name"     : "../results/solo_results_small.txt",      
                        "error_file_name"       : "../results/solo_error_small.txt",
                        "cost_file_name"        : "../results/solo_cost_small.txt",
                        "confusion_file_name"   : "../results/solo_confusion_small.txt",
                        "network_save_name"     : "../results/solo_network.pkl "
                    }                                                      
                   
    regularized_train (   
                            parent_params = parent_params,
                            child_params = child_params, 
                            joint_params = joint_params,
                            visual_params = visual_params,
                            dataset = (parent_dataset, child_dataset),
                            verbose = verbose
                    )                               
             
    solo(
                    arch_params             = child_arch_params,
                    optimization_params     = solo_optimization_params,
                    dataset                 = child_dataset, 
                    filename_params         = solo_filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = child_validate_after_epochs,
                    n_epochs                = child_n_epochs,
                    ft_epochs               = child_ft_epochs, 
                    verbose                 = verbose                                                
                )
                                                                                                        