#!/usr/bin/python
import sys
sys.path.insert(0, '../../Convolutional-Neural-Networks/')

from samosa.dataset import load_skdata_mnist
from samosa.dataset import load_skdata_mnist_noise1
from samosa.dataset import load_skdata_mnist_noise2
from samosa.dataset import load_skdata_mnist_noise3
from samosa.dataset import load_skdata_mnist_noise4
from samosa.dataset import load_skdata_mnist_noise5
from samosa.dataset import load_skdata_mnist_noise6
from samosa.dataset import load_skdata_mnist_bg_images
from samosa.dataset import load_skdata_mnist_bg_rand
from samosa.dataset import load_skdata_mnist_rotated
from samosa.dataset import load_skdata_mnist_rotated_bg
from samosa.dataset import load_skdata_cifar10
from samosa.dataset import preprocessing
 
import os
import cPickle
import time
import numpy
from random import randint


# Load initial data         
def setup_split (data_params, preprocess_params, outs, splits ):
    thismodule = sys.modules[__name__]
    print "... setting up dataset "
    # Load Batch 1.
    data_struct         = data_params # This command makes it possible to save down
    dataset             = data_params [ "loc" ]
    data_type           = data_params [ "type" ]
    height              = data_params [ "height" ]
    width               = data_params [ "width" ]
    batch_size          = data_params [ "batch_size" ]    
    load_batches        = data_params [ "load_batches"  ] * batch_size
    if load_batches < batch_size and (dataset == "caltech101" or dataset == "caltech256"):
        AssertionError("load_batches is improper for this dataset " + dataset)
    batches2train       = data_params [ "batches2train" ]
    batches2test        = data_params [ "batches2test" ]
    batches2validate    = data_params [ "batches2validate" ] 
    channels            = data_params [ "channels" ]
    
    start_time = time.clock()
    
    temp_dir = './_datasets/_dataset_' + str(randint(11111,99999))			
    os.mkdir(temp_dir)
    os.mkdir(temp_dir + "/train" )
    os.mkdir(temp_dir + "/test"  )
    os.mkdir(temp_dir + "/valid" )
    
    print "... creating dataset " + temp_dir 			    					
    # load skdata ( its a good library that has a lot of datasets)
    # Only skdata data as of now.... 
    # elif data_type == 'skdata':
    
    if (dataset == 'mnist' or 
        dataset == 'mnist_noise1' or 
        dataset == 'mnist_noise2' or
        dataset == 'mnist_noise3' or
        dataset == 'mnist_noise4' or
        dataset == 'mnist_noise5' or
        dataset == 'mnist_noise6' or
        dataset == 'mnist_bg_images' or
        dataset == 'mnist_bg_rand' or
        dataset == 'mnist_rotated' or
        dataset == 'mnist_rotated_bg' or
        dataset == 'cifar10') :

            print "... importing " + dataset + " from skdata"
            data = getattr(thismodule, 'load_skdata_' + dataset)()                
            print "... setting up dataset "
            print "... 		--> training data "			
            data_x, data_y, data_y1 = data[0]                
            # count the length of base and train arrays needed.
            locs = numpy.zeros(len(data_y), dtype = bool)

            for i in xrange(outs):
                if i in splits["base"]:
                    locs[data_y==i] = True
                if i in splits["shot"]:
		    temp = numpy.zeros(len(data_y), dtype = bool)
		    temp[data_y == i] = True
             	    count = 0
		    for j in xrange (temp.shape[0]):
			if temp[j] == True:
			    count = count + 1
			    if count > splits["p"]:		            
			        temp[j] = False			     
                    locs[temp] = True
            data_x = data_x[locs]
            data_y = data_y[locs]
            data_y1 = data_y1[locs]
            data_x = preprocessing ( data_x, height, width, channels, preprocess_params )                        					 
            f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')                
            obj = (data_x,data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()							                        
            n_train_images = data_x.shape[0]
            batches2train = numpy.floor(n_train_images / batch_size)
         
            print "... 		--> validation data "			
            data_x, data_y, data_y1 = data[1]                
            data_x = preprocessing ( data_x, height, width, channels, preprocess_params )                        					
            f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')                
            obj = (data_x,data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()	
            n_valid_images = data_x.shape[0]
            batches2validate = numpy.floor(n_valid_images / batch_size)                                   
                        
            print "... 		--> testing data "			
            data_x, data_y, data_y1 = data[2]                
            data_x = preprocessing ( data_x, height, width, channels, preprocess_params )                        					
            f = open(temp_dir + "/test/" + 'batch_' + str(0) + '.pkl', 'wb')                
            obj = (data_x,data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()							                        
            n_test_images = data_x.shape[0]
            batces2test = numpy.floor(n_test_images / batch_size)

            multi_load = False                      
            new_data_params = {
                "type"               : 'base',                                   
                "loc"                : temp_dir ,                                          
                "batch_size"         : batch_size,                                    
                "load_batches"       : -1,
                "batches2train"      : int(batches2train),                                      
                "batches2test"       : int(batches2test),                                     
                "batches2validate"   : int(batches2validate),                                       
                "height"             : height,                                      
                "width"              : width,                                       
                "channels"           : channels,
                "multi_load"		 : multi_load, 
                "n_train_batches"	 : 1,
                "n_test_batches"	 : 1,
                "n_valid_batches"	 : 1  						                                       
                }          

    if preprocess_params["gray"] is True:
        new_data_params ["channels"] = 1
        channels = 1 
    
    assert ( height * width * channels == numpy.prod(data_x.shape[1:]) )
    f = open(temp_dir  +  '/data_params.pkl', 'wb')
    cPickle.dump(new_data_params, f, protocol=2)
    f.close()	
  			  	
    end_time = time.clock()
    print "...         time taken is " + str(end_time - start_time) + " seconds"
		

    ## Boiler Plate ## 
if __name__ == '__main__':
              
    data_params = {
                   "type"               : 'skdata',                                   
                   "loc"                : 'mnist',                                          
                   "batch_size"         : 10,                                     
                   "load_batches"       : 1, 
                   "batches2train"      : 100,                                      
                   "batches2test"       : 20,                                      
                   "batches2validate"   : 20,                                        
                   "height"             : 28,                                       
                   "width"              : 28,                                       
                   "channels"           : 1                                       
                  }
                  
    preprocess_params = { 
                            "normalize"     : True,
                            "GCN"           : False,
                            "ZCA"           : False,
                            "gray"          : False,
                        }
  
                    
    splits = { 
                    "base"              : [],
                    "shot"              : [0,1,2,3,4,5,6,7,8,9],
                    "p"                 : 1
                }              
                      
    setup_split( data_params = data_params, preprocess_params = preprocess_params, outs = 10, splits = splits )
  
