import tensorflow as tf
import numpy as np
import load_data
import sys

from optparse import OptionParser

parser = OptionParser()

parser.add_option("--train_file", "--train_set_list", dest="train_file",
                  help="Required. Absolute path. Training data set (for now *.txt)", metavar="FILE")
                  
parser.add_option("--test_file", "--test_set_list", dest="test_file",
                  help="Required. Absolute path. Test data set (for now *.txt)", metavar="FILE")

parser.add_option("--nB", "--num_of_buyer",
                  help="Required. Num of buyer.", type="int", dest="nB")  

parser.add_option("--nS", "--num_of_seller",
                  help="Required. Num of seller.", type="int", dest="nS") 

parser.add_option("--outdir", "--outdir", dest="outdir",
                  help="Required. Output folder", metavar="STRING")
            
parser.add_option("--model_file", "--model_file", dest="model_file", default=False,
                  help="True or False. load model from a saved file. /tmp/model_.ckpt")

parser.add_option("--embedding_size", "--embedding_size",
                  help="Embedding size. Default is 16.", type="int", dest="embedding_size", default=16)
                  
parser.add_option("--batch_size", "--batch_size",
                  help="Batch size. Default is 10000.", type="int", dest="batch_size", default = 10000)
    
parser.add_option("--train_nbatch", "--train_nbatch",
                  help="Train n batches. Default is 5000, which means 5 eopches.", type="int", dest="train_nbatch", default = 5000)
                  
parser.add_option("--test_nbatch", "--test_nbatch",
                  help="Test n batches. Default is 231, which means 1 eopch.", type="int", dest="test_nbatch", default = 231)                

parser.add_option( "--omegaB", dest="omegaB",
                  help="Regularizer of Gb. Default is 1.", type="float", default=1)
  
parser.add_option( "--omegaS", dest="omegaS",
                  help="Regularizer of Gs. Default is 1.", type="float", default=1)

parser.add_option( "--lambdaB", dest="lambdaB",
                  help="Regularizer of Embedding. Default is 0.0003.", type="float", default=0.0003)    
 
parser.add_option( "--lambdaS", dest="lambdaS",
                  help="Regularizer of Embedding. Default is 0.01.", type="float", default=0.0003)
  
parser.add_option( "--alpha", dest="alpha",
                  help="Regularizer of Weights. Default is 0.1.", type="float", default=0.0003)
   
parser.add_option( "--lr", dest="lr",
                  help="Learning rate. Default is 0.002.", type="float", default=0.002) 
                 
parser.add_option("--other_ext", "--other_ext", dest="other_ext",
                  help="other_extensions", default="")   
               
(options, args) = parser.parse_args()

if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

nB = options.nB
nS = options.nS
batch_size = options.batch_size
train_nbatch = options.train_nbatch
test_nbatch = options.test_nbatch
lr = options.lr

hiddenLayer = [4*options.embedding_size, 2*options.embedding_size, options.embedding_size, 1]
omegaB = options.omegaB
omegaS = options.omegaS
lambdaB = options.lambdaB
lambdaS = options.lambdaS
alpha = options.alpha

file_ext = '_lr' + str(lr) + '_trainbatch' + str(train_nbatch) + '_omegab' + str(omegaB) \
          + '_omegas' + str(omegaS) + '_lambdab' + str(lambdaB) + '_lambdas' + str(lambdaS)\
          + '_alpha' + str(alpha)+ options.other_ext


#create a function, such that it generate data as mnist.train.next(batch_size)
#input data looks like: bid, sid, bid', sid', GBii', Rij, GSjj'
train_file = options.train_file
test_file = options.test_file

data = load_data.read_data_sets(train_file,test_file, validation_size = 0)

Bi = tf.sparse_placeholder(tf.float32)
Bi_p = tf.sparse_placeholder(tf.float32)
Sj = tf.sparse_placeholder(tf.float32)
Sj_p = tf.sparse_placeholder(tf.float32)

GB = tf.placeholder(tf.float32, [None, 1])
GS = tf.placeholder(tf.float32, [None, 1])
R = tf.placeholder(tf.float32, [None, 1])

kB = tf.placeholder(tf.float32, [None, 1])
kS = tf.placeholder(tf.float32, [None, 1])

#FIRST LAYER
w_B_1 = tf.Variable(tf.truncated_normal([nB, hiddenLayer[0]]) ,name="w_B_1")
bias_B_1 = tf.Variable(tf.zeros([hiddenLayer[0]]))
w_S_1 = tf.Variable(tf.truncated_normal([nS, hiddenLayer[0]]) ,name="w_S_1")
bias_S_1 = tf.Variable(tf.zeros([hiddenLayer[0]]))

layer_Bi_1 = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(Bi, w_B_1) + bias_B_1) 
layer_Bi_p_1 = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(Bi_p, w_B_1) + bias_B_1)

layer_Sj_1 = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(Sj, w_S_1) + bias_S_1)
layer_Sj_p_1 = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(Sj_p, w_S_1) + bias_S_1)

#SECOND LAYER
w_B_2 = tf.Variable(tf.truncated_normal([hiddenLayer[0], hiddenLayer[1]]) ,name="w_B_2")
bias_B_2 = tf.Variable(tf.zeros([hiddenLayer[1]]))
w_S_2 = tf.Variable(tf.truncated_normal([hiddenLayer[0], hiddenLayer[1]]) ,name="w_S_2")
bias_S_2 = tf.Variable(tf.zeros([hiddenLayer[1]]))

layer_Bi_2 = tf.nn.sigmoid(tf.matmul(layer_Bi_1, w_B_2) + bias_B_2)
layer_Bi_p_2 = tf.nn.sigmoid(tf.matmul(layer_Bi_p_1, w_B_2) + bias_B_2)

layer_Sj_2 = tf.nn.sigmoid(tf.matmul(layer_Sj_1, w_S_2) + bias_S_2)
layer_Sj_p_2 = tf.nn.sigmoid(tf.matmul(layer_Sj_p_1, w_S_2) + bias_S_2)

#THIRD LAYER
w_B_3 = tf.Variable(tf.truncated_normal([hiddenLayer[1], hiddenLayer[2]]) ,name="w_B_3")
bias_B_3 = tf.Variable(tf.zeros([hiddenLayer[2]]))
w_S_3 = tf.Variable(tf.truncated_normal([hiddenLayer[1], hiddenLayer[2]]) ,name="w_S_3")
bias_S_3 = tf.Variable(tf.zeros([hiddenLayer[2]]))
#
layer_Bi_3 = tf.nn.sigmoid(tf.matmul(layer_Bi_2, w_B_3) + bias_B_3)
layer_Bi_p_3 = tf.nn.sigmoid(tf.matmul(layer_Bi_p_2, w_B_3) + bias_B_3)

layer_Sj_3 = tf.nn.sigmoid(tf.matmul(layer_Sj_2, w_S_3) + bias_S_3)
layer_Sj_p_3 = tf.nn.sigmoid(tf.matmul(layer_Sj_p_2, w_S_3) + bias_S_3)

w_4 = tf.Variable(tf.truncated_normal([hiddenLayer[2], hiddenLayer[3]]) ,name="w_4")
bias_R_4 = tf.Variable(tf.zeros([hiddenLayer[3]]))
w_GB_4 = tf.Variable(tf.truncated_normal([hiddenLayer[2], hiddenLayer[3]]) ,name="w_GB_4")
bias_GB_4 = tf.Variable(tf.zeros([hiddenLayer[3]]))
w_GS_4 = tf.Variable(tf.truncated_normal([hiddenLayer[2], hiddenLayer[3]]) ,name="w_GS_4")
bias_GS_4 = tf.Variable(tf.zeros([hiddenLayer[3]]))

layer_R = tf.matmul(tf.multiply(layer_Bi_3, layer_Sj_3), tf.multiply(w_4, w_4)) + bias_R_4
layer_GB_4 = tf.matmul(tf.multiply(layer_Bi_3, layer_Bi_p_3),w_GB_4) + bias_GB_4
layer_GS_4 = tf.matmul(tf.multiply(layer_Sj_3, layer_Sj_p_3),w_GS_4) + bias_GS_4   #[batch_size, 1]

B_embedding_3 = np.zeros(nB*hiddenLayer[2]).reshape(nB, hiddenLayer[2])
S_embedding_3 = np.zeros(nS*hiddenLayer[2]).reshape(nS, hiddenLayer[2])


rmse = tf.reduce_sum(tf.square(layer_R - R))
GB_loss = tf.reduce_sum(tf.square( tf.multiply((layer_GB_4 - GB), kB) ))
GS_loss = tf.reduce_sum(tf.square( tf.multiply((layer_GS_4 - GS), kS) ))
weight_loss  = tf.reduce_sum(tf.square(w_B_1)) + tf.reduce_sum(tf.square(w_S_1)) + tf.reduce_sum(tf.square(bias_B_1))\
        + tf.reduce_sum(tf.square(bias_S_1))

loss = rmse + omegaB * GB_loss + omegaS * GS_loss + alpha * weight_loss



optimizer = tf.train.AdagradOptimizer(lr).minimize(loss)

# Train
shapeB = [batch_size, nB]
shapeS = [batch_size, nS]

logfile = open(options.outdir + '/log_' + file_ext + '.txt', 'w')

sess = tf.InteractiveSession()
if options.model_file is False:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
else:
    print("Load existing model file...")
    saver = tf.train.Saver()
    saver.restore(sess, options.outdir + "/model/model" + file_ext + ".ckpt")

i = 0

update_B_cmd = tf.matmul(layer_Bi_3, tf.reshape(tf.matrix_diag(tf.reshape(w_4, [hiddenLayer[2]])), [hiddenLayer[2], hiddenLayer[2]]))
update_S_cmd = tf.matmul(layer_Sj_3, tf.reshape(tf.matrix_diag(tf.reshape(w_4, [hiddenLayer[2]])), [hiddenLayer[2], hiddenLayer[2]]))

for _ in range(train_nbatch):


    batch_xy= data.train.next_batch(batch_size)   
    feed_dict={Bi: tf.SparseTensorValue(batch_xy[:,[0,1]], np.ones(batch_size), shapeB),\
                                   Bi_p: tf.SparseTensorValue(batch_xy[:,[0,3]], np.ones(batch_size), shapeB),\
                                   kB: batch_xy[:,[4]],\
                                   Sj: tf.SparseTensorValue(batch_xy[:,[0,2]], np.ones(batch_size), shapeS),\
                                   Sj_p: tf.SparseTensorValue(batch_xy[:,[0,5]], np.ones(batch_size) , shapeS),\
                                   kS: batch_xy[:,[6]],\
                                   GB: batch_xy[:,[7]],\
                                   R: batch_xy[:,[8]],\
                                   GS: batch_xy[:,[9]]\
                                   }
    _, update_B, update_S = sess.run([optimizer, update_B_cmd, update_S_cmd], feed_dict)
      
    B_embedding_3[batch_xy[:,1].astype(np.int32),:] = update_B
                            
    S_embedding_3[batch_xy[:,2].astype(np.int32),:] = update_S

    if data.train.epochs_completed > i:
        i += 1
        save_path = saver.save(sess, options.outdir +"/model/model" + file_ext + ".ckpt")
        print("epoch " + str(data.train.epochs_completed) + "finished!")
        logfile.write("epoch " + str(data.train.epochs_completed) + "finished!\n")
     
np.savetxt(options.outdir + '/B_embedding' + file_ext + '.txt', B_embedding_3.astype(np.float32))
np.savetxt(options.outdir + '/S_embedding' + file_ext + '.txt', S_embedding_3.astype(np.float32))

temp_rmse = 0
temp_GB_loss = 0
temp_GS_loss = 0
temp_weight_loss = 0


print("starting calculating testing performacne.")

for _ in range(test_nbatch):
    batch_xy= data.test.next_batch(batch_size)   
    feed_dict={Bi: tf.SparseTensorValue(batch_xy[:,[0,1]], np.ones(batch_size), shapeB),\
                                   Bi_p: tf.SparseTensorValue(batch_xy[:,[0,3]], np.ones(batch_size), shapeB),\
                                   kB: batch_xy[:,[4]],\
                                   Sj: tf.SparseTensorValue(batch_xy[:,[0,2]], np.ones(batch_size), shapeS),\
                                   Sj_p: tf.SparseTensorValue(batch_xy[:,[0,5]], np.ones(batch_size) , shapeS),\
                                   kS: batch_xy[:,[6]],\
                                   GB: batch_xy[:,[7]],\
                                   R: batch_xy[:,[8]],\
                                   GS: batch_xy[:,[9]]}
    temp_rmse += sess.run(rmse, feed_dict)
    temp_GB_loss += sess.run(GB_loss, feed_dict)
    temp_GS_loss += sess.run(GS_loss, feed_dict)
    temp_weight_loss += sess.run(weight_loss, feed_dict)

        
print ("rmse is: " + str(temp_rmse/(test_nbatch*batch_size)))
print ("GB_loss is: " + str(temp_GB_loss/(test_nbatch*batch_size)))
print ("GS_loss is: " + str(temp_GS_loss/(test_nbatch*batch_size)))
print ("weight_loss is: " + str(temp_weight_loss/(test_nbatch*batch_size)))

logfile.write ("rmse is: " + str(temp_rmse/(test_nbatch*batch_size)) + "\n")
logfile.write ("GB_loss is: " + str(temp_GB_loss/(test_nbatch*batch_size))+ "\n")
logfile.write ("GS_loss is: " + str(temp_GS_loss/(test_nbatch*batch_size))+ "\n")
logfile.write ("weight_loss is: " + str(temp_weight_loss/(test_nbatch*batch_size))+ "\n")

