# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import json
import sys
import os
from PIL import Image
import numpy as np
import configparser
import time
# import nvidia_smi

tf.executing_eagerly()
        
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
    def get_quantized_latent_value(self, flattened_inputs):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(flattened_inputs)
        flattened = tf.reshape(flattened_inputs, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        return quantized    
    
    
def get_encoder(latent_dim=16):
    encoder_inputs = keras.Input(shape=(28, 28, 3))
    x = encoder_inputs
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(latent_dim, 1, padding="same")(x)
    encoder_outputs = x
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16):
    latent_inputs = keras.Input(shape=get_encoder().output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def get_vqvae(latent_dim=16, num_embeddings=1024):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(28, 28, 3))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=16, num_embeddings=1024, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

        
   
    def call(self, inputs):
        return self.vqvae.call(inputs)
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
        
class MILPoolingLayer(layers.Layer):
    def __init__(self, data_dim=1024, beta = 0.05, pooling = True, pooling_type = 'noisy-and', pooling_gamma = 5.0, **kwargs):
        super(MILPoolingLayer, self).__init__(**kwargs)
        
        self.data_dim = data_dim
        self.beta = beta
        self.pooling_type = pooling_type
        self.pooling_gamma = pooling_gamma
        self.pooling = pooling
    
    # def build(self, input_shape):
        # Initialize the embeddings which we will quantize.
        
        if self.pooling_type == 'noisy-and':
            w_init = tf.random_uniform_initializer()

            self.b = tf.Variable(
                initial_value=w_init(
                    shape=(self.data_dim,), dtype="float32"
                ),
                trainable=True,
                # validatate_shape=True,
                name="NoisyAnd_b",
            )       
        
        # super(MILPoolingLayer, self).build(input_shape)  
        
    def call(self, x):
        # Calculate the mean value across all instances in the bag (batch)
        
        if self.pooling_type == 'noisy-and':
            mean_pij = tf.math.reduce_mean(x, axis=0, name="mean_p_ij") if self.pooling else x
            part1 = tf.math.sigmoid(self.pooling_gamma*(mean_pij-self.b))-tf.math.sigmoid(-self.pooling_gamma*self.b)
            part2 = tf.math.sigmoid(self.pooling_gamma * (1-self.b))-tf.math.sigmoid(-self.pooling_gamma*self.b)
            Pi = part1 / (part2 + keras.backend.epsilon())
            
        elif self.pooling_type == 'noisy-or':
            Pi = 1-tf.math.reduce_prod(1-x, axis=0) if self.pooling else x
            
        elif self.pooling_type == 'max':
            Pi = tf.math.reduce_max(x, axis=0) if self.pooling else x
            Pi = x
            
        elif self.pooling_type == 'isr':
            part1 = tf.math.reduce_sum(x/(1-x + keras.backend.epsilon()), axis=0) if self.pooling else x/(1-x + keras.backend.epsilon())
            Pi = part1/(1+part1+keras.backend.epsilon())
            
        elif self.pooling_type == 'gm':
            # Pi = tf.math.pow(tf.math.reduce_mean(tf.math.pow(x + keras.backend.epsilon(), self.model['gamma']), axis=0) + keras.backend.epsilon(), 1/self.model['gamma'])
            Pi = tf.math.pow(tf.math.pow(x + keras.backend.epsilon(), self.pooling_gamma) + keras.backend.epsilon(), 1/self.pooling_gamma)
        
        elif self.pooling_type == 'lse':
            Pi = (1/self.model['gamma'])*tf.math.log(tf.math.reduce_mean(tf.math.exp(self.model['gamma']*x), axis=0)) if self.pooling else (1/self.model['gamma'])*tf.math.log(tf.math.exp(self.pooling_gamma*x))
            
        Pi *= self.beta
        Pi *= tf.cast(tf.shape(x)[0], dtype=tf.float32)
        
        if self. pooling:
            Pi = tf.expand_dims(Pi, axis=0)

        return Pi
    
    
    
class STDNet(keras.models.Model):
    def __init__(self, 
                 input_dim = 784,
                 output_dim=3000, 
                 beta = 0.01, 
                 pooling = True,
                 pooling_type='noisy-and',
                 pooling_gamma=0.5, 
                 **kwargs):
        super(STDNet, self).__init__(**kwargs)
        
        self.input_layer = layers.Input(input_dim)
        self.fcnPerCellL1 = layers.Dense(output_dim, activation='sigmoid')
        self.fcnPerCellL2 = layers.Dense(output_dim, activation='sigmoid')
        self.milLayer = MILPoolingLayer(data_dim=output_dim, beta=beta, pooling=pooling, pooling_type=pooling_type, pooling_gamma=pooling_gamma)
        self.fcnOverallL1 = layers.Dense(output_dim, activation='sigmoid')
        self.fcnOverallL2 = layers.Dense(output_dim, activation='sigmoid')
        
        self.output_layer = self.call(self.input_layer)
        
        super(STDNet, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs)
        
    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )
        
    def call(self, inputs, training=False):      
        x = inputs
        x = self.fcnPerCellL1(x)
        x = self.fcnPerCellL2(x)
        x = self.milLayer(x)
        x = self.fcnOverallL1(x)
        x = self.fcnOverallL2(x)
        
        outputs = x
        
        return outputs      
        
        
if __name__ == '__main__':  
    sptx_stdnet_path = os.path.dirname(os.path.abspath(__file__))
    ini_filename = os.path.join(sptx_stdnet_path, 'sptx_stdnet.ini')

    model_config = configparser.ConfigParser()
    model_config.read(ini_filename)

    action = sys.argv[1]
    param_file_name =  sys.argv[2]
    result_file_name =  sys.argv[3]

    if action == 'model_list':
        return_dict = {'modelList':model_config.sections()}
        
        with open(result_file_name, "w") as outfile:
            json.dump(return_dict, outfile)
                 
    elif action == 'request_model_args':
        
        with open(param_file_name) as f:
            param = json.load(f)  
            
        model_name = param['modelName']
                    
        return_dict = {'samplingSize':int(model_config[model_name]['encoder_sampling_size']), 'pixelSize':float(model_config[model_name]['encoder_pixel_size'])}
        with open(result_file_name, "w") as outfile:
            json.dump(return_dict, outfile)
            
    elif action == 'run_prediction':
        # gpu_list = tf.config.list_physical_devices('GPU')
        # print("Num GPUs Available: ", len(gpu_list))
        #
        # minimal_gpu_memory_requirement = int(model_config['GENERAL']['minimal_gpu_memory_requirement'])
        # retry_times = int(model_config['GENERAL']['retry_times'])
        # delay_before_retry = int(model_config['GENERAL']['delay_before_retry'])
        #
        # for _ in range(retry_times):
        #
        #     chosen_gpu = -1
        #     max_free_mem= -1
        #
        #     for i, (gpu) in enumerate(gpu_list):
        #         nvidia_smi.nvmlInit()
        #
        #         print("Name:", gpu.name, "  Type:", gpu.device_type)
        #
        #         handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
        #         info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #
        #         print("Total memory:", info.total)
        #         print("Free memory:", info.free)
        #         print("Used memory:", info.used)
        #
        #         if info.free > max_free_mem:
        #             chosen_gpu = i
        #             max_free_mem = info.free
        #
        #         nvidia_smi.nvmlShutdown()
        #
        #     if max_free_mem < minimal_gpu_memory_requirement:
        #         print('Insufficient GPU memory!')
        #         time.sleep(delay_before_retry)
        #         continue
        #
        #     with tf.device('/CPU:'+str(i)):    
                
        start_time = time.time()
        print('=== Action ('+action+') Started at: '+str(start_time)+' ===')
                                   
        with open(param_file_name) as f:
            param = json.load(f)        
        
        model_name = param['modelName']
        encoder_model_filename = os.path.join(sptx_stdnet_path, model_config[model_name]['encoder_model_file'])
        encoder_data_variance = float(model_config[model_name]['encoder_data_variance'])
        encoder_latent_dim = int(model_config[model_name]['encoder_latent_dim'])
        encoder_num_embeddings = int(model_config[model_name]['encoder_num_embeddings'])
        decoder_model_filename = os.path.join(sptx_stdnet_path, model_config[model_name]['decoder_model_file'])
        decoder_gene_list = model_config[model_name]['decoder_gene_list'].split(',')
        decoder_input_dim = int(model_config[model_name]['decoder_input_dim'])
        decoder_output_dim = int(model_config[model_name]['decoder_output_dim'])
        decoder_pooling_type = model_config[model_name]['decoder_pooling_type']
        decoder_pooling_alpha = float(model_config[model_name]['decoder_pooling_alpha'])
        decoder_pooling_gamma = float(model_config[model_name]['decoder_pooling_gamma'])

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.compat.v1.Session(config=config)
    
        image_count = param['imageCount']
        sampling_size = int(model_config[model_name]['encoder_sampling_size'])
        
        vqvae_trainer = VQVAETrainer(encoder_data_variance, latent_dim=encoder_latent_dim, num_embeddings=encoder_num_embeddings)
        vqvae_trainer.vqvae.load_weights(encoder_model_filename)  
        vqvae_trainer.vqvae.summary()          
                
        vqvae_encoder = vqvae_trainer.vqvae.get_layer("encoder")
        vqvae_quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
        
        stdnet_nand = STDNet(input_dim = decoder_input_dim, output_dim = decoder_output_dim, pooling = False, pooling_type = decoder_pooling_type, pooling_gamma = decoder_pooling_gamma)
        stdnet_nand.load_weights(decoder_model_filename) 
        stdnet_nand.summary()    
        
        dataArray = []
        for i in range(image_count):
            fn = os.path.join(param['imageSetPath'], str(i)+'.png')
            
            img = Image.open(fn)
            img.load()
            img = img.resize((sampling_size, sampling_size), Image.BICUBIC)
            data = np.asarray( img, dtype="int32" )
            data = np.expand_dims(data, axis=0)
            dataArray.append(data)
            
        dataArray = np.concatenate(dataArray)
        dataArray = (dataArray / 255.0) - 0.5
    
        encoded_outputs = vqvae_encoder.predict(dataArray)
        flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
        X = vqvae_quantizer.get_quantized_latent_value(flat_enc_outputs)
        X = X.numpy().reshape((dataArray.shape[0], encoded_outputs.shape[1], encoded_outputs.shape[2], encoded_outputs.shape[3]))
        X = X.reshape(dataArray.shape[0], encoded_outputs.shape[1]*encoded_outputs.shape[2]*encoded_outputs.shape[3])
        
        y = (1.0/decoder_pooling_alpha)*stdnet_nand.predict(X, batch_size=X.shape[0])
    
        resultDict = {}
        for i in range(y.shape[0]):
            resultDict[str(i)] = {}
            for j in range(y.shape[1]):
                resultDict[str(i)][decoder_gene_list[j]] = float(y[i][j])
        
        with open(result_file_name, "w") as outfile:
            json.dump(resultDict, outfile)    
        
        finish_time = time.time()
        elapsed_time = time.time() - start_time
            
        print('=== Finished at: '+str(finish_time)+' / Elapsed Time:'+str(elapsed_time)+' ===')
                
            # break
