import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
import pandas as pd
import csv
import glob
import random 
from PIL import Image
import configparser
import argparse
import pprint

tf.executing_eagerly()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(description='Spatial Transcriptomics Deconvolution Network.')
parser.add_argument('-p', '--prefix', required=True, type=str, action='store', help='prefix to identify the task.')
parser.add_argument('-w', '--workspace', required=True, type=str, action='store', help='Workspace for storing all generated files.')
parser.add_argument('-i', '--image_dir', required=True, type=str, action='store', help='The folder of images.')
parser.add_argument('-s', '--sptx_file', required=True, type=str, action='store', help='Normalized gene expression data file.')
parser.add_argument('-g', '--gene_list_file', required=True, type=str, action='store', help='Gene list file.')
parser.add_argument('-d', '--dry_run', action='store_true', help='Enable dry run.')
parser.add_argument('-ee', '--encoder_epochs', default=100, type=int, action='store', help='Training epochs of encoder.')
parser.add_argument('-de', '--decoder_epochs', default=100, type=int, action='store', help='Training epochs of decoder.')
parser.add_argument('-ne', '--num_embeddings', default=1024, type=int, action='store', help='Number of the embeddings.')
parser.add_argument('-ld', '--latent_dim', default=16, type=int, action='store', help='Dimension of latent space.')
parser.add_argument('-px', '--pixel_size', default=0.5, type=float, action='store', help='Pixel size.')
parser.add_argument('-pt', '--pooling_type', default='noisy-and', type=str, action='store', help='Pooling type.')
parser.add_argument('-pa', '--pooling_alpha', default=0.1, type=float, action='store', help='Pooling alpha.')
parser.add_argument('-pb', '--pooling_beta', default=0.05, type=float, action='store', help='Pooling beta.')
parser.add_argument('-pg', '--pooling_gamma', default=5.0, type=float, action='store', help='Pooling gamma.')
parser.add_argument('-tb', '--tensorboard', action='store_true', help='Enable tensorboard.')
parser.add_argument('-cp', '--checkpoint', action='store_true', help='Enable checkpoints.')
parser.add_argument('-lg', '--log', action='store_true', help='Enable log.')

args = parser.parse_args()

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
        
class VQVAEDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 image_folder,  
                 augmentation = True,
                 batch_size = 128,
                 random_seed=1234,
                 k=5, 
                 m_list=[0,1,2,3,4]):
        'Initialization'
        
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.augmentation = augmentation
        image_list = glob.glob(os.path.join(image_folder, '*', '*.png'), recursive=True)
                
        random.seed(random_seed)            
        random.shuffle(image_list) 

        list_partitions = self.__partition(image_list, k)
        
        self.image_list = []
        
        for m in m_list:
            self.image_list += list_partitions[m]
        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        image_list = self.image_list[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X = self.__data_generation(image_list)
        
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        random.shuffle(self.image_list)        

    def __data_generation(self, image_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
                
        X_list = []
        for fn in image_list:
            img = Image.open(fn)
            ary = np.array(img)
            
            if self.augmentation:
                if random.randint(0, 1) == 1:
                    ary = np.flipud(ary)
                if random.randint(0, 1) == 1:
                    ary = np.fliplr(ary)            
                if random.randint(0, 1) == 1:
                    ary = np.transpose(ary, axes=(1,0,2))            
            
            ary = np.expand_dims(ary, axis=0)
            X_list.append(ary)

        X = np.concatenate(X_list, axis=0)
        X = X.astype(np.float32) / 255.0 - 0.5
        
        return X

    def __partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]
    
    def get_variance(self):
        
        image_list = self.image_list if len(self.image_list) < 10000 else random.choices(self.image_list, k=10000)
        
        X = self.__data_generation(image_list)
        
        var = np.var(X+0.5)
        return var
    


class STDNetDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 image_folder, 
                 stx_file, 
                 gene_list, 
                 vqvae_encoder, 
                 vqvae_quantizer, 
                 augmentation = True,
                 random_seed=1234, 
                 alpha=0.1, 
                 k=5, 
                 m_list=[0,1,2,3,4]):
        'Initialization'
        
        self.image_folder = image_folder
        self.gene_list = gene_list
        self.vqvae_encoder = vqvae_encoder
        self.vqvae_quantizer = vqvae_quantizer
        self.alpha = alpha
        self.augmentation = augmentation
        
        loc_list = []
        for name in glob.glob(os.path.join(self.image_folder, '*')):
            loc = os.path.basename(name)
            loc_list.append(loc)
    
                
        random.seed(random_seed)            
        random.shuffle(loc_list) 

        loc_list_partitions = self.__partition(loc_list, k)
        
        self.loc_list = []
        
        for m in m_list:
            self.loc_list += loc_list_partitions[m]
    
        self.stx_df = pd.read_csv(stx_file)
        
        self.stx_df = self.stx_df[self.stx_df['Unnamed: 0'].isin(gene_list)]
        self.stx_df.set_index('Unnamed: 0')
        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.loc_list)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        # Generate data
        X, y = self.__data_generation(index)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        random.shuffle(self.loc_list)        

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
                
        loc = self.loc_list[index]
        image_list = glob.glob(os.path.join(self.image_folder, loc, '*.png'))
        
        X_list = []
        for fn in image_list:
            img = Image.open(fn)
            ary = np.array(img)
            
            if self.augmentation:
                if random.randint(0, 1) == 1:
                    ary = np.flipud(ary)
                if random.randint(0, 1) == 1:
                    ary = np.fliplr(ary)            
                if random.randint(0, 1) == 1:
                    ary = np.transpose(ary, axes=(1,0,2))    
        
            ary = np.expand_dims(ary, axis=0)
            X_list.append(ary)

        data = np.concatenate(X_list, axis=0)
        data = data.astype(np.float32) / 255.0 - 0.5        
        
        encoded_outputs = self.vqvae_encoder.predict(data)
        flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
        X = self.vqvae_quantizer.get_quantized_latent_value(flat_enc_outputs)
        X = X.numpy().reshape((data.shape[0], encoded_outputs.shape[1], encoded_outputs.shape[2], encoded_outputs.shape[3]))
        X = X.reshape(data.shape[0], encoded_outputs.shape[1]*encoded_outputs.shape[2]*encoded_outputs.shape[3])
        
        y = self.alpha*self.stx_df[loc].to_numpy()
        
        return X, y

    def __partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

class MILPoolingLayer(layers.Layer):
    def __init__(self, data_dim=1024, beta = 0.05, pooling = True, pooling_type='noisy-and', pooling_gamma=5.0, **kwargs):
        super(MILPoolingLayer, self).__init__(**kwargs)
        
        self.data_dim = data_dim
        self.beta = beta
        self.pooling = pooling
        self.pooling_type = pooling_type
        self.pooling_gamma = pooling_gamma
    
        if self.pooling_type == 'noisy-and':
            w_init = tf.random_uniform_initializer()

            self.b = tf.Variable(
                initial_value=w_init(
                    shape=(self.data_dim,), dtype="float32"
                ),
                trainable=True,
                name="NoisyAnd_b",
            )       
            
    def get_config(self):
        return super().get_config()
        
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
            
        elif self.pooling_type == 'isr':
            part1 = tf.math.reduce_sum(x/(1-x + keras.backend.epsilon()), axis=0) if self.pooling else x/(1-x + keras.backend.epsilon())
            Pi = part1/(1+part1+keras.backend.epsilon())
            
        elif self.pooling_type == 'gm':
            part1 = tf.math.pow(x + keras.backend.epsilon(), self.pooling_gamma)
            part2 = tf.math.reduce_mean(part1, axis=0) if self.pooling else part1
            Pi = tf.math.pow(part2 + keras.backend.epsilon(), 1/self.pooling_gamma)
        
        elif self.pooling_type == 'lse':
            part1 = tf.math.reduce_mean(tf.math.exp(self.pooling_gamma*x), axis=0) if self.pooling else tf.math.exp(self.pooling_gamma*x)
            Pi = (1/self.pooling_gamma)*tf.math.log(part1)
            
        Pi *= self.beta
        Pi *= tf.cast(tf.shape(x)[0], dtype=tf.float32)
        
        if self.pooling:
            Pi = tf.expand_dims(Pi, axis=0)

        return Pi
    
    
class STDNet(keras.models.Model):
    def __init__(self, 
                 input_dim = 784,
                 output_dim=3000, 
                 beta = 0.01, 
                 pooling = True,
                 pooling_type='noisy-and',
                 pooling_gamma = 5.0,
                 **kwargs):
        super(STDNet, self).__init__(**kwargs)
        
        self.input_layer = layers.Input(input_dim)
        self.fcnPerCellL1 = layers.Dense(output_dim, activation='sigmoid')
        self.fcnPerCellL2 = layers.Dense(output_dim, activation='sigmoid')
        self.milLayer = MILPoolingLayer(data_dim=output_dim, beta=beta, pooling=pooling, pooling_type=pooling_type, pooling_gamma=pooling_gamma)
        self.fcnOverallL1 = layers.Dense(output_dim, activation='sigmoid')
        self.fcnOverallL2 = layers.Dense(output_dim, activation='sigmoid')
        self.output_layer = self.call(self.input_layer)
        
        self.mse = keras.losses.MeanSquaredError()
            
        self.loss_tracker = keras.metrics.Mean(name="loss")
    
        super(STDNet, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs)
 
    def build(self):
        # Initialize the graph
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
        
    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x = self.call(x)
            
            # Calculate the losses.
            loss = (
                self.mse(x, y)
            )
            
        # Backpropagation.
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking.
        self.loss_tracker.update_state(loss)

        # Log results.
        return {
            "loss": self.loss_tracker.result(),
        }
        
        



        
if __name__ == "__main__":

    
    
    vqvae_x_train = VQVAEDataGenerator(image_folder=args.image_dir)
    data_variance = vqvae_x_train.get_variance()
    
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=args.latent_dim, num_embeddings=args.num_embeddings)
    sampling_size = vqvae_trainer.vqvae.output.get_shape().as_list()[1]
    
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    
    vqvae_trainer_tensorboard = TensorBoard(
        log_dir=os.path.join(args.workspace, args.prefix+'-encoder_tensorboard_logs'),
        histogram_freq=1,
        write_images=True
        )
    
    vqvae_weights_filepath= os.path.join(args.workspace, args.prefix+'-encoder-{epoch:04d}-{loss:.4f}.model.h5')     
    vqvae_model_checkpoint = ModelCheckpoint(filepath=vqvae_weights_filepath,
                                       monitor='loss',
                                       verbose=1,
                                       mode='min',
                                       save_best_only=True,
                                       save_weights_only=False)
    
    vqvae_csvlogger = CSVLogger(args.prefix+'-encoder_logs.csv', append=True, separator=',')
    
    callbacks = []
    
    if args.tensorboard: callbacks.append(vqvae_trainer_tensorboard)
    if args.checkpoint: callbacks.append(vqvae_model_checkpoint)
    if args.log: callbacks.append(vqvae_csvlogger)
    
    vqvae_trainer.vqvae.summary()
    
    if not args.dry_run:
        vqvae_trainer.fit(vqvae_x_train, epochs=args.encoder_epochs, batch_size=128, shuffle=True,
                          callbacks=callbacks,
                          # use_multiprocessing=True      
                         )
        
        vqvae_trainer.vqvae.save_weights(os.path.join(args.workspace, args.prefix+'-encoder-final.model.h5'))
        
    decoder_input_dim = int(np.prod(vqvae_trainer.vqvae.get_layer('encoder').predict(vqvae_x_train[0]).shape[1:]))
    
    gene_list = []
    with open(args.gene_list_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row[0].strip()) > 0:
                gene_list.append(row[0])    

    vqvae_encoder = vqvae_trainer.vqvae.get_layer("encoder")
    vqvae_quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    
    stdnet_x_train = STDNetDataGenerator(image_folder=args.image_dir, stx_file=args.sptx_file, gene_list=gene_list, vqvae_encoder=vqvae_encoder, vqvae_quantizer=vqvae_quantizer, alpha=args.pooling_alpha)
        
    stdnet = STDNet(input_dim = decoder_input_dim, output_dim = len(gene_list), beta=args.pooling_beta, pooling = True, pooling_type=args.pooling_type, pooling_gamma=args.pooling_gamma)
    stdnet.compile(optimizer=keras.optimizers.Adam())
    
    stdnet_tensorboard = TensorBoard(log_dir=os.path.join(args.workspace, args.prefix+'-decoder_tensorboard_logs'), histogram_freq=1, write_images=True)
    
    stdnet_weights_filepath= os.path.join(args.workspace, args.prefix+'-decoder-{epoch:04d}-{loss:.4f}.model.h5')     
    
    stdnet_checkpoint = ModelCheckpoint(filepath=stdnet_weights_filepath,
                                       monitor='loss',
                                       verbose=1,
                                       mode='min',
                                       save_best_only=True,
                                       save_weights_only=False)
    
    stdnet_csvlogger = CSVLogger(args.prefix+'-decoder_logs.csv', append=True, separator=',')
    
    callbacks = []
    
    if args.tensorboard: callbacks.append(stdnet_tensorboard)
    if args.checkpoint: callbacks.append(stdnet_checkpoint)
    if args.log: callbacks.append(stdnet_csvlogger)
    
    stdnet.summary()
    
    if not args.dry_run:
        stdnet.fit(stdnet_x_train, epochs=args.decoder_epochs, shuffle=True,
                        callbacks=callbacks,   
                        # use_multiprocessing=True       
                        )
    
        stdnet.save_weights(os.path.join(args.workspace, args.prefix+'-decoder-final.model.h5'))
    

    model_config = configparser.ConfigParser()
    model_config.add_section(args.prefix)
    model_config.set(args.prefix, "encoder_model_file", args.prefix+'-encoder-final.model.h5')
    model_config.set(args.prefix, "encoder_data_variance", str(data_variance))
    model_config.set(args.prefix, "encoder_num_embeddings", str(args.num_embeddings))
    model_config.set(args.prefix, "encoder_latent_dim", str(args.latent_dim))
    model_config.set(args.prefix, "encoder_sampling_size", str(sampling_size))
    model_config.set(args.prefix, "encoder_pixel_size", str(args.pixel_size))
    model_config.set(args.prefix, "decoder_model_file", args.prefix+'-decoder-final.model.h5')
    model_config.set(args.prefix, "decoder_input_dim", str(decoder_input_dim))
    model_config.set(args.prefix, "decoder_output_dim", str(len(gene_list)))
    model_config.set(args.prefix, "decoder_pooling_type", args.pooling_type)
    model_config.set(args.prefix, "decoder_pooling_alpha", str(args.pooling_alpha))
    model_config.set(args.prefix, "decoder_pooling_beta", str(args.pooling_beta))        
    model_config.set(args.prefix, "decoder_pooling_gamma", str(args.pooling_gamma))
    
    print('\n')
    print(os.path.join(args.workspace, args.prefix+".ini"))
    
    pprint.pprint(model_config.items(args.prefix))
    
    if not args.dry_run:    
        with open(os.path.join(args.workspace, args.prefix+".ini"), 'w') as configfile:
            model_config.write(configfile)
        





    