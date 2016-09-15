#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Jesús Bocanegra Linares. prodboca@gmail.com
# Módulos necesarios
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves import range

# Para redimensionar:
#	- las convoluciones necesitan los datos de imagen formateados como un cubo (ancho x alto x número de canales)
#	- las etiquetas deben ser floats y codificadas en 1-hot.
tam_imagen = 28
n_clases = 10
n_canales = 1 # escala de grises
profundidad_pixel = 255.0

def cargar_letra():
	dataset = np.ndarray(shape=(1, tam_imagen, tam_imagen), dtype=np.float32)
	indice_imagen = 0
	ruta_imagen = "letra.png"
	try:
		imagen = (ndimage.imread(ruta_imagen).astype(float) - profundidad_pixel / 2) / profundidad_pixel
		if imagen.shape != (tam_imagen, tam_imagen):
			raise Exception('Unexpected image shape: %s' % str(imagen.shape))
	except IOError as e:
		print('No se pudo abrir: ', ruta_imagen, ':', e, '. Saltando.')
	return imagen

	
clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

with open('trained.bin', 'rb') as f:
	entrenado = pickle.load(f)
	# Cargar datos entrenados
	capa1_pesos = entrenado[0]
	capa1_sesgos = entrenado[1]
	capa2_pesos = entrenado[2]
	capa2_sesgos = entrenado[3]
	capa3_pesos = entrenado[4]
	capa3_sesgos = entrenado[5]
	capa4_pesos = entrenado[6]
	capa4_sesgos = entrenado[7]
	dataset_problema = cargar_letra()

  

def reformat(dataset, etiquetas):
  dataset = dataset.reshape((-1, tam_imagen, tam_imagen, n_canales)).astype(np.float32)
  etiquetas = (np.arange(n_clases) == etiquetas[:,None]).astype(np.float32)
  return dataset, etiquetas

dataset_problema = dataset_problema.reshape((-1, tam_imagen, tam_imagen, n_canales)).astype(np.float32)

def accuracy(predictions, etiquetas):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(etiquetas, 1))
          / predictions.shape[0])
		  
# Pequeña red con varias capas convolucionales seguidas de una capa completamente conectada.
# Limitamos la profundidad y el número de nodos conectados, porque las redes convolucionales conllevan un alto coste computacional.
tam_lote = 10
tam_parche = 5

# Gráfica por defecto:
graph = tf.Graph()

with graph.as_default():
  # Datos de entrada como una constante de tf.
  tf_dataset_problema = tf.constant(dataset_problema)
  
  # Definimos el modelo.
  def modelo(datos):
	# Primera capa convolucional
    conv1 = tf.nn.conv2d(datos, capa1_pesos, [1, 2, 2, 1], padding='SAME')
    oculta1 = tf.nn.relu(conv1 + capa1_sesgos)
	# Segunda capa convolucional
    conv2 = tf.nn.conv2d(oculta1, capa2_pesos, [1, 2, 2, 1], padding='SAME')
    oculta2 = tf.nn.relu(conv2 + capa2_sesgos)
    dimensiones = oculta2.get_shape().as_list()
    reshape = tf.reshape(oculta2, [dimensiones[0], dimensiones[1] * dimensiones[2] * dimensiones[3]])
    oculta3 = tf.nn.relu(tf.matmul(reshape, capa3_pesos) + capa3_sesgos)
	# Devuelve oculta3 * pesos4 + sesgos4
    return tf.matmul(oculta3, capa4_pesos) + capa4_sesgos
  
  # Prediction:
  prediccion_problema = tf.nn.softmax(modelo(tf_dataset_problema))
  

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  resultado = np.array(prediccion_problema.eval())
  print('Con probabilidad %f%% la letra es una: ' % resultado.max()*100)
  sys.exit(clases[resultado.argmax()])