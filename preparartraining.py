#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Jesús Bocanegra Linares. prodboca@gmail.com
# Módulos que hay que importar
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


# Descargar el dataset a la máquina
# Letras de la A a la J (10 clases). 28x28 píxeles.
# Training set y Test set

# Características de las imágenes. Cambiar si se aumentara el número de clases o el tamaño de las imágenes
num_classes = 10
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_path = "./grande/"
test_path = "./peq/"

train_folders = [
    os.path.join(train_path, d) for d in sorted(os.listdir(train_path))
    if os.path.isdir(os.path.join(train_path, d))]
test_folders = [
    os.path.join(test_path, d) for d in sorted(os.listdir(test_path))
    if os.path.isdir(os.path.join(test_path, d))]



# Extraer el dataset. Se creará una carpeta para cada clase
np.random.seed(1309)

# Cargar imágenes
def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Tensor del dataset completo:', dataset.shape)
  print('Media:', np.mean(dataset))
  print('Desviación típica:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

print(train_folders)
train_datasets = maybe_pickle(train_folders, 1000)
test_datasets = maybe_pickle(test_folders, 100)

# Tratar los datos
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	valid_dataset, valid_labels = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes
	print("num_classes", num_classes)
	print("valid_size", valid_size)
	print("train_size", train_size)
	
	print("vsize_per_class", vsize_per_class)
	print("tsize_per_class", tsize_per_class)
	
	print("DIMENSIONES: ", train_dataset.shape)
	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):
		print("Label: ", label, "; pickle_file: ", pickle_file)
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				# Mezclar las letras para que los juegos de validación y prueba sean aleatorios
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class
						
				train_letter = letter_set[vsize_per_class:end_l, :, :]

				train_dataset[start_t:end_t, :, :] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
	return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 100000
valid_size = 5000
test_size = 3000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Aleatorizar datos
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# Guardar datos
pickle_file = 'data.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

##RESHAPE!!
train_dataset = np.array(train_dataset)
train_labels = np.array(train_labels)


train_dataset = train_dataset.reshape((train_size,image_size*image_size));
valid_dataset = valid_dataset.reshape((valid_size,image_size*image_size));
test_dataset = test_dataset.reshape((test_size,image_size*image_size));

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
