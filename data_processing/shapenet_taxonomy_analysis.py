import numpy as np
import os
import settings
import matplotlib.pyplot as plt
'''
script analyzes taxonomy.json...counts number of classes
'''

shapenet_taxonomy_dir = os.path.join(settings.ROOT_DIR, 'data', 'ShapeNetCore.v2');

file = os.path.join(shapenet_taxonomy_dir, 'taxonomy.json');

import json
from pprint import pprint

with open(file) as data_file:
    data = json.load(data_file)
pprint(len(data))

histogram_of_classes = dict();
class_synset_id = dict();
for dict_object in data:
    key = dict_object['name'];

    histogram_of_classes[key] = dict_object['numInstances'];
    class_synset_id[key] = dict_object['synsetId'];

#get all classes with instances >1000
good_classes = list(); good_classes_pop = list();
desirable_synset_id = list();
for i in histogram_of_classes.keys():
    if(histogram_of_classes[i]> 1000):
        good_classes.append(i); good_classes_pop.append(histogram_of_classes[i]);
        desirable_synset_id.append(class_synset_id[i])
        print(i+', '+str(histogram_of_classes[i])+', '+class_synset_id[i])

## do an argsort
sorted_args = np.argsort(good_classes_pop);
for i in range(len(good_classes)):
    print(str(good_classes[sorted_args[i]])+', '+str(good_classes_pop[sorted_args[i]])+', '+str(desirable_synset_id[sorted_args[i]]))

print(len(good_classes))
print(good_classes)
print(desirable_synset_id)


print(histogram_of_classes.values())
plt.figure();
plt.hist(histogram_of_classes.values())
plt.show()




