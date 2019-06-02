import io
import os

#Import the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

#Instantiates a client
client = vision.ImageAnnotatorClient()

#The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    './images/AlmostRipeBanana.jpg')

#Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

#Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Lables')
for label in labels:
    print(label.description, label.score)

def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

localize_objects('./images/AlmostRipeBanana.jpg')
