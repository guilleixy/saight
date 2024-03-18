import boto3
import csv
import pprint

with open('demoCredentials.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        access_key_id = row[0]
        secret_access_key = row[1]

client = boto3.client('rekognition', region_name='eu-west-2', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

photo = 'images/demo4.jpg'

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

detect_faces = client.detect_faces(Image={'Bytes': source_bytes}, Attributes=['ALL'])

for faceDetail in detect_faces['FaceDetails']:
    print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) 
          + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
    print('Here are the other attributes:')
    print('Smile: ' + str(faceDetail['Smile']['Value']))
    print('Eyeglasses: ' + str(faceDetail['Eyeglasses']['Value']))
    print('Gender: ' + str(faceDetail['Gender']['Value']))

for faceDetail in detect_faces['FaceDetails']:
    pprint.pprint(faceDetail)
