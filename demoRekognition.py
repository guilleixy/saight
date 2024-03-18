import boto3
import csv

with open('demoCredentials.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        access_key_id = row[0]
        secret_access_key = row[1]

client = boto3.client('rekognition', region_name='eu-west-2', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

photo = 'images/demo5.jpg'

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

detect_objects = client.detect_labels(Image={'Bytes': source_bytes})
for label in detect_objects['Labels']:
    print(label['Name'] + ' : ' + str(label['Confidence']))

