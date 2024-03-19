import boto3
import csv

with open('demoCredentials.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        access_key_id = row[0]
        secret_access_key = row[1]

client = boto3.client('rekognition', region_name='eu-west-2', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
photo = 'images/demo6.jpg'

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()
detect_famous = client.recognize_celebrities(Image={'Bytes': source_bytes})
for celebrity in detect_famous['CelebrityFaces']:
    print(celebrity['Name'])    