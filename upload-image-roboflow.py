# -- coding: utf-8 --
# importing the roboflow Python Package
import glob
from roboflow import Roboflow
import os

# creating the Roboflow object
# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
api_key = os.environ["ROBOFLOW_PRIVATE_API_KEY"]
rf = Roboflow(api_key=api_key)

# using the workspace method on the Roboflow object
workspace = rf.workspace()

# identifying the project for upload
upload_project = workspace.project("signate-library-book-detect")


## DEFINITIONS
# roboflow params

# glob params
dir_name = "train"
image_dir = os.path.join(dir_name, "images")
annotation_dir = os.path.join(dir_name, "annotations")

# create image glob
image_glob = glob.glob(image_dir + '/*' + ".jpg")

annotation_glob = glob.glob(annotation_dir + '/*' + ".json")

## MAIN
# upload images
for image, annotation in zip(image_glob, annotation_glob):
    response = upload_project.upload(image, annotation)
    print(response)
