import glob, os
import argparse
import tempfile
from pdf2image import convert_from_path
from resizeimage import resizeimage
from PIL import Image
import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', required=True, help='Directory path of pdf files')


opt = parser.parse_args()
print(opt)


# Store pdf files names in a list
def list_pdf_files_names(path):
    path_list = []
    for file in glob.glob("*.pdf"):
        path_list.append(file)
        print(path_list)
    if path_list == []:
        print('\nWARNING! There is no pdf files in your directory!')

    return path_list

def convert_pdf(pdf_path_list):
    for pdf_path_file in pdf_path_list:
        pil_img_list = list_PIL_images(pdf_path_file)
        print('end conversion')
        pdf_file_name = create_new_img_folder(pdf_path_file)
        create_new_img_set(pil_img_list, pdf_file_name)

def list_PIL_images(pdf_path_file):
    print('start conversion')
    with tempfile.TemporaryDirectory() as path:
     images_from_path = convert_from_path(pdf_path_file, dpi=300, output_folder=path)
     return images_from_path

def create_new_img_folder(pdf_path_file):
    pdf_file_name = os.path.splitext(pdf_path_file)[0]
    if not os.path.exists(pdf_file_name):
        os.makedirs(pdf_file_name)
        print('New image folder', pdf_file_name, 'create.')

    return pdf_file_name

def create_new_img_set(pil_img_list, pdf_file_name):
    counter = 0
    #nbr_of_crop_by_img = 6
    #width = 256
    #height = 256
    for pil_img in pil_img_list:
        #ce = torchvision.transforms.CenterCrop(1000)
        #rc = torchvision.transforms.RandomCrop([256, 256])
        #pil_img = resizeimage.resize_crop(pil_img, [width, height])
        #pil_img = rc(ce(pil_img))
        #pil_img.thumbnail((256, 256))
        pil_img.save(str(pdf_file_name)+'/'+ str(pdf_file_name)+'img_'+str(counter), 'jpeg')
        counter += 1


pdf_file_list = list_pdf_files_names(opt.dir_path)
convert_pdf(pdf_file_list)
