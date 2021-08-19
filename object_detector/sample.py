import sys
sys.path.append(".")
import rpa as r
import cv2
import pytesseract
import argparse
import torch
import os
from train import predict, create_or_restore_training_state
from models import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

parser = argparse.ArgumentParser(description='Fill out some forms.')
parser.add_argument('--form_url', default='http://foersom.com/net/HowTo/data/OoPdfFormExample.pdf')
parser.add_argument('--data_file', default='./example_data.json')
parser.add_argument('--run_name', type=str, required=__name__ == '__main__')
parser.add_argument("--model_config_path",
                    type=str,
                    default="object_detector/config/yolov3.cfg",
                    help="path to model config file")
args = parser.parse_args()

config = ('-l eng --oem 1 --psm 3')

def get_values(field, data):
    """
    Cleans up the field title and extracts data
    """
    if "Address" in field and '1' in field:
        field = field[1:]
    field = field.split('(')[0]
    field = field.replace('.', '')
    field = field.replace("'", "")
    field = field.split('\n')[0]
    field = field.lower().strip().replace(' ', '_')

    if field == '':
        return '', ''

    if field == 'cami,_name':
        field = 'family_name'
    if field in ['ae_isi', 'wey.', 'einht', 'eset_fees._—', 'cami,_name', '—_house_nr', 'weys','wey', 'eset_fees_—', '|_speak_and_understand', 'uriving_license']:
        value = ''
    else:
        value = data[field]

    return value, field

def fill_out():
    """
    1. Loads the form
    2. Takes a screenshot of the form
    3. Passes the schreenshot to YOLO model to detect the bboxes
    4. Fills out the form
    """
    hard_coded_data = {
        "given_name": "Shayan",
        "family_name": "Kousha",
        "address_1": "661 University Ave",
        "address_2": "Suite 710",
        "postcode": "M5G 1M1",
        "city": "Toronto",
        "country": "Canada",
        "gender": "m",
        "height": "170",
        "favourite_colour": "b",
        "driving_license": True
    }
    
    r.init(visual_automation=True)
    r.url(args.form_url)
    r.wait(3.0)

    r.snap('page', 'pdf_form.jpg')
    checkpoint_dir = os.path.join("./checkpoints", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initiate model
    model = Darknet(args.model_config_path)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()))

    # load either non initialized or loaded state
    cur_epoch, model, optimizer, manager, prev_epoch_loss_list = create_or_restore_training_state(
        model, optimizer, dir_path=checkpoint_dir)

    model.eval()

    all_bboxes = predict('pdf_form.jpg', model, 'bbox')
    all_bboxes = np.array(all_bboxes).astype(int)

    filled_fields = []
    for i, box in enumerate(all_bboxes):
        adjusted_box = [int(box[0]-15), int(box[1] + 152), int(box[0] + box[2]), int(box[1] + box[3] + 152)]
        r.snap(adjusted_box[0], adjusted_box[1], adjusted_box[2], adjusted_box[3], f'field{i}.png')
        im = cv2.imread(f'field{i}.png')
        text = pytesseract.image_to_string(im, config=config).strip().replace(':', ' ')

        value, text = get_values(text, hard_coded_data)
        if value != '' and not (text in filled_fields):
            filled_fields.append(text)
            x_ratio = 0.97
            y_ratio = 0.5
            if text == 'address_1':
                x_ratio = 0.9
                y_ratio = 0.6
            if text == 'gender':
                x_ratio = 0.5
                y_ratio = 0.8
            if text == 'postcode':
                x_ratio = 0.6
            if text == 'driving_license':
                # x_ratio = 0.53
                y_ratio = 0.7
            if text == 'family_name':
                # x_ratio = 0.53
                y_ratio = 0.6

            x = int(adjusted_box[0] + x_ratio * (adjusted_box[2] - adjusted_box[0]))
            y = int(adjusted_box[1] + y_ratio * (adjusted_box[3] - adjusted_box[1]))
            if value and type(value) == bool:
                r.click(x, y)
            else:
                r.dclick(x, y)
                r.keyboard('[backspace]')
                r.type(x , y, value)

    r.snap('page', 'pdf_form_filled.jpg')
    r.close()

if __name__ == "__main__":
    fill_out()
