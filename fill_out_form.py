# Sample script to fill out a form
import rpa as r
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--update_test_image', action='store_true', default=False)
args = parser.parse_args()

def run_process(args):
    gt_image_path = './test/gt.png'
    result_image_path = './test/results.png'
    r.init()

    # r.url('https://ca.yahoo.com')
    # r.type('//*[@id="ybar-sbq"]', 'github')

    r.url('https://www.consumer.equifax.ca/personal/dispute-credit-report-form/')
    r.wait(5.0)
    r.fill_input("First Name", {'value': "Shayan"})
    r.fill_input("Middle Name", {'value': "N/A"})
    r.fill_input("Last Name", {'value': "Kousha"})
    r.fill_input("Social Insurance Number", {'value': "abcdefghi"})
    r.fill_input("Date of Birth", {
        'month': '6',
        'day': '5',
        'year': '2000'
    })

    if args.update_test_image:
        r.snap('page', gt_image_path)

    if args.test:
        r.snap('page', result_image_path)

        original = cv2.imread(gt_image_path)
        result = cv2.imread(result_image_path)

        difference = cv2.subtract(original, result)

        if np.mean(difference) == 0:
            print("test passed")
        else:
            print("test faile")

    r.close()

if __name__ == "__main__":
    run_process(args)
