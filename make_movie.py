import os
import sys
import cv2

results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
print(results_fullpath)

def make_movie(path):
    # read track plots
    images = [img for img in sorted(os.listdir(path)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    # save with 10fps to result dir
    video = cv2.VideoWriter(os.path.join(path, 'my_tracking_results.avi'), 0, 10, (width,height))

    for image in images:
        fname = os.path.join(path, image)
        video.write(cv2.imread(fname))
        os.remove(fname) # clean up

    cv2.destroyAllWindows()
    video.release()

make_movie(results_fullpath)