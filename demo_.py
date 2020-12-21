import argparse
import os, sys
import cv2
from facenet_pytorch import MTCNN
from utils.image_utils import *
from utils.run_audio_.AudioDetector  import start_audio, audio_inference

#################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
args = parser.parse_args()
#################################################################

mtcnn = MTCNN()

class Demo(object):
    def __init__(self, mtcnn, data):
        self.mtcnn = mtcnn
        self.data = data
        self.video = None

    def run(self, data):
        # video capture using data
        cap = cv2.VideoCapture(data)

        # audio start
        if args.image == '':
            ModelPath = './trained_model/'
            if args.video:
                self.video = args.video[:args.video.find('.')+1] + 'wav'
            start_audio(ModelPath, self.video)

        while True:
            # capture image from camera
            ret, frame = cap.read()

            try:
                # detect face box and probability
                boxes, probs = self.mtcnn.detect(frame, landmarks=False)

                # draw box on frame
                frame = draw_bbox(frame, boxes, probs)

                # perform only when face is detected
                if len(boxes) > 0:

                    # extract the face roi
                    rois = detect_rois(boxes)

                    for roi in rois:
                        (start_Y, end_Y, start_X, end_X) = roi
                        face = frame[start_Y:end_Y, start_X:end_X]

                        # run the classifier on bounding box
                        emotion_i = emotion_class(face)
                        gender_i = gender_class(face)
                        gaze_i = gaze_class(face)

                        # assign labeling
                        cv2.putText(frame, emotion_i, (end_X-50, start_Y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        cv2.putText(frame, gender_i, (end_X-50, start_Y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        cv2.putText(frame, gaze_i, (end_X-50, start_Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                
                # audio labeling
                if args.image == '':
                    gender_a, emotion_a = audio_inference(self.video)
                    cv2.putText(frame, gender_a, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, emotion_a, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

            except:
                pass

            # show the frame
            window = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow('Demo', window)
            
            # save image
            # cv2.imwrite('sample/sample.jpg', window)

            # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Interrupted by user!')
                break
        
        # clear program and close windows
        cap.release()
        cv2.destroyAllWindows()
        stop_audio()
        print('All done!')

if args.image:
    if not os.path.isfile(args.image):
        print("Input image file {} doesn't exist".format(args.image))
        sys.exit(1)
    fcd = Demo(mtcnn, args.image)
    fcd.run(args.image)
elif args.video:
    if not os.path.isfile(args.video):
        print("Input video file {} dosen't exist".format(args.video))
        sys.exit(1)
    os.system('ffmpeg -i {} -ar 48000 -f wav {}.wav'.format(args.video, args.video[:args.video.find('.')]))
    fcd = Demo(mtcnn, args.video)
    fcd.run(args.video)
else:
    fcd = Demo(mtcnn, args.image)
    fcd.run(args.src)
