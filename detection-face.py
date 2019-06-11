import cv2
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument('username', help='username', type=str)
parser.add_argument('mode', help='mode', type=str)

args = parser.parse_args()
username = args.username
mode = args.mode

path = ''

if mode == 'enroll':
    path = 'db/enrollment/' + username
elif mode == 'verify':
    path = 'db/verification/'
else: 
    exit(0)


if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(path+'/features'):
    os.mkdir(path+'/features')
if not os.path.exists(path+'/images'):
    os.mkdir(path+'/images')

def take_picture(number, username):

    cap = cv2.VideoCapture(0)
    cap.set(3, 640) #WIDTH
    cap.set(4, 480) #HEIGHT

    count = 1

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print(ret)
        
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(len(faces))
        # Display the resulting frame
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            if(number == 1):
                cv2.imwrite(path + str(username) + '.png', frame[y:y+h, x:x+w])
            else:
                cv2.imwrite(path + '/images/' + str(count) + '.png', frame[y:y+h, x:x+w])
            count = count + 1
            if (count > number):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # convert image to COLOR_BGR2GRAY
    if number == 1:
        image = cv2.imread(path + str(username) + '.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path + str(username) + '.png', gray)
    else:
        for i in range(1, count): 
            image = cv2.imread(path + '/images/' + str(i) + '.png')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path + '/images/' + str(i) + '.png', gray)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if mode == 'enroll':
    take_picture(10, username)
elif mode == 'verify':
    take_picture(1, username)