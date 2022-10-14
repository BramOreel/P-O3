import cv2

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

with open('getal.txt') as f:
    img_counter = int(f.readline())
    f.close()
    print(img_counter)


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        with open('getal.txt','w') as f:
            f.write(str(img_counter))
            f.close()

        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()