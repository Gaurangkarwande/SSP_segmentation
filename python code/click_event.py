# importing the module
import cv2
import numpy as np

# function to display the coordinates of
# of the points clicked on the image


def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensity = gray[x][y]
        print(x, ' ', y, ' ', intensity)

        # displaying the coordinates
        # on the image window
        text = '(' + str(x) + ',' + str(y) + ',' + str(intensity) + ')'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x, y), font,
                    0.5, (0, 0, 255), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":

    # reading the image
    img = cv2.imread('..\dataset\mri\image_5\image_5.jpg', 1)

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
