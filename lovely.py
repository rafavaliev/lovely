import random
import math
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys


# Checks if a circle with radius and coordinates is within specific borders + allowed overflow distance
def check_if_outside_image(x_coord=0, y_coord=0, xmax=0, ymax=0, radius=1):
    overflow_distance = radius / 3
    if x_coord + radius > xmax + overflow_distance:
        return True
    if x_coord - radius < 0 - overflow_distance:
        return True
    if y_coord + radius > ymax + overflow_distance:
        return True
    if (y_coord - radius) < 0 - overflow_distance:
        return True
    return False


# Checks if a circle with radius and coordinates intersects with existing circles and a face
def check_if_intersects(coordinates=[], safe_coordinates=[], x_coord=0, y_coord=0, radius=1, sub_element_radius=1):
    def _intersects(coords=[], _x_coord=0, _y_coord=0, _radius1=1, _radius2=1):
        for (safe_x, safe_y) in coords:
            distance_from_point_to_circle = math.sqrt((safe_x - _x_coord) ** 2 + (safe_y - _y_coord) ** 2)

            some_randomness = np.random.randint(0, 1)
            some_randomness = 1
            calculated_radius = some_randomness * (_radius1 + _radius2)
            if distance_from_point_to_circle < calculated_radius:
                return True
            # if ((x_coord - safe_x) ** 2 + (y_coord - safe_y) ** 2) <= (sub_element_radius * (1 - some_randomness)) ** 2:
            #     return False
            # if (x_coord + sub_element_radius) > xmax or (x_coord - sub_element_radius) < 0 or (
            #         y_coord + sub_element_radius) > ymax or (
            #         y_coord - sub_element_radius) < 0:
            #     return False
        return False

    # Check if point intersects with any of the safe coordinates(faces)
    if _intersects(safe_coordinates, x_coord, y_coord, radius, sub_element_radius):
        return True
    # Check if a points intersects with any of the other points that we already generated
    if _intersects(coordinates, x_coord, y_coord, sub_element_radius, sub_element_radius):
        return True

    return False


def get_face_coordinates(img_path=""):
    """
    Detects a face on the image.
    If image can't be read/open, raises an exception
    If there is 0 or more than 1 faces on the image, raises an exception
    :param img_path: path to the image
    :return: array of points(x,y and height and width of the face)

    """
    if imagePath == "":
        raise (Exception("No image path provided"))
    # find path of xml file containing haarcascade file
    casc_pathface = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    face_cascade = cv.CascadeClassifier(casc_pathface)
    # Find path to the image you want to detect face and pass it here
    try:
        image = cv.imread(imagePath)
    except Exception:
        raise (Exception("Could not open the image"))
    if image is None:
        raise (Exception("Image in path {} not found".format(imagePath)))
    # rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # convert image to Greyscale for haarcascade
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        raise (Exception("No faces found on the image"))
    if len(faces) > 1:
        raise (Exception("Too many faces found on the image: {0}".format(len(faces))))

    if faces[0][0] == 0 and faces[0][1] == 0 and faces[0][2] == 0 and faces[0][3]:
        raise (Exception("No faces found on the image"))

    return faces[0]


# Writes emojis to an image, returns image object back
def write_emojis_to_image(image_path="", sub_image_path="", coordinates=[], sub_image_radius=1):
    # Re-open the image to insert emojis and not stupid circles
    original_image = Image.open(image_path)

    # Open emoji that we want to insert
    heart = Image.open(sub_image_path)
    # Resize emoji to fit the face
    heart = heart.resize((int(sub_image_radius * 2), int(sub_image_radius * 2)))

    # No transparency mask specified, simulating a raster overlay
    back_im = original_image.copy()

    # We already generated coordinates for circles, so we insert emoji at those coordinates
    for (_x, _y) in coordinates:
        back_im.paste(heart, (_x - sub_image_radius, _y - sub_image_radius), heart)

    return back_im


# Just random points
def generate_uniform_random_points(image, wall=0, n_points=1000):
    """
    Generates a set of uniformly distributed points over the area of image
    :param image: image as an array
    :param n_points: int number of points to generate
    :return: array of points
    """
    ymax, xmax = image.shape[:2]
    return [(random.randint(0 + int(wall / 2), ymax - int(wall / 2)),
             random.randint(0 + int(wall / 2), ymax - int(wall / 2))) for n in range(n_points)]
    # return np.random.uniform(0+int(wall/2), ymax-int(wall/2), size=(n_points, 2))
    # return np.random.normal((0+int(wall/2), ymax-int(wall/2)), size=(n_points, 2))


# Get output image path, we add prefix(ex:'lovely') to the filename
def get_new_path(path, prefix="lovely"):
    parts = path.split(".")
    return "{}-{}.{}".format("".join(parts[:len(parts) - 1]), prefix, parts[-1])


# Smart math
def find_rectangle_centre(left, top, width, height):
    return int(left + width / 2), int(top + height / 2)


# Read path of the image
if len(sys.argv) <= 1 or sys.argv[1] == "":
    print("No image is passed. Example: python3 lovely.py /path/to/image.jpg")
    exit(0)
imagePath = sys.argv[1]

# Get face coordinates
x, y, w, h = 0, 0, 0, 0
try:
    x, y, w, h = get_face_coordinates(img_path=imagePath)
except Exception as e:
    print(e)
    exit(0)
print("Face found at: x: {}, y: {}, width: {}, height: {}".format(x, y, w, h))

# Read image
img = cv.imread(imagePath)
xmax, ymax, chan = img.shape


def get_number_of_emojis(image_width, face_width):
    return 5 if int(image_width / face_width) > 5 else int(image_width / face_width)


number_of_emojis = get_number_of_emojis(xmax, w)

face_centre = find_rectangle_centre(x, y, w, h)
face_diameter = w if w > h else h
face_radius = int(face_diameter / 2)

# Depending on the face size, there can fit none or too many emojis. let's try to fix it
heart_radius = xmax / 6
heart_radius = int(heart_radius)

# Generate image with random points

# Keep the points where we already put emojis + initial face coordinates
coordinates = []
safe_coordinates = [(face_centre[0], face_centre[1])]

# Show face
cv.circle(img, (face_centre[0], face_centre[1]), int(face_radius), (255, 0, 0), 5)
# Show face center
cv.rectangle(img, (face_centre[0], face_centre[1]), (face_centre[0] + 1, face_centre[1] + 1), (0, 255, 0), 5)

# Generate random points
points = generate_uniform_random_points(image=img, wall=face_radius, n_points=10000)

for (xx, yy) in points:
    xx = int(xx)
    yy = int(yy)

    # If a point is not safe, skip it
    if check_if_intersects(
            coordinates=coordinates,
            safe_coordinates=safe_coordinates,
            x_coord=xx,
            y_coord=yy,
            radius=face_radius,
            sub_element_radius=heart_radius
    ):
        continue
    if check_if_outside_image(x_coord=xx, y_coord=yy, xmax=xmax, ymax=ymax, radius=heart_radius):
        continue
    # Draw a circle on the image
    cv.circle(img, (xx, yy), int(heart_radius), (0, 0, 255), 5)
    coordinates.append((xx, yy))

# Show the image, wait for a key press and close the window
cv.imshow("Faces found", img)
cv.waitKey(0)
cv.destroyAllWindows()

result_img = write_emojis_to_image(
    image_path=imagePath,
    sub_image_path="heart.png",
    coordinates=coordinates,
    sub_image_radius=heart_radius
)
result_img.show()
# Save modified image with lovely-prefix in the filename
output_image_path = get_new_path(path=imagePath, prefix="lovely")

f = open(output_image_path, "w")
result_img.save(output_image_path)

print("finished.")
exit()
