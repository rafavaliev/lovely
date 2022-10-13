import random
import math
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys

# Load OpenCV classifier for face detection
# find path of xml file containing haarcascade file
casc_pathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
face_cascade = cv2.CascadeClassifier(casc_pathface)


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


def show_debug_image(coordinates, cv_image, emoji_radius, face_centre, face_coordinates, face_radius):
    # Show face and face center
    cv2.circle(cv_image, face_coordinates[0], int(face_radius), (255, 0, 0), 5)
    cv2.rectangle(cv_image, face_coordinates[0], (face_centre[0] + 1, face_centre[1] + 1), (0, 255, 0), 5)
    for (_x, _y) in coordinates:
        cv2.circle(cv_image, (_x, _y), int(emoji_radius), (0, 0, 255), 5)
    # Draw a circle on the image
    # Show the image, wait for a key press and close the window
    cv2.imshow("Faces found", cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Checks if a circle with radius and coordinates intersects with existing circles and a face
def check_if_intersects(coordinates=[], safe_coordinates=[], x_coord=0, y_coord=0, radius=1, sub_element_radius=1):
    def _intersects(coords=[], _x_coord=0, _y_coord=0, _radius1=1, _radius2=1):
        for (safe_x, safe_y) in coords:
            distance_from_point_to_circle = math.sqrt((safe_x - _x_coord) ** 2 + (safe_y - _y_coord) ** 2)

            some_randomness = 1  # np.random.randint(0, 1)
            calculated_radius = some_randomness * (_radius1 + _radius2)
            if distance_from_point_to_circle < calculated_radius:
                return True
        return False

    # Check if point intersects with any of the safe coordinates(faces)
    if _intersects(safe_coordinates, x_coord, y_coord, radius, sub_element_radius):
        return True
    # Check if a points intersects with any of the other points that we already generated
    if _intersects(coordinates, x_coord, y_coord, sub_element_radius, sub_element_radius):
        return True

    return False


# Gets face(s) coordinates from image
def get_face_coordinates(cv_image=None):
    """
    Detects a face on the image.
    :param cv_image: OpenCV representation of an image
    :return: array of points(x,y and height and width of the face)
    :raises: Exception if there is 0 or more than 1 faces on the image

    """

    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        raise (Exception("No faces found on the image"))
    if len(faces) > 1:
        raise (Exception("Too many faces found on the image: {0}".format(len(faces))))

    if faces[0][0] == 0 and faces[0][1] == 0 and faces[0][2] == 0 and faces[0][3]:
        raise (Exception("No faces found on the image"))

    return faces[0]


# Writes emojis to an image, returns image object back
def write_emojis_to_image(
        image=None,
        sub_image=None,
        _coordinates=[],
        sub_image_radius=1
):
    # Resize emoji to fit the face
    sub_image = sub_image.resize((int(sub_image_radius * 2), int(sub_image_radius * 2)))

    # No transparency mask specified, simulating a raster overlay
    back_im = image.copy()

    # We already generated coordinates for circles, so we insert emoji at those coordinates
    for (_x, _y) in _coordinates:
        back_im.paste(sub_image, (_x - sub_image_radius, _y - sub_image_radius), sub_image)

    return back_im


# Just random points
def generate_uniform_random_points(xmax=0, ymax=0, wall=0, n_points=1000):
    """
    Generates a set of uniformly distributed points over the area of image
    :param xmax: width of the image
    :param ymax: height of the image
    :param n_points: int number of points to generate
    :return: array of points
    """
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


# Generate validated points for emojis coordinates
def generate_validated_points(safe_coordinates=[], element_radius=1, sub_element_radius=1, max_x_coord=0,
                              max_y_coord=0):
    # Generate random points
    points = generate_uniform_random_points(xmax=max_x_coord, ymax=max_y_coord, wall=element_radius, n_points=10000)
    coordinates = []

    for (xx, yy) in points:
        xx = int(xx)
        yy = int(yy)

        # If a point is not safe, skip it
        if check_if_intersects(
                coordinates=coordinates,
                safe_coordinates=safe_coordinates,
                x_coord=xx,
                y_coord=yy,
                radius=element_radius,
                sub_element_radius=sub_element_radius
        ):
            continue
        if check_if_outside_image(
                x_coord=xx,
                y_coord=yy,
                xmax=max_x_coord,
                ymax=max_y_coord,
                radius=sub_element_radius
        ):
            continue

        coordinates.append((xx, yy))
    return coordinates


def emojify(image: Image = None, sub_image: Image = None, is_debug: bool = False) -> Image:
    if image is None:
        raise Exception("Image is not passed")
    if sub_image is None:
        raise Exception("Sub image is not passed")

    # Open image as OpenCV image
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Read image shape
    xmax, ymax = cv_image.shape[:2]

    # Get face coordinates
    x, y, w, h = get_face_coordinates(cv_image=cv_image)

    face_centre = find_rectangle_centre(x, y, w, h)
    face_diameter = w if w > h else h
    face_radius = int(face_diameter / 2)

    # Depending on the face size, there can fit none or too many emojis. let's try to fix it
    emoji_radius = int(xmax / 6)

    # Keep the points where we already put emojis + initial face coordinates
    face_coordinates = [(face_centre[0], face_centre[1])]

    coordinates = generate_validated_points(
        safe_coordinates=face_coordinates,
        element_radius=face_radius,
        sub_element_radius=emoji_radius,
        max_x_coord=xmax,
        max_y_coord=ymax
    )

    # If debug is enabled, show the image with circles, not emojis
    if is_debug:
        show_debug_image(coordinates, cv_image, emoji_radius, face_centre, face_coordinates, face_radius)

    result_img = write_emojis_to_image(
        image=image,
        sub_image=sub_image,
        _coordinates=coordinates,
        sub_image_radius=emoji_radius
    )
    if is_debug:
        result_img.show()

    return result_img


if __name__ == "__main__":

    # Check args
    if len(sys.argv) <= 1 or sys.argv[1] == "":
        print("No image is passed. Example: python3 lovely.py /path/to/image.jpg")
        exit(0)

    # Read path of the image
    image_path = sys.argv[1]
    if image_path == "":
        print("No image path provided")
        exit(0)

    # Read debug flag if any
    is_debug = False
    if len(sys.argv) == 3 and sys.argv[2] == "--debug":
        is_debug = True

    # Open image
    py_image = Image.open(image_path)
    # Open emoji that we want to insert
    py_sub_image = Image.open("heart.png")

    result_image = None  # type: Image
    try:
        result_image = emojify(image=py_image, sub_image=py_sub_image, is_debug=is_debug)
    except Exception as e:
        print(e)
        exit(0)

    # Save modified image with lovely-prefix in the filename
    output_image_path = get_new_path(path=image_path, prefix="lovely")

    f = open(output_image_path, "w+")
    f.write(result_image.tobytes())
    f.close()
    # result_image.save(output_image_path)
