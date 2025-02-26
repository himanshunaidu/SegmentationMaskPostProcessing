import cv2
import numpy as np
from segment_streets import find_objects, find_objects_tracking, segment_sidewalks

def load_image(image_path):
    return cv2.imread(image_path)

def load_depth(depth_path):
    return np.load(depth_path)

def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def test_find_objects():
    # Paths to your test images
    rgb_image_path = 'test_images/rgb_image.jpg'
    depth_image_path = 'test_images/depth_image.npy'
    mask_image_path = 'test_images/mask_image.png'

    # Load the images
    rgb_image = load_image(rgb_image_path)
    depth_image = load_depth(depth_image_path)
    mask_image = load_mask(mask_image_path)

    # Call the function
    object_boxes = find_objects(rgb_image, depth_image, mask_image)

    # Print the result
    print("Object Boxes:", object_boxes)

def test_find_objects_tracking():
    # Paths to your test images
    depth_image_path = 'test_images/depth_image.npy'
    mask_image_path = 'test_images/mask_image.png'

    # Load the images
    depth_image = load_depth(depth_image_path)
    mask_image = load_mask(mask_image_path)

    # Test coordinates and heading
    frame_lat = 37.7749  # Example latitude
    frame_lon = -122.4194  # Example longitude
    frame_heading = 90.0  # Example heading

    # Call the function
    object_list_dict = find_objects_tracking(depth_image, mask_image, frame_lat, frame_lon, frame_heading)

    # Print the result
    print("Object List Dict:", object_list_dict)

def test_segment_sidewalks():
    # Paths to your test images
    rgb_image_path = 'test_images/rgb_image.jpg'
    depth_image_path = 'test_images/depth_image.npy'
    mask_image_path = 'test_images/mask_image.png'

    # Load the images
    rgb_image = load_image(rgb_image_path)
    depth_image = load_depth(depth_image_path)
    mask_image = load_mask(mask_image_path)

    # Test coordinates and heading
    frame_lat = 37.7749  # Example latitude
    frame_lon = -122.4194  # Example longitude
    frame_heading = 90.0  # Example heading

    # Call the function
    sidewalks, width, est_gps = segment_sidewalks(rgb_image, depth_image, mask_image, frame_lat, frame_lon, frame_heading)

    # Print the result
    print("Sidewalks:", sidewalks)
    print("Width in meters:", width)
    print("Estimated GPS:", est_gps)

if __name__ == '__main__':
    test_find_objects()
    test_find_objects_tracking()
    test_segment_sidewalks()