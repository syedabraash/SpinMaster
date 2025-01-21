# Computer vision/graphics library
import cv2

# Gif writer
import imageio

# Display libraries
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# Calculations and Deep Learning library
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

! pip install -q git+https://github.com/tensorflow/docs

cyan = (255, 255, 0)
magenta = (255, 0, 255)

EDGE_COLORS = {
    (0, 1): magenta,
    (0, 2): cyan,
    (1, 3): magenta,
    (2, 4): cyan,
    (0, 5): magenta,
    (0, 6): cyan,
    (5, 7): magenta,
    (7, 9): cyan,
    (6, 8): magenta,
    (8, 10): cyan,
    (5, 6): magenta,
    (5, 11): cyan,
    (6, 12): magenta,
    (11, 12): cyan,
    (11, 13): magenta,
    (13, 15): cyan,
    (12, 14): magenta,
    (14, 16): cyan
}

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]

#initial_width, initial_height = (461,250)
WIDTH = HEIGHT = 256

def loop(frame, keypoints, threshold=0.11):
    """
    Main loop : Draws the keypoints and edges for each instance
    """

    # Loop through the results
    for instance in keypoints:
        # Draw the keypoints and get the denormalized coordinates
        denormalized_coordinates = draw_keypoints(frame, instance, threshold)
        # Draw the edges
        draw_edges(denormalized_coordinates, frame, EDGE_COLORS, threshold)

def draw_keypoints(frame, keypoints, threshold=0.11):
    """Draws the keypoints on a image frame"""

    # Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
    denormalized_coordinates = np.squeeze(np.multiply(keypoints, [WIDTH,HEIGHT,1]))
    #Iterate through the points
    for keypoint in denormalized_coordinates:
        # Unpack the keypoint values : y, x, confidence score
        keypoint_y, keypoint_x, keypoint_confidence = keypoint
        if keypoint_confidence > threshold:
            """"
            Draw the circle
            Note : A thickness of -1 px will fill the circle shape by the specified color.
            """
            cv2.circle(
                img=frame,
                center=(int(keypoint_x), int(keypoint_y)),
                radius=4,
                color=(255,0,0),
                thickness=-1
            )
    return denormalized_coordinates

def draw_edges(denormalized_coordinates, frame, edges_colors, threshold=0.11):
    """
    Draws the edges on a image frame
    """

    # Iterate through the edges
    for edge, color in edges_colors.items():
        # Get the dict value associated to the actual edge
        p1, p2 = edge
        # Get the points
        y1, x1, confidence_1 = denormalized_coordinates[p1]
        y2, x2, confidence_2 = denormalized_coordinates[p2]
        # Draw the line from point 1 to point 2, the confidence > threshold
        if (confidence_1 > threshold) & (confidence_2 > threshold):
            cv2.line(
                img=frame,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA # Gives anti-aliased (smoothed) line which looks great for curves
            )

def progress(value, max=100):
    """
    Returns an HTML progress bar with a certain value. Used within each step
    """


    return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value,
                max=max))

def load_gif():
    """
    Loads the gif and return its details
    """

    # Load the gif
    gif = cv2.VideoCapture("/content/rally 0.gif.gif")
    # Get the frame count
    frame_count = int(gif.get(cv2.CAP_PROP_FRAME_COUNT))
    # Display parameter
    print(f"Frame count: {frame_count}")

    """""
    Initialize the video writer
    We'll append each frame and its drawing to a vector, then stack all the frames to obtain a sequence (video).
    """
    output_frames = []

    # Get the initial shape (width, height)
    initial_shape = []
    initial_shape.append(int(gif.get(cv2.CAP_PROP_FRAME_WIDTH)))
    initial_shape.append(int(gif.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    return gif, frame_count, output_frames, initial_shape

def run_inference():
    """
    Runs inferences then starts the main loop for each frame
    """

    # Load the gif
    gif, frame_count, output_frames, initial_shape = load_gif()
    # Set the progress bar to 0. It ranges from the first to the last frame
    bar = display(progress(0, frame_count-1), display_id=True)

    # Loop while the gif is opened
    while gif.isOpened():

        # Capture the frame
        ret, frame = gif.read()

        # Exit if the frame is empty
        if frame is None:
            break

        # Retrieve the frame index
        current_index = gif.get(cv2.CAP_PROP_POS_FRAMES)

        # Copy the frame
        image = frame.copy()
        image = cv2.resize(image, (WIDTH,HEIGHT))
        # Resize to the target shape and cast to an int32 vector
        input_image = tf.cast(tf.image.resize_with_pad(image, WIDTH, HEIGHT), dtype=tf.int32)
        # Create a batch (input tensor)
        input_image = tf.expand_dims(input_image, axis=0)

        # Perform inference
        results = movenet(input_image)
        """
        Output shape :  [1, 6, 56] ---> (batch size), (instances), (xy keypoints coordinates and score from [0:50]
        and [ymin, xmin, ymax, xmax, score] for the remaining elements)
        First, let's resize it to a more convenient shape, following this logic :
        - First channel ---> each instance
        - Second channel ---> 17 keypoints for each instance
        - The 51st values of the last channel ----> the confidence score.
        Thus, the Tensor is reshaped without losing important information.
        """

        keypoints = results["output_0"].numpy()[:,:,:51].reshape((6,17,3))

        # Loop through the results
        loop(image, keypoints, threshold=0.11)

        # Get the output frame : reshape to the original size
        frame_rgb = cv2.cvtColor(
            cv2.resize(
                image,(initial_shape[0], initial_shape[1]),
                interpolation=cv2.INTER_LANCZOS4
            ),
            cv2.COLOR_BGR2RGB # OpenCV processes BGR images instead of RGB
        )

        # Add the drawings to the output frames
        output_frames.append(frame_rgb)

        # Update the progress bar
        bar.update(progress(current_index, frame_count-1))

    # Release the object
    gif.release()

    print("Completed !")

    return output_frames

output_frames = run_inference()

from tensorflow_docs.vis import embed

# Stack the output frames horizontally to compose a sequence
output = np.stack(output_frames, axis=0)
# Write the sequence to a gif
imageio.mimsave("./animation.gif", output, fps=15)
# Embed the output to the notebook
embed.embed_file("./animation.gif")
