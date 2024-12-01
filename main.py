from model import Model
from processor import VideoProcessor
from tracker import *


def main():
    model = Model()
    tracker = Tracker()
    inside_frame = [[(351, 1300), (1600, 590), (2400, 1001), (1280, 2400)]]
    out_frame = [[(1625, 600), (1650, 480), (1820, 530), (1780, 670)]]
    processor = VideoProcessor(model, tracker)
    processor.set_video("/Users/alex/Desktop/algorithms/al/samples/double_sliced_book.avi")
    processor.set_polygons(inside_frame, out_frame)
    processor.process_video(True)


if __name__ == '__main__':
    main()
