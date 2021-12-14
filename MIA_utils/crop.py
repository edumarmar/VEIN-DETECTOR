def crop(frame, width_frame, height_frame):
    width = frame.shape[1]
    height = frame.shape[0]
    range_x = [int(width / 2 - width_frame/2), int(width / 2 + width_frame/2)]
    range_y = [int(height / 2 - height_frame/2), int(height / 2 + height_frame/2)]
    frame = frame[range_y[0]: range_y[1], range_x[0]:range_x[1]]
    return frame
