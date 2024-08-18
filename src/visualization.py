import cv2


def draw_detection(frame, detection, color=(0, 255, 0), label=None):
    if detection is not None:
        if isinstance(detection, tuple) and len(detection) == 4:
            x, y, w, h = detection
        elif isinstance(detection, tuple) and len(detection) == 2:
            center_x, center_y = detection
            w, h = 20, 20  # Default size if only center is provided
            x, y = int(center_x - w / 2), int(center_y - h / 2)
        else:
            print(f"Unexpected detection format: {detection}")
            return

        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        if label:
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def visualize_results(frame, detection, debug_info, detector_type):
    result_frame = frame.copy()

    if detection is not None:
        color = (0, 255, 0)  # Default color
        if detector_type == "FRAME_DIFF":
            color = (0, 255, 0)
        elif detector_type == "MOG2":
            color = (255, 0, 0)
        elif detector_type == "OPT_FLOW":
            color = (0, 0, 255)

        draw_detection(result_frame, detection, color=color, label=detector_type)

    if 'optical_flow' in debug_info and 'motion_vectors' in debug_info['optical_flow'] and 'points' in debug_info[
        'optical_flow']:
        draw_motion_vectors(result_frame, debug_info['optical_flow']['motion_vectors'],
                            debug_info['optical_flow']['points'], color=(0, 255, 255))

    return result_frame


def draw_motion_vectors(frame, vectors, points, color=(0, 255, 255)):
    for i, (vector, point) in enumerate(zip(vectors, points)):
        start_point = tuple(point.ravel().astype(int))
        end_point = tuple((point + vector * 3).ravel().astype(int))
        cv2.arrowedLine(frame, start_point, end_point, color, 1, tipLength=0.2)


def draw_mog2_mask(frame, mask, scope):
    x, y = scope.top_left
    w, h = scope.size

    mask_resized = cv2.resize(mask, (w, h))
    mask_overlay = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    roi = frame[y:y + h, x:x + w]

    if roi.shape != mask_overlay.shape:
        print(f"ROI shape: {roi.shape}, Mask overlay shape: {mask_overlay.shape}")
        return

    cv2.addWeighted(roi, 0.7, mask_overlay, 0.3, 0, roi)
    frame[y:y + h, x:x + w] = roi


def draw_scope(frame, scope, color=(0, 255, 0)):
    x, y = scope.top_left
    w, h = scope.size
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def create_debug_image(frame, detection, debug_info, detector_type, scope):
    result_frame = visualize_results(frame, detection, debug_info, detector_type)
    draw_scope(result_frame, scope)

    if detector_type == "MOG2" and 'mask' in debug_info:
        draw_mog2_mask(result_frame, debug_info['mask'], scope)

    return result_frame


def draw_bbox(frame, bbox, color):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
