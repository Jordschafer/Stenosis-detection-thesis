from tqdm import tqdm
import torch
from torchvision.ops import box_iou
from typing import List, Dict

def apply_sequence_consistency_alignment(
        sequence_detections: List[Dict[str, torch.Tensor]],
        t_iou: float = 0.3,
        t_frame: int = 3,
        t_score_interp: float = 0.1,
        max_frame_gap_for_linking: int = 1,
        debug: bool = False
) -> List[Dict[str, torch.Tensor]]:
    """
    Applies Sequence Consistency Alignment (SCA) to a sequence of detections.
    """

    num_total_frames = len(sequence_detections)
    if num_total_frames == 0:
        return []

    # 🔧 Determine and normalize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for frame in sequence_detections:
        frame['boxes'] = frame['boxes'].to(device)
        frame['scores'] = frame['scores'].to(device)
        frame['labels'] = frame['labels'].to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    if debug:
        print(f"SCA starting on {num_total_frames} frames using device {device}.")

    tracks = []
    used_detections_coords = set()

    # First tqdm: Track building
    for frame_idx in tqdm(range(num_total_frames), desc="SCA: Building Tracks"):
        current_frame = sequence_detections[frame_idx]
        for det_idx in range(current_frame['boxes'].shape[0]):
            if (frame_idx, det_idx) in used_detections_coords:
                continue

            box = current_frame['boxes'][det_idx]
            score = current_frame['scores'][det_idx]
            label = current_frame['labels'][det_idx]

            track = [(frame_idx, {'box': box, 'score': score, 'label': label})]
            used_detections_coords.add((frame_idx, det_idx))

            last_idx = frame_idx
            last_box = box
            last_label = label

            while True:
                best_match = None
                for offset in range(1, max_frame_gap_for_linking + 2):
                    next_idx = last_idx + offset
                    if next_idx >= num_total_frames:
                        break
                    next_frame = sequence_detections[next_idx]
                    for next_det_idx in range(next_frame['boxes'].shape[0]):
                        if (next_idx, next_det_idx) in used_detections_coords:
                            continue
                        next_box = next_frame['boxes'][next_det_idx]
                        next_label = next_frame['labels'][next_det_idx]
                        if next_label != last_label:
                            continue
                        iou = box_iou(last_box.unsqueeze(0), next_box.unsqueeze(0))[0, 0].item()
                        if iou >= t_iou:
                            best_match = (next_idx, next_det_idx, next_box, next_frame['scores'][next_det_idx], next_label)
                            break
                    if best_match:
                        break

                if best_match:
                    next_idx, next_det_idx, next_box, next_score, next_label = best_match
                    track.append((next_idx, {'box': next_box, 'score': next_score, 'label': next_label}))
                    used_detections_coords.add((next_idx, next_det_idx))
                    last_idx = next_idx
                    last_box = next_box
                    last_label = next_label
                else:
                    break

            if len(track) >= t_frame:
                tracks.append(track)

    if debug:
        print(f"SCA built {len(tracks)} tracks.")

    # Second tqdm: Interpolation
    interpolated_tracks = []
    for track in tqdm(tracks, desc="SCA: Interpolating Tracks"):
        track.sort(key=lambda x: x[0])
        frame_range = range(track[0][0], track[-1][0] + 1)
        track_map = {idx: data for idx, data in track}
        interpolated = []

        for f in frame_range:
            if f in track_map:
                interpolated.append((f, track_map[f]))
            else:
                # Find bounding boxes before and after for interpolation
                prev = next((data for idx, data in reversed(track) if idx < f), None)
                succ = next((data for idx, data in track if idx > f), None)
                if prev and succ:
                    alpha = (f - track[0][0]) / (track[-1][0] - track[0][0])
                    box_interp = prev['box'] * (1 - alpha) + succ['box'] * alpha
                    interpolated.append((f, {
                        'box': box_interp,
                        'score': torch.tensor(t_score_interp, device=device),
                        'label': prev['label']
                    }))

        interpolated_tracks.append(interpolated)

    # Reconstruct sequence detections
    output_sequence = []
    for f in range(num_total_frames):
        frame_boxes, frame_scores, frame_labels = [], [], []
        for track in interpolated_tracks:
            for idx, det in track:
                if idx == f:
                    frame_boxes.append(det['box'])
                    frame_scores.append(det['score'])
                    frame_labels.append(det['label'])
        if frame_boxes:
            output_sequence.append({
                'boxes': torch.stack(frame_boxes),
                'scores': torch.stack(frame_scores),
                'labels': torch.stack(frame_labels)
            })
        else:
            output_sequence.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32, device=device),
                'scores': torch.empty((0,), dtype=torch.float32, device=device),
                'labels': torch.empty((0,), dtype=torch.int64, device=device),
            })

    return output_sequence
