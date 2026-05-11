"""Generate HSV-based labels from compressed HDF5 episodes."""
"""HSV-based label generation for compressed ManiSkill HDF5 episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

import argparse
from pathlib import Path

@dataclass(frozen=True)
class HSVThresholds:
    hue_min: float
    hue_max: float
    sat_min: float
    val_min: float
    min_pixel_ratio: float

class CompressedHDF5LabelGenerator:
    """Generate HSV-based labels from compressed HDF5 episodes."""

    def __init__(
        self,
        crop_region: Optional[Tuple[int, int, int, int]] = (100, 200, 300, 400),
        resize: Optional[Tuple[int, int]] = (640, 480),
    ) -> None:
        self.crop_region = crop_region
        self.resize = resize
        self.frame_indices: list[int] = []
        self.episode_ids: list[str] = []

    def _decode_compressed(self, compressed_data: np.ndarray) -> Optional[np.ndarray]:
        img_bgr = cv2.imdecode(np.frombuffer(compressed_data, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        if self.crop_region is not None:
            y0, y1, x0, x1 = self.crop_region
            img_bgr = img_bgr[y0:y1, x0:x1]
        if self.resize is not None:
            img_bgr = cv2.resize(img_bgr, self.resize)
        return img_bgr

    def _camera_key(self, camera: str) -> str:
        if camera == "wrist":
            return "cam_right_wrist_rgb"
        if camera == "high":
            return "cam_high_rgb"
        raise ValueError(f"Unsupported camera: {camera}")

    def extract_rgb_frames_from_episodes(
        self,
        data_dir: Path,
        camera: str = "high",
        sample_index: int = 279,
    ) -> tuple[list[np.ndarray], list[str]]:
        hdf5_files = sorted(Path(data_dir).glob("episode_*.hdf5"))
        frames: list[np.ndarray] = []
        episode_ids: list[str] = []

        cam_key = self._camera_key(camera)
        for hdf5_file in hdf5_files:
            episode_id = hdf5_file.stem
            with h5py.File(hdf5_file, "r") as handle:
                raw_compressed = handle[f"obs/{cam_key}"][sample_index]
            img_bgr = self._decode_compressed(raw_compressed)
            if img_bgr is None:
                continue
            frames.append(img_bgr)
            episode_ids.append(episode_id)
        return frames, episode_ids

    def interactive_hsv_tuning(
        self,
        sample_frames: Sequence[np.ndarray],
        sample_ids: Sequence[str],
    ) -> HSVThresholds:
        current_frame_idx = 0
        window_name = "HSV Tuning - n/p next/prev, q quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar("Hue Min", window_name, 140, 180, lambda _: None)
        cv2.createTrackbar("Hue Max", window_name, 160, 180, lambda _: None)
        cv2.createTrackbar("Sat Min", window_name, 76, 255, lambda _: None)
        cv2.createTrackbar("Val Min", window_name, 51, 255, lambda _: None)
        cv2.createTrackbar("Min Ratio %", window_name, 15, 100, lambda _: None)

        while True:
            frame_bgr = sample_frames[current_frame_idx]

            h_min = cv2.getTrackbarPos("Hue Min", window_name)
            h_max = cv2.getTrackbarPos("Hue Max", window_name)
            s_min = cv2.getTrackbarPos("Sat Min", window_name)
            v_min = cv2.getTrackbarPos("Val Min", window_name)
            ratio_threshold = cv2.getTrackbarPos("Min Ratio %", window_name) / 100.0

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                np.array([h_min, s_min, v_min]),
                np.array([h_max, 255, 255]),
            )

            pixel_ratio = np.sum(mask > 0) / mask.size
            result = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

            status = "MATCH" if pixel_ratio >= ratio_threshold else "NO MATCH"
            text_lines = [
                f"Episode: {sample_ids[current_frame_idx]} ({current_frame_idx + 1}/{len(sample_frames)})",
                f"Match: {pixel_ratio:.1%}",
                f"Status: {status}",
            ]

            for i, text in enumerate(text_lines):
                color = (0, 255, 0) if pixel_ratio >= ratio_threshold else (0, 0, 255)
                cv2.putText(result, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(window_name, result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                current_frame_idx = (current_frame_idx + 1) % len(sample_frames)
            elif key == ord("p"):
                current_frame_idx = (current_frame_idx - 1) % len(sample_frames)

        cv2.destroyAllWindows()

        return HSVThresholds(
            hue_min=h_min * 2.0,
            hue_max=h_max * 2.0,
            sat_min=s_min / 255.0,
            val_min=v_min / 255.0,
            min_pixel_ratio=ratio_threshold,
        )

    def generate_color_labels_from_episodes(
        self,
        data_dir: Path,
        camera: str = "wrist",
        target_hue_range: Tuple[float, float] = (280, 320),
        target_sat_min: float = 0.3,
        target_val_min: float = 0.2,
        min_pixel_ratio: float = 0.15,
    ) -> np.ndarray:
        hdf5_files = sorted(Path(data_dir).glob("episode_*.hdf5"))
        color_labels: list[int] = []
        cam_key = self._camera_key(camera)

        for hdf5_file in hdf5_files:
            episode_id = hdf5_file.stem
            with h5py.File(hdf5_file, "r") as handle:
                if f"obs/{cam_key}" not in handle:
                    continue
                compressed_rgb = handle[f"obs/{cam_key}"][()]

            for t, raw_compressed in enumerate(compressed_rgb):
                img_bgr = self._decode_compressed(raw_compressed)
                if img_bgr is None:
                    label = 0
                else:
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                    h = hsv[:, :, 0].astype(np.float32) * 2.0
                    s = hsv[:, :, 1].astype(np.float32) / 255.0
                    v = hsv[:, :, 2].astype(np.float32) / 255.0

                    h_mask = (h >= target_hue_range[0]) & (h <= target_hue_range[1])
                    s_mask = s >= target_sat_min
                    v_mask = v >= target_val_min
                    color_mask = h_mask & s_mask & v_mask

                    pixel_ratio = np.sum(color_mask) / color_mask.size
                    label = int(pixel_ratio >= min_pixel_ratio)

                color_labels.append(label)
                self.episode_ids.append(episode_id)
                self.frame_indices.append(t)

        return np.array(color_labels)

    def export_dataset(
        self,
        output_path: Path,
        color_labels: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        np.savez(
            output_path,
            color_labels=color_labels,
            frame_indices=np.array(self.frame_indices),
            episode_ids=np.array(self.episode_ids),
            metadata=json.dumps(metadata or {}),
        )

def load_hsv_thresholds(path: Path) -> HSVThresholds:
    with open(path, "r") as f:
        raw = json.load(f)
    return HSVThresholds(
        hue_min=raw["hue_min"],
        hue_max=raw["hue_max"],
        sat_min=raw["sat_min"],
        val_min=raw.get("val_min", 0.2),
        min_pixel_ratio=raw["min_pixel_ratio"],
    )

def save_hsv_thresholds(path: Path, thresholds: HSVThresholds) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "hue_min": thresholds.hue_min,
                "hue_max": thresholds.hue_max,
                "sat_min": thresholds.sat_min,
                "val_min": thresholds.val_min,
                "min_pixel_ratio": thresholds.min_pixel_ratio,
            },
            f,
            indent=2,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HSV labels from compressed HDF5")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--camera", choices=["wrist", "high"], default="wrist")
    parser.add_argument("--tune_hsv", action="store_true")
    parser.add_argument("--output", default="labels.npz")
    parser.add_argument("--sample_index", type=int, default=279)
    parser.add_argument("--hsv_thresholds", default="hsv_thresholds.json")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    generator = CompressedHDF5LabelGenerator()

    thresholds_path = data_dir / args.hsv_thresholds
    if args.tune_hsv:
        frames, ids = generator.extract_rgb_frames_from_episodes(
            data_dir, camera=args.camera, sample_index=args.sample_index
        )
        if not frames:
            raise RuntimeError("No frames found for HSV tuning")

        thresholds = generator.interactive_hsv_tuning(frames, ids)
        save_hsv_thresholds(thresholds_path, thresholds)
        return

    thresholds = load_hsv_thresholds(thresholds_path)
    color_labels = generator.generate_color_labels_from_episodes(
        data_dir,
        camera=args.camera,
        target_hue_range=(thresholds.hue_min, thresholds.hue_max),
        target_sat_min=thresholds.sat_min,
        target_val_min=thresholds.val_min,
        min_pixel_ratio=thresholds.min_pixel_ratio,
    )
    metadata = {
        "data_dir": str(data_dir),
        "camera": args.camera,
        "num_total_frames": len(color_labels),
        "hsv_thresholds": thresholds.__dict__,
    }
    generator.export_dataset(Path(args.output), color_labels, metadata)

if __name__ == "__main__":
    main()
