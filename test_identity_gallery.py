from __future__ import annotations

import glob

import numpy as np
import torch

from config import (
    IDENTITY_GALLERY_PATH,
    TEST_PERSON_A_ID,
    TEST_PERSON_A_IMAGE_GLOB,
    TEST_PERSON_B_ID,
    TEST_PERSON_B_IMAGE_GLOB,
)
from identity_gallery import IdentityGallery


def _sorted_images(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def _print_pairwise_distances(labels: list[str], distance_matrix: np.ndarray) -> None:
    print("Embedding labels:")
    for idx, label in enumerate(labels):
        print(f"{idx}: {label}")

    print("\nFull pairwise distance matrix:")
    print(distance_matrix)


def _average_distance_summary(labels: list[str], distance_matrix: np.ndarray) -> tuple[float, float]:
    label_array = np.array(labels)
    same_person = label_array[:, None] == label_array[None, :]
    different_person = ~same_person
    off_diagonal = ~np.eye(len(labels), dtype=bool)

    within_distances = distance_matrix[same_person & off_diagonal]
    between_distances = distance_matrix[different_person]

    average_within = float(np.mean(within_distances))
    average_between = float(np.mean(between_distances))
    return average_within, average_between


def main() -> None:
    person_a_images = _sorted_images(TEST_PERSON_A_IMAGE_GLOB)
    person_b_images = _sorted_images(TEST_PERSON_B_IMAGE_GLOB)

    if not person_a_images:
        raise RuntimeError(f"No images found for {TEST_PERSON_A_ID}: {TEST_PERSON_A_IMAGE_GLOB}")
    if not person_b_images:
        raise RuntimeError(f"No images found for {TEST_PERSON_B_ID}: {TEST_PERSON_B_IMAGE_GLOB}")

    gallery = IdentityGallery()
    gallery.enroll_from_images(TEST_PERSON_A_ID, person_a_images)
    gallery.enroll_from_images(TEST_PERSON_B_ID, person_b_images)
    gallery.save(IDENTITY_GALLERY_PATH)

    labels = []
    embeddings = []
    for person_id, person_embeddings in gallery.embeddings.items():
        for embedding in person_embeddings:
            labels.append(person_id)
            embeddings.append(embedding)

    all_embeddings = torch.cat(embeddings, dim=0)
    distance_matrix = gallery.reid.compute_distance(all_embeddings, all_embeddings)

    _print_pairwise_distances(labels, distance_matrix)

    average_within, average_between = _average_distance_summary(labels, distance_matrix)
    print(f"\nAverage within-person distance: {average_within}")
    print(f"Average between-person distance: {average_between}")
    print(
        "Average within-person distance is smaller than average between-person distance:",
        average_within < average_between,
    )


if __name__ == "__main__":
    main()
