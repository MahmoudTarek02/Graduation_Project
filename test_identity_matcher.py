from __future__ import annotations

import glob

import cv2

from config import (
    IDENTITY_GALLERY_PATH,
    TEST_PERSON_A_ID,
    TEST_PERSON_A_MATCH_IMAGE_GLOB,
    TEST_PERSON_B_ID,
    TEST_PERSON_B_MATCH_IMAGE_GLOB,
    TEST_KNOWN_MATCH_MAX_IMAGES,
    TEST_KNOWN_MATCH_MIN_IMAGES,
    TEST_UNKNOWN_IMAGE_GLOB,
    TEST_UNKNOWN_MATCH_MIN_IMAGES,
    TEST_UNKNOWN_PERSON_ID,
)
from identity_gallery import IdentityGallery
from identity_matcher import IdentityMatcher


def _sorted_images(patterns: str | tuple[str, ...]) -> list[str]:
    if isinstance(patterns, str):
        patterns = (patterns,)

    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(pattern))

    return sorted(image_paths)


def _print_result(test_name: str, image_path: str, expected_id, actual_id, score) -> bool:
    passed = actual_id == expected_id
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} {test_name}: image={image_path}, "
        f"expected={expected_id}, actual={actual_id}, score={score}"
    )
    return passed


def _run_known_person_tests(
    matcher: IdentityMatcher,
    person_id: str,
    image_paths: list[str],
) -> list[bool]:
    if len(image_paths) < TEST_KNOWN_MATCH_MIN_IMAGES:
        print(
            f"FAIL known person {person_id}: found {len(image_paths)} test images, "
            f"need at least {TEST_KNOWN_MATCH_MIN_IMAGES}"
        )
        return [False]

    results = []
    for image_path in image_paths[:TEST_KNOWN_MATCH_MAX_IMAGES]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"FAIL known person {person_id}: cannot read image={image_path}")
            results.append(False)
            continue

        matched_id, score = matcher.match_image(image)
        results.append(_print_result(f"known person {person_id}", image_path, person_id, matched_id, score))

    return results


def _run_unknown_person_tests(
    matcher: IdentityMatcher,
    image_paths: list[str],
) -> list[bool]:
    if len(image_paths) < TEST_UNKNOWN_MATCH_MIN_IMAGES:
        print(
            f"FAIL unknown person {TEST_UNKNOWN_PERSON_ID}: found {len(image_paths)} test images, "
            f"need at least {TEST_UNKNOWN_MATCH_MIN_IMAGES}"
        )
        return [False]

    results = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"FAIL unknown person {TEST_UNKNOWN_PERSON_ID}: cannot read image={image_path}")
            results.append(False)
            continue

        matched_id, score = matcher.match_image(image)
        results.append(_print_result(f"unknown person {TEST_UNKNOWN_PERSON_ID}", image_path, None, matched_id, score))

    return results


def main() -> None:
    gallery = IdentityGallery.from_file(IDENTITY_GALLERY_PATH)
    matcher = IdentityMatcher(gallery)

    results = []
    results.extend(
        _run_known_person_tests(
            matcher,
            TEST_PERSON_A_ID,
            _sorted_images(TEST_PERSON_A_MATCH_IMAGE_GLOB),
        )
    )
    results.extend(
        _run_known_person_tests(
            matcher,
            TEST_PERSON_B_ID,
            _sorted_images(TEST_PERSON_B_MATCH_IMAGE_GLOB),
        )
    )
    results.extend(
        _run_unknown_person_tests(
            matcher,
            _sorted_images(TEST_UNKNOWN_IMAGE_GLOB),
        )
    )

    passed = all(results)
    print("PASS identity matcher tests" if passed else "FAIL identity matcher tests")
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
