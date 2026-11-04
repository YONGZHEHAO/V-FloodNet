"""A batch must not silently run unreviewed people or apply assumed heights."""
import unittest
from scripts.batch_people_depth import reviewed_targets


class ReviewedBatchTests(unittest.TestCase):
    def test_only_explicit_boolean_review_is_approved(self):
        rows = [{"person_id": "approved", "reviewed_for_mesh": True, "height_cm": None},
                {"person_id": "unreviewed", "reviewed_for_mesh": False},
                {"person_id": "string_flag", "reviewed_for_mesh": "true"}]
        self.assertEqual([r["person_id"] for r in reviewed_targets({"selected": rows})], ["approved"])

    def test_duplicate_ids_are_rejected_before_inference(self):
        row = {"person_id": "same", "reviewed_for_mesh": True}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reviewed_targets({"selected": [row, row.copy()]})

    def test_batch_without_reviewed_targets_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No visually reviewed"):
            reviewed_targets({"selected": [{"person_id": "no", "reviewed_for_mesh": False}]})

    def test_nonempty_heights_require_single_person_verified_entry(self):
        with self.assertRaisesRegex(ValueError, "verified height"):
            reviewed_targets({"selected": [{"person_id": "height", "reviewed_for_mesh": True, "height_cm": 175.4}]})


if __name__ == "__main__":
    unittest.main()
