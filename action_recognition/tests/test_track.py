import unittest
import numpy as np
from ..tracker import Track, Person
from ..util import COCOKeypoints


class TestTrack(unittest.TestCase):

    def setUp(self):
        self.person = Person(np.array([[1, 2], [0, 0], [4, 5], [4, 3]]))
        other_person_1 = Person(np.array([[4, 5], [1, 2], [10, 23], [4, 1]]))
        other_person_2 = Person(np.array([[4, 10], [10, 2], [1, 3], [4, 1]]))
        other_person_3 = Person(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
        mostly_zero_person = Person(np.array([[0, 0], [0, 0], [3, 4], [0, 0]]))
        self.people = [self.person, other_person_1, other_person_2,
                       other_person_3, mostly_zero_person]

    def test_recently_updated(self):
        track = Track()
        track.add_person(self.people[0], 10)

        self.assertTrue(track.recently_updated(0))
        self.assertTrue(track.recently_updated(10))
        self.assertFalse(track.recently_updated(20))

    def test_get_average_speed_in_window(self):
        track = Track()
        track.add_person(self.people[0], 10)
        track.add_person(self.people[0], 11)
        track.add_person(self.people[0], 12)
        track.add_person(self.people[0], 13)
        track.add_person(self.people[0], 14)

        speed = track.get_average_speed_in_window()
        self.assertEqual(speed, 0)

        track = Track()
        track.add_person(self.people[0], 0)
        track.add_person(self.people[4], 1)
        track.add_person(self.people[1], 2)
        speed = track.get_average_speed_in_window()

        distance = self.people[0].distance(self.people[4])
        distance += self.people[4].distance(self.people[1])
        manual_speed = distance / 2
        self.assertEqual(speed, manual_speed)

    def test_add_person(self):
        track = Track()
        track.add_person(self.people[0], 0)
        self.assertEqual(track.last_frame_update, 0)
        track.add_person(self.people[0], 10)
        self.assertEqual(track.last_frame_update, 10)

    def test_get_keypoint_path(self):
        track = Track()
        track.add_person(self.people[0], 0)
        track.add_person(self.people[1], 1)
        track.add_person(self.people[2], 2)
        track.add_person(self.people[3], 3)
        track.add_person(self.people[4], 4)

        keypoint_path = track.get_keypoint_path(COCOKeypoints.RElbow.value)

        manual_path = [
            self.people[0][COCOKeypoints.RElbow.value],
            self.people[1][COCOKeypoints.RElbow.value],
            self.people[2][COCOKeypoints.RElbow.value]
        ]
        np.testing.assert_array_equal(keypoint_path, manual_path)

    def test_fill_missing_frames(self):
        track = Track()
        track.add_person(self.people[0], 0)
        track.add_person(self.people[4], 4)

        track.fill_missing_frames()

        diff = self.people[4].keypoints - self.people[0].keypoints
        step = diff / 4
        p1 = Person(self.people[0].keypoints + step)
        p2 = Person(self.people[0].keypoints + step * 2)
        p3 = Person(self.people[0].keypoints + step * 3)
        manual_track = [
            self.people[0].keypoints,
            p1.keypoints,
            p2.keypoints,
            p3.keypoints,
            self.people[4].keypoints
        ]
        keypoint_track = [p.keypoints for p in track.track]
        np.testing.assert_array_equal(keypoint_track, manual_track)
        np.testing.assert_array_equal(track.frame_assigned, [0, 1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
