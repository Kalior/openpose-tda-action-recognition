import unittest
import numpy as np
from tracker import Path, Person
from util import COCOKeypoints


class TestPath(unittest.TestCase):

    def setUp(self):
        self.person = Person(np.array([[1, 2], [0, 0], [4, 5], [4, 3]]))
        other_person_1 = Person(np.array([[4, 5], [1, 2], [10, 23], [4, 1]]))
        other_person_2 = Person(np.array([[4, 10], [10, 2], [1, 3], [4, 1]]))
        other_person_3 = Person(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
        mostly_zero_person = Person(np.array([[0, 0], [0, 0], [3, 4], [0, 0]]))
        self.people = [self.person, other_person_1, other_person_2,
                       other_person_3, mostly_zero_person]

    def test_is_relevant(self):
        path = Path()
        path.add_person(self.people[0], 10)

        self.assertTrue(path.is_relevant(0))
        self.assertTrue(path.is_relevant(10))
        self.assertFalse(path.is_relevant(20))

    def test_get_average_speed_in_window(self):
        path = Path()
        path.add_person(self.people[0], 10)
        path.add_person(self.people[0], 11)
        path.add_person(self.people[0], 12)
        path.add_person(self.people[0], 13)
        path.add_person(self.people[0], 14)

        speed = path.get_average_speed_in_window()
        self.assertEqual(speed, 0)

        path = Path()
        path.add_person(self.people[0], 0)
        path.add_person(self.people[4], 1)
        path.add_person(self.people[1], 2)
        speed = path.get_average_speed_in_window()

        distance = self.people[0].distance(self.people[4])
        distance += self.people[4].distance(self.people[1])
        manual_speed = distance / 2
        self.assertEqual(speed, manual_speed)

    def test_add_person(self):
        path = Path()
        path.add_person(self.people[0], 0)
        self.assertEqual(path.last_frame_update, 0)
        path.add_person(self.people[0], 10)
        self.assertEqual(path.last_frame_update, 10)

    def test_get_keypoint_path(self):
        path = Path()
        path.add_person(self.people[0], 0)
        path.add_person(self.people[1], 1)
        path.add_person(self.people[2], 2)
        path.add_person(self.people[3], 3)
        path.add_person(self.people[4], 4)

        keypoint_path = path.get_keypoint_path(COCOKeypoints.RElbow.value)

        manual_path = [
            self.people[0][COCOKeypoints.RElbow.value],
            self.people[1][COCOKeypoints.RElbow.value],
            self.people[2][COCOKeypoints.RElbow.value]
        ]
        np.testing.assert_array_equal(keypoint_path, manual_path)

if __name__ == '__main__':
    unittest.main()
