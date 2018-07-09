import unittest
import numpy as np
from tracker import Tracker, Person, PathVisualiser


class TestTracker(unittest.TestCase):

    def setUp(self):
        self.person = np.array([[1, 2], [0, 0], [4, 5], [4, 3]])
        other_person_1 = np.array([[4, 5], [1, 2], [10, 23], [4, 1]])
        other_person_2 = np.array([[4, 10], [10, 2], [1, 3], [4, 1]])
        other_person_3 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.mostly_zero_person = np.array([[0, 0], [0, 0], [3, 4], [0, 0]])

        self.people = np.array([self.person, other_person_1, other_person_2, other_person_3])

        self.no_people = np.array([])

        self.far_away_person = np.array([[10000, 10000], [10000, 10000],
                                         [10000, 10000], [10000, 10000]])
        self.far_away_people = np.array([self.far_away_person])

        self.tracker = Tracker(no_openpose=True)

    def test_filter_nonzero(self):
        a = np.array([[1, 2, 3, 4], [0, 0, 2, 4]])
        b = np.array([[1, 0, 0, 4], [3, 2, 4, 0]])
        a, b = self.tracker.filter_nonzero(a, b)
        self.assertEqual(np.count_nonzero(a), a.size)
        self.assertEqual(np.count_nonzero(b), b.size)

    def test_find_assignments(self):
        assignments, _ = self.tracker.find_assignments(self.people, self.people)
        self.assertListEqual(list(assignments[0]), list(assignments[1]))

    def test_update_paths(self):
        other_people_order = np.array(
            [self.people[2], self.people[1], self.people[3], self.people[0], self.far_away_person])

        people = np.append(self.people, self.far_away_people, axis=0)

        self.tracker.people_paths = [[Person(i, p)] for i, p in enumerate(people)]
        self.tracker.path_indices = {i: i for i, _ in enumerate(people)}
        self.tracker.person_counter = len(people)

        assignments, distances = self.tracker.find_assignments(other_people_order, people)
        self.tracker.update_paths(distances, assignments, people)

        path_indices = self.tracker.path_indices
        # Do the paths get assigned correctly
        self.assertTrue(len(path_indices) == 5 and
                        path_indices[2] == 0 and
                        path_indices[1] == 1 and
                        # 0, but too far away to claim they are the same person
                        path_indices[3] == 5 and
                        path_indices[0] == 3 and
                        path_indices[4] == 4)

        assignments, distances = self.tracker.find_assignments(people, other_people_order)
        self.tracker.update_paths(distances, assignments, people)

        path_indices = self.tracker.path_indices
        # Reverse order, see that the indices still correspond to the correct
        # path
        self.assertTrue(len(path_indices) == 5 and
                        path_indices[0] == 0 and
                        path_indices[1] == 1 and
                        # 2, but too far away to claim they are the same person
                        path_indices[2] == 6 and
                        path_indices[3] == 3 and
                        path_indices[4] == 4)

        people_paths = self.tracker.people_paths
        # Check a path so that it also contains the correct people
        self.assertTrue(np.array_equal(people_paths[0][0].keypoints, self.people[0]) and
                        np.array_equal(people_paths[0][1].keypoints, self.people[2]) and
                        np.array_equal(people_paths[0][2].keypoints, self.people[0]))

        # Check that every person in the same path has the same path_index
        self.assertTrue(all(all(person.path_index == path[0].path_index
                                for person in path)
                            for path in self.tracker.people_paths))

    def test_get_nonzero_keypoint(self):
        person = Person(0, self.mostly_zero_person)
        keypoint = person.get_nonzero_keypoint()
        self.assertTrue(np.array_equal(keypoint, [3, 4]))

if __name__ == '__main__':
    unittest.main()
