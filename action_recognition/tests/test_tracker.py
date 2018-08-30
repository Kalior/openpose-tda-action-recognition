import unittest
import numpy as np
from ..tracker import Tracker, Person, TrackVisualiser, Track


class TestTracker(unittest.TestCase):

    def setUp(self):
        self.person = Person(np.array([[1, 2], [0, 0], [4, 5], [4, 3]]))
        other_person_1 = Person(np.array([[4, 5], [1, 2], [10, 23], [4, 1]]))
        other_person_2 = Person(np.array([[4, 10], [10, 2], [1, 3], [4, 1]]))
        other_person_3 = Person(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
        self.mostly_zero_person = Person(np.array([[0, 0], [0, 0], [3, 4], [0, 0]]))

        self.people = [self.person, other_person_1, other_person_2, other_person_3]

        self.no_people = []

        self.far_away_person = Person(np.array([[10000, 10000], [10000, 10000],
                                                [10000, 10000], [10000, 10000]]))
        self.far_away_people = [self.far_away_person]

        self.tracker = Tracker(detector=None)

    def test_filter_nonzero(self):
        a = np.array([[1, 2, 3, 4], [0, 0, 2, 4]])
        b = np.array([[1, 0, 0, 4], [3, 2, 4, 0]])
        a, b = self.person._filter_nonzero(a, b)
        self.assertEqual(np.count_nonzero(a), a.size)
        self.assertEqual(np.count_nonzero(b), b.size)

    def test_find_assignments(self):
        self.tracker.tracks = [Track() for p in self.people]
        assignments, _, _ = self.tracker._find_assignments(self.people, self.people, 0)
        self.assertListEqual(list(assignments[0]), list(assignments[1]))

    def test_update_tracks(self):
        other_people_order = [self.people[2], self.people[1],
                              self.people[3], self.people[0], self.far_away_person]

        people = self.people + self.far_away_people

        assignments, distances, removed_people = self.tracker._find_assignments(people, [], 0)
        self.tracker._update_tracks(distances, assignments, people + removed_people, [], 0)

        assignments, distances, removed_people = self.tracker._find_assignments(
            other_people_order, people, 1)
        self.tracker._update_tracks(distances, assignments,
                                    other_people_order + removed_people, people, 1)

        clone_people = [Person(p.keypoints) for p in people]
        assignments, distances, removed_people = self.tracker._find_assignments(
            clone_people, other_people_order, 2)
        self.tracker._update_tracks(distances, assignments, clone_people +
                                    removed_people, other_people_order, 2)

        tracks = self.tracker.tracks

        manual_path = [self.people[0].keypoints,
                       self.people[0].keypoints, self.people[0].keypoints]
        # Check a track so that it also contains the correct people
        np.testing.assert_array_equal([p.keypoints for p in tracks[0]], manual_path)

        # Check that every person in the same track has the same track_index
        self.assertTrue(all(all(person.track_index == track[0].track_index
                                for person in track)
                            for track in self.tracker.tracks))

if __name__ == '__main__':
    unittest.main()
