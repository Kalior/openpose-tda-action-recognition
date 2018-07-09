class Path(object):

    def __init__(self):
        self.path = []
        self.last_frame_update = -1

    def __getitem__(self, item):
        return self.path[item]

    def add_person(self, person, current_frame):
        self.last_frame_update = current_frame
        self.path.append(person)

    def get_last_person(self):
        return self.path[-1]

    def get_average_speed_in_window(self, window_size=-1):
        if window_size == -1:
            window_size = len(self.path)

        distance = 0
        for i in range(len(self.path) - window_size + 1, len(self.path)):
            distance += self.path[i - 1].distance(self.path[i])

        speed = distance / window_size
        return speed

    def is_relevant(self, current_frame):
        # Paths are relevant if they are repeated
        return self.last_frame_update > current_frame - 10
