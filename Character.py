class Character:
    charVal = None
    pixels = []

    def __init__(self, data_list):
        self.charVal = data_list[1]
        self.pixels = list(map(int, data_list[6:]))