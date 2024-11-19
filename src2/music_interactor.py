class MusicInteractor():
    def __init__(self):
        self.roomSize = 0.0
        self.gain = 0.0
        self.low = 10
        self.high = 20000

    def hand2gain(self, hand):
        if hand["posScaled"] != None:
            self.gain = 15 - 30 * hand["posScaled"]["y"]
    
    def hand2reverb(self, hand):
        if hand["openness"] != None:
            if hand["openness"] < 0.0:
                self.roomSize = 0.0
            elif hand["openness"] > 1.0:
                self.roomSize = 1.0
            else:
                self.roomSize = hand["openness"]
    
    def hand2filter(self, lefthand, righthand):
        if lefthand["gesture"] != None:
            if lefthand["gesture"] == "Pointing_Up":
                self.low = 10 * 2.2 ** (lefthand["posScaled"]["x"] * 10)

        if righthand["gesture"] != None:
            if righthand["gesture"] == "Pointing_Up":
                self.high = 10 * 2.2 ** (righthand["posScaled"]["x"] * 10)