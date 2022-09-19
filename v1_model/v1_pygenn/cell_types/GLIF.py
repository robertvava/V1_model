class LIF:
    def __init__(self, V, Vthresh, Vreset, Vrest):
        self.V = V 
        self.Vthresh = Vthresh
        self.Vreset = Vreset
        self.Vrest = Vrest


class GLIF(LIF):
    def __init__(self, V, Vthresh, Vreset, Vrest):
        super.__init__(V, Vthresh, Vreset, Vrest)

class GLIF2(GLIF):
    def __init__(self, V, Vthresh, Vreset, Vrest):
        super.__init__(V, Vthresh, Vreset, Vrest)

class GLIF3(GLIF):
    def __init__(self, V, Vthresh, Vreset, Vrest):
        super.__init__(V, Vthresh, Vreset, Vrest)

class GLIF4(GLIF):
    def __init__(self, V, Vthresh, Vreset, Vrest):
        super.__init__(V, Vthresh, Vreset, Vrest)

class GLIF5(GLIF):
    def __init__(self, V, Vthresh, Vreset, Vrest):
        super.__init__(V, Vthresh, Vreset, Vrest)
    