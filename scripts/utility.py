class Progress_bar():
    def __init__(self, total, current=0, bar_length=100):
        self._total = total
        self._current = current
        self._bar_length = bar_length
        
    def update(self, current):
        self._current = current
        percent = self._current / self._total
        hashes = '#' * int(round(percent * self._bar_length))
        spaces = ' ' * (self._bar_length - len(hashes))
        print(f"\rProgress: [{hashes}{spaces}] {int(round(percent * 100))}% ", end='\r')
        
    def finish(self):
        print("\nProgress finished")
        
    def reset(self):
        self._current = 0  