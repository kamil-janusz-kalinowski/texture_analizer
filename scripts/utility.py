from IPython.display import display, HTML
from abc import ABC, abstractmethod
import logging


class Progress_displayer(ABC):
    @abstractmethod
    def disp_progress(self):
        pass
    
    @abstractmethod
    def disp_finish(self):
        pass

class Terminal_displayer(Progress_displayer):
    def disp_progress(self, current, total):
        percent = current / total
        hashes = '#' * int(round(percent * self._bar_length))
        spaces = ' ' * (self._bar_length - len(hashes))
        print(f"\rProgress: [{hashes}{spaces}] {int(round(percent * 100))}% ", end='\r')
        
    def disp_finish(self):
        print("\nProgress finished")

class Log_displayer(Progress_displayer):
    def disp_progress(self, current, total):
        percent = current / total
        hashes = '#' * int(round(percent * self._bar_length))
        spaces = ' ' * (self._bar_length - len(hashes))
        logging.info(f"Progress: [{hashes}{spaces}] {int(round(percent * 100))}%")
        
    def disp_finish(self):
        logging.info("Progress finished")

class Jupyter_displayer(Progress_displayer):
    def disp_progress(self, current, total):
        percent = current / total
        hashes = '#' * int(round(percent * self._bar_length))
        spaces = ' ' * (self._bar_length - len(hashes))
        display(HTML(f"<p>Progress: [{hashes}{spaces}] {int(round(percent * 100))}%</p>"))
        
    def disp_finish(self):
        display(HTML("<p>Progress finished</p>"))


class Progress_bar():
    def __init__(self, total, current=0, bar_length=100, mode_display='terminal'):
        self._total = total
        self._current = current
        self._bar_length = bar_length
        self._displayer = None
        
        assert mode_display in ['terminal', 'log', 'jupyter'], 'mode_display must be one of: terminal, log, jupyter'
        
        if mode_display == 'terminal':
            self._displayer = Terminal_displayer()
        elif mode_display == 'log':
            self._displayer = Log_displayer()
        elif mode_display == 'jupyter':
            self._displayer = Jupyter_displayer()
    
    def update(self, current):
        self._current = current
        self._displayer.disp_progress(self._current, self._total)
        
    def finish(self):
        self._displayer.disp_finish()
        
    def reset(self):
        self._current = 0

