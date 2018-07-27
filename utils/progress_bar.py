from datetime import datetime, timedelta

from progress.bar import Bar as BarInterface


class Bar(BarInterface):
    message = 'Loading'
    fill = '*'
    suffix = 'DONE: %(percent).1f%% (%(index)d of %(max)d) - ETA: %(long_eta_td)s - ELAPSED: %(long_elapsed_td)s'
    start_time = None

    def __init__(self, message, **kwargs):
        super().__init__(message, **kwargs)
        self.start_time = datetime.now()
        self.long_term_avg_eta = None

    def next(self, **kwargs):
        # do all the default stuff, and also update our average
        super(Bar, self).next(**kwargs)
        self.update_average()

    def update_average(self):
        avg = self.elapsed / self.index
        self.long_term_avg_eta = int(avg * self.remaining)

    @property
    def long_eta_td(self):
        if self.long_term_avg_eta:
            s = str(timedelta(seconds=self.long_term_avg_eta))
            return s
        return '??:??:??'

    @property
    def long_elapsed_td(self):
        s = str(timedelta(seconds=self.elapsed))
        return s
