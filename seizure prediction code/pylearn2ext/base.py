class PerformanceMetric(object):

    def __init__(self, threshold, predict_seizure_seconds,
                 total_hours, total_seizures,
                 tp, fp, fp_range, latency):
        self.threshold = threshold
        self.predict_seizure_seconds = predict_seizure_seconds
        self.total_hours = total_hours
        self.total_seizures = total_seizures
        self.tp = tp
        self.fp = fp
        self.fp_range = fp_range
        self.latency = latency