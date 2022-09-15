import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import logging
import typing as t

from .utils import iou, convert_bbox_to_z, convert_x_to_bbox
from .types import BoundingBox, Track


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    '''The linear assignment module tries to minimise the total assignment cost.
    In our case we pass -iou_matrix as we want to maximise the total IOU between track predictions and the frame detection.'''
    matched_rows, matched_cols = linear_sum_assignment(iou_matrix, maximize=True)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_rows:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_cols:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m_r, m_c in zip(matched_rows, matched_cols):
        if iou_matrix[m_r, m_c] < iou_threshold:
            unmatched_detections.append(m_r)
            unmatched_trackers.append(m_c)
        else:
            matches.append((m_r, m_c))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.stack(matches)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.additional_info = []
        self.predict_num = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if bbox != []:
            self.kf.update(convert_bbox_to_z(bbox))
            self.predict_num = 0
        else:
            self.predict_num += 1

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1][0]

    def get_state(self) -> BoundingBox:
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)[0]


class Sort:

    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, img_size, additional_info, predict_num) -> t.Sequence[Track]:
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if dets != []:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    trk.additional_info.append(additional_info[d[0]])

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                trk.additional_info.append(additional_info[i])
                logging.info("new Tracker: {0}".format(trk.id + 1))
                self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # ret.append(np.concatenate((d, [trk.id + 1], np.mean(trk.additional_info, axis=0))).reshape(1, -1))
                ret.append(Track(
                    track_id=trk.id + 1,
                    descriptors=trk.additional_info,
                    person=None,
                    tracking_box=d.astype(np.int32)
                ))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update >= self.max_age or trk.predict_num >= predict_num
                    or d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]):
                logging.info('remove tracker: {0}'.format(trk.id + 1))
                self.trackers.pop(i)
        return ret
