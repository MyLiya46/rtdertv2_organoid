import os
import sys
from copy import deepcopy
import motmetrics as mm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from TrackEval import trackeval
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
from pathlib import Path
import glob
import csv
import configparser


class HOTA(trackeval.metrics._base_metric._BaseMetric):
    """Class which implements the HOTA metrics.
    See: https://link.springer.com/article/10.1007/s11263-020-01375-2
    """

    def __init__(self, config=None):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    # @_timing.time
    def eval_sequence(self, data):
        """Calculates the HOTA metrics for one sequence"""

        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float64)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA'] = np.ones((len(self.array_labels)), dtype=np.float64)
            res['LocA(0)'] = 1.0
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                for a, alpha in enumerate(self.array_labels):
                    res['HOTA_FN'][a] += len(gt_ids_t)
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Calculate and accumulate basic statistics
            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)
                res['HOTA_TP'][a] += num_matches
                res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
                res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
                if num_matches > 0:
                    res['LocA'][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                    matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

        # Calculate association scores (AssA, AssRe, AssPr) for the alpha value.
        # First calculate scores per gt_id/tracker_id combo and then average over the number of detections.
        for a, alpha in enumerate(self.array_labels):
            matches_count = matches_counts[a]
            ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
            res['AssA'][a] = np.sum(matches_count * ass_a) / np.maximum(1, res['HOTA_TP'][a])
            ass_re = matches_count / np.maximum(1, gt_id_count)
            res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1, res['HOTA_TP'][a])
            ass_pr = matches_count / np.maximum(1, tracker_id_count)
            res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res['HOTA_TP'][a])

        # Calculate final scores
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_array_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items()
                     if (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)

        for field in self.float_fields + self.float_array_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values() if
                                      (v['HOTA_TP'] + v['HOTA_FN'] + v['HOTA_FP'] > 0 + np.finfo('float').eps).any()],
                                     axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_array_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in ['AssRe', 'AssPr', 'AssA']:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        loca_weighted_sum = sum([all_res[k]['LocA'] * all_res[k]['HOTA_TP'] for k in all_res.keys()])
        res['LocA'] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])

        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)']*res['LocA(0)']
        return res

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        for name, style in zip(self.float_array_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('alpha')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in self.float_array_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='lower left')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()

    def calculate_box_ious(self, bboxes1, bboxes2, do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """
        # layout: (x0, y0, w, h)
        bboxes1 = deepcopy(bboxes1)
        bboxes2 = deepcopy(bboxes2)

        # layout: (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo('float').eps
            ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

            return ioas
        else:
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
            intersection[union <= 0 + np.finfo('float').eps] = 0
            union[union <= 0 + np.finfo('float').eps] = 1
            ious = intersection / union
            return ious


class MotChallenge2DBox(trackeval.datasets._base_dataset._BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = trackeval.utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # '{gt_folder}/{seq}/gt/gt.txt'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = trackeval.utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        self.gt_set = gt_set
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = ['organoid']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise trackeval.utils.TrackEvalException('Attempted to evaluate an invalid class. Only pedestrian class is valid.')
        # self.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
        #                                'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
        #                                'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
        # self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        # self.seq_list, self.seq_lengths = self._get_seq_info()
        # if len(self.seq_list) < 1:
        #     raise trackeval.utils.TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        # for seq in self.seq_list:
        #     if not self.data_is_zipped:
        #         curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
        #         if not os.path.isfile(curr_file):
        #             print('GT file not found ' + curr_file)
        #             raise trackeval.utils.TrackEvalException('GT file not found for sequence: ' + seq)
        # if self.data_is_zipped:
        #     curr_file = os.path.join(self.gt_fol, 'data.zip')
        #     if not os.path.isfile(curr_file):
        #         print('GT file not found ' + curr_file)
        #         raise trackeval.utils.TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        # if self.config['TRACKERS_TO_EVAL'] is None:
        #     self.tracker_list = os.listdir(self.tracker_fol)
        # else:
        #     self.tracker_list = self.config['TRACKERS_TO_EVAL']

        self.tracker_list = [f'{self.tracker_fol}/bopredict.txt']
        self.seq_list = []
        # if self.config['TRACKER_DISPLAY_NAMES'] is None:
        #     self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        # elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
        #         len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
        #     self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        # else:
        #     raise trackeval.utils.TrackEvalException('List of tracker files and tracker display names do not match.')
        #
        # for tracker in self.tracker_list:
        #     if self.data_is_zipped:
        #         curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
        #         if not os.path.isfile(curr_file):
        #             print('Tracker file not found: ' + curr_file)
        #             raise trackeval.utils.TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
        #     else:
        #         for seq in self.seq_list:
        #             curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
        #             if not os.path.isfile(curr_file):
        #                 print('Tracker file not found: ' + curr_file)
        #                 raise trackeval.utils.TrackEvalException(
        #                     'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
        #                         curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            # If sequence length is 'None' tries to read sequence length from .ini files.
            for seq, seq_length in seq_lengths.items():
                if seq_length is None:
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise trackeval.utils.TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])

        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] is None:
                    seqmap_file = os.path.join(self.config['GT_FOLDER'], 'seqmaps', self.gt_set + '.txt')
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.gt_set + '.txt')
            if not os.path.isfile(seqmap_file):
                print('no seqmap found: ' + seqmap_file)
                raise trackeval.utils.TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == '':
                        continue
                    seq = row[0]
                    seq_list.append(seq)
                    ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                    if not os.path.isfile(ini_file):
                        raise trackeval.utils.TrackEvalException('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        zip_file = None

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(tracker, is_zipped=self.data_is_zipped, zip_file=zip_file)

        # Convert data to required format
        num_timesteps = len(read_data.keys())
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str( t+ 1) for t in range(num_timesteps)]

        for t in range(num_timesteps):
            time_key = str(t+1)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=np.float64)
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])
                    # else:
                    #     raise trackeval.utils.TrackEvalException(
                    #         'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                    #             seq, t))
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 6].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        # raw_data['seq'] = seq
        return raw_data

    # @_timing.time
    def get_preprocessed_seq_data(self, raw_data):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        # self._check_unique_ids(raw_data)

        # distractor_class_names = ['person_on_vehicle', 'static_person', 'distractor', 'reflection']
        # if self.benchmark == 'MOT20':
        #     distractor_class_names.append('non_mot_vehicle')
        # distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        # cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            similarity_scores = raw_data['similarity_scores'][t]

            # Evaluation is ONLY valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise trackeval.utils.TrackEvalException(
                    'Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at '
                    'timestep %i.' % (np.max(tracker_classes), raw_data['seq'], t))

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            # to_remove_tracker = np.array([], np.int32)

            # if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
            #     # Check all classes are valid:
            #     # invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
            #     # if len(invalid_classes) > 0:
            #     #     print(' '.join([str(x) for x in invalid_classes]))
            #     #     raise(trackeval.utils.TrackEvalException('Attempting to evaluate using invalid gt classes. '
            #     #                              'This warning only triggers if preprocessing is performed, '
            #     #                              'e.g. not for MOT15 or where prepropressing is explicitly disabled. '
            #     #                              'Please either check your gt data, or disable preprocessing. '
            #     #                              'The following invalid classes were found in timestep ' + str(t) + ': ' +
            #     #                              ' '.join([str(x) for x in invalid_classes])))
            #
            #     matching_scores = similarity_scores.copy()
            #     matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
            #     match_rows, match_cols = linear_sum_assignment(-matching_scores)
            #     actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
            #     match_rows = match_rows[actually_matched_mask]
            #     match_cols = match_cols[actually_matched_mask]
            #
            #     is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
            #     to_remove_tracker = match_cols[is_distractor_class]

            # Apply preprocessing to remove all unwanted tracker dets.
            # data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            # data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            # data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            # similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)
            # if self.do_preproc and self.benchmark != 'MOT15':
            #     gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
            #                       (np.equal(gt_classes, cls_id))
            # else:
            #     # There are no classes for MOT15
            #     gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            # data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            # data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            # data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['tracker_confidences'][t] = tracker_confidences
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores
            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int32)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int32)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)

        data['num_timesteps'] = raw_data['num_timesteps']
        # data['seq'] = raw_data['seq']

        # Ensure again that ids are unique per timestep after preproc.
        # self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            # logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            print('No ground truth for {}, skipping.'.format(k))

    return accs, names


def mota_eval(predict):
    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    gt_type = '_val_half'
    # gt_type = ''
    # print('gt_type', gt_type)
    # gtfiles = glob.glob(
    #     os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
    # print('gt_files', gtfiles)
    # tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]
    gtfiles = glob.glob('../data/img_216/annotations/MOT/gt.txt')
    tsfiles = glob.glob(f'../data/img_216/annotations/MOT/{predict}')

    # logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    # logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    # logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    # logger.info('Loading files.')

    '''file storage:yourdata/
    --video1/gt/gt.txt          # video1的gt文件
    --video2/gt/gt.txt          # video2的gt文件
    --video1.txt                # video1的test文件
    --video2.txt                # video2的test文件
    '''
    # gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in gtfiles])
    # ts = OrderedDict(
    #     [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in
    #      tsfiles])
    '''file storage:img216/annotations/MOT/
    --gt.txt          # gt文件
    --predict.txt      # test文件
    '''
    gt = OrderedDict([(Path(f).parts[-4], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in gtfiles])
    ts = OrderedDict(
        [(os.path.splitext(Path(f).parts[-4])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in
         tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    # logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters,
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    # logger.info('Completed')


def hota_eval():
    # hota = sqrt(detA*assA)
    # data = {'num_tracker_dets':20,'num_gt_dets':20,'num_gt_ids':20,'num_tracker_ids':20,'similarity_scores':0.5,}
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    # parser = argparse.ArgumentParser()
    # for setting in config.keys():
    #     if type(config[setting]) == list or type(config[setting]) == type(None):
    #         parser.add_argument("--" + setting, nargs='+')
    #     else:
    #         parser.add_argument("--" + setting)
    # args = parser.parse_args().__dict__
    # for setting in args.keys():
    #     if args[setting] is not None:
    #         if type(config[setting]) == type(True):
    #             if args[setting] == 'True':
    #                 x = True
    #             elif args[setting] == 'False':
    #                 x = False
    #             else:
    #                 raise Exception('Command line parameter ' + setting + 'must be True or False')
    #         elif type(config[setting]) == type(1):
    #             x = int(args[setting])
    #         elif type(args[setting]) == type(None):
    #             x = None
    #         elif setting == 'SEQ_INFO':
    #             x = dict(zip(args[setting], [None] * len(args[setting])))
    #         else:
    #             x = args[setting]
    #         config[setting] = x
    config['GT_FOLDER'] = r"D:\Workspace\Organoid_Tracking\tracking_labeled\stomach_cancer_labeled\annotations\MOT"
    config['SKIP_SPLIT_FOL'] = True
    config['GT_LOC_FORMAT'] = '{gt_folder}/gt.txt'
    config['TRACKERS_FOLDER'] = r"D:\Workspace\Organoid_Tracking\organoid_tracking\rtdetrv2_organoid\output\exp_track\rtdetrv2_r50vd_organoid_epoch50_freeze3stage_20250516-162630"
    config['LOG_ON_ERROR'] = 'error.txt'
    config['METRICS'] = ['HOTA']
    config['CLASSES_TO_EVAL'] = ['organoid']

    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}


    # Run code
    # evaluator = trackeval.Evaluator(eval_config)
    data_loader = MotChallenge2DBox(dataset_config)
    hota = HOTA(metrics_config)
    class_list = ['organoid']
    raw_gt_data = data_loader._load_raw_file(r"D:\Workspace\Organoid_Tracking\tracking_labeled\stomach_cancer_labeled\annotations\MOT\gt.txt", is_gt=True)
    raw_tracker_data = data_loader._load_raw_file(r"D:\Workspace\Organoid_Tracking\organoid_tracking\rtdetrv2_organoid\output\exp_track\rtdetrv2_r50vd_organoid_epoch50_freeze3stage_20250516-162630\predict.txt", is_gt=False)
    raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

    # Calculate similarities for each timestep.
    similarity_scores = []
    for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
        ious = data_loader._calculate_similarities(gt_dets_t, tracker_dets_t)
        similarity_scores.append(ious)
    raw_data['similarity_scores'] = similarity_scores
    res_dict = {}
    data = data_loader.get_preprocessed_seq_data(raw_data)
    res_dict['HOTA'] = hota.eval_sequence(data)
    for name in hota.summary_fields:
        if name in hota.float_array_fields:
            print(f'{name}:{"{0:1.5g}".format(100 * np.mean(res_dict["HOTA"][name]))}%')
        elif name in hota.float_fields:
            print(f'{name}:{"{0:1.5g}".format(100 * float(res_dict["HOTA"][name]))}%')
    print(res_dict)

        # res_dict[cls]['Count'] = Count.eval_sequence(data)
    # return res_dict
    # evaluator.evaluate(dataset_list, [HOTA(metrics_config)])


if __name__ == '__main__':
    hota_eval()
    # mota_eval('bypredict75.txt')
