# Functors for processing annotations.
# Example functor - collecting statistics on PA and annotations:
# python data_sources/pedestrians/process_pedestrian_annotations.py
# --reharvest-dir ~/pedestrian_intent/reharvests_for_annotation/ --annotation-dir
# ~/pedestrian_intent/annotation_results/pedestrian-intent-labeling-small7-2000/
# ~/pedestrian_intent/annotation_results/pedestrian-intent-labeling-small6-1000/
# --annotation-processing-module data_sources.pedestrians.annotation_functor
# --annotation-processing-class AnnotationEvaluationFunctor --augmented-protobufs-dir
# ~/augmented_reharvests --consolidate --output-file ./annotation_evaluations.json
#
#
import copy
import json
import os
import subprocess
from abc import ABC, abstractmethod
from collections import OrderedDict
from html.parser import HTMLParser

import IPython
import matplotlib.pyplot as plt
import numpy as np
from colorama import Back, Fore, Style

from data_sources.pedestrians.pedestrian_annotation_consolidation import canfloat


def to_float(x):
    if type(x) is str and (x.lower() == "o" or x.lower() == "0o"):
        x = 0
    elif type(x) is str and x.lower() == "-":
        x = -1.0
    else:
        x = float(x)
    return x


class AnnotationFunctor(ABC):
    def __init__(self, params):
        self.params = params
        pass

    @abstractmethod
    def process(self, reharvester_jsn, annotation_labels, annotation_video_pathname):
        pass

    @abstractmethod
    def finalize(self):
        pass


def add_count(results, key):
    if key not in results:
        results[key] = 0
    results[key] += 1
    return results


class AnnotationEvaluationFunctor(AnnotationFunctor):
    def __init__(self, params, keyname="q5-a"):
        """
        Get evaluation annotations -- both overall statistics and per-case annotation/intent prediction by PA, and pointers to the tlog and video w/ bounding box.
        :param params:
        :param keyname: the name of the question to use as "is going to cross" - q5-a or q5-c
        """
        super().__init__(params)
        self.keyname = keyname
        self.evaluations = {}
        self.counts = {}

    def process(self, reharvester_jsn, annotation_labels, annotation_video_pathname):

        is_corrupt = annotation_labels["q0-a"]["on"]
        is_misshaped = annotation_labels["q0-b"]["on"]
        is_weird_bbox = annotation_labels["q0-c"]["on"]
        is_too_fast = annotation_labels["q0-d"]["on"]
        is_partial = annotation_labels["q0-f"]["on"]
        is_nonpedestrian = annotation_labels["q0-e"]["on"]
        is_multiple_pedestrians = annotation_labels["q0-g"]["on"]
        is_on_road_edge = annotation_labels["q7"]["on"]
        is_on_crosswalk = annotation_labels["q6"]["on"]
        is_parking_lot = annotation_labels["q10-b-ii"]["on"]
        is_relevant = annotation_labels["q3"]["on"]
        is_uniform_answer = True
        if "std" in annotation_labels[self.keyname] and to_float(annotation_labels[self.keyname]["std"]) > 0:
            is_uniform_answer = False
        if self.keyname not in annotation_labels:
            return
        is_valid = (
            (not is_corrupt)
            and (not is_weird_bbox)
            and (not is_too_fast)
            and (not is_misshaped)
            and (not is_nonpedestrian)
            and is_uniform_answer
            and (is_on_road_edge or is_on_crosswalk)
            and (not is_parking_lot)
            and (not is_multiple_pedestrians)
            and is_relevant
        )
        if is_too_fast:
            add_count(self.counts, "is_too_fast")
        if is_weird_bbox:
            add_count(self.counts, "is_weird_bbox")
        if is_nonpedestrian:
            add_count(self.counts, "is_nonpedestrian")
        if is_misshaped:
            add_count(self.counts, "is_misshaped")
        if is_multiple_pedestrians:
            add_count(self.counts, "is_multiple_pedestrians")
        if not (is_on_road_edge or is_on_crosswalk):
            add_count(self.counts, "not (is_on_road_edge or is_on_crosswalk)")
        if not is_uniform_answer:
            add_count(self.counts, "not_is_uniform_answer")

        if self.keyname in annotation_labels and "crossing_probability" in reharvester_jsn:
            if "matched_examples" not in self.evaluations:
                self.evaluations["matched_examples"] = OrderedDict()
            if "standard_deviations" not in self.evaluations:
                self.evaluations["standard_deviations"] = []
            if "std" in annotation_labels[self.keyname]:
                self.evaluations["standard_deviations"].append(annotation_labels[self.keyname]["std"])
            elif "votes" in annotation_labels[self.keyname] and len(annotation_labels[self.keyname]) > 1:
                std = np.std([float(x) for x in annotation_labels[self.keyname]["votes"] if canfloat(x)])
                self.evaluations["standard_deviations"].append(std)
            if "votes" in annotation_labels[self.keyname]:
                votes = annotation_labels[self.keyname]["votes"]
                for i in range(len(votes)):
                    if votes[i] == "1-":
                        votes[i] = "-1"
                    if votes[i].endswith("sec"):
                        votes[i] = votes[i][:-3]
                    if votes[i].endswith("s"):
                        votes[i] = votes[i][:-1]
                non_negative = [to_float(x) >= 0 for x in votes]
                if np.std(non_negative) > 0.25:
                    is_valid = False

            if "value" in annotation_labels[self.keyname] and is_valid:
                question_value = annotation_labels[self.keyname]["value"]
                if type(question_value) is str and question_value.strip().lower() == "missing":
                    question_value = -1
                question_value = float(question_value)
                crossing_probability = reharvester_jsn["crossing_probability"]
                source_tlog = reharvester_jsn["source_tlog"]
                timestamp = reharvester_jsn["timestamp"]
                identifier = OrderedDict()
                identifier["source_tlog"] = source_tlog
                identifier["timestamp"] = timestamp
                identifier["video"] = annotation_video_pathname
                identifier_str = str(identifier)
                self.evaluations["matched_examples"][identifier_str] = np.array([crossing_probability, question_value])

    def finalize(self):
        matched_examples_arr = np.array(list(self.evaluations["matched_examples"].values()))
        pa_crossing = matched_examples_arr[:, 0] > 0.2
        annotator_crossing = matched_examples_arr[:, 1] >= 0
        # conf_matrix = np.array([[np.sum(pa_crossing*annotator_crossing), np.sum(pa_crossing*(1-annotator_crossing))],
        #                         [np.sum((1-pa_crossing)*annotator_crossing), np.sum((1-pa_crossing)*(1-annotator_crossing))]])
        class_ratio = 1
        conf_matrix = np.array(
            [
                [
                    np.sum(pa_crossing * annotator_crossing),
                    class_ratio * np.sum(pa_crossing * (1 - annotator_crossing)),
                ],
                [
                    np.sum((1 - pa_crossing) * annotator_crossing),
                    class_ratio * np.sum((1 - pa_crossing) * (1 - annotator_crossing)),
                ],
            ]
        )
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)
        precision = conf_matrix[0, 0] / np.sum(conf_matrix[0, :])
        fpr = conf_matrix[0, 1] / np.sum(conf_matrix[:, 1])
        npv = conf_matrix[1, 1] / np.sum(conf_matrix[1, :])
        accuracy_crossing = np.sum(pa_crossing * annotator_crossing) / np.sum(annotator_crossing)
        accuracy_non_crossing = np.sum((1 - pa_crossing) * (1 - annotator_crossing)) / np.sum(1 - annotator_crossing)
        gmean = np.sqrt(accuracy_crossing * accuracy_non_crossing)
        recall = conf_matrix[0, 0] / np.sum(conf_matrix[:, 0])
        f1 = 2.0 / (1.0 / precision + 1.0 / recall)
        fraction_crossing = np.sum(conf_matrix[:, 0]) / np.sum(conf_matrix)
        # plt.plot(matched_examples_arr[:,0],matched_examples_arr[:,1],'.')
        print("Confusion matrix: " + str(conf_matrix))
        print("fraction crossing = " + str(fraction_crossing))
        print("accuracy = " + str(accuracy))
        print("precision = " + str(precision))
        print("npv = " + str(npv))
        print("fpr = " + str(fpr))
        print("recall = " + str(recall))
        print("g-mean = " + str(gmean))
        print("f1 score = " + str(f1))
        for key in self.counts:
            print(key + " = " + str(self.counts[key]))
        if self.params["output_file"] is not None:

            examples = copy.deepcopy(self.evaluations["matched_examples"])
            for key in examples:
                examples[key] = examples[key].tolist()

            with open(self.params["output_file"], "w") as fp:
                json.dump(examples, fp, indent=2)


class AnnotationStatisticsFunctor(AnnotationFunctor):
    """
    Get statistics of annotations -- to evaluate how sure are the annotators and where they fail.
    """

    def __init__(self, params):
        super().__init__(params)
        self.evaluations = {}

    def process(self, reharvester_jsn, annotation_labels, annotation_video_pathname):
        for key in annotation_labels:
            if key not in self.evaluations:
                self.evaluations[key] = {"standard_deviations": []}
            if "std" in annotation_labels[key]:
                self.evaluations[key]["standard_deviations"].append(annotation_labels[key]["std"])
            elif "votes" in annotation_labels[key] and len(annotation_labels[key]) > 1:
                self.evaluations[key]["standard_deviations"].append(
                    np.std([float(x) for x in annotation_labels[key]["votes"] if canfloat(x)])
                )

    def finalize(self):
        # plot the standard deviation of different questions
        standard_deviations = []
        keys = sorted(list(self.evaluations.keys()))
        for key in keys:
            standard_deviations.append(self.evaluations[key]["standard_deviations"])
            print(key + ": " + str(np.mean([x > 0 for x in self.evaluations[key]["standard_deviations"]])))

        plt.boxplot(standard_deviations)
        plt.xticks(range(len(standard_deviations)), list(self.evaluations.keys()))
        plt.show()
        import IPython

        IPython.embed(header="finalize")


class QueryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.questions = {}
        self.intermediate_tag = None
        self.intermediate_attrs = None
        self.intermediate_data = None

    def handle_starttag(self, tag, attrs):
        self.intermediate_tag = tag
        self.intermediate_attrs = dict(attrs)

    def handle_endtag(self, tag):
        if tag == "crowd-checkbox":
            self.questions[self.intermediate_attrs["name"]] = {"data": self.intermediate_data, "type": "checkbox"}
        elif tag == "crowd-text-area":
            self.questions[self.intermediate_attrs["name"]] = {"data": self.intermediate_data, "type": "open"}
        self.intermediate_tag = None

    def handle_data(self, data):
        self.intermediate_data = data


class VisualizationFunctor(object):
    def __init__(self, params):
        super().__init__(params)
        self.annotation_query_filename = params["annotation_query"]
        self.parser = QueryParser()
        if not self.annotation_query_filename == "" and self.annotation_query_filename is not None:
            with open(self.annotation_query_filename, "r") as myfile:
                data = myfile.read().replace("\n", "")

                self.annotation = self.parser.feed(data)
                self.questions = self.parser.questions
        else:
            raise (Exception("Annotation query missing."))

    def form_answer_string(self, annotation_labels, name, attr):
        if name in annotation_labels:
            if attr is not None:
                return str(annotation_labels[name][attr])
            else:
                return str(annotation_labels[name])
        else:
            return "missing"

    def print_question(self, annotation_labels, name, attr):
        if name in annotation_labels:
            answer = self.form_answer_string(annotation_labels, name, attr)
        else:
            answer = "MISSING"
        print(
            Fore.GREEN
            + Style.BRIGHT
            + "Question "
            + name
            + ": {}\n{}".format(self.questions[name]["data"], Fore.RED + Style.BRIGHT + answer)
        )
        print(Style.RESET_ALL)

    def process(self, reharvester_jsn, annotation_labels, annotation_video_pathname, annotation_query_filename):

        pedestrian_intent_vector = reharvester_jsn["pedestrian_intent_vector"]
        pedestrian_awareness_vector = reharvester_jsn["pedestrian_awareness_vector"]
        vlc_proc = subprocess.Popen(["vlc", "--quiet", annotation_video_pathname])
        print("annotation_video_pathname: {}".format(str(annotation_video_pathname)))
        print("pedestrian_intent_vector: {}".format(str(pedestrian_intent_vector)))
        print("pedestrian_awareness_vector: {}".format(str(pedestrian_awareness_vector)))
        source_tlog = reharvester_jsn["source_tlog"]
        if not os.path.exists(source_tlog):
            print("tlog missing: " + os.path.exists(source_tlog))
        for name in ["q5-b", "q5-d", "q6", "q7", "q8", "q9"]:
            self.print_question(annotation_labels, name, "on")
        for name in ["q5-a", "q5-c"]:
            self.print_question(annotation_labels, name, None)

        IPython.embed(header="Introspect or use ctrl+D to skip. VLC window will close once you continue.")
        vlc_proc.terminate()

    def finalize(self):
        pass
