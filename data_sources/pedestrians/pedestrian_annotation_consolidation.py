import json

import numpy as np


def canfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def tofloat(value):
    try:
        return float(value)
    except ValueError:
        if value.lower() == "missing" or value.lower() == "-":
            value = -1.0
        elif value.lower() == "o":
            value = 0.0
        else:
            raise ValueError(str(value))
    return value


def consolidate_annotations(jsn, timing_questions=["q5-a", "q5-c"], open_text=["q2-notes"]):
    aggregated = {}
    workers = set()
    standard_deviations = []
    for jsn_entry in jsn:
        ann = jsn_entry["consolidatedAnnotation"]["content"]
        task_id = list(ann.keys())[0]
        image_id = ann[task_id]["imageSource"]["s3Uri"]
        if image_id not in aggregated:
            response = {}
            response["imageSource"] = ann[task_id]["imageSource"]
            response["labels"] = {}
        else:
            response = aggregated[image_id]
        worker_id = ann[task_id]["workerId"]
        if worker_id not in workers:
            workers.add(worker_id)
        labels = ann[task_id]["labels"]
        for question_key in labels:
            if question_key not in response["labels"]:
                response["labels"][question_key] = {"votes": [], "voters": []}
            response["labels"][question_key]["votes"].append(labels[question_key])
            response["labels"][question_key]["voters"].append(worker_id)
        for question_key in timing_questions:
            if question_key not in response["labels"]:
                response["labels"][question_key] = {"votes": [], "voters": []}
            if question_key not in labels:
                response["labels"][question_key]["votes"].append("-1.0")
                response["labels"][question_key]["voters"].append(worker_id)

        aggregated[image_id] = response

    for image_id in aggregated:
        response = aggregated[image_id]
        for question_key in list(set(list(response["labels"]) + timing_questions)):
            if (
                type(response["labels"][question_key]["votes"][0]) is dict
                and "on" in response["labels"][question_key]["votes"][0]
            ):
                votes = [float(x["on"]) for x in response["labels"][question_key]["votes"]]
                majority_vote = np.bincount(np.array(votes, dtype=np.int64)).argmax()
                standard_deviation = np.std(votes)
                standard_deviations.append(standard_deviation)
                response["labels"][question_key]["on"] = np.bool(majority_vote)
                response["labels"][question_key]["std"] = standard_deviation
            elif question_key in timing_questions:
                values = response["labels"][question_key]["votes"]
                validity = [(type(x) is str and x.lower() in ["missing", "o", "-"]) or canfloat(x) for x in values]
                majority_vote_validity = np.bincount(np.array(validity, dtype=np.int64)).argmax()
                valid_values = [tofloat(x) for x, y in zip(values, validity) if y]
                standard_deviation = np.std(np.sign(np.array(valid_values) + 1e-5))
                # import IPython;
                # IPython.embed(header='check votes')
                value = -1.0
                if len(valid_values) == 0:
                    import IPython

                    IPython.embed(header="check valid values")
                if not (np.sum(validity)) == len(valid_values):
                    pass
                if min(valid_values) < 0 and max(valid_values) >= 0:
                    import IPython

                    IPython.embed(header="check votes")
                if any(validity):
                    value = np.median(valid_values)

                response["labels"][question_key]["value"] = str(value)
                response["labels"][question_key]["std"] = str(standard_deviation)
            elif question_key in open_text:
                response["labels"][question_key]["value"] = response["labels"][question_key]["votes"]
            else:
                response["labels"][question_key]["value"] = response["labels"][question_key]["votes"][0]
        aggregated[image_id] = response
    # import IPython;IPython.embed()
    stats = {"standard_deviations": standard_deviations}
    return aggregated, stats
