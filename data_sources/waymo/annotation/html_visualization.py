import argparse
import glob
import json
import os
import typing

# Number of sentences workers can input in an annotation task.
MAX_NUM_SENTENCES = 10


def parse_args(args=None):
    parser = argparse.ArgumentParser("Create HTML visualization for the annotations")
    parser.add_argument("--annotation-json-dir", type=str, help="Path to jsons containing the workers' responses.")
    parser.add_argument("--video-dir", type=str, help="Path to store the annotation videos.")
    parser.add_argument("--output-html-path", type=str, help="Path prefix to store the output HTML files.")
    parser.add_argument(
        "--num-videos-per-html", type=int, default=250, help="Number of videos to show in one HTML page."
    )

    args = parser.parse_args(args)
    return args


def find_value_by_key(response: dict, key: str) -> typing.Union[str, None]:
    """Find the value with the given key in the response dictionary.

    Parameters
    ----------
    response: dict
        The dictionary containing consolidated worker response.
    key: str
        The target key for the value.

    Returns
    -------
    item; str or None
        The found value string or None if not found.
    """
    if key in response:
        return response[key]
    for value in response.values():
        if type(value) != dict:
            continue
        item = find_value_by_key(value, key)
        if item is not None:
            return item
    return None


def gen_video_tag(video_dir: str, video_name) -> str:
    """Generate the HTML video tag."""
    video_path = os.path.join(video_dir, video_name)
    video_tag = """
        {}<br />
        <video width=\"400\" height=\"400\" controls>
            <source src=\"{}\" type=\"video/mp4\">
            Your browser does not support the video tag.
        </video><br />\n""".format(
        video_name, video_path
    )
    return video_tag


def gen_annotation_block(labels: dict) -> str:
    """Generate the HTML block for the annotation."""
    html = ""
    for s in range(MAX_NUM_SENTENCES):
        s_idx = "s{}".format(s)
        start_idx = "t{}".format(s)
        end_idx = "e{}".format(s)
        if s_idx not in labels or start_idx not in labels or end_idx not in labels:
            break
        line = "{}-{}: {}<br />\n".format(labels[start_idx], labels[end_idx], labels[s_idx])
        html += line
    html += "<br /><hr />"
    return html


def process_responses(args: argparse.Namespace) -> None:
    """Go through all jsons to generate a HTML file that lists videos with annotations.

    Parameters
    ----------
    args: argparse.Namespace
        The parameters for processing the worker response. See parse_args for the list of arguments.
    """
    annotaiton_files = list(glob.glob(args.annotation_json_dir + "*.json"))
    htmls = []
    n_annotations = 0
    for annotation_filename in annotaiton_files:
        with open(annotation_filename, "r") as afile:
            responses = json.load(afile)
            for response in responses:
                html = ""
                video_path = find_value_by_key(response, "s3Uri")
                video_name = os.path.split(video_path)[1]
                labels = find_value_by_key(response, "labels")
                html = gen_video_tag(args.video_dir, video_name)
                html += gen_annotation_block(labels)
                htmls.append(html)
                n_annotations += 1
                if n_annotations % args.num_videos_per_html == 0:
                    html_idx = int(n_annotations / args.num_videos_per_html)
                    html_path = args.output_html_path + "viz_{}.html".format(html_idx)
                    with open(html_path, "w") as fp:
                        fp.write("\n".join(htmls))
                        htmls = []
    # Save to html file
    html_idx = int(n_annotations / args.num_videos_per_html) + 1
    html_path = args.output_html_path + "viz_{}.html".format(html_idx)
    with open(html_path, "w") as fp:
        fp.write("\n".join(htmls))


def main():
    args = parse_args()
    process_responses(args)


if __name__ == "__main__":
    main()
