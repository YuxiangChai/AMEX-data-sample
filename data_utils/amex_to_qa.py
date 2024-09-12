import argparse
import json
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def check_create_dir(path: Path) -> Path:
    if not path.is_dir():
        path.mkdir(parents=True)
    return path


def descale_coord(width: int, height: int, x_: int, y_: int) -> tuple:
    max_edge = max((width, height))

    if width < height:
        x_origin = (height - width) // 2
        y_origin = 0
    else:
        x_origin = 0
        y_origin = (width - height) // 2

    x = x_ * max_edge - x_origin
    y = y_ * max_edge - y_origin
    return x, y


def scale_coord(
    x: int, y: int, width: int, height: int, pad_x: int, pad_y: int
) -> tuple:
    x = x + pad_x
    y = y + pad_y
    sx = x / width
    sy = y / height
    return sx, sy


def get_pad_info(width: int, height: int) -> tuple:
    if height > width:
        pad_x = int((height - width) / 2)
        pad_y = 0
        width = height
    else:
        pad_x = 0
        pad_y = int((width - height) / 2)
        height = width

    return pad_x, pad_y, width, height


def parse_l1_qa(root_dir: Path) -> list:
    qa = []
    anno_jsons = sorted(list((root_dir / "element_anno").glob("*.json")))
    for anno in tqdm(anno_jsons, desc="Parsing Level 1 QA"):
        with open(anno, "r") as f:
            data = json.load(f)

        # get image size
        image_name = data["image_path"]
        image = Image.open(root_dir / "screenshot" / image_name)
        image_width, image_height = image.size

        # get element boxes
        clickable_elements = data["clickable_elements"]
        scrollable_elements = data["scrollable_elements"]
        clickable_bboxes = [e["bbox"] for e in clickable_elements]
        scrollable_bboxes = [e["bbox"] for e in scrollable_elements]

        # transform boxes and add to qa
        transformed_clickable_bboxes = []
        for e in clickable_bboxes:
            pad_x, pad_y, pad_width, pad_height = get_pad_info(
                image_width, image_height
            )
            x1, y1 = scale_coord(e[0], e[1], pad_width, pad_height, pad_x, pad_y)
            x2, y2 = scale_coord(e[2], e[3], pad_width, pad_height, pad_x, pad_y)

            transformed_clickable_bboxes.append([x1, y1, x2, y2])

        transformed_scrollable_bboxes = []
        for e in scrollable_bboxes:
            pad_x, pad_y, pad_width, pad_height = get_pad_info(
                image_width, image_height
            )
            x1, y1 = scale_coord(e[0], e[1], pad_width, pad_height, pad_x, pad_y)
            x2, y2 = scale_coord(e[2], e[3], pad_width, pad_height, pad_x, pad_y)

            transformed_scrollable_bboxes.append([x1, y1, x2, y2])

        clickable_answer = ",".join(
            [
                f"[{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]"
                for x1, y1, x2, y2 in transformed_clickable_bboxes
            ]
        )
        scrollable_answer = ",".join(
            [
                f"[{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]"
                for x1, y1, x2, y2 in transformed_scrollable_bboxes
            ]
        )

        qa.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Identify all clickable elements on the screen and provide their 2D bounding boxes in the format of [x1,y1,x2,y2].",
                    },
                    {"from": "gpt", "value": clickable_answer},
                ],
                "image": (root_dir / "screenshot" / image_name).as_posix(),
            }
        )
        qa.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "Identify all scrollable areas on the screen and provide their 2D bounding boxes in the format of [x1,y1,x2,y2].",
                    },
                    {"from": "gpt", "value": scrollable_answer},
                ],
                "image": (root_dir / "screenshot" / image_name).as_posix(),
            }
        )

    return qa


def parse_l2_qa(root_dir: Path) -> list:
    qa = []
    anno_jsons = sorted(list((root_dir / "element_anno").glob("*.json")))
    for anno in tqdm(anno_jsons, desc="Parsing Level 2 QA"):
        with open(anno, "r") as f:
            data = json.load(f)

        # add page description to qa
        if "page_caption" in data:
            page_desc = data["page_caption"]

            qa.append(
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Provide a one-sentence caption for the provided screenshot.",
                        },
                        {"from": "gpt", "value": page_desc},
                    ],
                    "image": (root_dir / "screenshot" / data["image_path"]).as_posix(),
                }
            )

        # get image size
        image_name = data["image_path"]
        image = Image.open(root_dir / "screenshot" / image_name)
        image_width, image_height = image.size

        # get element functionality and add to qa
        clickable_elements = data["clickable_elements"]

        for e in clickable_elements:
            if "functionality" in e and e["functionality"] != "":
                bbox = e["bbox"]
                pad_x, pad_y, pad_width, pad_height = get_pad_info(
                    image_width, image_height
                )
                x1, y1 = scale_coord(
                    bbox[0], bbox[1], pad_width, pad_height, pad_x, pad_y
                )
                x2, y2 = scale_coord(
                    bbox[2], bbox[3], pad_width, pad_height, pad_x, pad_y
                )
                qa.append(
                    {
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"What is the functionality of the element at: [{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]?",
                            },
                            {"from": "gpt", "value": e["functionality"]},
                        ],
                        "image": (root_dir / "screenshot" / image_name).as_posix(),
                    }
                )

    return qa


def parse_swipe_direction(touch: list, lift: list, device_dim: list) -> str:
    x1, y1 = touch
    x2, y2 = lift
    width, height = device_dim

    if abs(x2 - x1) / width > abs(y2 - y1) / height:
        if x2 > x1:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if y2 > y1:
            return "DOWN"
        else:
            return "UP"


def parse_action(action: dict) -> str:
    if action["action"] == "SWIPE":
        direction = parse_swipe_direction(
            action["touch_coord"], action["lift_coord"], action["device_dim"]
        )
        return f"SWIPE[{direction}]"
    elif action["action"] == "TAP":
        pad_x, pad_y, pad_width, pad_height = get_pad_info(
            action["device_dim"][0], action["device_dim"][1]
        )
        x, y = scale_coord(
            action["touch_coord"][0],
            action["touch_coord"][1],
            pad_width,
            pad_height,
            pad_x,
            pad_y,
        )
        return f"CLICK[{x:.4f},{y:.4f}]"
    elif action["action"] == "TYPE":
        return f"TYPE[{action['type_text']}]"
    elif action["action"] == "PRESS_BACK":
        return "PRESS_BACK"
    elif action["action"] == "PRESS_HOME":
        return "PRESS_HOME"
    elif action["action"] == "PRESS_ENTER":
        return "PRESS_ENTER"
    elif action["action"] == "TASK_COMPLETE":
        return "TASK_COMPLETE"
    elif action["action"] == "TASK_IMPOSSIBLE":
        return "TASK_IMPOSSIBLE"


def parse_l3_qa(root_dir: Path) -> list:
    qa = {
        "train": [],
        "test": [],
    }
    anno_jsons = sorted(list((root_dir / "instruction_anno").glob("*.json")))

    base_prompt = "Given the task instruction, please specify the action to take on the current screen."

    for anno in tqdm(anno_jsons, desc="Parsing Level 3 QA"):
        with open(anno, "r") as f:
            data = json.load(f)

        instruction = data["instruction"]

        if instruction.startswith("Open"):
            app_name = re.findall(r"Open (.+?)\.", instruction)[0].lower()
            # print(app_name)
            # exit()
            if app_name in [
                "citymapper",
                "gmail",
                "booking",
                "microsoft to do",
                "yelp",
                "signal",
                "booking",
                "youtube music",
                "shein",
                "nbc news",
            ]:
                subset_split = "test"
            else:
                subset_split = "train"
        else:
            subset_split = "train"

        history_action = []

        for step in data["steps"]:
            action_answer = parse_action(step)

            user_prompt = (
                base_prompt
                + f"\nTask: {instruction}\nHistory Actions: {','.join(history_action)}"
            )

            qa[subset_split].append(
                {
                    "conversations": [
                        {"from": "human", "value": user_prompt},
                        {"from": "gpt", "value": action_answer},
                    ],
                    "image": (root_dir / "screenshot" / step["image_path"]).as_posix(),
                }
            )

            history_action.append(action_answer)
    return qa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory of AMEX dataset, which contains screenshot, element_anno and instruction_anno",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory of QA json files, default to the same as root_dir",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        choices=["all", "l1", "l2", "l3"],
        help="select the level of AMEX to convert to QA, choose from ['all', 'l1', 'l2', 'l3']",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if args.output_dir is None:
        output_dir = root_dir
    else:
        output_dir = Path(args.output_dir)

    if args.level == "all":
        all_qa = []
        l1_qa = parse_l1_qa(root_dir)
        l2_qa = parse_l2_qa(root_dir)
        l3_qa = parse_l3_qa(root_dir)
        all_qa.extend(l1_qa)
        all_qa.extend(l2_qa)
        all_qa.extend(l3_qa["train"])
        test_qa = l3_qa["test"]
    elif args.level == "l1":
        all_qa = parse_l1_qa(root_dir)
        test_qa = None
    elif args.level == "l2":
        all_qa = parse_l2_qa(root_dir)
        test_qa = None
    elif args.level == "l3":
        l3_qa = parse_l3_qa(root_dir)
        all_qa = l3_qa["train"]
        test_qa = l3_qa["test"]

    with open(root_dir / f"{args.level}_train_qa.json", "w") as f:
        json.dump(all_qa, f, indent=4, ensure_ascii=False)

    if test_qa is not None:
        with open(root_dir / f"{args.level}_test_qa.json", "w") as f:
            json.dump(test_qa, f, indent=4, ensure_ascii=False)
