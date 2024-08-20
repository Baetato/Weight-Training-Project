import argparse
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=bool,  # 부울값으로 받도록 수정
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()

def main(args):
    # 모델 로드 및 이미지 처리
    model = FastSAM(args.model_path)
    input = Image.open(args.img_path).convert("RGB")
    everything_results = model(input, device=args.device, retina_masks=args.retina, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    
    # 프롬프트 처리
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
    ann = None
    
    # Box Prompt 처리
    if args.box_prompt and isinstance(args.box_prompt, list) and len(args.box_prompt) > 0:
        if isinstance(args.box_prompt[0], list) and len(args.box_prompt[0]) == 4:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
        else:
            print("Invalid box_prompt format. Skipping box_prompt processing.")

    # Text Prompt 처리
    if ann is None and args.text_prompt:
        ann = prompt_process.text_prompt(text=args.text_prompt)
    
    # Point Prompt 처리
    if ann is None and args.point_prompt and not all(point == [0, 0] for point in args.point_prompt):
        ann = prompt_process.point_prompt(points=args.point_prompt, pointlabel=args.point_label)

    # Default Prompt 처리
    if ann is None:
        ann = prompt_process.everything_prompt()
    
    # 바운딩 박스 추출 및 출력
    if 'bbox' in ann:
        bounding_boxes = ann['bbox']
        print("Bounding Boxes:")
        for bbox in bounding_boxes:
            print(bbox)
    else:
        print("No bounding boxes found.")

    # 바운딩 박스 이미지에 그리기 (옵션)
    prompt_process.plot(
        annotations=ann,
        output_path=args.output + args.img_path.split("/")[-1],
        bboxes=bounding_boxes if 'bbox' in ann else None
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
