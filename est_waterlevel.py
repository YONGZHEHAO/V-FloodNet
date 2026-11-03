import argparse
import os
from glob import glob
import torch

def get_parser():
    parser = argparse.ArgumentParser(description='Estimate Water Level')
    parser.add_argument('--test-name', type=str, required=True,
                        help='Name of the test video')
    parser.add_argument('--test-path', type=str, required=True,
                        help='Input image directory.')
    parser.add_argument('--out-dir', default='output/waterlevel',
                        help='A file or directory to save output results.')
    parser.add_argument('--opt', type=str, required=True, choices=['people', 'stopsign', 'ref'],
                        help='Estimation options. "people", "stopsign", or "ref"')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Object detector device. Human mesh estimation still requires CUDA.')
    parser.add_argument('--sign-size-cm', type=float, default=79.0,
                        help='Actual stop sign plate size in cm; upstream default is 79 cm.')
    parser.add_argument('--pole-height-cm', type=float, default=215.9,
                        help='Actual pole height in cm; upstream default is 215.9 cm.')

    return parser.parse_args()


def main(args):

    args.water_mask_dir = os.path.join('./output/segs/', args.test_name, 'mask')
    img_list = sorted(glob(os.path.join(args.test_path, '*.jpg')) + glob(os.path.join(args.test_path, '*.png')))
    water_mask_list = sorted(glob(os.path.join(args.water_mask_dir, '*.png')))
    if not img_list:
        raise ValueError(f'No jpg/png images found in {args.test_path}')
    masks_by_name = {os.path.splitext(os.path.basename(path))[0]: path for path in water_mask_list}
    image_names = [os.path.splitext(os.path.basename(path))[0] for path in img_list]
    if len(set(image_names)) != len(image_names):
        raise ValueError('Image names must be unique, including across jpg and png extensions.')
    missing = [name for name in image_names if name not in masks_by_name]
    if missing:
        raise ValueError(f'Missing water masks for {missing}. Run test_image_seg.py with the same test-name first.')
    water_mask_list = [masks_by_name[name] for name in image_names]
    out_dir = os.path.join(args.out_dir, f'{args.test_name}_{args.opt}')
    os.makedirs(out_dir, exist_ok=True)

    record_dir = './records/groundtruth'

    if args.opt in ['people', 'stopsign']:
        device = getattr(args, 'device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but no CUDA device is available.')
        if args.opt == 'people' and device != 'cuda':
            raise RuntimeError('The upstream human mesh pipeline requires CUDA, MeshTransformer and licensed SMPL files. Use the stop sign CPU route or a GPU environment.')
        from estimation.object_detection import est_by_obj_detection, stopsign_meta
        sign_size_cm = getattr(args, 'sign_size_cm', 79.0)
        pole_height_cm = getattr(args, 'pole_height_cm', 215.9)
        if sign_size_cm <= 0 or pole_height_cm <= 0:
            raise ValueError('Reference dimensions must be positive, in cm.')
        stopsign_meta.update(size=sign_size_cm, pole_height=pole_height_cm)
        est_by_obj_detection(img_list, water_mask_list, out_dir, args.opt, device=device)
    elif args.opt == 'ref':
        from estimation.reference_tracking import est_by_reference
        est_by_reference(img_list, water_mask_list, out_dir, record_dir, args.test_name)
    else:
        raise NotImplementedError(args.opt)

if __name__ == '__main__':

    _args = get_parser()
    print(_args)

    main(_args)
