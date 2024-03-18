import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='v5', help='dataset.yaml path')
    parser.add_argument('--save_dir', type=str, default='saved_patches/', help='save')
    parser.add_argument('--patch_name', type=str, default='patch_v3.jpg', help='name')
    opt = parser.parse_args()

    return opt

opt = parse_opt()
save_path = opt.save_dir
save_p_img_batch = os.path.join(
                    save_path,
                    opt.patch_name)
print(save_p_img_batch)
