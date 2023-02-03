from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from train_net_video import *
from detectron2.engine import DefaultTrainer

from PIL import Image 
import numpy as np 

# colors = [0, 0, 0, 255, 0, 0]

class Demo():
    def __init__(self, config_file, weights):  
        cfg = CfgNode()
        cfg = CfgNode(cfg.load_yaml_with_base(config_file))
        model = DefaultTrainer.build_model(cfg)
        model.eval() 
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(weights)
        self.model = model 

    def refer(self, image_path, expression, out_root):
        image = Image.open(image_path)
        aspect_ratio = image.width / image.height
        new_height = 640
        new_width = int(new_height * aspect_ratio)
        image = image.resize((new_width, new_height))
        input = dict() 
        input["image"] = [torch.as_tensor(np.ascontiguousarray(np.array(image).transpose(2, 0, 1))).cuda()]
        input["expression"] = expression
        out = self.model([input])
        print(out["pred_scores"])
        msk = Image.fromarray(out["pred_masks"][0][0].detach().cpu().numpy().astype(np.uint8) * 255)
        over = Image.fromarray(overlay_davis(np.array(image), np.array(msk) // 255))
        
        out_name = image_path.split('/')[-1].split(".")[0] + "_" + expression + ".png"
        out_name = os.path.join(out_root, out_name)
        # msk.save(out_name)
        over.save(out_name)

        del image
        del input
        del out
        del msk
        del over 

        return out_name

def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)
